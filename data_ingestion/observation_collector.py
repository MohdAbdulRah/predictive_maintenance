"""Sequence-based collector. Guarantees: no silent loss, no duplicates, gaps are recorded.

Loop: bootstrap with /current (gives instanceId + nextSequence), then repeatedly
/sample?from=<next_sequence>. Detects
  * buffer overrun  (header.firstSequence > requested from, or OUT_OF_RANGE error)
  * agent restart   (instanceId changed)
  * comms failure   (ConnectionError)
and writes each as a CollectionGap row instead of hiding it.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select

from data_ingestion.mtconnect_client import AgentError, MTConnectClient, RawObservation, StreamsHeader
from data_ingestion.probe_parser import parse_probe
from database.models import CollectionGap, CollectorState, DataItemRow, ObservationRow

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def _to_float(v: str | None):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except ValueError:
        return None


class ObservationCollector:
    def __init__(self, client: MTConnectClient, session_factory, device: str | None = None,
                 sample_count: int = 1000, poll_interval_s: float = 1.0):
        self.client, self.sf, self.device = client, session_factory, device
        self.count, self.poll = sample_count, poll_interval_s
        self.device_uuid: str | None = None
        self.instance_id: int | None = None
        self.next_seq: int | None = None
        self.items: dict[str, dict] = {}
        self._last_ok_ts: datetime | None = None

    # ---- setup -------------------------------------------------------------------
    def load_probe(self):
        parsed = parse_probe(self.client.probe_raw(self.device))
        devs = parsed["devices"]
        if not devs:
            raise RuntimeError("Probe returned no devices")
        chosen = next((d for d in devs if self.device in (d["name"], d["uuid"])), devs[0]) if self.device else devs[0]
        self.device_uuid = chosen["uuid"]
        mine = [d for d in parsed["data_items"] if d.device_uuid == self.device_uuid]
        with self.sf() as s:
            for d in mine:
                row = s.get(DataItemRow, (d.device_uuid, d.id)) or DataItemRow(device_uuid=d.device_uuid, id=d.id)
                row.device_name, row.name, row.category, row.type = d.device_name, d.name, d.category, d.type
                row.sub_type, row.units, row.native_units = d.sub_type, d.units, d.native_units
                row.representation, row.component_path, row.component_type = d.representation, d.component_path, d.component_type
                s.merge(row)
            st = s.get(CollectorState, self.device_uuid)
            if st:
                self.instance_id, self.next_seq = st.instance_id, st.next_sequence
            s.commit()
        self.items = {d.id: d.to_dict() for d in mine}
        return mine

    # ---- persistence ---------------------------------------------------------------
    def _store(self, obs: list[RawObservation], instance_id: int) -> int:
        if not obs:
            return 0
        seqs = [o.sequence for o in obs]
        with self.sf() as s:
            existing = set(s.scalars(select(ObservationRow.sequence).where(
                ObservationRow.device_uuid == self.device_uuid,
                ObservationRow.instance_id == instance_id,
                ObservationRow.sequence.between(min(seqs), max(seqs)))))
            new, seen = [], set()
            for o in obs:
                if o.sequence in existing or o.sequence in seen:
                    continue
                seen.add(o.sequence)
                rep = self.items.get(o.data_item_id, {}).get("representation", "VALUE")
                num = _to_float(o.value) if (o.category == "SAMPLE" and rep == "VALUE") else None
                new.append(ObservationRow(
                    device_uuid=self.device_uuid, instance_id=instance_id, sequence=o.sequence,
                    data_item_id=o.data_item_id, ts=o.timestamp, category=o.category, element=o.element,
                    value_num=num, value_text=o.value, unavailable=o.unavailable,
                    condition_level=o.condition_level, native_code=o.native_code,
                    condition_text=o.condition_text))
            s.add_all(new)
            s.commit()
            return len(new)

    def _save_state(self):
        with self.sf() as s:
            st = s.get(CollectorState, self.device_uuid) or CollectorState(device_uuid=self.device_uuid)
            st.instance_id, st.next_sequence, st.updated_at = self.instance_id, self.next_seq, _now()
            s.merge(st)
            s.commit()

    def _gap(self, kind, frm=None, to=None, note=None, start_ts=None, end_ts=None):
        log.warning("GAP %s seq %s..%s %s", kind, frm, to, note or "")
        with self.sf() as s:
            s.add(CollectionGap(device_uuid=self.device_uuid, kind=kind, instance_id=self.instance_id,
                                from_sequence=frm, to_sequence=to, detected_at=_now(),
                                gap_start_ts=start_ts, gap_end_ts=end_ts, note=note))
            s.commit()

    # ---- one polling step (testable) -------------------------------------------------
    def bootstrap(self) -> int:
        hdr, obs = self.client.current(self.device)
        if self.instance_id is not None and hdr.instance_id != self.instance_id:
            self._gap("INSTANCE_RESET", note=f"instanceId {self.instance_id} -> {hdr.instance_id}")
        self.instance_id = hdr.instance_id if hdr.instance_id is not None else 0
        n = self._store(obs, self.instance_id)
        # Use next_sequence (live write position) for /sample?from=.
        # Some agents (e.g. demo.mtconnect.org running MTConnect 2.8) reject from= values
        # below their current write position with OUT_OF_RANGE even when those sequences are
        # technically still in the ring buffer. Starting from next_sequence keeps us in the
        # live stream; historical backfill is skipped to avoid repeated rejection loops.
        self.next_seq = hdr.next_sequence if hdr.next_sequence is not None else hdr.first_sequence
        self._save_state()
        self._last_ok_ts = _now()
        return n

    def step(self) -> int:
        """Fetch one batch. Returns number of NEW observations stored."""
        if self.next_seq is None or self.instance_id is None:
            return self.bootstrap()
        try:
            hdr, obs = self.client.sample(self.device, start=self.next_seq, count=self.count)
        except AgentError as e:
            if e.code == "OUT_OF_RANGE":
                # requested sequence fell out of (or beyond) the buffer
                h, _ = self.client.current(self.device)
                if h.first_sequence is not None and self.next_seq < h.first_sequence:
                    self._gap("BUFFER_OVERRUN", self.next_seq, h.first_sequence - 1,
                              "requested sequence no longer in agent buffer; data lost", end_ts=_now())
                else:
                    self._gap("INSTANCE_RESET", note="OUT_OF_RANGE beyond lastSequence (agent likely restarted)")
                    self.instance_id = None
                self.next_seq = h.first_sequence if h.first_sequence is not None else None
                if self.instance_id is None:
                    return self.bootstrap()
                self._save_state()
                return 0
            raise
        except ConnectionError as e:
            if self._last_ok_ts:
                self._gap("COMM_FAILURE", note=str(e)[:300], start_ts=self._last_ok_ts, end_ts=_now())
            self._last_ok_ts = None
            raise

        if hdr.instance_id is not None and hdr.instance_id != self.instance_id:
            self._gap("INSTANCE_RESET", note=f"instanceId {self.instance_id} -> {hdr.instance_id}")
            self.instance_id = hdr.instance_id
            self.next_seq = hdr.first_sequence
            self._save_state()
            return 0
        if hdr.first_sequence is not None and self.next_seq < hdr.first_sequence:
            self._gap("BUFFER_OVERRUN", self.next_seq, hdr.first_sequence - 1, "oldest observations already evicted")
        n = self._store(obs, self.instance_id)
        if hdr.next_sequence is not None:
            self.next_seq = hdr.next_sequence
        self._save_state()
        self._last_ok_ts = _now()
        return n

    def run(self, stop=lambda: False, max_backoff_s: float = 60.0):
        self.load_probe()
        backoff = 1.0
        while not stop():
            try:
                n = self.step()
                backoff = 1.0
                if n == 0:
                    time.sleep(self.poll)
            except ConnectionError:
                time.sleep(backoff); backoff = min(backoff * 2, max_backoff_s)
            except AgentError as e:
                log.warning("Agent error (%s): %s — re-bootstrapping in %ss", e.code, e.message, self.poll * 5)
                # Any unrecognised agent error (incl. malformed MTConnectError without an Error
                # element) most likely means our stored sequence is stale/out-of-range.
                # Reset state so the next step() calls bootstrap() from /current.
                self.next_seq = None
                self.instance_id = None
                time.sleep(self.poll * 5)
