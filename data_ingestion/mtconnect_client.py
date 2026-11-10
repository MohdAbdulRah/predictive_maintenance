"""Thin, defensive HTTP client for an MTConnect agent (probe / current / sample)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from lxml import etree

log = logging.getLogger(__name__)


def _local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


class AgentError(Exception):
    """Agent answered with an MTConnectError document."""
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


@dataclass
class StreamsHeader:
    instance_id: int | None
    first_sequence: int | None
    last_sequence: int | None
    next_sequence: int | None
    buffer_size: int | None
    creation_time: str | None


@dataclass
class RawObservation:
    device_uuid: str | None
    data_item_id: str
    sequence: int
    timestamp: datetime
    category: str            # SAMPLE | EVENT | CONDITION (from the Streams doc)
    element: str             # e.g. Position, Execution, Normal, Fault
    value: str | None        # None when absent
    unavailable: bool
    condition_level: str | None = None   # NORMAL/WARNING/FAULT/UNAVAILABLE for CONDITION
    native_code: str | None = None
    native_severity: str | None = None
    condition_text: str | None = None


def parse_ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_streams(xml_bytes: bytes) -> tuple[StreamsHeader, list[RawObservation]]:
    root = etree.fromstring(xml_bytes)
    if _local(root.tag) == "MTConnectError":
        # MTConnect 1.x: <Error errorCode="...">message</Error>
        # MTConnect 2.x: <Errors><OutOfRange errorCode="..."><ErrorMessage>msg</ErrorMessage></OutOfRange></Errors>
        for e in root.iter():
            local = _local(e.tag)
            if local == "Error" and e.get("errorCode"):
                raise AgentError(e.get("errorCode", "UNKNOWN"), (e.text or "").strip())
            if local not in ("MTConnectError", "Header", "Errors") and e.get("errorCode"):
                # 2.x element like OutOfRange, InvalidRequest, etc.
                msg_el = next((c for c in e if _local(c.tag) == "ErrorMessage"), None)
                msg = (msg_el.text or "").strip() if msg_el is not None else (e.text or "").strip()
                raise AgentError(e.get("errorCode", "UNKNOWN"), msg)
        raise AgentError("UNKNOWN", "MTConnectError without Error element")

    hdr = StreamsHeader(None, None, None, None, None, None)
    obs: list[RawObservation] = []
    for el in root:
        if _local(el.tag) == "Header":
            hdr = StreamsHeader(
                _int(el.get("instanceId")), _int(el.get("firstSequence")),
                _int(el.get("lastSequence")), _int(el.get("nextSequence")),
                _int(el.get("bufferSize")), el.get("creationTime"))
    for cat_el in root.iter():
        name = _local(cat_el.tag)
        if name not in ("Samples", "Events", "Condition"):
            continue
        category = {"Samples": "SAMPLE", "Events": "EVENT", "Condition": "CONDITION"}[name]
        dev_uuid = None
        parent = cat_el.getparent()
        while parent is not None:
            if _local(parent.tag) == "DeviceStream":
                dev_uuid = parent.get("uuid") or parent.get("name")
                break
            parent = parent.getparent()
        for o in cat_el:
            tag = _local(o.tag)
            did, seq, ts = o.get("dataItemId"), _int(o.get("sequence")), o.get("timestamp")
            if did is None or seq is None or ts is None:
                continue   # malformed observation: skip, never invent
            text = (o.text or "").strip()
            unavailable = text.upper() == "UNAVAILABLE"
            level = None
            if category == "CONDITION":
                level = tag.upper()
                unavailable = level == "UNAVAILABLE"
            obs.append(RawObservation(
                device_uuid=dev_uuid, data_item_id=did, sequence=seq, timestamp=parse_ts(ts),
                category=category, element=tag, value=None if unavailable or text == "" else text,
                unavailable=unavailable, condition_level=level,
                native_code=o.get("nativeCode"), native_severity=o.get("nativeSeverity"),
                condition_text=text or None if category == "CONDITION" else None))
    return hdr, obs


class MTConnectClient:
    def __init__(self, base_url: str, timeout: float = 10.0, max_retries: int = 5,
                 session: requests.Session | None = None):
        self.base = base_url.rstrip("/")
        self.timeout, self.max_retries = timeout, max_retries
        self.s = session or requests.Session()

    def _get(self, path: str, params: dict | None = None) -> bytes:
        delay, last = 1.0, None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.s.get(f"{self.base}/{path.lstrip('/')}", params=params, timeout=self.timeout)
                # MTConnect errors can arrive with 4xx and an XML body: return body so caller can parse it
                if r.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                return r.content
            except (requests.RequestException,) as e:
                last = e
                log.warning("GET %s failed (attempt %d/%d): %s", path, attempt, self.max_retries, e)
                time.sleep(min(delay, 30)); delay *= 2
        raise ConnectionError(f"Agent unreachable after {self.max_retries} attempts: {last}")

    def probe_raw(self, device: str | None = None) -> bytes:
        return self._get(f"{device}/probe" if device else "probe")

    def current_raw(self, device: str | None = None, at: int | None = None) -> bytes:
        p = {"at": at} if at is not None else None
        return self._get(f"{device}/current" if device else "current", p)

    def sample_raw(self, device: str | None = None, start: int | None = None, count: int = 1000) -> bytes:
        p = {"count": count}
        if start is not None:
            p["from"] = start
        return self._get(f"{device}/sample" if device else "sample", p)

    def current(self, device=None, at=None):
        return parse_streams(self.current_raw(device, at))

    def sample(self, device=None, start=None, count=1000):
        return parse_streams(self.sample_raw(device, start, count))
