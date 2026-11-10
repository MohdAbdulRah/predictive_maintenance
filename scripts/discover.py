"""Phase 0 - inspect the REAL agent before modelling.

    python -m scripts.discover --url https://demo.mtconnect.org            # live agent
    python -m scripts.discover --probe probe.xml --current current.xml      # saved files

Writes artifacts/discovery_report.json and prints a human summary. Paste the summary back
to your assistant so the config/state mapping is based on facts, not assumptions."""
import argparse, collections, json
from pathlib import Path

from config import ROOT
from data_ingestion.mtconnect_client import MTConnectClient, parse_streams
from data_ingestion.probe_parser import parse_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url"); ap.add_argument("--probe"); ap.add_argument("--current"); ap.add_argument("--sample")
    ap.add_argument("--device"); ap.add_argument("--save-raw", action="store_true")
    a = ap.parse_args()
    if a.url:
        c = MTConnectClient(a.url)
        probe_b, cur_b = c.probe_raw(a.device), c.current_raw(a.device)
        samp_b = c.sample_raw(a.device, count=500)
        if a.save_raw:
            for n, b in (("probe", probe_b), ("current", cur_b), ("sample", samp_b)):
                (ROOT / "artifacts" / f"{n}.xml").write_bytes(b)
    else:
        probe_b = Path(a.probe).read_bytes()
        cur_b = Path(a.current).read_bytes() if a.current else None
        samp_b = Path(a.sample).read_bytes() if a.sample else None

    parsed = parse_probe(probe_b)
    items = parsed["data_items"]
    rep = {"agent_header": parsed["header"], "devices": parsed["devices"],
           "counts_by_category": dict(collections.Counter(i.category for i in items)),
           "data_items": [i.to_dict() for i in items]}
    by_type = collections.defaultdict(list)
    for i in items:
        by_type[(i.category, i.type)].append(i)
    print(f"Devices: {[d['name'] for d in parsed['devices']]}")
    print(f"DataItems: {len(items)}  {rep['counts_by_category']}\n")
    for (cat, typ), lst in sorted(by_type.items()):
        units = sorted({i.units or '-' for i in lst})
        print(f"{cat:9s} {typ:28s} n={len(lst):3d} units={units}  e.g. id={lst[0].id} @ {lst[0].component_path}")
    for label, b in (("current", cur_b), ("sample", samp_b)):
        if not b: continue
        h, obs = parse_streams(b)
        rep[f"{label}_header"] = h.__dict__
        un = sum(o.unavailable for o in obs)
        print(f"\n/{label}: {len(obs)} observations, {un} UNAVAILABLE; seq {h.first_sequence}..{h.last_sequence} next={h.next_sequence} "
              f"buffer={h.buffer_size} instanceId={h.instance_id}")
        ex = collections.Counter(o.value for o in obs if o.element.upper() == "EXECUTION")
        if ex:
            rep[f"{label}_execution_values"] = dict(ex); print("EXECUTION values seen:", dict(ex))
        rep[f"{label}_numeric_ids_with_data"] = sorted({o.data_item_id for o in obs if o.category == "SAMPLE" and not o.unavailable})
    out = ROOT / "artifacts" / "discovery_report.json"
    out.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
