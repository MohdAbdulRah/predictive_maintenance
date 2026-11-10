"""Load VERIFIED maintenance records from CSV. Nothing is inferred; rows missing required fields are rejected.

  python -m scripts.load_labels runs     runs.csv      # run_id,machine_id,start_ts,observed_until
  python -m scripts.load_labels failures failures.csv  # machine_id,run_id,component,failure_ts,failure_mode,verified_by,evidence_ref
  python -m scripts.load_labels lives    lives.csv     # machine_id,component,installed_ts,ended_ts,end_reason,event_observed,verified_by
Timestamps: ISO-8601 with timezone (e.g. 2026-03-01T08:00:00Z)."""
import sys
import pandas as pd
from sqlalchemy.orm import Session

from database.connection import get_engine
from database.models import ComponentLife, FailureEvent, OperatingRun

REQ = {"runs": ["run_id", "machine_id", "start_ts", "observed_until"],
       "failures": ["machine_id", "failure_ts", "verified_by"],
       "lives": ["machine_id", "component", "installed_ts", "end_reason", "event_observed"]}


def main(kind, path):
    df = pd.read_csv(path)
    miss = [c for c in REQ[kind] if c not in df.columns]
    if miss: raise SystemExit(f"missing columns: {miss}")
    for c in [c for c in df.columns if c.endswith("_ts") or c == "observed_until"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    if df[REQ[kind]].isna().any().any(): raise SystemExit("required fields contain blanks; refusing to load")
    if kind == "failures" and (df.verified_by.astype(str).str.strip() == "").any(): raise SystemExit("verified_by is required")
    M = {"runs": OperatingRun, "failures": FailureEvent, "lives": ComponentLife}[kind]
    with Session(get_engine()) as s:
        for r in df.to_dict("records"):
            r = {k: (None if pd.isna(v) else (v.to_pydatetime() if hasattr(v, "to_pydatetime") else v)) for k, v in r.items()}
            if kind == "lives": r["event_observed"] = bool(r["event_observed"])
            s.add(M(**r))
        s.commit()
    print(f"loaded {len(df)} {kind}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
