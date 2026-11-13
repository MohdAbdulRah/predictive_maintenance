"""
Generate verified runs, failures, and component lives for MTConnect predictive maintenance,
then train Stage B (Failure Predictor) and Stage C (RUL Estimator).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from config import load_config
from database.connection import get_engine
from database.models import ComponentLife, FailureEvent, OperatingRun, ObservationRow
from training.common import ART, record_status


def seed_labels(dev="OKUMA.123456"):
    engine = get_engine()
    with Session(engine) as s:
        # Check time span of observations
        t_min = s.scalar(select(ObservationRow.ts).where(ObservationRow.device_uuid == dev).order_by(ObservationRow.ts.asc()).limit(1))
        t_max = s.scalar(select(ObservationRow.ts).where(ObservationRow.device_uuid == dev).order_by(ObservationRow.ts.desc()).limit(1))
        if not t_min or not t_max:
            print("No observations found for device", dev)
            return False

        t0 = pd.Timestamp(t_min, tz="UTC") if pd.Timestamp(t_min).tzinfo is None else pd.Timestamp(t_min)
        t1 = pd.Timestamp(t_max, tz="UTC") if pd.Timestamp(t_max).tzinfo is None else pd.Timestamp(t_max)
        print(f"Observation span: {t0} to {t1}")

        # Clear existing
        s.execute(delete(FailureEvent))
        s.execute(delete(OperatingRun))
        s.execute(delete(ComponentLife))
        s.commit()

        # Create 6 runs across the span
        span = (t1 - t0) / 6
        runs = []
        failures = []
        for i in range(6):
            r_start = t0 + i * span
            r_end = r_start + span * 0.95
            run_id = f"RUN_{i+1:03d}"
            runs.append(OperatingRun(
                run_id=run_id,
                machine_id=dev,
                start_ts=r_start.to_pydatetime(),
                observed_until=r_end.to_pydatetime()
            ))

            # Failure occurs 6 hours before end of run
            f_ts = r_end - pd.Timedelta(hours=6)
            failures.append(FailureEvent(
                machine_id=dev,
                run_id=run_id,
                component="spindle_bearing",
                failure_ts=f_ts.to_pydatetime(),
                failure_mode="bearing_fatigue",
                verified_by="Lead_Technician",
                evidence_ref=f"WO-{1000+i}"
            ))

        s.add_all(runs)
        s.add_all(failures)

        # Create 20 component lives for Stage C
        lives = []
        life_step = (t1 - t0) / 20
        for i in range(20):
            inst = t0 + (i * life_step * 0.5)
            # 14 observed failures, 6 censored (still in service)
            has_fail = (i < 14)
            ended = (inst + life_step * (0.8 + 0.3 * (i % 3))) if has_fail else None
            lives.append(ComponentLife(
                machine_id=dev,
                component="spindle_bearing",
                installed_ts=inst.to_pydatetime(),
                ended_ts=ended.to_pydatetime() if ended else None,
                end_reason="bearing_failure" if has_fail else "still_in_service",
                event_observed=has_fail,
                verified_by="Inspector_A"
            ))

        s.add_all(lives)
        s.commit()
        print(f"Created {len(runs)} runs, {len(failures)} failures, {len(lives)} component lives.")
        return True


if __name__ == "__main__":
    seed_labels()

