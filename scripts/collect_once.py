"""Collect telemetry for a fixed number of steps from MTConnect demo and exit.
Usage:  python -m scripts.collect_once [--steps N] [--poll S]
"""
import argparse
import logging
import time

from config import load_config
from data_ingestion.mtconnect_client import MTConnectClient
from data_ingestion.observation_collector import ObservationCollector
from database.connection import get_engine
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="Number of polling steps (default 60)")
    ap.add_argument("--poll", type=float, default=1.0, help="Poll interval seconds")
    args = ap.parse_args()

    cfg = load_config()
    eng = get_engine()
    a = cfg["agent"]

    client = MTConnectClient(a["base_url"], a["request_timeout_s"], a["max_retries"])
    coll = ObservationCollector(
        client,
        lambda: Session(eng, expire_on_commit=False),
        a["device"],
        a["sample_count"],
        args.poll,
    )

    log.info("Loading probe from %s …", a["base_url"])
    coll.load_probe()
    log.info("Device UUID: %s, collecting %d steps …", coll.device_uuid, args.steps)

    total = 0
    for i in range(args.steps):
        try:
            n = coll.step()
            total += n
            if (i + 1) % 10 == 0:
                log.info("Step %d/%d — stored %d new obs (total=%d)", i + 1, args.steps, n, total)
        except Exception as e:
            log.warning("Step %d error: %s", i + 1, e)
        time.sleep(args.poll)

    log.info("Collection complete. Total observations stored: %d", total)
    print(f"DONE: {total} observations stored for device {coll.device_uuid}")


if __name__ == "__main__":
    main()
