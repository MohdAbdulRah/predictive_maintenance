"""
Multi-round bootstrap collector. Calls /current N times with a delay,
storing each snapshot. This is robust against fast ring-buffer overruns on
demo.mtconnect.org where /sample sequences are immediately stale.

Usage:  python -m scripts.collect_bootstrap [--rounds N] [--delay S]
"""
import argparse, logging, time
from config import load_config
from data_ingestion.mtconnect_client import MTConnectClient
from data_ingestion.observation_collector import ObservationCollector
from database.connection import get_engine
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30, help="Number of /current snapshots")
    ap.add_argument("--delay", type=float, default=3.0, help="Seconds between snapshots")
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
        args.delay,
    )

    log.info("Loading probe...")
    coll.load_probe()
    log.info("Device UUID: %s", coll.device_uuid)

    total = 0
    for i in range(args.rounds):
        # Reset state to force bootstrap on every round
        coll.next_seq = None
        coll.instance_id = None
        n = coll.bootstrap()
        total += n
        log.info("Round %d/%d: stored %d new obs (cumulative=%d)", i + 1, args.rounds, n, total)
        if i < args.rounds - 1:
            time.sleep(args.delay)

    log.info("Bootstrap collection done. Total new observations: %d", total)
    print(f"DONE: {total} total new observations for device {coll.device_uuid}")


if __name__ == "__main__":
    main()
