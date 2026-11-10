"""Run the collector:  python -m scripts.collect"""
import logging
from config import load_config
from data_ingestion.mtconnect_client import MTConnectClient
from data_ingestion.observation_collector import ObservationCollector
from database.connection import get_engine, session_scope  # noqa
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    cfg = load_config()
    eng = get_engine()
    a = cfg["agent"]
    client = MTConnectClient(a["base_url"], a["request_timeout_s"], a["max_retries"])
    ObservationCollector(client, lambda: Session(eng, expire_on_commit=False), a["device"],
                         a["sample_count"], a["poll_interval_s"]).run()
