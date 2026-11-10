import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("PM_CONFIG", ROOT / "config" / "machine_config.yaml"))
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        cfg["database"]["url"] = db_url
    return cfg
