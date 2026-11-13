"""Check DB contents."""
from database.connection import get_engine
from sqlalchemy import text

eng = get_engine()
with eng.connect() as c:
    n_obs = c.execute(text("SELECT COUNT(*) FROM observations")).scalar()
    n_items = c.execute(text("SELECT COUNT(*) FROM data_items")).scalar()
    print(f"observations: {n_obs}")
    print(f"data_items: {n_items}")
    span = c.execute(text("SELECT MIN(ts), MAX(ts) FROM observations")).fetchone()
    print(f"time span: {span[0]} to {span[1]}")
    rows = c.execute(text("SELECT data_item_id, value_num FROM observations WHERE value_num IS NOT NULL LIMIT 5")).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]}")
