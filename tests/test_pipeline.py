import copy
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from config import load_config
from data_ingestion.mtconnect_client import AgentError, MTConnectClient, parse_streams
from data_ingestion.observation_collector import ObservationCollector
from data_ingestion.probe_parser import parse_probe
from database.models import Base, CollectionGap, ObservationRow
from tests.simulator import FakeAgent, PROBE, error_xml, gen_events, streams_xml


def make_db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    Base.metadata.create_all(eng)
    return eng, (lambda: Session(eng, expire_on_commit=False))


def make_collector(agent, sf, count=200):
    sess = type("S", (), {"get": lambda self, url, params=None, timeout=None: agent.get(url, params, timeout)})()
    c = ObservationCollector(MTConnectClient("http://x", session=sess, max_retries=1), sf, sample_count=count)
    c.load_probe()
    return c


# ---------- parsing ----------
def test_probe_catalogue_discovered_not_assumed():
    p = parse_probe(PROBE)
    ids = {d.id: d for d in p["data_items"]}
    assert ids["load"].category == "SAMPLE" and ids["load"].units == "PERCENT"
    assert ids["exec"].category == "EVENT" and ids["cond1"].category == "CONDITION"
    assert "Rotary[C]" in ids["load"].component_path


def test_streams_unavailable_and_error():
    xml = streams_xml([(1, "2026-01-01T00:00:00Z", "S:Load", "load", "UNAVAILABLE"),
                       (2, "2026-01-01T00:00:01Z", "C:Fault", "cond1", "")])
    h, o = parse_streams(xml)
    assert o[0].unavailable and o[0].value is None
    assert o[1].condition_level == "FAULT"
    with pytest.raises(AgentError) as e:
        parse_streams(error_xml("OUT_OF_RANGE"))
    assert e.value.code == "OUT_OF_RANGE"


# ---------- collector ----------
def test_collector_no_duplicates_and_resumes():
    ev = gen_events(600); agent = FakeAgent(ev, buffer=100000); eng, sf = make_db()
    c = make_collector(agent, sf); agent.advance(300)
    c.bootstrap()
    for _ in range(5): c.step()
    agent.advance(len(ev))
    for _ in range(20): c.step()
    c.step(); c.step()                      # idempotent when nothing new
    with sf() as s:
        n = s.query(ObservationRow).count()
        keys = s.execute(select(ObservationRow.sequence)).scalars().all()
    assert len(keys) == len(set(keys))
    assert n >= len(ev) - 10               # bootstrap /current may precede; nothing after it may be lost
    with sf() as s:
        assert s.query(CollectionGap).count() == 0


def test_buffer_overrun_is_recorded_not_hidden():
    ev = gen_events(1200); agent = FakeAgent(ev, buffer=50); eng, sf = make_db()
    c = make_collector(agent, sf, count=50); agent.advance(100); c.bootstrap()
    agent.advance(400)                       # agent produces far more than the buffer holds
    c.step()
    with sf() as s:
        gaps = s.scalars(select(CollectionGap)).all()
    assert any(g.kind == "BUFFER_OVERRUN" and g.to_sequence > g.from_sequence for g in gaps)


def test_instance_reset_recorded():
    ev = gen_events(300); agent = FakeAgent(ev, buffer=10000); eng, sf = make_db()
    c = make_collector(agent, sf); agent.advance(100); c.bootstrap(); c.step()
    agent.instance = 2; agent.advance(50); c.step()
    with sf() as s:
        assert any(g.kind == "INSTANCE_RESET" for g in s.scalars(select(CollectionGap)))


# ---------- preprocessing / features ----------
def fill_db(n_seconds=7200, drift_after=None, seed=0):
    ev = gen_events(n_seconds, seed=seed, drift_after=drift_after)
    agent = FakeAgent(ev, buffer=10**7); eng, sf = make_db()
    c = make_collector(agent, sf, count=5000); agent.advance(len(ev)); c.bootstrap()
    while c.step() > 0: pass          # catch up fully, as the real run loop would
    return eng, sf, c


def prep(sf, cfg=None):
    from preprocessing.preprocessing import build_grid, load_catalogue, load_gaps, load_observations, select_signals
    from feature_engineering.rolling_features import build_features
    cfg = cfg or load_config()
    with sf() as s:
        obs = load_observations(s, "TEST-1"); cat = load_catalogue(s, "TEST-1"); gaps = load_gaps(s, "TEST-1", obs)
    sigs = select_signals(cat, obs, cfg)
    grid = build_grid(obs, cat, gaps, sigs, cfg)
    return obs, cat, sigs, grid, build_features(grid, sigs, cfg)


def test_constant_signal_excluded_and_state_derived():
    eng, sf, _ = fill_db(3600)
    obs, cat, sigs, grid, feats = prep(sf)
    assert "const" not in sigs and set(sigs) == {"load", "spd", "temp"}
    assert set(grid.__state.unique()) <= {"active", "idle", "unknown"} and "active" in set(grid.__state)


def test_features_are_backward_looking_only():
    eng, sf, _ = fill_db(3600)
    obs, cat, sigs, grid, feats = prep(sf)
    from feature_engineering.rolling_features import build_features
    cfg = load_config(); cut = len(grid) // 2
    g2 = grid.copy(); g2.iloc[cut + 1:, g2.columns.get_loc("load")] += 1e6      # corrupt the FUTURE
    f2 = build_features(g2, sigs, cfg)
    col = [c for c in feats.columns if c.startswith("load__300s")]
    a, b = feats.iloc[:cut + 1][col], f2.iloc[:cut + 1][col]
    pd.testing.assert_frame_equal(a, b)


def test_state_aware_stats_do_not_mix_idle_and_active():
    eng, sf, _ = fill_db(3600)
    obs, cat, sigs, grid, feats = prep(sf)
    act = feats[(feats.__state == "active")]["load__300s_mean"].dropna()
    idl = feats[(feats.__state == "idle")]["load__300s_mean"].dropna()
    assert act.mean() > 40 and idl.mean() < 10      # blended stats would sit between them


# ---------- anomaly ----------
def test_anomaly_pipeline_plumbing_and_evidence():
    from feature_engineering.rolling_features import feature_columns
    from models.anomaly_detector import AnomalyDetector
    cfg = load_config(); cfg = copy.deepcopy(cfg); cfg["anomaly"]["min_training_rows_per_state"] = 100
    eng, sf, _ = fill_db(14400, seed=1, drift_after=11000)
    obs, cat, sigs, grid, feats = prep(sf, cfg)
    cut = grid.index[0] + pd.Timedelta(seconds=9000)
    det = AnomalyDetector(cfg).fit(feats[feats.index < cut], feature_columns(feats))
    sc = det.score(feats[feats.index > grid.index[0] + pd.Timedelta(seconds=11500)])
    act = sc[(sc.operating_state == "active") & (sc.status != "not_scored")]
    assert (act.status == "anomalous").mean() > 0.5
    pre = det.score(feats[(feats.index > cut + pd.Timedelta(seconds=300)) & (feats.index < grid.index[0] + pd.Timedelta(seconds=10800))])
    pre = pre[pre.status != "not_scored"]
    assert (pre.status == "anomalous").mean() < 0.10          # un-shifted held-out data should rarely alert
    top = act[act.status == "anomalous"].iloc[-1].top_features
    assert any(f["feature"].startswith("load__") for f in top)          # evidence points at the shifted signal
    assert "not a failure" in sc.disclaimer.iloc[0].lower() or "not a confirmed fault" in sc.disclaimer.iloc[0].lower()


# ---------- supervised labels ----------
def test_failure_labels_censoring_rules():
    from models.failure_predictor import build_failure_labels, enforce_monotone
    t0 = pd.Timestamp("2026-01-01", tz="UTC"); times = pd.date_range(t0, periods=100, freq="1h")
    fail = [t0 + pd.Timedelta(hours=50)]; end = t0 + pd.Timedelta(hours=99)
    lab = build_failure_labels(times, fail, end, [1, 6, 24, 72])
    assert lab.loc[times[49], "y_1h"] == 1 and lab.loc[times[40], "y_1h"] == 0 and lab.loc[times[40], "y_24h"] == 1
    assert lab.loc[times[10], "y_6h"] == 0
    assert np.isnan(lab.loc[times[98], "y_6h"])                  # follow-up incomplete -> unknown, never 0
    nofail = build_failure_labels(times, [], end, [24, 72])
    assert nofail.loc[times[10], "y_24h"] == 0 and np.isnan(nofail.loc[times[90], "y_24h"])
    assert enforce_monotone({1: .3, 6: .1, 24: .2, 72: .5}) == {1: .3, 6: .3, 24: .3, 72: .5}


def test_training_refuses_without_verified_labels(tmp_path):
    from training import train_failure, train_rul
    eng, sf, _ = fill_db(3600)
    with sf() as s:
        assert train_failure.run(session=s)["status"] == "not_trained"
        assert train_rul.run(session=s)["status"] == "not_trained"


# ---------- RUL plumbing (synthetic lives; no accuracy claim) ----------
def test_rul_gate_and_conditional_survival():
    from models.rul_estimator import InsufficientLives, RULEstimator, assess_lives
    cfg = load_config(); rng = np.random.default_rng(0)
    assert assess_lives(pd.DataFrame({"event": []}), cfg)["status"] == "not_trained"
    n = 60; x = rng.normal(size=n); dur = rng.weibull(2, n) * 100 * np.exp(-0.3 * x); ev = rng.random(n) < 0.7
    X = pd.DataFrame({"x": x})
    with pytest.raises(InsufficientLives):
        RULEstimator(cfg).fit(X.iloc[:5], dur[:5], ev[:5], ["x"])
    m = RULEstimator(cfg, backend="cox").fit(X, dur, ev, ["x"])
    r = m.predict(X.iloc[:2], age_h=10)[0]
    pf = r["p_fail_within_h"]
    assert pf[1] <= pf[6] <= pf[24] <= pf[72]


def test_operating_hours_uses_state_not_wallclock():
    from models.rul_estimator import operating_hours_between
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    st = pd.Series(["active", "idle", "active"], index=[t0, t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=5)])
    h, cov = operating_hours_between(st, t0, t0 + pd.Timedelta(hours=7), t0 + pd.Timedelta(hours=7))
    assert abs(h - 4.0) < 1e-6 and cov > 0.99


# ---------- API ----------
def test_api_never_fabricates(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import database.connection as dbc, api.service as svc
    monkeypatch.setattr(svc, "ART", tmp_path)
    eng, sf, _ = fill_db(1800)
    dbc._engine = eng; dbc._Session = sf
    from api.main import app
    cl = TestClient(app)
    assert cl.get("/api/predictions/failure-risk").json()["status"] == "not_trained"
    assert cl.get("/api/predictions/rul").json()["status"] == "not_trained"
    assert cl.get("/api/predictions/anomalies").json()["status"] == "not_trained"
    live = cl.get("/api/machine/live").json()
    assert live["status"] == "ok" and any(o["data_item_id"] == "load" and o["units"] == "PERCENT" for o in live["observations"])
    assert cl.get("/api/machine/history", params={"data_item_id": "load"}).json()["points"]
    assert cl.get("/api/model/status").json()["stages"]["failure"]["artifact_present"] is False
