"""SYNTHETIC test fixtures. These are NOT machine data and are never used to report performance.
They exist only to verify that the code paths (parsing, gaps, dedupe, gating) behave correctly."""
from datetime import datetime, timedelta, timezone
import numpy as np

PROBE = b"""<?xml version="1.0"?>
<MTConnectDevices xmlns="urn:mtconnect.org:MTConnectDevices:1.7">
 <Header instanceId="1" bufferSize="1000" creationTime="2026-01-01T00:00:00Z"/>
 <Devices><Device id="d1" name="TestMill" uuid="TEST-1">
  <DataItems><DataItem id="avail" type="AVAILABILITY" category="EVENT"/></DataItems>
  <Components>
   <Controller id="ctl"><DataItems><DataItem id="exec" type="EXECUTION" category="EVENT"/>
        <DataItem id="cond1" type="SYSTEM" category="CONDITION"/></DataItems></Controller>
   <Axes id="ax"><Components><Rotary id="c" name="C"><DataItems>
        <DataItem id="spd" type="ROTARY_VELOCITY" category="SAMPLE" units="REVOLUTION/MINUTE"/>
        <DataItem id="load" type="LOAD" category="SAMPLE" units="PERCENT"/>
        <DataItem id="temp" type="TEMPERATURE" category="SAMPLE" units="CELSIUS"/>
        <DataItem id="const" type="POSITION" category="SAMPLE" units="MILLIMETER"/>
   </DataItems></Rotary></Components></Axes>
  </Components></Device></Devices></MTConnectDevices>"""


def streams_xml(obs, instance=1, first=1, last=None, nxt=None, buffer=1000):
    """obs: list of (seq, ts, kind, item_id, value). kind in Sample/Event/Condition level names."""
    last = last if last is not None else max([o[0] for o in obs], default=0)
    nxt = nxt if nxt is not None else last + 1
    s = [f'<MTConnectStreams xmlns="urn:mtconnect.org:MTConnectStreams:1.7"><Header instanceId="{instance}" bufferSize="{buffer}" '
         f'firstSequence="{first}" lastSequence="{last}" nextSequence="{nxt}" creationTime="2026-01-01T00:00:00Z"/>'
         '<Streams><DeviceStream name="TestMill" uuid="TEST-1"><ComponentStream component="X" componentId="x">']
    for kind, tag in (("S", "Samples"), ("E", "Events"), ("C", "Condition")):
        items = [o for o in obs if o[2][0] == kind]
        if not items: continue
        s.append(f"<{tag}>")
        for seq, ts, k, did, val in items:
            el = k.split(":")[1]
            body = "" if kind == "C" else val
            s.append(f'<{el} dataItemId="{did}" sequence="{seq}" timestamp="{ts}">{body}</{el}>')
        s.append(f"</{tag}>")
    s.append("</ComponentStream></DeviceStream></Streams></MTConnectStreams>")
    return "".join(s).encode()


def error_xml(code, text=""):
    return (f'<MTConnectError xmlns="urn:mtconnect.org:MTConnectError:1.7"><Header instanceId="1"/>'
            f'<Errors><Error errorCode="{code}">{text}</Error></Errors></MTConnectError>').encode()


def gen_events(n_seconds=7200, seed=0, drift_after=None, start=datetime(2026, 1, 1, tzinfo=timezone.utc), seq0=1):
    """Simulated machine: alternating ACTIVE/READY blocks; load/temp/spd ~ state-dependent noise."""
    rng = np.random.default_rng(seed); out = []; seq = seq0
    state = "READY"; temp = 30.0
    for t in range(0, n_seconds, 2):
        ts = (start + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if t % 600 == 0:
            state = "ACTIVE" if state == "READY" else "READY"
            out.append((seq, ts, "E:Execution", "exec", state)); seq += 1
        act = state == "ACTIVE"
        load = (55 + rng.normal(0, 4)) if act else (3 + rng.normal(0, 0.5))
        spd = (3000 + rng.normal(0, 30)) if act else 0.0
        temp += 0.02 * ((34.0 if act else 30.0) - temp) + rng.normal(0, 0.05)   # stationary, mean-reverting
        if drift_after is not None and t >= drift_after:
            load += 25; temp += 0.0
        out.append((seq, ts, "S:Load", "load", f"{load:.3f}")); seq += 1
        out.append((seq, ts, "S:RotaryVelocity", "spd", f"{spd:.2f}")); seq += 1
        out.append((seq, ts, "S:Temperature", "temp", f"{temp:.3f}")); seq += 1
        if t % 60 == 0:
            out.append((seq, ts, "S:Position", "const", "0.0")); seq += 1
    return out


class FakeAgent:
    """Serves probe/current/sample from a list of events with a bounded ring buffer."""
    def __init__(self, events, buffer=1000, instance=1):
        self.ev, self.buffer, self.instance, self.cursor = events, buffer, instance, 0

    def advance(self, n): self.cursor = min(len(self.ev), self.cursor + n)
    def visible(self): return self.ev[:self.cursor]
    def window(self):
        v = self.visible(); return v[-self.buffer:] if v else []

    def get(self, path, params=None, timeout=None):
        class R: pass
        r = R(); r.status_code = 200
        v, w = self.visible(), self.window()
        last = v[-1][0] if v else 0; first = w[0][0] if w else 1
        if path.endswith("probe"): r.content = PROBE
        elif path.endswith("current"):
            latest = {}
            for o in v: latest[(o[2], o[3])] = o
            r.content = streams_xml(list(latest.values()), self.instance, first, last, last + 1)
        else:
            frm = int(params.get("from", first)); cnt = int(params.get("count", 1000))
            if frm < first - 0 and v and frm < first:
                pass
            if v and (frm > last + 1):
                r.content = error_xml("OUT_OF_RANGE", "from beyond lastSequence")
            else:
                sel = [o for o in w if o[0] >= frm][:cnt]
                nxt = (sel[-1][0] + 1) if sel else max(frm, last + 1)
                r.content = streams_xml(sel, self.instance, first, last, nxt)
        return r
