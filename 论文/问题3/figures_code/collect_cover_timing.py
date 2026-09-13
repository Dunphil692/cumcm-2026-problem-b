# -*- coding: utf-8 -*-
"""为图6采集覆盖点首次访问时序 + 首次清除时序（200 局）。"""
import random, math, json, sys
sys.path.insert(0, "/Users/zhuchen/建模大赛/问题三/code")
import problem3_belief as B
from problem3_simulation_model import generate_instance, census_stops

COVERS = census_stops()
SNAP = 40.0
state = {}

orig_move = B.Robot.move_to
def move(self, x, y):
    t = self.time
    r = orig_move(self, x, y)
    for i, (cx, cy) in enumerate(COVERS):
        if math.hypot(x - cx, y - cy) <= SNAP and state["fv"][i] is None:
            state["fv"][i] = t
    return r
B.Robot.move_to = move

orig_clear = B.Robot.clear
def clear(self, ch):
    r = orig_clear(self, ch)
    if r and state["fc"] is None:
        state["fc"] = self.time
    return r
B.Robot.clear = clear

def one(idx):
    rng = random.Random((2027 + 100003 * (idx + 1)) % (2**31 - 1))
    w = generate_instance(rng)
    r = B.run_belief(w, fidelity="high")
    return {
        "trial": idx, "time_s": r["time_s"], "remaining": r["remaining"],
        "first_visit": [None if v is None else round(v, 1) for v in state["fv"]],
        "first_clear": None if state["fc"] is None else round(state["fc"], 1),
        "cover_visits": r["cover_visits"],
        "cover_at_first_clear": r["cover_visits_at_first_clear"],
    }

rows = []
for idx in range(200):
    state["fv"] = [None] * len(COVERS)
    state["fc"] = None
    rows.append(one(idx))
json.dump({"rows": rows}, open("/Users/zhuchen/建模大赛/问题三/论文/figures_code/cover_timing.json", "w"))
print("saved", len(rows), "trials")
