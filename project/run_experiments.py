"""Experiments: (1) solution quality / speed of MILP vs greedy vs RL, (2) sensitivity & scenario analysis.

Usage:  python run_experiments.py [quality|analysis|all]
Outputs CSV files in results/.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from evcharge.generate import feasible_instance
from evcharge.heuristics import greedy, local_search
from evcharge.instance import make_instance
from evcharge.milp import solve_milp
from evcharge.rl import RLAgent

MILP_TL = 60
SIZES = [(10, 3, 1), (20, 4, 2), (30, 6, 3), (40, 8, 4), (50, 10, 4)]   # (n EVs, m FCS, K vans)
LAYOUTS = ["R", "C", "RC"]
SEEDS = range(5)
SOFT = 30.0       # overflow penalty (cost per queued gun-hour) used in the soft-capacity analysis


def load_agent():
    d = json.load(open("results/rl_policy.json"))
    a = RLAgent(seed=123)
    a.w = np.array(d["w"])
    return a


def _row(inst, method, sol, rt, extra=None):
    r = dict(instance=inst.name, n=inst.n, m=inst.m, K=inst.K, method=method, runtime=rt,
             cost=np.nan if sol is None else sol.cost, feasible=sol is not None and sol.feasible)
    if sol is not None:
        r.update(sol.parts)
        r["vans_used"] = sum(1 for x in sol.routes if x)
        r["mcs_evs"] = sum(len(x) for x in sol.routes)
        r["fcs_evs"] = inst.n - r["mcs_evs"]
    if extra:
        r.update(extra)
    return r


def run_one(args):
    """All methods on one instance (executed in a worker process)."""
    n, m, K, seed, layout = args
    inst = feasible_instance(n=n, m=m, K=K, layout=layout, seed=seed)
    agent = load_agent()
    rows = []
    t = time.time(); sol, info = solve_milp(inst, time_limit=MILP_TL)
    rows.append(_row(inst, "MILP", sol, time.time() - t, dict(gap=info.get("gap"), bound=info.get("bound"),
                                                              optimal=info.get("optimal"))))
    t = time.time(); g = greedy(inst); rows.append(_row(inst, "Greedy", g, time.time() - t))
    t = time.time(); gl = local_search(inst, g) if g else None
    rows.append(_row(inst, "Greedy+LS", gl, time.time() - t + rows[-1]["runtime"]))
    t = time.time(); r0 = agent.solve(inst, samples=0, polish=False); rows.append(_row(inst, "RL (greedy decode)", r0, time.time() - t))
    t = time.time(); r1 = agent.solve(inst, samples=30, polish=True); rows.append(_row(inst, "RL (30 samples + LS)", r1, time.time() - t))
    return rows


def quality():
    jobs = [(n, m, K, 100 * s + n, LAYOUTS[s % 3]) for (n, m, K) in SIZES for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for k, out in enumerate(ex.map(run_one, jobs)):
            rows += out
            print(f"[{k + 1}/{len(jobs)}] {out[0]['instance']} MILP {out[0]['cost']:.1f} ({out[0]['runtime']:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/quality_all.csv", index=False)
    return df


# ------------------------------------------------------------------ analysis
def _solve_soft(inst, tl=30):
    sol, info = solve_milp(inst.with_(overflow_cost=SOFT), time_limit=tl)
    return sol, info


def analysis_job(args):
    kind, val, seed, extra = args
    base = dict(n=30, m=5, K=3, layout="RC", seed=seed)
    kw = dict(base)
    K = extra.get("K", 3)
    if kind == "K":
        K = int(val)
    if kind == "cap":
        kw["cap_scale"] = val
    if kind == "demand":
        kw["demand_scale"] = val
    if kind == "grid":
        kw["demand_scale"], kw["cap_scale"] = val
    if kind == "layout":
        kw["layout"] = val
    if kind == "fleet_grid":
        kw["demand_scale"], kw["cap_scale"] = val
        K = extra["K"]
    kw["K"] = K
    inst = make_instance(**kw)
    sol, info = _solve_soft(inst)
    hard, hinfo = solve_milp(inst, time_limit=20)
    row = dict(kind=kind, value=str(val), seed=seed, K=K, runtime=info["runtime"], hard_feasible=hard is not None,
               hard_cost=np.nan if hard is None else hard.cost)
    if sol is not None:
        row.update(cost=sol.cost, **sol.parts, vans_used=sum(1 for r in sol.routes if r),
                   mcs_evs=sum(len(r) for r in sol.routes), gap=info.get("gap"))
        used = np.array([sum(inst.dur_f[i] for i, j in enumerate(sol.fcs_of) if j == s) for s in range(inst.m)])
        row["fcs_util"] = float(used.sum() / inst.cap_f.sum())
        row["fcs_util_std"] = float(np.std(used / inst.cap_f))
    return row


def analysis():
    jobs = []
    seeds = range(6)
    for s in seeds:
        for K in range(0, 7):
            jobs.append(("K", K, s, {}))
        for c in (0.6, 0.8, 1.0, 1.2, 1.5):
            jobs.append(("cap", c, s, {}))
        for d in (0.6, 0.8, 1.0, 1.2, 1.4):
            jobs.append(("demand", d, s, {}))
        for l in ("R", "C", "RC"):
            jobs.append(("layout", l, s, {}))
    # scenario grid: demand x station capacity, with and without the fleet
    for s in range(4):
        for d in (0.7, 1.0, 1.3):
            for c in (0.7, 1.0, 1.3):
                for K in (0, 3):
                    jobs.append(("fleet_grid", (d, c), s, {"K": K}))
    rows = []
    with ProcessPoolExecutor(max_workers=3) as ex:
        for k, r in enumerate(ex.map(analysis_job, jobs)):
            rows.append(r)
            if k % 20 == 0:
                print(f"[{k + 1}/{len(jobs)}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/analysis_all.csv", index=False)
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("quality", "all"):
        quality()
    if what in ("analysis", "all"):
        analysis()
