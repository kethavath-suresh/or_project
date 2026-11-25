"""Solution representation, feasibility check and exact cost evaluation."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Solution:
    fcs_of: list          # fcs_of[i] = station index (0..m-1) or -1 if served by an MCS (i 0-based)
    routes: list          # routes[k] = list of 1-based EV ids visited by van k
    cost: float = np.nan
    parts: dict = None
    feasible: bool = True
    msg: str = ""


def route_schedule(inst, route):
    """Return (feasible, start_times, finish_time_back_at_depot, load)."""
    t, prev, load, starts = 0.0, 0, 0.0, []
    for i in route:
        t = max(t + inst.tau[prev, i], inst.e[i - 1])
        if t > inst.l[i - 1] + 1e-9:
            return False, starts, np.inf, load
        starts.append(t)
        t += inst.s[i - 1]
        load += inst.q[i - 1]
        prev = i
    back = t + inst.tau[prev, 0]
    ok = back <= inst.T + 1e-9 and load <= inst.Q + 1e-9
    return ok, starts, back, load


def evaluate(inst, fcs_of, routes):
    """Exact cost of a candidate plan; checks every constraint of the model."""
    n, m = inst.n, inst.m
    msg, feasible = [], True
    served = np.zeros(n, int)
    for i, j in enumerate(fcs_of):
        if j >= 0:
            served[i] += 1
    for r in routes:
        for i in r:
            served[i - 1] += 1
    if not np.all(served == 1):
        feasible = False; msg.append("every EV must be served exactly once")
    if sum(1 for r in routes if r) > inst.K:
        feasible = False; msg.append("too many vans")
    ev_trav = inst.c_ev * sum(inst.d_fcs[i, j] for i, j in enumerate(fcs_of) if j >= 0)
    incon_f = inst.lam_ev * sum(inst.d_fcs[i, j] / inst.speed for i, j in enumerate(fcs_of) if j >= 0)
    mcs_trav = fixed = batt = wait = 0.0
    for r in routes:
        if not r:
            continue
        ok, starts, back, load = route_schedule(inst, r)
        if not ok:
            feasible = False; msg.append("route violates time/battery")
        fixed += inst.f_mcs
        seq = [0] + r + [0]
        mcs_trav += inst.c_mcs * sum(inst.d_mcs[a, b] for a, b in zip(seq[:-1], seq[1:]))
        batt += inst.c_batt * load
        wait += inst.lam_wait * sum(st - inst.e[i - 1] for st, i in zip(starts, r)) if np.isfinite(back) else 0
    over, over_cost = 0.0, 0.0
    for j in range(m):
        used = sum(inst.dur_f[i] for i, jj in enumerate(fcs_of) if jj == j)
        ov = max(0.0, used - inst.cap_f[j])
        if ov > 1e-9:
            if inst.overflow_cost is None:
                feasible = False; msg.append(f"FCS {j} over capacity")
            else:
                over += ov; over_cost += inst.overflow_cost * ov
    parts = dict(ev_travel=ev_trav, ev_inconv=incon_f, mcs_travel=mcs_trav, mcs_fixed=fixed,
                 mcs_energy=batt, mcs_wait=wait, overflow=over_cost)
    cost = sum(parts.values())
    return Solution(list(fcs_of), [list(r) for r in routes], cost, parts, feasible, "; ".join(msg))
