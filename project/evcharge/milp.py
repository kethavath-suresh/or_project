"""Exact MILP (arc-based two-index VRPTW-style routing + EV->FCS assignment), solved with HiGHS (highspy).

Sets:   EVs i in 1..n, depot 0, stations j in 0..m-1
Vars:   z[i,j] in {0,1}  EV i charges at FCS j
        x[a,b] in {0,1}  a van drives directly a -> b   (a,b in 0..n)
        t[i]   >= 0      time a van starts charging EV i
        u[i]   >= 0      cumulative energy delivered on the van route up to and incl. i
        o[j]   >= 0      overflow gun-hours at FCS j (only when overflow_cost is set)
Min:    EV driving + EV detour inconvenience + van driving + van dispatch + van energy
        + EV waiting penalty (+ overflow penalty)
"""
import time
import numpy as np
import highspy
from .solution import evaluate

INT = highspy.HighsVarType.kInteger


def _sum(terms):
    it = iter(terms)
    try:
        acc = next(it)
    except StopIteration:
        return 0
    for e in it:
        acc = acc + e
    return acc


def solve_milp(inst, time_limit=60, gap=1e-4, msg=False, threads=1):
    n, m, K = inst.n, inst.m, inst.K
    N = range(1, n + 1)
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(msg))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(gap))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("random_seed", 0)
    inf = highspy.kHighsInf

    z = {(i, j): h.addVariable(lb=0, ub=1, type=INT) for i in N for j in range(m)}
    arcs = [(a, b) for a in range(n + 1) for b in range(n + 1) if a != b]
    # prune arcs that can never be used (time-window / battery infeasible)
    use = []
    for a, b in arcs:
        if a == 0 and b == 0:
            continue
        if b == 0 or a == 0:
            use.append((a, b)); continue
        if inst.e[a - 1] + inst.s[a - 1] + inst.tau[a, b] > inst.l[b - 1] + 1e-9: continue
        if inst.q[a - 1] + inst.q[b - 1] > inst.Q: continue
        use.append((a, b))
    x = {ab: h.addVariable(lb=0, ub=1, type=INT) for ab in use} if K > 0 else {}
    t = {i: h.addVariable(lb=inst.e[i - 1], ub=inst.l[i - 1]) for i in N}
    u = {i: h.addVariable(lb=inst.q[i - 1], ub=inst.Q) for i in N}
    soft = inst.overflow_cost is not None
    o = {j: h.addVariable(lb=0, ub=inf) for j in range(m)} if soft else {}

    # ---- objective
    obj = _sum((inst.c_ev + inst.lam_ev / inst.speed) * inst.d_fcs[i - 1, j] * z[i, j]
               for i in N for j in range(m))
    if K > 0:
        obj = obj + _sum(inst.c_mcs * inst.d_mcs[a, b] * x[a, b] for (a, b) in use)
        obj = obj + _sum(inst.f_mcs * x[0, j] for j in N if (0, j) in x)
        obj = obj + _sum(inst.c_batt * inst.q[b - 1] * x[a, b] for (a, b) in use if b != 0)
        obj = obj + _sum(inst.lam_wait * t[i] for i in N)
        const = -inst.lam_wait * float(np.sum(inst.e))
    else:
        const = 0.0
    if soft:
        obj = obj + _sum(inst.overflow_cost * o[j] for j in range(m))

    # ---- (1) every EV served exactly once
    for i in N:
        inflow = _sum(x[a, i] for a in range(n + 1) if (a, i) in x)
        h.addConstr(_sum(z[i, j] for j in range(m)) + inflow == 1)
    if K > 0:
        for i in N:                                           # (2) flow conservation
            h.addConstr(_sum(x[a, i] for a in range(n + 1) if (a, i) in x)
                        == _sum(x[i, b] for b in range(n + 1) if (i, b) in x))
        h.addConstr(_sum(x[0, j] for j in N) <= K)            # (3) fleet size
        h.addConstr(_sum(x[0, j] for j in N) == _sum(x[j, 0] for j in N))
        for i in N:
            qi, si = inst.q[i - 1], inst.s[i - 1]
            if (0, i) in x:                                   # (5) leaving depot at 0
                h.addConstr(t[i] >= inst.tau[0, i] * x[0, i])
            if (i, 0) in x:                                   # return before end of horizon
                M = max(0.0, inst.l[i - 1] + si + inst.tau[i, 0] - inst.T)
                h.addConstr(t[i] + (si + inst.tau[i, 0]) * 1 <= inst.T + M * (1 - x[i, 0]))
        for (a, b) in use:
            if a == 0 or b == 0:
                continue
            h.addConstr(u[b] >= u[a] + inst.q[b - 1] - inst.Q * (1 - x[a, b]))      # (4) battery
            Mt = max(0.0, inst.l[a - 1] + inst.s[a - 1] + inst.tau[a, b] - inst.e[b - 1])
            h.addConstr(t[b] >= t[a] + inst.s[a - 1] + inst.tau[a, b] - Mt * (1 - x[a, b]))
    # ---- (6) FCS capacity (gun-hours = guns x available time)
    for j in range(m):
        lhs = _sum(inst.dur_f[i - 1] * z[i, j] for i in N)
        h.addConstr(lhs <= inst.cap_f[j] + (o[j] if soft else 0))

    t0 = time.time()
    h.minimize(obj)
    rt = time.time() - t0
    ms = h.getModelStatus()
    info = dict(status=str(ms).split(".")[-1], runtime=rt, n=n, m=m, K=K)
    inf_ = h.getInfo()
    if ms == highspy.HighsModelStatus.kInfeasible or inf_.primal_solution_status != 2:
        info.update(obj=np.nan, bound=np.nan, gap=np.nan, feasible=False)
        return None, info
    fcs_of = [-1] * n
    for i in N:
        for j in range(m):
            if h.val(z[i, j]) > 0.5:
                fcs_of[i - 1] = j
    routes = []
    if K > 0:
        succ = {a: [] for a in range(n + 1)}
        for (a, b), var in x.items():
            if h.val(var) > 0.5:
                succ[a].append(b)
        for s0 in succ[0]:
            r, cur = [], s0
            while cur != 0:
                r.append(cur); cur = succ[cur][0]
            routes.append(r)
    sol = evaluate(inst, fcs_of, routes)
    objv = inf_.objective_function_value + const
    bound = inf_.mip_dual_bound + const
    info.update(obj=objv, recomputed=sol.cost, feasible=sol.feasible, bound=bound,
                gap=(objv - bound) / max(1e-9, abs(objv)),
                optimal=(ms == highspy.HighsModelStatus.kOptimal))
    return sol, info
