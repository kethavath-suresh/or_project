"""Constructive builder shared by the greedy baseline and the RL agent, plus local search."""
import numpy as np
from .solution import evaluate, route_schedule

N_FEAT = 10
BIG = 1000.0   # internal overflow penalty (cost / gun-hour) used to steer constructive search away from dead ends


def relax(inst):
    """Copy of the instance whose FCS capacity is soft (huge penalty). Heuristics search here; the final
    plan is re-evaluated on the real (hard-capacity) instance via finalize()."""
    return inst if inst.overflow_cost is not None else inst.with_(overflow_cost=BIG)


def finalize(inst, sol):
    if sol is None:
        return None
    out = evaluate(inst, sol.fcs_of, sol.routes)
    return out if out.feasible else None


class Builder:
    """Sequentially assigns EVs. Action set per EV: each FCS j (if capacity allows) and each van k
    (at its cheapest feasible insertion position)."""

    def __init__(self, inst, order=None):
        self.inst = inst
        self.order = order if order is not None else [int(i) for i in np.argsort(-inst.q, kind='stable')]  # largest demand first (bin-packing friendly)
        self.reset()

    def reset(self):
        I = self.inst
        self.used = np.zeros(I.m)
        self.routes = [[] for _ in range(I.K)]
        self.fcs_of = [-1] * I.n
        self.cur_route_cost = [0.0] * I.K
        self.load = np.zeros(I.K)

    def _pending(self, i):
        """EVs not yet assigned when deciding EV i (i itself included)."""
        k = self.order.index(i)
        return set(self.order[k:])

    def _rcost(self, r):
        I = self.inst
        if not r:
            return 0.0
        ok, starts, back, load = route_schedule(I, r)
        seq = [0] + r + [0]
        c = I.f_mcs + I.c_mcs * sum(I.d_mcs[a, b] for a, b in zip(seq[:-1], seq[1:]))
        c += I.c_batt * load + I.lam_wait * sum(st - I.e[i - 1] for st, i in zip(starts, r))
        return c

    def _best_insert(self, k, ev):
        r = self.routes[k]
        best = None
        base = self.cur_route_cost[k]
        for p in range(len(r) + 1):
            nr = r[:p] + [ev] + r[p:]
            ok, _, _, _ = route_schedule(self.inst, nr)
            if not ok:
                continue
            d = self._rcost(nr) - base
            if best is None or d < best[0]:
                best = (d, p)
        return best

    def actions(self, i):
        """List of (kind, idx, pos, delta_cost) and feature matrix for 0-based EV i."""
        I = self.inst
        acts, feats, last_resort = [], [], []
        scale = 20.0
        # global "pressure": unassigned demand (gun-hours) relative to the free FCS gun-hours left
        pend = float(sum(I.dur_f[a] for a in range(I.n) if a in self._pending(i)))
        free = float(np.maximum(I.cap_f - self.used, 0).sum()) + 1e-6
        pr = min(pend / free, 3.0)
        for j in range(I.m):
            over = self.used[j] + I.dur_f[i] - I.cap_f[j]
            if over > 1e-9 and I.overflow_cost is None:
                continue
            dc = (I.c_ev + I.lam_ev / I.speed) * I.d_fcs[i, j]
            if over > 1e-9:
                if I.overflow_cost >= BIG:          # internal relaxation: overflow only as a last resort
                    last_resort.append((j, dc)); continue
                dc += I.overflow_cost * min(over, I.dur_f[i])
            rem = max(0.0, (I.cap_f[j] - self.used[j] - I.dur_f[i]) / I.cap_f[j])
            feats.append([1, dc / scale, rem, self.used[j] / I.cap_f[j], 0, 0, 0, 0, pr, 0])
            acts.append(("F", j, None, dc))
        for k in range(I.K):
            if self.load[k] + I.q[i] > I.Q + 1e-9:
                continue
            b = self._best_insert(k, i + 1)
            if b is None:
                continue
            dc, p = b
            empty = 1.0 if not self.routes[k] else 0.0
            nr = self.routes[k][:p] + [i + 1] + self.routes[k][p:]
            _, _, back, _ = route_schedule(I, nr)
            feats.append([0, 0, 0, 0, dc / scale, empty, (I.Q - self.load[k] - I.q[i]) / I.Q, (I.T - back) / I.T, 0, pr])
            acts.append(("M", k, p, dc))
        if not acts:
            for j, dc in last_resort:
                feats.append([1, dc / scale, 0.0, 1.0, 0, 0, 0, 0, pr, 0])
                acts.append(("F", j, None, dc + BIG))
        return acts, np.array(feats, float)

    def apply(self, i, act):
        kind, idx, p, _ = act
        I = self.inst
        if kind == "F":
            self.fcs_of[i] = idx
            self.used[idx] += I.dur_f[i]
        else:
            self.routes[idx].insert(p, i + 1)
            self.cur_route_cost[idx] = self._rcost(self.routes[idx])
            self.load[idx] += I.q[i]

    def result(self):
        return evaluate(self.inst, self.fcs_of, self.routes)


def greedy(inst, orders=("urgent", "demand")):
    """Cheapest-feasible-insertion greedy (baseline); best over a few EV orderings."""
    rel = relax(inst)
    best = None
    for o in orders:
        order = [int(i) for i in (np.argsort(rel.l) if o == "urgent" else np.argsort(-rel.q))]
        b = Builder(rel, order)
        ok = True
        for i in b.order:
            acts, _ = b.actions(i)
            if not acts:
                ok = False
                break
            b.apply(i, min(acts, key=lambda a: a[3]))
        if ok:
            c = finalize(inst, b.result())
            if c is not None and (best is None or c.cost < best.cost):
                best = c
    return best


def _builder_from(inst, fo, rt):
    b = Builder(inst)
    b.fcs_of = list(fo)
    b.routes = [list(r) for r in rt]
    b.used = np.array([sum(inst.dur_f[a] for a in range(inst.n) if fo[a] == j) for j in range(inst.m)])
    b.cur_route_cost = [b._rcost(r) for r in b.routes]
    b.load = np.array([sum(inst.q[a - 1] for a in r) for r in b.routes])
    return b


def local_search(inst, sol, max_pass=15):
    """Local search on the relaxed instance; returns a plan feasible for the real instance (or None)."""
    rel = relax(inst)
    out = _local_search(rel, evaluate(rel, sol.fcs_of, sol.routes), max_pass)
    return finalize(inst, out) or finalize(inst, sol)


def _local_search(inst, sol, max_pass=15):
    """Relocate (remove an EV, re-insert at best/2nd/3rd best place on any resource) + intra-route 2-opt."""
    best = sol
    for _ in range(max_pass):
        improved = False
        for i in range(inst.n):
            fo = list(best.fcs_of)
            rt = [list(r) for r in best.routes]
            if fo[i] >= 0:
                fo[i] = -1
            else:
                for r in rt:
                    if (i + 1) in r:
                        r.remove(i + 1)
            acts, _ = _builder_from(inst, fo, rt).actions(i)
            for a in sorted(acts, key=lambda a: a[3])[:3]:
                bb = _builder_from(inst, fo, rt)
                bb.apply(i, a)
                cand = bb.result()
                if cand.feasible and cand.cost < best.cost - 1e-9:
                    best, improved = cand, True
                    break
        for k in range(len(best.routes)):
            r = best.routes[k]
            for a in range(len(r)):
                for c in range(a + 1, len(r)):
                    nr = r[:a] + r[a:c + 1][::-1] + r[c + 1:]
                    rts = [list(x) for x in best.routes]
                    rts[k] = nr
                    cand = evaluate(inst, best.fcs_of, rts)
                    if cand.feasible and cand.cost < best.cost - 1e-9:
                        best, improved = cand, True
                        r = nr
        if not improved:
            break
    return best
