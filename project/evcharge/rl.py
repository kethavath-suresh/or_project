"""Learning-to-optimize: REINFORCE-trained softmax policy over assignment actions.

The agent builds a plan EV by EV (most urgent first). For every EV it scores each feasible action
(charge at FCS j / insert into van k at its cheapest position) with a linear policy
    pi(a|s) = softmax(w . phi(s, a))
where phi are instance-size-independent features (normalized marginal cost, remaining station capacity,
van-empty flag, battery and time slack). Because the features do not depend on instance size, ONE trained
policy transfers to unseen instances. Training: REINFORCE with a per-instance mean baseline over a batch
of rollouts, Adam optimizer. (numpy only - no deep-learning framework is installed.)
"""
import numpy as np
from .heuristics import Builder, N_FEAT, local_search, relax, finalize


FAIL_PER_EV = 15.0


def _softmax(s):
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()


def rollout(inst, w, rng, greedy_pick=False):
    b = Builder(inst)
    traj = []
    for i in b.order:
        acts, F = b.actions(i)
        if not acts:
            return None, traj
        p = _softmax(F @ w)
        a = int(np.argmax(p)) if greedy_pick else int(rng.choice(len(acts), p=p))
        traj.append((F, p, a))
        b.apply(i, acts[a])
    return b.result(), traj


class RLAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(N_FEAT)       # start from an uninformed policy (not the greedy rule)
        self.history = []
        self.val_history = []

    def train(self, make_inst, iters=300, batch=16, lr=0.05, log_every=25, val_fn=None, val_every=50):
        m1 = np.zeros(N_FEAT)
        m2 = np.zeros(N_FEAT)
        b1, b2 = 0.9, 0.999
        for it in range(1, iters + 1):
            inst = relax(make_inst(it))
            costs, grads, fails = [], [], 0
            for _ in range(batch):
                sol, traj = rollout(inst, self.w, self.rng)
                if sol is None or not sol.feasible or sol.parts["overflow"] > 1e-9:
                    c = inst.n * FAIL_PER_EV     # dead end (needed last-resort overflow): capped failure penalty
                    fails += 1
                else:
                    c = sol.cost
                g = np.zeros(N_FEAT)
                for F, p, a in traj:
                    g += F[a] - p @ F            # grad log pi(a|s)
                costs.append(c)
                grads.append(g)
            costs = np.array(costs)
            base = costs.mean()
            adv = (costs - base) / max(costs.std(), 1e-6 * max(base, 1.0), 1e-3)
            adv = np.clip(adv, -3, 3)
            grad = -sum(a * g for a, g in zip(adv, grads)) / batch    # ascent direction for E[-cost]
            m1 = b1 * m1 + (1 - b1) * grad
            m2 = b2 * m2 + (1 - b2) * grad ** 2
            self.w += lr * (m1 / (1 - b1 ** it)) / (np.sqrt(m2 / (1 - b2 ** it)) + 1e-8)
            self.history.append((it, base / inst.n, fails))
            if log_every and it % log_every == 0:
                print(f"  it {it:4d}  mean cost/EV {base / inst.n:7.3f}  failed {fails}/{batch}", flush=True)
            if val_fn is not None and (it % val_every == 0 or it == 1):
                v = val_fn(self)
                self.val_history.append((it, v))
                print(f"     validation mean cost (greedy decode): {v:.2f}", flush=True)
        return self

    def solve(self, inst, samples=30, polish=True):
        """Greedy decode + best of `samples` stochastic rollouts, optionally polished by local search."""
        rel = relax(inst)
        cands = [rollout(rel, self.w, self.rng, greedy_pick=True)[0]]
        cands += [rollout(rel, self.w, self.rng)[0] for _ in range(samples)]
        cands = [c for c in (finalize(inst, c) for c in cands if c is not None) if c is not None]
        if not cands:
            # no hard-feasible rollout: polish the least-overflowing relaxed plan and re-check
            cands = [rollout(rel, self.w, self.rng, greedy_pick=True)[0]]
            return local_search(inst, cands[0]) if polish and cands[0] is not None else None
        best = min(cands, key=lambda c: c.cost)
        return local_search(inst, best) if polish else best
