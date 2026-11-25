"""Train the REINFORCE policy on randomly generated instances (seeds disjoint from the test seeds)."""
import json, time, sys
import numpy as np
from evcharge.instance import make_instance
from evcharge.generate import feasible_instance
from evcharge.rl import RLAgent, rollout
from evcharge.heuristics import relax

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


POOL_SIZE = 150


def _gen(it):
    r = np.random.default_rng(10_000 + it)
    n = int(r.integers(12, 41))
    m = max(3, round(n / 5))
    K = int(r.integers(1, 5))
    return feasible_instance(n=n, m=m, K=K, layout=str(r.choice(["R", "C", "RC"])), seed=10_000 + it,
                             demand_scale=float(r.uniform(0.7, 1.2)), cap_scale=float(r.uniform(0.9, 1.4)))


_POOL = []


def make_train_instance(it):
    return _POOL[it % len(_POOL)]


VAL = [feasible_instance(n=n, m=max(3, round(n / 5)), K=k, layout=l, seed=50_000 + s)
       for s, (n, k, l) in enumerate([(20, 2, "R"), (25, 2, "C"), (30, 3, "RC"), (30, 3, "C"), (35, 3, "RC"),
                                     (20, 1, "RC"), (40, 4, "R"), (25, 3, "RC")])]


def val_fn(agent):
    cs = []
    for inst in VAL:
        s, _ = rollout(relax(inst), agent.w, agent.rng, greedy_pick=True)
        cs.append(min(s.cost, inst.n * 15.0) / inst.n)
    return float(np.mean(cs))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        _POOL.extend(ex.map(_gen, range(POOL_SIZE)))
    print("instance pool ready", flush=True)
    t0 = time.time()
    agent = RLAgent(seed=0)
    print("training REINFORCE policy ...", flush=True)
    agent.train(make_train_instance, iters=ITERS, batch=16, lr=0.03, log_every=100, val_fn=val_fn, val_every=50)
    print(f"done in {time.time() - t0:.0f}s, weights = {np.round(agent.w, 3).tolist()}")
    json.dump({"w": agent.w.tolist(), "history": agent.history, "val_history": agent.val_history, "train_seconds": time.time() - t0},
              open("results/rl_policy.json", "w"))
