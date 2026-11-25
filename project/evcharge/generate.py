"""Feasible-instance generation: draw seeds until the MILP finds at least one feasible plan."""
from .instance import make_instance
from .milp import solve_milp


def feasible_instance(max_tries=40, check_time=3.0, **kw):
    seed0 = kw.pop("seed", 0)
    for t in range(max_tries):
        inst = make_instance(seed=seed0 + 1000 * t, **kw)
        sol, info = solve_milp(inst, time_limit=check_time)
        if sol is not None and sol.feasible:
            return inst
    raise RuntimeError(f"no feasible instance found for {kw}")
