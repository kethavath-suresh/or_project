"""Instance definition and Solomon-style synthetic generator."""
from dataclasses import dataclass, field, replace
import numpy as np


@dataclass
class Instance:
    n: int                      # number of EV requests
    m: int                      # number of fixed charging stations (FCS)
    K: int                      # number of mobile charging vans (MCS)
    depot: np.ndarray           # (2,) km
    ev_xy: np.ndarray           # (n,2) km
    fcs_xy: np.ndarray          # (m,2) km
    q: np.ndarray               # (n,) charging demand kWh
    e: np.ndarray               # (n,) earliest service time at EV (h)  (request release)
    l: np.ndarray               # (n,) latest service start when served by an MCS (h)
    cap_f: np.ndarray           # (m,) available gun-hours of each FCS in the horizon
    T: float = 4.0              # planning horizon (h), evening peak
    Q: float = 120.0            # MCS battery capacity (kWh)
    speed: float = 30.0         # km/h, same for all vans (and EVs)
    p_f: float = 22.0           # FCS power (kW)  (L2)
    p_m: float = 22.0           # MCS charging power (kW)
    guns: int = 2               # charging points per FCS
    # cost parameters
    c_ev: float = 0.5           # cost / km of EV driving to an FCS
    c_mcs: float = 1.5          # cost / km of MCS driving
    f_mcs: float = 20.0         # fixed dispatch cost per van used
    c_batt: float = 0.05        # cost / kWh delivered by MCS (battery wear + recharge)
    lam_ev: float = 4.0         # inconvenience / hour of EV detour to an FCS
    lam_wait: float = 6.0       # inconvenience / hour an EV waits for an MCS after release
    overflow_cost: float | None = None   # None => FCS capacity is a hard constraint
    name: str = "inst"

    def __post_init__(self):
        P = np.vstack([self.depot[None], self.ev_xy, self.fcs_xy])
        D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
        n = self.n
        # index map: 0 depot, 1..n EVs, n+1..n+m FCS
        self.D_all = D
        self.d_mcs = D[: n + 1, : n + 1]               # depot+EV matrix (km)
        self.d_fcs = D[1: n + 1, n + 1:]               # EV -> FCS (km)
        self.tau = self.d_mcs / self.speed             # travel time van (h)
        self.dur_f = self.q / self.p_f                 # gun-hours when at FCS
        self.s = self.q / self.p_m                     # service time with MCS (h), index 0..n-1

    # convenience: 1-indexed EV helpers (0 = depot)
    def s1(self, i):
        return 0.0 if i == 0 else self.s[i - 1]

    def with_(self, **kw):
        return replace(self, **kw)


def make_instance(n=20, m=4, K=3, layout="RC", seed=0, T=4.0, demand_scale=1.0,
                  cap_scale=1.0, area=20.0, **kw):
    """Solomon-style synthetic instance.
    layout: 'R' random, 'C' clustered, 'RC' mixed (as in Solomon R/C/RC classes).
    FCS availability is heterogeneous (some stations busy -> uneven utilization)."""
    rng = np.random.default_rng(seed)
    depot = np.array([area / 2, area / 2])
    if layout == "R":
        ev = rng.uniform(0, area, (n, 2))
    else:
        nc = max(2, n // 6)
        centers = rng.uniform(2, area - 2, (nc, 2))
        nclu = n if layout == "C" else n // 2
        lab = rng.integers(0, nc, nclu)
        ev_c = centers[lab] + rng.normal(0, 1.5, (nclu, 2))
        ev_r = rng.uniform(0, area, (n - nclu, 2))
        ev = np.clip(np.vstack([ev_c, ev_r]), 0, area)
    fcs = rng.uniform(0.1 * area, 0.9 * area, (m, 2))
    q = np.clip(rng.normal(20, 7, n), 8, 40) * demand_scale
    e = rng.uniform(0, T - 1.5, n)
    width = rng.uniform(0.75, 1.5, n)
    l = np.minimum(e + width, T - 0.3)
    avail = rng.uniform(0.3, 0.8, m)       # fraction of gun-time not yet booked
    cap_f = 2 * T * avail * cap_scale
    return Instance(n=n, m=m, K=K, depot=depot, ev_xy=ev, fcs_xy=fcs, q=q, e=e, l=l,
                    cap_f=cap_f, T=T, name=f"{layout}-n{n}-m{m}-K{K}-s{seed}", **kw)
