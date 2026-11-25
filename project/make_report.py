"""Build figures and markdown tables from results/*.csv"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evcharge.instance import make_instance
from evcharge.milp import solve_milp

R = "results/"
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
COL = {"MILP": "#1f4e79", "Greedy": "#bbbbbb", "Greedy+LS": "#e08e0b",
       "RL (greedy decode)": "#8fce8f", "RL (30 samples + LS)": "#2a9d3a"}
out = []


def md(df, floatfmt=2):
    return df.to_markdown(floatfmt=f".{floatfmt}f")


# ---------------------------------------------------------------- RL training curve
d = json.load(open(R + "rl_policy.json"))
h = np.array(d["history"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
w = 25
ax[0].plot(h[:, 0], h[:, 1], alpha=.25, color="C0")
ax[0].plot(h[w - 1:, 0], np.convolve(h[:, 1], np.ones(w) / w, "valid"), color="C0")
ax[0].set(xlabel="training iteration", ylabel="mean sampled cost per EV", title="REINFORCE training")
v = np.array(d["val_history"], dtype=float)
ax[1].plot(v[:, 0], v[:, 1], "o-", color="C2")
ax[1].set(xlabel="training iteration", ylabel="validation cost per EV (greedy decode)", title="Validation (8 held-out instances)")
plt.tight_layout(); plt.savefig(R + "fig_rl_training.png"); plt.close()

# ---------------------------------------------------------------- quality experiment
q = pd.read_csv(R + "quality_all.csv")
q["size"] = q["n"].astype(str) + "/" + q["m"].astype(str) + "/" + q["K"].astype(str)
ref = q[q.method == "MILP"].set_index("instance")
q["milp_cost"] = q["instance"].map(ref["cost"])
q["milp_bound"] = q["instance"].map(ref["bound"])
q["best_known"] = q.groupby("instance")["cost"].transform("min")
q["gap_best"] = 100 * (q["cost"] - q["best_known"]) / q["best_known"]
q["gap_bound"] = 100 * (q["cost"] - q["milp_bound"]) / q["milp_bound"]
order = ["MILP", "Greedy", "Greedy+LS", "RL (greedy decode)", "RL (30 samples + LS)"]
t1 = q.groupby(["n", "method"]).agg(cost=("cost", "mean"), gap_best=("gap_best", "mean"),
                                    gap_bound=("gap_bound", "mean"), time=("runtime", "mean"),
                                    feas=("feasible", "mean")).reset_index()
t1["method"] = pd.Categorical(t1["method"], order)
t1 = t1.sort_values(["n", "method"])
t1.to_csv(R + "table_quality.csv", index=False)
mi = q[q.method == "MILP"].groupby("n").agg(solved_opt=("optimal", "mean"), milp_gap=("gap", "mean"),
                                           milp_time=("runtime", "mean")).reset_index()
mi.to_csv(R + "table_milp.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ns = sorted(q.n.unique()); wd = 0.16
for k, mth in enumerate(order):
    sub = t1[t1.method == mth].set_index("n").reindex(ns)
    ax[0].bar(np.arange(len(ns)) + (k - 2) * wd, sub["gap_best"], wd, label=mth, color=COL[mth])
    ax[1].plot(ns, sub["time"], "o-", label=mth, color=COL[mth])
ax[0].set_xticks(range(len(ns))); ax[0].set_xticklabels(ns)
ax[0].set(xlabel="# EVs", ylabel="gap to best-known cost (%)", title="Solution quality")
ax[1].set(xlabel="# EVs", ylabel="runtime (s)", yscale="log", title="Computation time")
ax[1].legend(fontsize=7)
ax[2].bar(range(len(ns)), mi["milp_gap"] * 100, color=COL["MILP"])
ax[2].set_xticks(range(len(ns))); ax[2].set_xticklabels(ns)
ax[2].set(xlabel="# EVs", ylabel="MILP optimality gap at 60 s (%)", title="MILP gap at time limit")
plt.tight_layout(); plt.savefig(R + "fig_quality.png"); plt.close()

# ---------------------------------------------------------------- analysis
a = pd.read_csv(R + "analysis_all.csv")
a["cost_f"] = a["cost"]


def curve(kind, xlab, fname, numeric=True):
    s = a[a.kind == kind].copy()
    s["x"] = s["value"].astype(float) if numeric else s["value"]
    g = s.groupby("x").agg(cost=("cost", "mean"), mcs_evs=("mcs_evs", "mean"), vans=("vans_used", "mean"),
                           overflow=("overflow", "mean"), util=("fcs_util", "mean"),
                           hard_feas=("hard_feasible", "mean")).reset_index()
    return g


gK = curve("K", "vans", "")
gC = curve("cap", "", "")
gD = curve("demand", "", "")
gL = curve("layout", "", "", numeric=False)
for nm, g in [("K", gK), ("cap", gC), ("demand", gD), ("layout", gL)]:
    g.to_csv(R + f"table_analysis_{nm}.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
ax[0].plot(gK.x, gK.cost, "o-", color="C0"); ax[0].set(xlabel="number of mobile chargers K", ylabel="optimal total cost",
                                                        title="Cost vs fleet size")
ax2 = ax[0].twinx(); ax2.plot(gK.x, gK.mcs_evs, "s--", color="C1"); ax2.set_ylabel("EVs served by MCS", color="C1"); ax2.grid(False)
ax[1].stackplot(gK.x, gK.mcs_evs, 30 - gK.mcs_evs, labels=["served by MCS", "served by FCS"], colors=["#e08e0b", "#1f4e79"], alpha=.8)
ax[1].set(xlabel="number of mobile chargers K", ylabel="EVs (of 30)", title="Who serves the EVs"); ax[1].legend(fontsize=7)
ax[2].bar(gK.x, gK.overflow, color="#c0392b"); ax[2].set(xlabel="number of mobile chargers K", ylabel="FCS overflow penalty in cost", title="Unserved / queued FCS load (soft cap)")
plt.tight_layout(); plt.savefig(R + "fig_fleet_size.png"); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ax[0].plot(gC.x, gC.cost, "o-"); ax[0].set(xlabel="FCS capacity scale", ylabel="optimal cost", title="Station capacity")
ax2 = ax[0].twinx(); ax2.plot(gC.x, gC.mcs_evs, "s--", color="C1"); ax2.set_ylabel("EVs served by MCS", color="C1"); ax2.grid(False)
ax[1].plot(gD.x, gD.cost, "o-"); ax[1].set(xlabel="charging demand scale", ylabel="optimal cost", title="Charging demand")
ax2 = ax[1].twinx(); ax2.plot(gD.x, gD.mcs_evs, "s--", color="C1"); ax2.set_ylabel("EVs served by MCS", color="C1"); ax2.grid(False)
ax[2].bar(gL.x, gL.mcs_evs, color=["C0", "C1", "C2"]); ax[2].set(xlabel="EV spatial layout (Solomon R / C / RC)", ylabel="EVs served by MCS", title="Spatial layout")
plt.tight_layout(); plt.savefig(R + "fig_sensitivity.png"); plt.close()

# scenario grid: savings of fleet (K=3) vs no fleet (K=0)
fgrid = a[a.kind == "fleet_grid"].copy()
fgrid[["d", "c"]] = fgrid["value"].str.strip("()").str.split(",", expand=True).astype(float)
p = fgrid.pivot_table(index=["d", "c", "seed"], columns="K", values="cost").reset_index()
p["saving_pct"] = 100 * (p[0] - p[3]) / p[0]
sg = p.groupby(["d", "c"]).agg(saving=("saving_pct", "mean"), cost0=(0, "mean"), cost3=(3, "mean")).reset_index()
sg.to_csv(R + "table_scenario_grid.csv", index=False)
hard = fgrid[fgrid.K == 0].groupby(["d", "c"]).hard_feasible.mean().reset_index()
M = sg.pivot(index="d", columns="c", values="saving")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
im = ax[0].imshow(M.values, origin="lower", cmap="YlOrRd")
ax[0].set_xticks(range(M.shape[1])); ax[0].set_xticklabels(M.columns); ax[0].set_yticks(range(M.shape[0])); ax[0].set_yticklabels(M.index)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax[0].text(j, i, f"{M.values[i, j]:.0f}%", ha="center", va="center", fontsize=9)
ax[0].set(xlabel="FCS capacity scale", ylabel="demand scale", title="Cost saving from 3 MCS vs none"); ax[0].grid(False)
H = hard.pivot(index="d", columns="c", values="hard_feasible")
im2 = ax[1].imshow(H.values, origin="lower", cmap="Greens", vmin=0, vmax=1)
ax[1].set_xticks(range(H.shape[1])); ax[1].set_xticklabels(H.columns); ax[1].set_yticks(range(H.shape[0])); ax[1].set_yticklabels(H.index)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax[1].text(j, i, f"{100 * H.values[i, j]:.0f}%", ha="center", va="center", fontsize=9)
ax[1].set(xlabel="FCS capacity scale", ylabel="demand scale", title="FCS-only plan can serve everyone (hard cap)"); ax[1].grid(False)
plt.tight_layout(); plt.savefig(R + "fig_scenarios.png"); plt.close()

# ---------------------------------------------------------------- example solution map
inst = make_instance(n=30, m=5, K=3, layout="RC", seed=2)
sol, info = solve_milp(inst, time_limit=30)
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.scatter(*inst.ev_xy.T, c="k", s=14, label="EV request", zorder=3)
ax.scatter(*inst.fcs_xy.T, marker="s", s=90, c="#1f4e79", label="FCS", zorder=3)
ax.scatter(*inst.depot, marker="*", s=220, c="gold", edgecolor="k", label="MCS depot", zorder=4)
for i, j in enumerate(sol.fcs_of):
    if j >= 0:
        ax.plot(*zip(inst.ev_xy[i], inst.fcs_xy[j]), color="#1f4e79", alpha=.25, lw=.8)
for k, r in enumerate([r for r in sol.routes if r]):
    pts = [inst.depot] + [inst.ev_xy[i - 1] for i in r] + [inst.depot]
    ax.plot(*zip(*pts), "-o", color=f"C{k + 1}", lw=2, ms=5, label=f"MCS route {k + 1} ({len(r)} EVs)", zorder=2)
ax.set(title=f"Optimal plan, cost {sol.cost:.1f}", xlabel="km", ylabel="km"); ax.legend(fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(R + "fig_example_solution.png"); plt.close()
print("figures written")
