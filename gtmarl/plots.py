"""IEEE-style figures and LaTeX table for Sec. V.

python -m gtmarl.plots --runs /data/$USER/gtmarl_runs --out /data/$USER/gtmarl_runs/figs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL_W = 3.5   # IEEE single column [in]
LABEL = {"gippo": "Proposed (graph-aided IPPO)", "ippo": "IPPO w/o GNN", "mappo": "MAPPO (team reward)",
         "greedy": "Greedy shortest path", "myopic": "Myopic best response"}
STYLE = {"gippo": ("C3", "o", "-"), "ippo": ("C0", "s", "--"), "mappo": ("C2", "^", "-."),
         "greedy": ("C7", "x", ":"), "myopic": ("C1", "d", ":")}


def _read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def _smooth(y, k=10):
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([y[:k - 1], c])


def fig_convergence(root, out):
    curves = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "main", "*_N20_M4_s*"))):
        algo = os.path.basename(d).split("_")[0]
        p = os.path.join(d, "progress.csv")
        if os.path.exists(p):
            rows = _read_csv(p)
            curves[algo].append({k: np.array([float(r[k]) if r.get(k) not in (None, "", "nan") else np.nan
                                              for r in rows]) for k in ("iter", "train/J", "train/success",
                                                                        "train/viol")})
    if not curves:
        return
    fig, axs = plt.subplots(1, 3, figsize=(2 * COL_W, 1.6))
    for algo, runs in curves.items():
        c, m, ls = STYLE[algo]
        n = min(len(r["iter"]) for r in runs)
        x = runs[0]["iter"][:n] + 1
        for ax, key, lab in zip(axs, ("train/J", "train/success", "train/viol"),
                                ("Mean individual cost $J_i$", "Mission success rate",
                                 "Rate-violation ratio")):
            Y = np.stack([_smooth(r[key][:n]) for r in runs])
            mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
            ax.plot(x, mu, color=c, ls=ls, label=LABEL[algo])
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
            ax.set_xlabel("Training iteration")
            ax.set_ylabel(lab)
    axs[0].legend(loc="upper right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(out, "fig_convergence.pdf"))
    plt.close(fig)


def _group(rows, key="N"):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r["method"]][int(r[key])].append(r)
    return g


def fig_scalability(res, out):
    rows = []
    for f in ("scal_transfer.csv", "scal_retrained.csv", "scal_heur.csv"):
        p = os.path.join(res, f)
        if os.path.exists(p):
            for r in _read_csv(p):
                if f == "scal_transfer.csv":
                    r["method"] = r["method"] + ("" if int(r["N"]) == int(r["N_train"]) else "_tr")
                rows.append(r)
    if not rows:
        return
    # proposed: transfer (no retraining) curve; MAPPO: retrained per N
    g = _group(rows)
    metrics = [("success", "Mission success rate"), ("viol", "Rate-violation ratio"),
               ("detour", r"Detour $T_i/T_i^{\min}$"), ("jain_util", "Jain's fairness index")]
    fig, axs = plt.subplots(1, 4, figsize=(2 * COL_W, 1.55))
    for method in ("gippo", "mappo", "myopic", "greedy"):
        series = defaultdict(list)
        for key in (method, method + "_tr"):
            for N, rs in g.get(key, {}).items():
                series[N] += rs
        if not series:
            continue
        Ns = sorted(series)
        c, mk, ls = STYLE[method]
        for ax, (k, lab) in zip(axs, metrics):
            vals = [np.nanmean([float(r[k]) for r in series[N]]) for N in Ns]
            sds = [np.nanstd([float(r[k]) for r in series[N]]) for N in Ns]
            ax.errorbar(Ns, vals, yerr=sds, color=c, marker=mk, ls=ls, capsize=2, label=LABEL[method])
            ax.set_xlabel("Number of UAVs $N$")
            ax.set_ylabel(lab)
    axs[0].legend(loc="lower left")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(out, "fig_scalability.pdf"))
    plt.close(fig)


def fig_nash_gap(res, out):
    pts = defaultdict(list)
    final = defaultdict(list)
    for f in glob.glob(os.path.join(res, "br_*.json")):
        d = json.load(open(f))
        it = int(d["ckpt"].split("_")[1].split(".")[0])
        if d["algo"] == "gippo":
            pts[it].append(d["gap_rel"])
        if it == max(1500, it):
            final[d["algo"]].append(d["gap_rel"])
    if not pts:
        return
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.5), gridspec_kw={"width_ratios": [1.6, 1]})
    its = sorted(pts)
    mu = [100 * np.mean(pts[i]) for i in its]
    sd = [100 * np.std(pts[i]) for i in its]
    axs[0].errorbar(its, mu, yerr=sd, color=STYLE["gippo"][0], marker="o", capsize=2)
    axs[0].set_xlabel("Training iteration")
    axs[0].set_ylabel("Relative BR gap (%)")
    algos = [a for a in ("gippo", "ippo", "mappo") if final.get(a)]
    axs[1].bar(range(len(algos)), [100 * np.mean(final[a]) for a in algos],
               yerr=[100 * np.std(final[a]) for a in algos], color=[STYLE[a][0] for a in algos], capsize=2)
    axs[1].set_xticks(range(len(algos)))
    axs[1].set_xticklabels(["Prop.", "IPPO", "MAPPO"][:len(algos)] if algos == ["gippo", "ippo", "mappo"]
                           else algos)
    axs[1].set_ylabel("Final gap (%)")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(out, "fig_nash_gap.pdf"))
    plt.close(fig)


def fig_traj(res, out):
    files = [f for f in ("traj_gippo.npz", "traj_mappo.npz") if os.path.exists(os.path.join(res, f))]
    if not files:
        return
    fig, axs = plt.subplots(1, len(files), figsize=(COL_W * len(files) / 2 * 1.0, 1.75), squeeze=False)
    for ax, f in zip(axs[0], files):
        d = np.load(os.path.join(res, f))
        serve = d["serve"][:, :, 0].T
        ax.imshow(serve, origin="lower", cmap="Pastel1", alpha=0.6, interpolation="nearest")
        g = d["gbs_vox"]
        ax.scatter(g[:, 0], g[:, 1], marker="^", s=30, c="k", zorder=5)
        tr = d["traj"]
        for i in range(tr.shape[1]):
            ax.plot(tr[:, i, 0], tr[:, i, 1], lw=0.7)
            ax.plot(*d["orig"][i, :2], "o", ms=2, c="k")
            ax.plot(*d["dest"][i, :2], "*", ms=3.5, c="r" if not d["arrived"][i] else "g")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(LABEL[f.split("_")[1].split(".")[0]], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_trajectories.pdf"))
    plt.close(fig)


def table_main(res, out):
    p = os.path.join(res, "main.csv")
    if not os.path.exists(p):
        return
    g = defaultdict(list)
    for r in _read_csv(p):
        g[r["method"]].append(r)
    cols = [("success", "Success", 1), ("detour", "Detour", 1), ("J", "$J_i$", 1), ("welfare", "$\\sum_i J_i$", 1),
            ("viol", "Viol.", 1), ("jain_util", "Jain", 1)]
    lines = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for m in ("gippo", "ippo", "mappo", "myopic", "greedy"):
        if m not in g:
            continue
        cells = []
        for k, _, _ in cols:
            v = np.array([float(r[k]) for r in g[m]])
            cells.append(f"{np.nanmean(v):.3f}" + (f" $\\pm$ {np.nanstd(v):.3f}" if len(v) > 1 else ""))
        lines.append(LABEL[m] + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out, "table_main.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = os.path.join(a.runs, "results")
    out = a.out or os.path.join(a.runs, "figs")
    os.makedirs(out, exist_ok=True)
    for fn in (fig_convergence,):
        fn(a.runs, out)
    for fn in (fig_scalability, fig_nash_gap, fig_traj, table_main):
        fn(res, out)
    print("figures in", out)


if __name__ == "__main__":
    main()
