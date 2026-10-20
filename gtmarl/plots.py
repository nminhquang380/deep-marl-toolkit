"""IEEE-style figures and LaTeX tables for Sec. V of manuscript v22.

python -m gtmarl.plots --runs gtmarl/data/gtmarl_runs            # -> <runs>/figs

Outputs (only those whose inputs exist):
  fig_convergence.pdf   V-B  return, stage potential, entropy, BR gap vs training
  table_main.tex        V-C  MSR, detour, P_out, Jain, social cost, BR gap
  fig_scalability.pdf   V-D  MSR / P_out / detour vs N for B = 4, 9 (+ inference time)
  fig_quant_mu.pdf      V-E  effect of L and of mu
  table_ablation.tex    V-F
  fig_trajectories.pdf  V-G  SP vs proposed + cell utilisation over time
"""
from __future__ import annotations

import argparse
import csv
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
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.5     # IEEE single column [in]
DCOL = 7.16   # double column [in]
LABEL = {"gippo": "Proposed", "gippo_tr": "Proposed (trained at $N=30$)", "ippo_mlp": "IPPO-MLP",
         "mappo": "CTDE-MAPPO", "greedy": "SP", "myopic": "MBR"}
STYLE = {"gippo": ("C3", "o", "-"), "gippo_tr": ("C3", "o", "--"), "ippo_mlp": ("C0", "s", "-"),
         "mappo": ("C2", "^", "-."), "greedy": ("C7", "x", ":"), "myopic": ("C1", "d", ":")}


def rd(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def smooth(y, k=15):
    if len(y) < k:
        return y
    return np.concatenate([y[:k - 1], np.convolve(y, np.ones(k) / k, mode="valid")])


def ms(vals):
    v = np.array([fnum(x) for x in vals], dtype=float)
    return np.nanmean(v), (np.nanstd(v) if np.sum(~np.isnan(v)) > 1 else 0.0)


def br_results(res):
    out = defaultdict(lambda: defaultdict(list))         # algo -> ckpt iter -> [gap_rel]
    for f in glob.glob(os.path.join(res, "br_*.json")):
        d = json.load(open(f))
        c = d["ckpt"]
        it = int(c.split("_")[1].split(".")[0]) if c.split("_")[1].split(".")[0].isdigit() else -1
        out[d["algo"]][it].append(d["gap_rel"])
    return out


# ------------------------------------------------------------------ V-B
def fig_convergence(root, res, out):
    curves = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "main", "*_N30_M4_s*"))):
        algo = os.path.basename(d).split("_N30")[0]
        rows = rd(os.path.join(d, "progress.csv"))
        if rows:
            curves[algo].append({k: np.array([fnum(r.get(k)) for r in rows]) for k in
                                 ("iter", "train/shaped_return", "train/potential", "loss/entropy")})
    br = br_results(res)
    if not curves:
        return
    fig, ax = plt.subplots(1, 4, figsize=(DCOL, 1.55))
    for algo in ("gippo", "ippo_mlp", "mappo"):
        runs = curves.get(algo)
        if not runs:
            continue
        c, mk, ls = STYLE[algo]
        n = min(len(r["iter"]) for r in runs)
        x = runs[0]["iter"][:n] + 1
        for a, key in zip(ax[:3], ("train/shaped_return", "train/potential", "loss/entropy")):
            Y = np.stack([smooth(r[key][:n]) for r in runs])
            mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
            a.plot(x, mu, color=c, ls=ls, label=LABEL[algo])
            a.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    for a, lab in zip(ax[:3], ("Training return", r"Stage potential $\phi_x$", "Policy entropy")):
        a.set_xlabel("Iteration"); a.set_ylabel(lab)
    ax[0].legend(loc="lower right")
    its = sorted(i for i in br.get("gippo", {}) if i > 0)
    if its:
        m = [100 * np.mean(br["gippo"][i]) for i in its]
        s = [100 * np.std(br["gippo"][i]) for i in its]
        ax[3].errorbar(its, m, yerr=s, color=STYLE["gippo"][0], marker="o", capsize=2)
    ax[3].set_xlabel("Iteration"); ax[3].set_ylabel(r"BR gap $\hat\epsilon$ (%)")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(os.path.join(out, "fig_convergence.pdf")); plt.close(fig)


# ------------------------------------------------------------------ V-C
def table_main(res, out):
    rows = [r for r in rd(os.path.join(res, "main.csv")) if not r.get("tag")]
    if not rows:
        return
    br = br_results(res)
    g = defaultdict(list)
    for r in rows:
        g[r["method"]].append(r)
    cols = [("success", "MSR", 3), ("detour", "Detour", 3), ("p_out", "$P_{\\rm out}$", 4),
            ("jain_load", "Jain", 3), ("social_cost", "$\\sum_i J_i$", 1)]
    L = ["\\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", "\\toprule",
         "Method & " + " & ".join(c[1] for c in cols) + " & $\\hat\\epsilon$ (\\%) \\\\", "\\midrule"]
    for m in ("greedy", "myopic", "ippo_mlp", "mappo", "gippo"):
        if m not in g:
            continue
        cells = []
        for k, _, p in cols:
            mu, sd = ms([r[k] for r in g[m]])
            cells.append(f"{mu:.{p}f}" + (f" $\\pm$ {sd:.{p}f}" if sd > 0 else ""))
        gaps = [v for it, vs in br.get(m, {}).items() if it in (1500, -1) for v in vs]
        cells.append(f"{100 * np.mean(gaps):.1f}" if gaps else "--")
        L.append(LABEL[m] + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table_main.tex"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ V-D
def fig_scalability(res, out):
    data = {}
    for B in (4, 9):
        series = defaultdict(lambda: defaultdict(list))
        for r in rd(os.path.join(res, f"scal_transfer_B{B}.csv")):
            series["gippo_tr"][int(r["N"])].append(r)
        for r in rd(os.path.join(res, "scal_retrained.csv")):
            if int(r["M"]) == B and r["method"] in ("gippo", "mappo") and not r.get("tag"):
                series[r["method"]][int(r["N"])].append(r)
        for r in rd(os.path.join(res, f"scal_heur_B{B}.csv")):
            series[r["method"]][int(r["N"])].append(r)
        data[B] = series
    if not any(data[B] for B in data):
        return
    metrics = [("success", "MSR"), ("p_out", r"$P_{\rm out}$"), ("detour", "Detour factor")]
    fig, ax = plt.subplots(2, 3, figsize=(DCOL, 3.0), sharex=True)
    for i, B in enumerate((4, 9)):
        for m in ("gippo", "gippo_tr", "mappo", "myopic", "greedy"):
            ser = data[B].get(m)
            if not ser:
                continue
            Ns = sorted(ser)
            c, mk, ls = STYLE[m]
            for j, (k, lab) in enumerate(metrics):
                v = [ms([r[k] for r in ser[N]]) for N in Ns]
                ax[i, j].errorbar(Ns, [a for a, _ in v], yerr=[b for _, b in v], color=c, marker=mk,
                                  ls=ls, capsize=2, label=LABEL[m])
                ax[i, j].set_ylabel(lab)
        ax[i, 0].set_title(f"$B={B}$", fontsize=8, loc="left")
    for j in range(3):
        ax[1, j].set_xlabel("Number of UAVs $N$")
    ax[0, 0].legend(loc="lower left")
    fig.tight_layout(h_pad=0.3, w_pad=0.4)
    fig.savefig(os.path.join(out, "fig_scalability.pdf")); plt.close(fig)
    # inference time (per UAV) vs N
    lines = ["N,B,infer_us_mean,infer_us_std"]
    for B in (4, 9):
        for N, rs in sorted(data[B].get("gippo_tr", {}).items()):
            mu, sd = ms([r["infer_us"] for r in rs])
            lines.append(f"{N},{B},{mu:.1f},{sd:.1f}")
    open(os.path.join(out, "inference_time.csv"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ V-E
def fig_quant_mu(res, out):
    q = defaultdict(list)
    for r in rd(os.path.join(res, "quant.csv")):
        q[int(r["L"])].append(r)
    qh = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(res, "quant_heur_L*.csv")):
        for r in rd(f):
            qh[r["method"]][int(r["L"])].append(r)
    mu = defaultdict(list)
    for r in rd(os.path.join(res, "mu.csv")):
        mu[float(r["mu"])].append(r)
    if not q and not mu:
        return
    fig, ax = plt.subplots(1, 3, figsize=(DCOL, 1.6))
    for name, ser in (("gippo", q), ("greedy", qh.get("greedy", {})), ("myopic", qh.get("myopic", {}))):
        if not ser:
            continue
        Ls = sorted(ser)
        c, mk, ls = STYLE[name]
        for a, k in zip(ax[:2], ("success", "p_out")):
            v = [ms([r[k] for r in ser[L]]) for L in Ls]
            a.errorbar(Ls, [x for x, _ in v], yerr=[y for _, y in v], color=c, marker=mk, ls=ls,
                       capsize=2, label=LABEL[name])
    ax[0].set_ylabel("MSR"); ax[1].set_ylabel(r"$P_{\rm out}$")
    for a in ax[:2]:
        a.set_xlabel("SNR levels $L$"); a.set_xticks([1, 2, 3, 4])
    ax[0].legend(loc="lower right")
    if mu:
        mus = sorted(mu)
        d = [ms([r["detour"] for r in mu[m]]) for m in mus]
        p = [ms([r["p_out"] for r in mu[m]]) for m in mus]
        ax[2].errorbar([x for x, _ in d], [x for x, _ in p], xerr=[y for _, y in d], yerr=[y for _, y in p],
                       color=STYLE["gippo"][0], marker="o", capsize=2)
        eta = 1.5
        for m, (dx, _), (py, _) in zip(mus, d, p):
            ax[2].annotate(f"{m / eta:g}$\\eta$", (dx, py), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax[2].set_xlabel("Detour factor"); ax[2].set_ylabel(r"$P_{\rm out}$")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(os.path.join(out, "fig_quant_mu.pdf")); plt.close(fig)


# ------------------------------------------------------------------ V-F
def table_ablation(res, out):
    rows = rd(os.path.join(res, "abl.csv"))
    if not rows:
        return
    g = defaultdict(list)
    for r in rows:
        g[r.get("tag") or "full"].append(r)
    names = [("full", "Proposed (full)"), ("nococell", "w/o co-cell indicator"),
             ("nobeta", "w/o neighbour bearer $\\beta_j$"), ("nopos", "w/o own position $\\mathbf{q}_i$"),
             ("noshape", "w/o shaping $r_i^{\\rm nav}$"), ("lostpen", "penalty $\\beta_i-\\kappa_i$"),
             ("w0", "constant failure cost")]
    cols = [("success", "MSR", 3), ("detour", "Detour", 3), ("p_out", "$P_{\\rm out}$", 4),
            ("social_cost", "$\\sum_i J_i$", 1)]
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
         "Variant & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for key, lab in names:
        if key not in g:
            continue
        cells = []
        for k, _, p in cols:
            mu, sd = ms([r[k] for r in g[key]])
            cells.append(f"{mu:.{p}f}" + (f" $\\pm$ {sd:.{p}f}" if sd > 0 else ""))
        L.append(lab + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "table_ablation.tex"), "w").write("\n".join(L) + "\n")


# ------------------------------------------------------------------ V-G
def fig_traj(res, out):
    files = [(n, os.path.join(res, f)) for n, f in (("greedy", "traj_sp.npz"), ("gippo", "traj_gippo.npz"))
             if os.path.exists(os.path.join(res, f))]
    if not files:
        return
    fig, ax = plt.subplots(1, len(files) + 1, figsize=(DCOL, 2.2),
                           gridspec_kw={"width_ratios": [1] * len(files) + [1.3]})
    for a, (name, f) in zip(ax, files):
        d = np.load(f)
        a.imshow(d["serve"][:, :, 0].T, origin="lower", cmap="Pastel1", alpha=0.6, interpolation="nearest")
        g = d["gbs_vox"]
        a.scatter(g[:, 0], g[:, 1], marker="^", s=25, c="k", zorder=5)
        tr = d["traj"]
        for i in range(tr.shape[1]):
            a.plot(tr[:, i, 0], tr[:, i, 1], lw=0.5)
            a.plot(*d["dest"][i, :2], "*", ms=3, c="g" if d["arrived"][i] else "r")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(LABEL[name], fontsize=8)
        D = d["D"] / float(d["K"])                       # (T+1, B)
        c, _, ls = STYLE[name]
        ax[-1].plot(D.max(1), color=c, ls=ls, label=f"{LABEL[name]}: max$_b$")
        ax[-1].plot(D.min(1), color=c, ls=ls, alpha=0.5, lw=0.8, label=f"{LABEL[name]}: min$_b$")
    ax[-1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[-1].set_xlabel("Slot $t$"); ax[-1].set_ylabel("Cell utilisation $D_b/K_b$")
    ax[-1].legend(loc="upper right")
    fig.tight_layout(w_pad=0.4)
    fig.savefig(os.path.join(out, "fig_trajectories.pdf")); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = os.path.join(a.runs, "results")
    out = a.out or os.path.join(a.runs, "figs")
    os.makedirs(out, exist_ok=True)
    fig_convergence(a.runs, res, out)
    table_main(res, out)
    fig_scalability(res, out)
    fig_quant_mu(res, out)
    table_ablation(res, out)
    fig_traj(res, out)
    print("figures/tables in", out, ":", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
