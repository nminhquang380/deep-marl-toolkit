"""Figures of Section V from the sweep results (parts A-D).

python figs.py RES_DIR TRAJ_DIR OUT_DIR
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RES, TRAJ, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- style (IEEE column, Times-like serif)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "lines.markersize": 4, "grid.linewidth": 0.4, "grid.color": "#d9d9d6", "axes.grid": True,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#d9d9d6",
    "legend.handlelength": 1.8, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
COL = 3.5                                   # IEEE single column [in]
GREEN, RED, GOLD, GRAY, DARK = "#17643a", "#e65a3c", "#cf9a0b", "#8a8984", "#52514e"
STY = {  # method -> (label, color, marker, linestyle)
    "egat_cc": ("EGAT-IPPO", GREEN, "o", "-"),
    "egat_tr": ("EGAT-IPPO, trained at $N{=}50$", GREEN, "o", "--"),
    "ippo_mlp": ("IPPO-MLP", RED, "s", "-"),
    "mappo": ("CTDE-MAPPO", GOLD, "^", "-"),
    "greedy": ("SP", DARK, "x", ":"),
    "myopic": ("MBR", GRAY, "d", "-."),
}


def csv(path):
    return pd.read_csv(os.path.join(RES, path))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def curves(pattern, col):
    """Evaluation curves (every 50 iterations) of the runs matching pattern: iterations, matrix (runs x points)."""
    ds = [pd.read_csv(f).dropna(subset=[col]) for f in sorted(glob.glob(os.path.join(RES, pattern, "progress.csv")))]
    n = min(len(d) for d in ds)
    it = ds[0]["iter"].values[:n] + 1
    return it, np.stack([d[col].values[:n] for d in ds])


# =============================================================== Fig. 3: convergence (V-B)
def fig_convergence():
    rows = [("eval/success", "MSR"), ("eval/J", r"Average cost $\bar{J}$"),
            ("eval/potential", r"Stage potential $\bar{\phi}$")]
    fig, ax = plt.subplots(3, 1, figsize=(COL, 3.9), sharex=True)
    for key, pat in (("egat_cc", "A/main2400/egat_cc_N50_M4_s*" if glob.glob(os.path.join(RES, "A/main2400/egat_cc_N50_M4_s*")) else "A/main/egat_cc_N50_M4_s*"), ("ippo_mlp", "A/main2400/ippo_mlp_N50_M4_s*")):
        lab, c, m, ls = STY[key]
        for a, (col, _) in zip(ax, rows):
            it, Y = curves(pat, col)
            mu, sd = Y.mean(0), Y.std(0, ddof=1)
            a.plot(it, mu, color=c, ls=ls, label=lab)
            a.fill_between(it, mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    for a, (_, yl) in zip(ax, rows):
        a.set_ylabel(yl)
    ax[0].set_ylim(0, 1.02)
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for a, ticks in ((ax[1], [20, 50, 100, 200]), (ax[2], [1, 2, 5, 10])):
        a.set_yscale("log"); a.yaxis.set_major_locator(FixedLocator(ticks)); a.yaxis.set_major_formatter(fmt)
        a.yaxis.set_minor_locator(NullLocator())
    ax[-1].set_xlabel("Training iteration")
    ax[-1].set_xlim(0, 2400)
    ax[0].legend(loc="center right")
    for a, t in zip(ax, "abc"):
        a.text(0.985, 0.93, f"({t})", transform=a.transAxes, va="top", ha="right", fontsize=8)
    fig.align_ylabels(ax)
    fig.tight_layout(h_pad=0.3)
    save(fig, "fig_convergence.pdf")


# =============================================================== Fig. 4: scalability (V-D)
def scal_data():
    main, n30, scal, s24, ip24 = (csv("A/results/main.csv"), csv("A/results/n30.csv"), csv("B/results/scal.csv"),
                                  csv("B/results/scal2400.csv"), csv("A/results/ippo2400.csv"))
    out = {}
    def agg(df, N):
        return df[df.N == N]
    egat = {30: n30[n30.method == "egat_cc"], 40: scal[(scal.method == "egat_cc") & (scal.N == 40)],
            50: main[(main.method == "egat_cc") & (main.N == 50)], 60: s24[s24.method == "egat_cc"]}
    egat_tr = {N: main[(main.method == "egat_cc") & (main.N == N)] for N in (30, 40, 50, 60)}
    ippo = {40: scal[(scal.method == "ippo_mlp") & (scal.N == 40)], 50: ip24, 60: s24[s24.method == "ippo_mlp"]}
    sp = {N: main[(main.method == "greedy") & (main.N == N)] for N in (30, 40, 50, 60)}
    mbr = {N: main[(main.method == "myopic") & (main.N == N)] for N in (30, 40, 50, 60)}
    return {"egat_cc": egat, "egat_tr": egat_tr, "ippo_mlp": ippo, "greedy": sp, "myopic": mbr}


def fig_scalability():
    data = scal_data()
    fig, ax = plt.subplots(1, 2, figsize=(COL * 2, 2.0))
    for key in ("greedy", "myopic", "ippo_mlp", "egat_tr", "egat_cc"):
        lab, c, m, ls = STY[key]
        Ns = sorted(data[key])
        for a, col, sc in ((ax[0], "J", 1.0), (ax[1], "p_out", 100.0)):
            mu = np.array([data[key][N][col].mean() * sc for N in Ns])
            sd = np.array([data[key][N][col].std(ddof=1) * sc if len(data[key][N]) > 1 else 0 for N in Ns])
            mfc = "white" if key == "egat_tr" else c
            a.errorbar(Ns, mu, yerr=sd, color=c, marker=m, ls=ls, mfc=mfc, capsize=2, elinewidth=0.6,
                       label=lab, zorder=3 if key.startswith("egat") else 2)
    ax[0].set_ylabel(r"Average cost $\bar{J}$"); ax[1].set_ylabel(r"Outage probability $P_{\mathrm{out}}$ [%]")
    for a in ax:
        a.set_xlabel("Number of UAVs $N$"); a.set_xticks([30, 40, 50, 60]); a.set_yscale("log")
    h, l = ax[0].get_legend_handles_labels()
    order = [4, 3, 2, 0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.08), frameon=False, columnspacing=1.2)
    for a, t in zip(ax, "ab"):
        a.text(0.01, 0.97, f"({t})", transform=a.transAxes, va="top", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_scalability.pdf")


# =============================================================== Fig. 5: completion-time CDF vs eta (V-E)
def fig_deadline():
    eta, ref = csv("B/results/eta.csv"), csv("A/results/ref1200.csv")
    grid = np.round(np.arange(1.0, 3.01, 0.1), 1)
    fcols = [f"F@{x:.1f}" for x in grid]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    shades = {1.2: "#9fd1ae", 1.5: "#4fa36b", 2.0: "#17643a", 3.0: "#0b3a1f"}
    mk = {1.2: "v", 1.5: "o", 2.0: "s", 3.0: "D"}
    for e in (1.2, 1.5, 2.0, 3.0):
        df = ref if e == 1.5 else eta[(eta.method == "egat_cc") & (np.isclose(eta.eta_eval, e))]
        F = df[fcols].values.mean(0)
        ax.plot(grid, F, color=shades[e], marker=mk[e], markevery=2, ms=3, label=rf"EGAT-IPPO, $\eta={e:g}$")
        ax.axvline(e, color=shades[e], lw=0.6, ls=":")
    sp = eta[(eta.method == "greedy")].iloc[0]
    ax.plot(grid, sp[fcols].values.astype(float), color=DARK, ls=":", marker="x", markevery=2, ms=3, label="SP (any $\\eta$)")
    ax.set_xlabel(r"Normalized completion time $T_i / T_i^{\min}$")
    ax.set_ylabel(r"$\Pr[T_i / T_i^{\min} \leq x]$")
    ax.set_xlim(1.0, 3.0); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig_deadline_cdf.pdf")


# =============================================================== Fig. 6: penalty trade-off (V-F)
def fig_penalty():
    pen, main = csv("C/results/pen1600.csv"), csv("A/results/main.csv")
    fam = {  # family -> (label, color, marker, linestyle)
        "marginal": ("Marginal (proposed)", GREEN, "o", "-"),
        "indicator": ("Violation indicator", RED, "s", "--"),
        "shortfall": ("Normalized shortfall", GOLD, "^", "-."),
        "lost": ("Lost RBs", DARK, "d", ":"),
    }
    OFFS = {("marginal", 4.5): (-5, -10), ("marginal", 9.0): (5, -9), ("marginal", 2.25): (5, 2),
            ("indicator", 22.0): (-5, 6), ("indicator", 11.0): (5, 0), ("indicator", 5.5): (5, 0),
            ("shortfall", 50.0): (-5, 4), ("shortfall", 100.0): (5, 2), ("shortfall", 200.0): (-3, 5),
            ("lost", 2.25): (5, 0), ("lost", 4.5): (5, 0), ("lost", 9.0): (5, 0)}
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    ref = main[(main.method == "egat_cc") & (main.N == 50)]
    for f, (lab, c, m, ls) in fam.items():
        d = pen[pen.penalty == f][["mu", "detour", "p_out"]].copy()
        if f == "marginal":
            d = pd.concat([d, pd.DataFrame([{"mu": 4.5, "detour": ref.detour.mean(), "p_out": ref.p_out.mean()}])])
            ax.errorbar([ref.detour.mean()], [100 * ref.p_out.mean()], xerr=[ref.detour.std(ddof=1)],
                        yerr=[100 * ref.p_out.std(ddof=1)], color=c, capsize=2, elinewidth=0.7, ls="none", zorder=4)
        d = d.sort_values("mu")
        ax.plot(d.detour, 100 * d.p_out, color=c, ls=ls, lw=1.0, marker=m, ms=5.5, mec="white", mew=0.6,
                label=lab, zorder=5 if f == "marginal" else 3)
        for _, r in d.iterrows():
            off = OFFS.get((f, float(r.mu)), (5, 0))
            ax.annotate(f"{r.mu:g}", (r.detour, 100 * r.p_out), xytext=off, textcoords="offset points",
                        fontsize=6.5, color=DARK, ha="left" if off[0] >= 0 else "right", va="center")
    ax.set_xlabel(r"Detour factor $T_i / T_i^{\min}$")
    ax.set_ylabel(r"Outage probability $P_{\mathrm{out}}$ [%]")
    ax.set_xlim(1.085, 1.215); ax.set_ylim(0.3, 2.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, columnspacing=1.2)
    ax.text(0.99, 0.97, r"labels: weight $\mu$", transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=GRAY)
    fig.tight_layout()
    save(fig, "fig_penalty.pdf")


# =============================================================== Fig. 7: trajectories (V-G)
def fig_traj(seed=1001):
    files = [("greedy", f"traj_sp_{seed}.npz"), ("egat_cc", f"traj_egat_{seed}.npz")]
    HOT = "#d62728"
    fig = plt.figure(figsize=(COL * 2, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.06)
    ax_u = fig.add_subplot(gs[0, 2])
    for k, (name, f) in enumerate(files):
        d = np.load(os.path.join(TRAJ, f))
        a = fig.add_subplot(gs[0, k])
        serve, z = d["serve"], 1
        a.contour(np.arange(60), np.arange(60), serve[:, :, z].T, levels=np.arange(0.5, serve.max()),
                  colors=[DARK], linewidths=0.6)                                   # cell boundaries
        tr, D, K = d["traj"], d["D"], float(d["K"])
        arrived, Tarr = d["arrived"], d["T_arr"]
        hot, n_over = [], 0
        for i in range(tr.shape[1]):
            T = Tarr[i] if arrived[i] else tr.shape[0] - 1
            p = tr[:T + 1, i]
            a.plot(p[:, 0], p[:, 1], color="#c9c8c4", lw=0.35, zorder=1)
            cells = serve[p[1:, 0], p[1:, 1], p[1:, 2]]
            over = D[np.arange(1, T + 1), cells] > K
            n_over += over.sum()
            hot.append(p[1:][over])
        hot = np.concatenate(hot)
        a.scatter(hot[:, 0], hot[:, 1], s=5, c=HOT, linewidths=0, alpha=0.85, zorder=3)
        g = d["gbs_vox"]
        a.scatter(g[:, 0], g[:, 1], marker="^", s=26, c="k", edgecolors="white", linewidths=0.5, zorder=4)
        a.set_xlim(-0.5, 59.5); a.set_ylim(-0.5, 59.5); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for sp_ in a.spines.values():
            sp_.set_color(GRAY)
        a.set_title(f"({'ab'[k]}) {STY[name][0]}", fontsize=8, pad=2)
        a.text(0.03, 0.97, f"{n_over} UAV-slots\nin overload", transform=a.transAxes, ha="left", va="top",
               fontsize=6.5, color=HOT, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=5)
        u = D / K
        c, ls = (DARK, ":") if name == "greedy" else (GREEN, "-")
        ax_u.plot(np.arange(len(u)), u.max(1), color=c, ls=ls, lw=1.2, label=STY[name][0])
    ax_u.axhline(1.0, color=HOT, lw=0.7, ls="--")
    ax_u.text(len(u) - 1, 1.02, "RB budget", fontsize=6.5, color=HOT, va="bottom", ha="right")
    ax_u.set_xlabel("Slot $t$")
    ax_u.set_ylabel(r"$\max_{b}\, D_b(t)/K_b$")
    ax_u.yaxis.set_label_position("right"); ax_u.yaxis.tick_right()
    ax_u.set_title("(c) Load of the most loaded cell", fontsize=7.5, pad=2)
    ax_u.legend(loc="upper right")
    handles = [Line2D([], [], color=HOT, marker="o", ls="none", ms=3, label="UAV in an overloaded cell"),
               Line2D([], [], color="#c9c8c4", lw=0.8, label="Trajectory"),
               Line2D([], [], color=DARK, lw=0.6, label="Cell boundary"),
               Line2D([], [], color="k", marker="^", ls="none", ms=5, label="GBS")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.33, 0.05), frameon=False,
               fontsize=6.5, columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig_trajectories.pdf")


if __name__ == "__main__":
    fig_convergence()
    fig_scalability()
    fig_deadline()
    fig_penalty()
    fig_traj()
