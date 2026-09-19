"""08 - The PINN figures of the paper, from the outputs of 04_pinn_train.py.

  results/fig_holdout_mb10.png   fig:result1   held-out MB = 10 uM: data, mechanistic model,
                                               PINN mean of 3 seeds and seed range
  results/fig_PINN_parity.png    fig:PINN_parity  test fold (46 rows), PINN seed 0
  results/fig_pinn_loss.png      (not in the paper) training loss of seed 0, with the
                                               epoch at which the learning rate is first halved

R2 values in legends are rounded to two decimals, half up. 600 dpi.
Usage:  python 08_figures.py [--epochs 1100]
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kinetics as K

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=1100)
a = ap.parse_args()

DPI = 600
plt.rcParams.update({"font.size": 7, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5})
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def r2s(x):
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


D = K.RESULTS / f"pinn_e{a.epochs}"
seeds = [dict(np.load(p, allow_pickle=True)) for p in sorted(D.glob("seed*.npz"))]
s0 = next(s for s in seeds if int(s["seed"]) == 0)

# ---- held-out MB = 10 uM ---------------------------------------------------------------
hold = K.load_holdout()
cond = K.condition(hold, "mb_10uM")
t_h, y_h = K.curve(hold, "mb_10uM")
t_d = s0["t_dense"]
dense = np.vstack([s["hold_dense"] for s in seeds])
pts = np.vstack([s["hold_pts"] for s in seeds])
mech_d = K.predict_nah(cond, t_d)
r2_mp = K.r2(y_h, pts.mean(0))
r2_mech = K.r2(y_h, K.predict_nah(cond, t_h))

fig, ax = plt.subplots(figsize=(3.5, 2.8))
h_sc = ax.scatter(t_h, y_h, s=18, color=C[0], edgecolors="black", linewidths=0.5, alpha=0.9, zorder=4)
h_me, = ax.plot(t_d, mech_d, color=C[2], ls="--", lw=1.3, zorder=2)
h_pi, = ax.plot(t_d, dense.mean(0), color=C[1], ls="--", lw=1.3, zorder=3)
h_bd = ax.fill_between(t_d, dense.min(0), dense.max(0), color=C[1], alpha=0.22, lw=0, zorder=1)
ax.set_xlabel("Time (min)"); ax.set_ylabel("NAH (µM)")
ax.set_xlim(0, 62); ax.set_ylim(0, 32); ax.grid(alpha=0.3)
ax.legend([h_sc, h_me, h_pi, h_bd],
          ["Experimental results", f"Mechanistic model (R² = {r2s(r2_mech)})",
           f"PINN, mean of {len(seeds)} seeds (R² = {r2s(r2_mp)})",
           f"PINN seed range ({len(seeds)} seeds)"],
          loc="upper left", frameon=True, framealpha=0.9, fontsize=5.5, handlelength=1.6,
          handletextpad=0.5, labelspacing=0.3, borderpad=0.4, borderaxespad=0.4)
fig.tight_layout(); fig.savefig(K.RESULTS / "fig_holdout_mb10.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---- parity, test fold -----------------------------------------------------------------
df = K.load_curves()
raw = df[K.COLS].values.astype(float)
te = s0["test_idx"]
tt, pp = raw[te, 5], s0["pinn_test"]
mm = K.predict_rows(raw, te)
lim = 1.05 * max(tt.max(), pp.max(), mm.max())
fig, ax = plt.subplots(figsize=(3.5, 3.3))
ax.plot([0, lim], [0, lim], "k--", lw=0.8, zorder=1, label="Parity line")
ax.scatter(tt, pp, s=18, color=C[0], edgecolors="black", linewidths=0.4, zorder=3,
           label=f"PINN (R² = {r2s(K.r2(tt, pp))})")
ax.scatter(tt, mm, s=18, marker="^", color=C[1], edgecolors="black", linewidths=0.4, zorder=2,
           label=f"Mechanistic model (R² = {r2s(K.r2(tt, mm))})")
ax.set_xlabel("Measured NAH (µM)"); ax.set_ylabel("Predicted NAH (µM)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.legend(loc="upper left", frameon=True, framealpha=0.9)
fig.tight_layout(); fig.savefig(K.RESULTS / "fig_PINN_parity.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# ---- training loss ----------------------------------------------------------
def first_lr_halving(loss, patience):
    """Replays SciANN's default ReduceLROnPlateau (monitor = training loss, patience =
    epochs/10, factor 0.5): the first epoch at which the learning rate is halved."""
    best, wait = np.inf, 0
    for e, v in enumerate(loss, start=1):
        if v < best:
            best, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                return e
    return None


L = s0["loss"]
ep = np.arange(1, len(L) + 1)
w = 25
lg = np.log10(L)
run = 10 ** np.array([lg[max(0, i - w // 2):i + w // 2 + 1].mean() for i in range(len(L))])
e_lr = first_lr_halving(L, max(10, len(L) // 10))
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.semilogy(ep, L, lw=0.5, color=C[1], alpha=0.35, label="Per-epoch loss")
ax.semilogy(ep, run, lw=1.3, color=C[1], label=f"Running mean ({w} epochs)")
if e_lr:
    ax.axvline(e_lr, color="grey", lw=0.8, ls="--")
    ax.text(e_lr + 10, 6e-3, "learning rate\nhalved", fontsize=6, va="top", color="dimgrey")
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.set_xlim(0, len(L)); ax.grid(alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(0.62, 1.0), frameon=True, framealpha=0.9,
          fontsize=5.5, handlelength=1.6, labelspacing=0.3, borderpad=0.4)
fig.tight_layout(); fig.savefig(K.RESULTS / "fig_pinn_loss.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"held-out: PINN (mean prediction) R2 = {r2_mp:.4f} -> {r2s(r2_mp)};  mechanistic {r2_mech:.4f} -> {r2s(r2_mech)}")
print(f"parity:   PINN R2 = {K.r2(tt, pp):.4f};  mechanistic {K.r2(tt, mm):.4f}")
print(f"loss:     learning rate first halved at epoch {e_lr}")
