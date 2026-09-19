"""05 - PINN versus mechanistic model on the test fold and the held-out MB = 10 uM curve.

No training: reads the per-seed PINN outputs written by 04_pinn_train.py and evaluates the
mechanistic model (kinetics.py) on the same rows.

Reproduces (paper):
  * test fold (46 rows), fig:PINN_parity:  PINN R2 = 0.96 (seed 0 in the figure; mean
    +/- sd over seeds printed too) versus mechanistic model 0.90;
  * refit control: mechanistic constants re-estimated on the 181 training rows only,
    R2 on the test fold still 0.90;
  * held-out MB = 10 uM, fig:result1:  PINN R2 = 0.79 (R2 of the 3-seed mean prediction,
    figure legend; the mean of per-seed R2 is printed as well) versus mechanistic 0.26;
    60-min plateau PINN 23.8 uM (range over seeds), mechanistic 28.6 uM, measured 18.0 uM;
  * Appendix, epoch sensitivity: 60-min plateau at 800 / 1100 / 1700 epochs (if those
    runs exist) and their spread against the 2.6 uM reference-condition reproducibility.

Usage:  python 05_compare.py [--epochs 1100] [--no-refit]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import kinetics as K

REF_REPRO = 2.6        # uM, reproducibility at the reference condition (Appendix)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=1100)
ap.add_argument("--no-refit", action="store_true")
a = ap.parse_args()


def load_seeds(epochs):
    d = K.RESULTS / f"pinn_e{epochs}"
    return [dict(np.load(p, allow_pickle=True)) for p in sorted(d.glob("seed*.npz"))]


df = K.load_curves()
raw = df[K.COLS].values.astype(float)
tr, te = train_test_split(np.arange(len(raw)), test_size=K.TEST_SIZE, random_state=K.SPLIT_SEED)
y_te = raw[te, 5]
seeds = load_seeds(a.epochs)
if not seeds:
    raise SystemExit(f"no results/pinn_e{a.epochs}/seed*.npz - run 04_pinn_train.py first")
out = {"epochs": a.epochs, "seeds": [int(s["seed"]) for s in seeds]}

# ---- test fold ------------------------------------------------------------------------
mech_te = K.predict_rows(raw, te)
r2_mech_te = K.r2(y_te, mech_te)
r2_pinn_te = np.array([float(s["r2_testfold"]) for s in seeds])
print(f"TEST FOLD ({len(tr)} train / {len(te)} test rows)")
print(f"  PINN, per seed          : " + "  ".join(f"{v:.4f}" for v in r2_pinn_te))
print(f"  PINN, mean +/- sd       : {r2_pinn_te.mean():.4f} +/- {r2_pinn_te.std(ddof=1):.4f}")
print(f"  mechanistic model       : {r2_mech_te:.4f}")
out.update(r2_testfold_pinn=r2_pinn_te.tolist(), r2_testfold_mech=r2_mech_te)
np.savez(K.RESULTS / "05_testfold_mech.npz", test_idx=te, true_test=y_te, mech_test=mech_te)

if not a.no_refit:
    p = K.refit(raw, tr)
    r2_refit = K.r2(y_te, K.predict_rows(raw, te, p))
    print(f"  mechanistic, refit on the {len(tr)} training rows: {r2_refit:.4f}   "
          f"(k4 = {p['k4']:.3g}, Keq3 = {p['Keq3']:.3f}, k6 = {p['k6']:.4g}, k8 = {p['k8']:.4g})")
    out.update(r2_testfold_mech_refit=r2_refit, refit_constants=p)

# ---- held-out MB = 10 uM --------------------------------------------------------------
hold = K.load_holdout()
cond = K.condition(hold, "mb_10uM")
t_h, y_h = K.curve(hold, "mb_10uM")
mech_h = K.predict_nah(cond, t_h)
r2_mech_h = K.r2(y_h, mech_h)
pts = np.array([s["hold_pts"] for s in seeds])
r2_seed = np.array([K.r2(y_h, p) for p in pts])
r2_meanpred = K.r2(y_h, pts.mean(0))
pl = pts[:, -1]
print(f"\nHELD-OUT MB = 10 uM (withheld from both models)")
print(f"  PINN R2, per seed       : " + "  ".join(f"{v:+.4f}" for v in r2_seed))
print(f"  PINN R2, mean +/- sd    : {r2_seed.mean():.4f} +/- {r2_seed.std(ddof=1):.4f}")
print(f"  PINN R2 of mean pred.   : {r2_meanpred:.4f}   (figure legend)")
print(f"  mechanistic model R2    : {r2_mech_h:.4f}")
print(f"  60-min plateau: PINN {pl.mean():.1f} uM ({pl.min():.1f}-{pl.max():.1f}), "
      f"mechanistic {mech_h[-1]:.1f} uM, measured {y_h[-1]:.1f} uM")
out.update(r2_holdout_pinn=r2_seed.tolist(), r2_holdout_pinn_meanpred=r2_meanpred,
           r2_holdout_mech=r2_mech_h, plateau60_pinn=pl.tolist(),
           plateau60_mech=float(mech_h[-1]), plateau60_measured=float(y_h[-1]))

# ---- epoch sensitivity (Appendix) -----------------------------------------------------
rows = []
for d in sorted(K.RESULTS.glob("pinn_e*"), key=lambda p: int(p.name[6:])):
    ss = load_seeds(int(d.name[6:]))
    if not ss:
        continue
    p60 = np.array([float(s["plateau60"]) for s in ss])
    rows.append({"epochs": int(d.name[6:]), "n_seeds": len(ss), "plateau60_mean": p60.mean(),
                 "plateau60_sd": p60.std(ddof=1) if len(ss) > 1 else np.nan,
                 "r2_holdout_mean": np.mean([float(s["r2_holdout"]) for s in ss]),
                 "r2_testfold_mean": np.mean([float(s["r2_testfold"]) for s in ss])})
if len(rows) > 1:
    tab = pd.DataFrame(rows)
    spread = tab.plateau60_mean.max() - tab.plateau60_mean.min()
    print("\nEPOCH SENSITIVITY (held-out 60-min plateau)")
    print(tab.round(4).to_string(index=False))
    print(f"  spread of the mean plateau across epoch counts = {spread:.2f} uM "
          f"(reference-condition reproducibility {REF_REPRO} uM)")
    tab.to_csv(K.RESULTS / "05_epoch_sensitivity.csv", index=False)
    out.update(plateau_spread_across_epochs=spread)

json.dump(out, open(K.RESULTS / "05_compare.json", "w"), indent=1, default=float)
