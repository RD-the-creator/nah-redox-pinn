"""04 - Train the PINN on the outer split and score it on the test fold and the held-out
MB = 10 uM curve.

The PINN (pinn.FINAL) is trained on 181 of the 227 rows (outer split, random_state = 0)
for each seed. For every seed this saves, to results/pinn_e{EPOCHS}/seed{s}.npz:
    test_idx, pinn_test, r2_testfold       the 46 test rows
    hold_pts, r2_holdout, plateau60        the MB = 10 uM curve at its 14 measured times
    t_dense, hold_dense                    the same curve on a 0-60 min grid (241 points)
    loss                                   training loss per epoch (08_figures.py)
Nothing is computed from these here: 05_compare.py and 08_figures.py read them.

Usage:  python 04_pinn_train.py [--seeds 0 1 2] [--epochs 1100]
        (env PINN_THREADS=n to fix the TensorFlow thread count)
About 10-20 min per seed on a laptop CPU at 1100 epochs.
Reported runs: seeds 0-2 at 1100 epochs (paper), and at 800 and 1700 epochs for the
epoch-sensitivity check in the Appendix.
"""
import argparse
import os
import time

_T = os.environ.get("PINN_THREADS")
if _T:
    os.environ.setdefault("OMP_NUM_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np

import kinetics as K
import pinn

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=pinn.EPOCHS)
a = ap.parse_args()
pinn.set_threads()

out = K.RESULTS / f"pinn_e{a.epochs}"
out.mkdir(exist_ok=True)
tr, te = pinn.outer_split()
hold = K.load_holdout()
cond = K.condition(hold, "mb_10uM")
t_hold, y_hold = K.curve(hold, "mb_10uM")
t_dense = np.linspace(0, 60, 241)
print(f"outer split: {len(tr)} train / {len(te)} test rows; epochs = {a.epochs}")

for s in a.seeds:
    t0 = time.time()
    o = pinn.build(tr, s, epochs=a.epochs, return_nets=True, **pinn.FINAL)
    pinn_test = o["predict"](te)
    hold_pts = o["predict_curve"](cond, t_hold)
    res = dict(seed=s, epochs=a.epochs, test_idx=te, true_test=pinn.RAW[te, 5],
               pinn_test=pinn_test, r2_testfold=K.r2(pinn.RAW[te, 5], pinn_test),
               t_hold=t_hold, y_hold=y_hold, hold_pts=hold_pts,
               r2_holdout=K.r2(y_hold, hold_pts), plateau60=hold_pts[-1],
               t_dense=t_dense, hold_dense=o["predict_curve"](cond, t_dense),
               loss=np.asarray(o["history"]["loss"], float),
               provenance="04_pinn_train.py")
    np.savez(out / f"seed{s}.npz", **res)
    print(f"seed {s}: {(time.time() - t0) / 60:.1f} min   test fold R2 = {res['r2_testfold']:.4f}   "
          f"held-out R2 = {res['r2_holdout']:+.4f}   60-min plateau = {res['plateau60']:.1f} uM")
