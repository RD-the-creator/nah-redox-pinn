"""06 - PINN versus mechanistic model over 50 random 80/20 splits.

For each split s = 0..49: train_test_split(rows, test_size=0.2, random_state=s); one PINN
(pinn.FINAL, 1100 epochs, training seed = s) and the mechanistic model with k4, Keq3, k6,
k8 re-estimated on that split's training rows only (kinetics.refit), so that neither model
has seen the test rows. Paired comparison over the splits with a Wilcoxon signed-rank test.

Reproduces (paper, PINN section and Appendix): PINN R2 0.98 versus 0.93 averaged over the
50 splits, higher on all 50 (paired difference +0.053 +/- 0.013, p ~ 2e-15).

Splits are independent and checkpointed to results/split50.csv, so the script can be
stopped and restarted, or run in parallel:  --worker i --n-workers n  (then run once more
with no arguments to aggregate).

Usage:  python 06_split50.py [--ode-only] [--splits 0 1 2] [--worker i --n-workers n]
About 10-20 min per PINN split on a laptop CPU; the ODE refit takes about a minute.
"""
import argparse
import os

_T = os.environ.get("PINN_THREADS")
if _T:
    os.environ.setdefault("OMP_NUM_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

import kinetics as K

N_SPLITS = 50
ap = argparse.ArgumentParser()
ap.add_argument("--ode-only", action="store_true")
ap.add_argument("--splits", type=int, nargs="+")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n-workers", type=int, default=1)
ap.add_argument("--epochs", type=int, default=1100)
a = ap.parse_args()

df = K.load_curves()
raw = df[K.COLS].values.astype(float)
CSV = K.RESULTS / ("split50.csv" if a.n_workers == 1 else f"split50_w{a.worker}.csv")
done = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=["split", "r2_pinn", "r2_ode"])
todo = a.splits if a.splits is not None else [s for s in range(N_SPLITS) if s % a.n_workers == a.worker]

if not a.ode_only:
    import pinn
    pinn.set_threads()

for s in todo:
    row = done[done.split == s]
    have_ode = len(row) and pd.notna(row.r2_ode.iloc[0])
    have_pinn = len(row) and pd.notna(row.r2_pinn.iloc[0])
    if have_ode and (have_pinn or a.ode_only):
        continue
    tr, te = train_test_split(np.arange(len(raw)), test_size=0.2, random_state=s)
    t0 = time.time()
    r_ode = row.r2_ode.iloc[0] if have_ode else K.r2(raw[te, 5], K.predict_rows(raw, te, K.refit(raw, tr)))
    r_pinn = row.r2_pinn.iloc[0] if have_pinn else np.nan
    if not a.ode_only and not have_pinn:
        r_pinn = K.r2(raw[te, 5], pinn.build(tr, s, epochs=a.epochs, **pinn.FINAL)(te))
    done = pd.concat([done[done.split != s],
                      pd.DataFrame([{"split": s, "r2_pinn": r_pinn, "r2_ode": r_ode}])])
    done.sort_values("split").to_csv(CSV, index=False)
    print(f"split {s:2d}: {(time.time() - t0) / 60:4.1f} min   PINN {r_pinn:.4f}   ODE {r_ode:.4f}", flush=True)

# ---- aggregate (single-worker file, or all worker files once complete) ----------------
parts = sorted(K.RESULTS.glob("split50_w*.csv"))
if a.n_workers == 1 and parts:
    allr = pd.concat([pd.read_csv(p) for p in parts] + [done]).drop_duplicates("split", keep="last")
    allr.sort_values("split").to_csv(K.RESULTS / "split50.csv", index=False)
else:
    allr = done
d = allr.dropna().sort_values("split")
if len(d) < 2:
    raise SystemExit(0)
diff = d.r2_pinn.values - d.r2_ode.values
print(f"\n{len(d)} splits with both models")
print(f"  PINN        : {d.r2_pinn.mean():.4f} +/- {d.r2_pinn.std():.4f}")
print(f"  mechanistic : {d.r2_ode.mean():.4f} +/- {d.r2_ode.std():.4f}")
print(f"  paired diff : {diff.mean():+.4f} +/- {diff.std(ddof=1):.4f};  PINN higher on "
      f"{(diff > 0).sum()}/{len(d)};  Wilcoxon p = {stats.wilcoxon(diff).pvalue:.2e}")
