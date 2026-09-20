"""Shared data loading and the mechanistic kinetic model.

Everything the scripts need about the experiments and the mechanism is defined here once:
the data files, the reaction network, and the fitted rate constants.

Units throughout: concentrations in uM (fuel in the data files in mM), time in minutes.
NAH is the reduced nicotinamide analogue (1-benzyl-1,4-dihydro-3-carbamoylpyridine).

Mechanism (state vector [F, PC, FPC, C, S, P, C2, C2D]):
    1.  F  + PC  <=> FPC          k1, k2      formate binds the Rh(III) catalyst
    2.  FPC       -> C  + CO2     k3          hydride formation (rate-limiting)
    3.  S  + C   <=> P  + PC      k4, k5      hydride transfer to NA+ (S) giving NAH (P)
    4.  P  + C2   -> S  + C2D     k6          oxidation of NAH by methylene blue (C2)
    5.  C2D       -> C2           k8          re-oxidation of reduced MB by O2
Steps 1-3 alone are the reduction half-cycle (6 species, no MB).

Paper notation: k1, k-1, k2, K3 = k3/k-3, k4, k5' correspond here to
k1, k2, k3, Keq3 = k4/k5, k6, k8.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

COLS = ["fuel_mM", "sub_uM", "rh_uM", "mb_uM", "t_min", "nah_uM"]
SPLIT_SEED = 0          # the outer 80/20 split used for the reported test fold
TEST_SIZE = 0.2


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_curves():
    """The 16 training time-courses (227 rows). Row order matters: the random splits are
    taken over row indices, so it is kept exactly as recorded."""
    return pd.read_csv(DATA / "curves.csv")


def load_holdout():
    """The MB = 10 uM time-course, withheld from both models."""
    return pd.read_csv(DATA / "holdout_mb10.csv")


def load_initial_rates():
    return pd.read_csv(DATA / "initial_rates.csv")


def curve(df, curve_id):
    """(t, NAH) arrays for one curve."""
    g = df[df.curve_id == curve_id]
    return g.t_min.values.astype(float), g.nah_uM.values.astype(float)


def condition(df, curve_id):
    """[fuel_mM, sub_uM, rh_uM, mb_uM] of one curve."""
    return df[df.curve_id == curve_id][COLS[:4]].iloc[0].values.astype(float)


# Curves used by the mechanistic fits, in the order they were fitted.
FUEL_SERIES = ["fuel_20mM", "fuel_10mM", "fuel_5mM", "fuel_2p5mM"]
RH_SERIES = ["rh_40uM", "rh_60uM", "rh_80uM", "rh_100uM"]
SUB_SERIES = ["sub_100uM", "sub_80uM", "sub_60uM", "sub_40uM"]
MB_SERIES = ["mb_20uM", "mb_12uM", "mb_8uM", "mb_9uM", "rh_40uM"]   # MB = 20, 12, 8, 9, 0 uM


# --------------------------------------------------------------------------------------
# Fitted rate constants (uM, min). Provenance:
#   k2/k1 ratio and k3 ........ 01_initial_rates.py
#   k4, Keq3 .................. 02_reduction_fit.py   (only Keq3 is identifiable)
#   k6, k8 .................... 03_oxidation_fit.py   (fitted without the MB = 10 uM curve)
# --------------------------------------------------------------------------------------
K_EQ1 = 3.37e-5                 # k1/k2, uM^-1
FITTED = dict(
    k1=70.0,                    # fixed: not identifiable from NAH(t)
    k3=2.077e-3 * 60,           # s^-1 -> min^-1
    k4=3285.7,                  # not identifiable on its own
    Keq3=1.525,
    k6=0.02276,
    k8=0.1685,
)


def full_constants(p=None):
    """Complete constant set (k1..k8) from the free ones, deriving k2 and k5."""
    p = dict(FITTED if p is None else p)
    p["k2"] = p["k1"] / K_EQ1
    p["k5"] = p["k4"] / p["Keq3"]
    return p


def rates_reduction(y, t, k1, k2, k3, k4, k5):
    """Reduction half-cycle, state [F, PC, FPC, C, S, P]."""
    F, PC, FPC, C, S, P = y
    r1f, r1r, r2 = k1 * F * PC, k2 * FPC, k3 * FPC
    r3f, r3r = k4 * S * C, k5 * P * PC
    return [-r1f + r1r, -r1f + r1r + r3f - r3r, r1f - r1r - r2,
            r2 - r3f + r3r, -r3f + r3r, r3f - r3r]


def rates_full(y, t, k1, k2, k3, k4, k5, k6, k8):
    """Full cycle, state [F, PC, FPC, C, S, P, C2, C2D]; step 4 irreversible."""
    F, PC, FPC, C, S, P, C2, C2D = y
    r1f, r1r, r2 = k1 * F * PC, k2 * FPC, k3 * FPC
    r3f, r3r = k4 * S * C, k5 * P * PC
    r4f, r5d = k6 * P * C2, k8 * C2D
    return [-r1f + r1r, -r1f + r1r + r3f - r3r, r1f - r1r - r2, r2 - r3f + r3r,
            -r3f + r3r + r4f, r3f - r3r - r4f, -r4f + r5d, r4f - r5d]


def predict_nah(cond, times, p=None):
    """NAH (uM) for a condition [fuel_mM, sub_uM, rh_uM, mb_uM] at the given times, using
    the full mechanism. Each time is integrated from t = 0 separately (as in all reported
    held-out comparisons)."""
    c = full_constants(p)
    args = (c["k1"], c["k2"], c["k3"], c["k4"], c["k5"], c["k6"], c["k8"])
    F, S, Rh, MB = cond
    y0 = [F * 1e3, Rh, 0.0, 0.0, S, 0.0, MB, 0.0]
    return np.array([0.0 if t <= 0 else
                     odeint(rates_full, y0, [0.0, t], args=args, mxstep=10000)[-1, 5]
                     for t in np.atleast_1d(times)])


def predict_rows(raw, idx, p=None):
    """NAH (uM) for rows `idx` of the data matrix (columns as COLS)."""
    return np.array([predict_nah(raw[i, :4], [raw[i, 4]], p)[0] for i in idx])


def refit(raw, train_idx):
    """Re-estimate k4, Keq3, k6, k8 on the training rows only (k1, k3 fixed), for a
    held-out comparison in which neither model has seen the test rows."""
    from lmfit import Parameters, minimize
    y = raw[train_idx, 5]

    def resid(q):
        p = dict(FITTED, k4=q["k4"].value, Keq3=q["Keq3"].value,
                 k6=q["k6"].value, k8=q["k8"].value)
        return predict_rows(raw, train_idx, p) - y

    q = Parameters()
    for k in ("k4", "Keq3", "k6", "k8"):
        q.add(k, value=FITTED[k], min=0)
    r = minimize(resid, q, method="leastsq", nan_policy="omit", max_nfev=20000)
    return dict(FITTED, **{k: r.params[k].value for k in ("k4", "Keq3", "k6", "k8")})


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
