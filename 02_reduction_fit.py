"""02 - Global fit of the reduction half-cycle.

Fits the species-balance ODEs of steps 1-3 (kinetics.rates_reduction) to the twelve
reduction time-courses (fuel, Rh-catalyst and substrate series, MB = 0), with k3 and
k1/k2 fixed from 01 and k1 fixed at a representative value.

Reproduces (paper, 'Fuel-driven Reduction'): in-sample R2 = 0.93; K3 = 1.53 +/- 0.10
(Keq3 here); the k1 sensitivity scan showing R2 and Keq3 flat over six decades of k1;
the conversion plots of the three series; and Figure S2 (1000x fuel and substrate).

Usage:  python 02_reduction_fit.py [--no-scan]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, fit_report
from scipy.integrate import odeint

import kinetics as K

df = K.load_curves()
RUNS = []                       # (curve_id, y0 = [F, PC, FPC, C, S, P], t, NAH)
for cid in K.FUEL_SERIES + K.RH_SERIES + K.SUB_SERIES:
    F, S, Rh, _ = K.condition(df, cid)
    t, y = K.curve(df, cid)
    RUNS.append((cid, [F * 1e3, Rh, 0.0, 0.0, S, 0.0], t, y))


def params(k1=K.FITTED["k1"]):
    p = Parameters()
    p.add("k1", value=k1, vary=False)
    p.add("k2", expr=f"k1 / {K.K_EQ1}")
    p.add("k3", value=K.FITTED["k3"], vary=False)
    p.add("k4", value=2e2, min=0)
    p.add("Keq3", value=1.52, min=0)
    p.add("k5", expr="k4 / Keq3")
    return p


def simulate(p, y0, t):
    a = tuple(p[k].value for k in ("k1", "k2", "k3", "k4", "k5"))
    return odeint(K.rates_reduction, y0, t, args=a, mxstep=10000)[:, 5]


def objective(p):
    return np.concatenate([simulate(p, y0, t) - y for _, y0, t, y in RUNS])


def overall_r2(p):
    obs = np.concatenate([y for *_, y in RUNS])
    return K.r2(obs, np.concatenate([simulate(p, y0, t) for _, y0, t, _ in RUNS]))


res = minimize(objective, params(), method="leastsq", nan_policy="omit",
               max_nfev=100000, calc_covar=True)
print(fit_report(res, show_correl=True))
R2 = overall_r2(res.params)
kq, kq_err = res.params["Keq3"].value, res.params["Keq3"].stderr
print(f"\nOverall in-sample R2 = {R2:.4f}   (12 reduction time-courses)")
print(f"Keq3 = {kq:.3f} +/- {kq_err:.3f}  ({100 * kq_err / kq:.1f} %)")

out = {"r2": R2, "Keq3": kq, "Keq3_stderr": kq_err, "k4": res.params["k4"].value}

# --- k1 sensitivity scan: k1 is not identifiable ------------------------------------
if "--no-scan" not in sys.argv:
    rows = []
    for k1 in [0.06, 0.6, 6, 70, 600, 6000, 60000]:            # uM^-1 min^-1
        r = minimize(objective, params(k1), method="leastsq", nan_policy="omit",
                     max_nfev=100000, calc_covar=True)
        rows.append({"k1 (uM^-1 min^-1)": k1, "k1 (M^-1 s^-1)": round(k1 / 60 * 1e6),
                     "Keq3": round(r.params["Keq3"].value, 4),
                     "Keq3_stderr": round(r.params["Keq3"].stderr or 0, 4),
                     "R2": round(overall_r2(r.params), 6)})
    scan = pd.DataFrame(rows)
    print("\nk1 sensitivity scan (R2 and Keq3 should be flat):")
    print(scan.to_string(index=False))
    scan.to_csv(K.RESULTS / "02_k1_scan.csv", index=False)

json.dump(out, open(K.RESULTS / "02_reduction_fit.json", "w"), indent=1)

# --- conversion plots, one per series ------------------------------------------------
for name, ids, lab in [("fuel", K.FUEL_SERIES, "Fuel {:.1f} mM"),
                       ("rh", K.RH_SERIES, "Rh-catalyst {:.0f} µM"),
                       ("substrate", K.SUB_SERIES, "Substrate {:.0f} µM")]:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, (cid, y0, t, y) in enumerate([r for r in RUNS if r[0] in ids]):
        S0 = y0[4]
        v = {"fuel": y0[0] / 1e3, "rh": y0[1], "substrate": S0}[name]
        ax.scatter(t, y / S0, s=14, color=f"C{i}", edgecolors="black", linewidths=0.4,
                   label=lab.format(v), zorder=3)
        tt = np.linspace(0, t.max(), 200)
        ax.plot(tt, simulate(res.params, y0, tt) / S0, color=f"C{i}", lw=1)
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Conversion ($P/S_0$)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(K.RESULTS / f"02_conversion_{name}.png", dpi=600); plt.close(fig)

# --- Figure S2: 1000x fuel and substrate --------------------------------------------
y0b = [20e3, 40.0, 0, 0, 80.0, 0]
y0h = [1000 * 20e3, 40.0, 0, 0, 1000 * 80.0, 0]
tt = np.linspace(0, 180, 1200)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.plot(tt, simulate(res.params, y0b, tt) / 80.0, label="Reference conditions")
ax.plot(tt, simulate(res.params, y0h, tt) / 80e3, label="1000× fuel and substrate")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Conversion ($P/S_0$)"); ax.grid(alpha=0.3)
ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(K.RESULTS / "02_scaled_conversion.png", dpi=600); plt.close(fig)
