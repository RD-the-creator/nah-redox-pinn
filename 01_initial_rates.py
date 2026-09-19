"""01 - Initial-rate analysis of catalyst activation.

Fits v = K1 k2 [F][Rh] / (1 + K1 [F]) to the initial rates of the fuel and Rh-catalyst
series (data/initial_rates.csv), first on the fuel series alone and then jointly on both.

Reproduces (paper, 'Fuel-driven Reduction'): k2 = 2.08e-3 s^-1, K1 = 33.8 M^-1,
K_M = 1/K1 ~ 30 mM, and fig:F_fit / Figure S1.
The constants carried forward into the ODE model (kinetics.py) are k3 = 2.077e-3 s^-1 x 60
and k1/k2 = K1 / 1e6 = 3.37e-5 uM^-1 (script notation).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import kinetics as K

d = K.load_initial_rates()
conv = 1e-6 / 60.0                                   # uM/min -> M/s
fu, rh = d[d.series == "fuel"], d[d.series == "rh"]
F1, Cat1, v1 = fu.fuel_mM.values * 1e-3, fu.rh_uM.values * 1e-6, fu.v0_uM_per_min.values * conv
F2, Cat2, v2 = rh.fuel_mM.values * 1e-3, rh.rh_uM.values * 1e-6, rh.v0_uM_per_min.values * conv


def mm(F, k, KM):
    return k * F / (KM + F)


def rate_model(x, K_eq, k):
    F, Cat = x
    return K_eq * k * F * Cat / (1.0 + K_eq * F)


# fuel series alone (normalised by catalyst)
(k_1, KM_1), cov1 = curve_fit(mm, F1, v1 / Cat1, p0=[2e-3, 0.03], bounds=([0, 0], [np.inf, np.inf]))
k_1_err, KM_1_err = np.sqrt(np.diag(cov1))

# joint fit, both series
F_all, Cat_all, v_all = np.r_[F1, F2], np.r_[Cat1, Cat2], np.r_[v1, v2]
(K_eq, k_j), cov2 = curve_fit(rate_model, (F_all, Cat_all), v_all, p0=[1 / KM_1, k_1],
                              bounds=([0, 0], [np.inf, np.inf]), maxfev=20000)
K_eq_err, k_j_err = np.sqrt(np.diag(cov2))
r2_joint = K.r2(v_all, rate_model((F_all, Cat_all), K_eq, k_j))
r2_fuel = K.r2(v1 / Cat1, mm(F1, k_1, KM_1))

print("Fuel series alone:  k = %.3e +/- %.1e s^-1   K1 = %.1f M^-1   KM = %.1f mM   R2 = %.4f"
      % (k_1, k_1_err, 1 / KM_1, KM_1 * 1e3, r2_fuel))
print("Joint fit:          k = %.3e +/- %.1e s^-1   K1 = %.1f +/- %.1f M^-1   R2 = %.4f"
      % (k_j, k_j_err, K_eq, K_eq_err, r2_joint))
print("Carried into kinetics.py:  k3 = %.4f min^-1,  k1/k2 = %.3e uM^-1"
      % (K.FITTED["k3"], K.K_EQ1))

json.dump({"fuel_only": {"k_s": k_1, "K1_M": 1 / KM_1, "KM_mM": KM_1 * 1e3, "r2": r2_fuel},
           "joint": {"k_s": k_j, "K1_M": K_eq, "r2": r2_joint}},
          open(K.RESULTS / "01_initial_rates.json", "w"), indent=1)

# figures: fig:F_fit and Figure S1
Fr = np.linspace(0, 22e-3, 200)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.scatter(F1 * 1e3, v1 / Cat1, s=18, color="C0", edgecolors="black", linewidths=0.5, zorder=3,
           label="Experimental results")
ax.plot(Fr * 1e3, mm(Fr, k_1, KM_1), "k-", lw=1.2, label="Fit")
ax.set_xlabel("Formate (mM)"); ax.set_ylabel("$v_0$ / [Rh] (s$^{-1}$)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(K.RESULTS / "01_fuel_fit.png", dpi=600); plt.close(fig)

Cr = np.linspace(0, 110e-6, 200)
slope = K_eq * k_j * F2[0] / (1 + K_eq * F2[0])
fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.scatter(Cat2 * 1e6, v2, s=18, color="C0", edgecolors="black", linewidths=0.5, zorder=3,
           label="Experimental results")
ax.plot(Cr * 1e6, slope * Cr, "k-", lw=1.2, label="Joint fit")
ax.set_xlabel("Rh-catalyst (µM)"); ax.set_ylabel("$v_0$ (M s$^{-1}$)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0)); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(K.RESULTS / "01_catalyst_linearity.png", dpi=600); plt.close(fig)
