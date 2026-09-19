"""03 - Fit of the oxidation module (full cycle).

With the reduction constants fixed (kinetics.FITTED), fits k6 (NAH + MB) and k8 (MB
regeneration) to the five MB-variation time-courses (MB = 20, 12, 8, 9, 0 uM). The
MB = 10 uM curve is NOT used: it is the held-out condition. A multi-start search avoids
a local minimum; a profile-likelihood scan over k8 shows how k6 compensates (Figure S3).

Reproduces (paper, 'The Full Cycle'): in-sample R2 = 0.94; k4 = 3.8e2 M^-1 s^-1 (+/-20 %)
and k5 = 2.8e-3 s^-1 (+/-11 %) in paper notation (k6, k8 here); the correlation of about
-0.89 between them; the MB conversion plot and Figure S3 (profile likelihood).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, fit_report
from scipy.integrate import odeint

import kinetics as K

df = K.load_curves()
RUNS = []                        # (MB0, y0, t, NAH)
for cid in K.MB_SERIES:
    F, S, Rh, MB = K.condition(df, cid)
    t, y = K.curve(df, cid)
    RUNS.append((MB, [F * 1e3, Rh, 0.0, 0.0, S, 0.0, MB, 0.0], t, y))


def params(k6=0.1, k8=0.1, vary_k8=True):
    p = Parameters()
    p.add("k1", value=K.FITTED["k1"], vary=False)
    p.add("k2", expr=f"k1 / {K.K_EQ1}")
    p.add("k3", value=K.FITTED["k3"], vary=False)
    p.add("k4", value=K.FITTED["k4"], vary=False)
    p.add("Keq3", value=K.FITTED["Keq3"], vary=False)
    p.add("k5", expr="k4 / Keq3")
    p.add("k6", value=k6, min=0)
    p.add("k8", value=k8, min=0, vary=vary_k8)
    return p


def simulate(p, y0, t):
    a = tuple(p[k].value for k in ("k1", "k2", "k3", "k4", "k5", "k6", "k8"))
    return odeint(K.rates_full, y0, t, args=a, mxstep=10000)[:, 5]


def objective(p):
    return np.concatenate([simulate(p, y0, t) - y for _, y0, t, y in RUNS])


# multi-start: keep the lowest chi-square, then refit from it with covariance
best = minimize(objective, params(), method="leastsq", nan_policy="omit", max_nfev=100000)
for k6_0, k8_0 in [(0.1, 0.1), (0.01, 0.15), (0.05, 0.2), (0.5, 0.3), (1.0, 0.05)]:
    r = minimize(objective, params(k6_0, k8_0), method="leastsq", nan_policy="omit",
                 max_nfev=100000)
    if r.chisqr < best.chisqr:
        best = r
res = minimize(objective, params(best.params["k6"].value, best.params["k8"].value),
               method="leastsq", nan_policy="omit", max_nfev=100000, calc_covar=True)
print(fit_report(res, show_correl=True))

obs = np.concatenate([y for *_, y in RUNS])
R2 = K.r2(obs, np.concatenate([simulate(res.params, y0, t) for _, y0, t, _ in RUNS]))
k6, k8 = res.params["k6"], res.params["k8"]
corr = res.params["k6"].correl.get("k8") if res.params["k6"].correl else None
print(f"\nOverall in-sample R2 = {R2:.4f}   (5 MB time-courses; MB = 10 uM withheld)")
print(f"k6 = {k6.value:.5f} uM^-1 min^-1 = {k6.value * 1e6 / 60:.3g} M^-1 s^-1  (+/- {100 * k6.stderr / k6.value:.0f} %)")
print(f"k8 = {k8.value:.4f} min^-1       = {k8.value / 60:.3g} s^-1       (+/- {100 * k8.stderr / k8.value:.0f} %)")
print(f"corr(k6, k8) = {corr:.3f}")
print(f"(kinetics.FITTED carries k6 = {K.FITTED['k6']}, k8 = {K.FITTED['k8']})")

# profile likelihood over k8 (Figure S3)
rows = []
for k8v in np.logspace(np.log10(0.02), np.log10(2.0), 25):
    r = minimize(objective, params(k6.value, k8v, vary_k8=False), method="leastsq",
                 nan_policy="omit", max_nfev=100000)
    rows.append({"k8": k8v, "k6": r.params["k6"].value, "delta_chi2": r.chisqr - res.chisqr})
prof = pd.DataFrame(rows)
prof.to_csv(K.RESULTS / "03_k8_profile.csv", index=False)

json.dump({"r2": R2, "k6": k6.value, "k6_stderr": k6.stderr, "k8": k8.value,
           "k8_stderr": k8.stderr, "corr_k6_k8": corr},
          open(K.RESULTS / "03_oxidation_fit.json", "w"), indent=1)

# conversion plot, MB series
fig, ax = plt.subplots(figsize=(3.5, 2.8))
for i, (MB, y0, t, y) in enumerate(sorted(RUNS, key=lambda r: r[0])):
    ax.scatter(t, y / y0[4], s=14, color=f"C{i}", edgecolors="black", linewidths=0.4,
               label=f"MB {MB:.0f} µM", zorder=3)
    tt = np.linspace(0, t.max(), 200)
    ax.plot(tt, simulate(res.params, y0, tt) / y0[4], color=f"C{i}", lw=1)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Conversion ($P/S_0$)"); ax.grid(alpha=0.3)
ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(K.RESULTS / "03_conversion_mb.png", dpi=600); plt.close(fig)

# Figure S3
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
a1.plot(prof.k8, prof.delta_chi2, "k-o", ms=3)
a1.axhline(3.84, color="C3", ls="--", lw=0.9, label="95 % threshold")
a1.axvline(k8.value, color="C0", ls=":", label="Best fit")
a1.set_xscale("log"); a1.set_xlabel("$k_8$ (min$^{-1}$)"); a1.set_ylabel(r"$\Delta\chi^2$")
a1.grid(alpha=0.3); a1.legend(fontsize=6)
a2.plot(prof.k8, prof.k6, "k-o", ms=3)
a2.axvline(k8.value, color="C0", ls=":", label="Best fit")
a2.set_xscale("log"); a2.set_xlabel("$k_8$ (min$^{-1}$)"); a2.set_ylabel("$k_6$ (µM$^{-1}$ min$^{-1}$)")
a2.grid(alpha=0.3); a2.legend(fontsize=6)
fig.tight_layout(); fig.savefig(K.RESULTS / "03_k8_profile.png", dpi=600); plt.close(fig)
