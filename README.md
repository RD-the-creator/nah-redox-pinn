# Code and data: Kinetic Modeling of Nicotinamide's Catalytically-governed Redox Cycle Using PINNs

This folder reproduces every number and model figure in the paper: the mechanistic
(ODE) model, the PINN, and the comparison between the two.

## Setup

```
pip install -r requirements.txt        # Python 3.10
```

Scripts 01–03 and 05 need only the scientific Python stack. Scripts 04, 06 (without
`--ode-only`) and 07 train the PINN and need TensorFlow 2.10 and SciANN 0.7.0.1.
Run every script from this folder. All outputs go to `results/`.

## Data (`data/`)

| File | Contents |
|---|---|
| `curves.csv` | The 16 time-courses used for fitting and training: 227 rows of `curve_id, fuel_mM, sub_uM, rh_uM, mb_uM, t_min, nah_uM`. Keep the row order: the 80/20 splits are taken over row indices. |
| `holdout_mb10.csv` | The MB = 10 µM time-course (14 points), withheld from both models. |
| `initial_rates.csv` | Initial rates of the fuel and Rh-catalyst series. |

NAH is the reduced nicotinamide analogue. Units are µM and minutes, and fuel is in mM.

## Scripts

| Script | What it does | Paper |
|---|---|---|
| `kinetics.py` | Shared module: data loaders, the reaction network, the fitted rate constants, the ODE predictor, and the re-fit of the free constants on a subset of rows. | Mechanism |
| `pinn.py` | Shared module: the PINN, defined once. The reported configuration is `pinn.FINAL` (mass-balance bound on the plateau, 100 time + 1600 condition-space collocation points, 40-30-20-10, Taylor order 2) at 1100 epochs. | PINN section |
| `01_initial_rates.py` | Saturation fit of the initial rates. | k₂ = 2.08×10⁻³ s⁻¹, K₁ = 33.8 M⁻¹, K_M ≈ 30 mM; fig:F_fit, Fig. S1 |
| `02_reduction_fit.py` | Global fit of the reduction half-cycle to the 12 reduction curves, plus the k₁ sensitivity scan. | R² = 0.93, K₃ = 1.53 ± 0.10; conversion plots; Fig. S2 |
| `03_oxidation_fit.py` | Fit of k₄ and k₅ (paper notation) to the five MB curves, excluding MB = 10 µM, plus the profile likelihood. | R² = 0.94, 3.8×10² M⁻¹ s⁻¹ (±20 %), 2.8×10⁻³ s⁻¹ (±11 %); Fig. S3 |
| `04_pinn_train.py` | Trains the PINN on the outer split (181/46 rows) for seeds 0–2 and saves test-fold and held-out predictions and the loss history. | Input to 05 and 08 |
| `05_compare.py` | PINN vs mechanistic model on the test fold and the held-out curve, the refit control, and the epoch-sensitivity table. | 0.96 vs 0.90; refit 0.90; 0.79 vs 0.26; 23.8 / 28.6 / 18.0 µM; Appendix plateau spread 0.46 µM |
| `06_split50.py` | 50 random 80/20 splits. For each split: one PINN (seed = split index) and the mechanistic model re-fitted on that split's training rows. Paired Wilcoxon test. | 0.98 vs 0.93, PINN higher on 50/50 |
| `07_epoch_selection.py` | Inner-validation sweep on the training rows only, with the pre-declared 1-sd rule. `--config M0` is the sweep that selected 1100. `--config M2` is the repeat on the final configuration. | Appendix |
| `08_figures.py` | The PINN figures, built from the `04` outputs. | fig:result1, fig:PINN_parity, Fig. S4 |

Notation: the scripts number the rate constants by elementary step, and the paper uses its own names. `kinetics.py` has the mapping. For example, the script's k6 and k8 are the paper's k₄ and k₅, and `Keq3` is K₃.

## Reported runs (`results/`)

PINN training is slow: roughly 10–20 min per fit on a laptop CPU, and several hours per seed for the 07 sweeps. So `results/` already holds the outputs of the runs reported in the paper:

- `pinn_e1100/seed{0,1,2}.npz` are the reported 1100-epoch fits. For seeds 1 and 2, the held-out curves and the test-fold scores come from two separate runs that used the same settings. Each file records this in its `provenance` field.
- `pinn_e800/`, `pinn_e1700/` hold the per-seed scalars from the epoch-sensitivity check.
- `split50.csv` holds the per-split R² of both models.
- `epoch_sweep_M0.csv`, `epoch_sweep_M2.csv` are the two validation sweeps.

With these files, `python 05_compare.py`, `python 06_split50.py --ode-only` and `python 08_figures.py` reproduce the paper's numbers and figures without any training. The figures are pixel-identical to those in the paper. Re-running 04, 06 or 07 overwrites the corresponding files.

## Notes on reproducibility

- **PINN results depend slightly on the TensorFlow thread count.** Seeds are fixed, but the thread count changes the floating-point summation order. This moves R² by about 0.001–0.005 and the 60-min plateau by a few hundredths of a µM. Set `PINN_THREADS=n` to fix the thread count.
- **Some ODE re-fits differ in the fourth decimal.** k4 (script notation) is not identifiable, so the least-squares re-fit can end at different k4 values with the same fit quality. As a result, a few split-wise ODE R² values can differ in the fourth decimal between library versions (e.g. split 1: 0.9240 stored, 0.9243 re-computed). For the same reason, the standard error of K₃ from `02` ranges from 0.097 to 0.109 depending on the lmfit/scipy version. K₃ itself and R² do not change.
- **The original 1100-epoch selection used an earlier module.** The `07 --config M0` sweep that selected 1100 epochs was run with an earlier version of `pinn.py` that had the same model and preprocessing. A re-run should reproduce the selection, though not every digit of the stored curve.
- **The M2 repeat sweep rises slowly and does not plateau.** Mean inner-validation R² goes from 0.985 at 800 epochs to 0.992 at 3000 (0.987 at 1100). Applied to this sweep, the 1-sd rule would select 1700. The held-out plateau at 800 / 1100 / 1700 epochs differs by 0.46 µM (05).
- **Row order matters.** The splits are taken over row indices, so `curves.csv` must be used as shipped.

## Licence

MIT License, see `LICENSE` — this covers everything in the repository: the scripts, the
data in `data/` and the generated figures in `results/`.

If you use this code or data, please cite the paper; `CITATION.cff` has the details.
