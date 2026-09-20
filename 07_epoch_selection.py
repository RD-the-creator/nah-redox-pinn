"""07 - Choice of the number of training epochs (inner validation).

Only the 181 training rows of the outer split are used; neither the test fold nor the
MB = 10 uM curve is consulted. Per seed, the training rows are split again 80/20
(random_state = seed) into inner-train and inner-validation rows; scaling and centring are
fitted on inner-train only. One continuous run per seed records the inner-validation R2
every 100 epochs. Pre-declared rule: the smallest epoch count whose mean inner-validation
R2 lies within one standard deviation of the best mean.

  --config M0   5 seeds to 4000 epochs: the selection that fixed 1100 epochs (Appendix).
  --config M2   3 seeds to 3000 epochs: the repeat on the final configuration, showing
                the validation curve rises slowly and does not plateau (see README).

Usage:  python 07_epoch_selection.py --config M0|M2 [--seeds ...] [--max-epochs ...]
Writes results/epoch_sweep_{config}.csv. Several hours per seed at 3000-4000 epochs.
"""
import argparse
import os

_T = os.environ.get("PINN_THREADS")
if _T:
    os.environ.setdefault("OMP_NUM_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _T)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import random
import time

import numpy as np
import pandas as pd
import sciann as sn
import tensorflow as tf
from sklearn.model_selection import train_test_split

import kinetics as K
import pinn

ap = argparse.ArgumentParser()
ap.add_argument("--config", choices=["M0", "M2"], required=True)
ap.add_argument("--seeds", type=int, nargs="+")
ap.add_argument("--max-epochs", type=int)
ap.add_argument("--every", type=int, default=100)
a = ap.parse_args()
CFG = {**pinn.BASE, **getattr(pinn, a.config)}
SEEDS = a.seeds or ([0, 1, 2, 3, 4] if a.config == "M0" else [0, 1, 2])
MAXEP = a.max_epochs or (4000 if a.config == "M0" else 3000)
pinn.set_threads()


class InnerVal(tf.keras.callbacks.Callback):
    def __init__(self, NAH, prep, vidx, with_hi, every):
        super().__init__()
        self.NAH, self.prep, self.vidx, self.every = NAH, prep, vidx, every
        self.vin = prep.net_inputs(vidx, with_hi)[:9]
        self.rec = []

    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e % self.every == 0:
            pv = self.prep.invert(self.vidx, self.NAH.eval(self.vin))
            self.rec.append((e, K.r2(pinn.RAW[self.vidx, 5], pv),
                             float((logs or {}).get("loss", np.nan))))


def run(seed):
    outer_tr, _ = pinn.outer_split()
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    inner_tr, inner_val = train_test_split(outer_tr, test_size=0.2, random_state=seed)
    prep = pinn.Prep(inner_tr, no_leak=True)
    with_hi = CFG["pinf_mode"] == "mass_balance"
    ins, NAH, L, _, _ = pinn._graph(list(CFG["arch"]), CFG["taylor_order"], CFG["pinf_mode"], prep.lo)
    m = sn.SciModel(ins, [sn.Data(NAH), sn.Data(L)], loss_func="mse")
    x, y, w = pinn.training_arrays(prep, inner_tr, CFG["n_colloc"], CFG["n_cond"], rng, with_hi)
    cb = InnerVal(NAH, prep, inner_val, with_hi, a.every)
    m.train(x_true=x, y_true=y, weights=w, learning_rate=1e-3, batch_size=32,
            epochs=MAXEP, verbose=0, callbacks=[cb])
    return cb.rec


recs = {}
for s in SEEDS:
    t0 = time.time()
    recs[s] = run(s)
    print(f"seed {s}: {(time.time() - t0) / 60:.0f} min, final inner-val R2 {recs[s][-1][1]:.4f}", flush=True)

ep = [e for e, _, _ in recs[SEEDS[0]]]
val = np.array([[r for _, r, _ in recs[s]] for s in SEEDS])
loss = np.array([[l for _, _, l in recs[s]] for s in SEEDS]).mean(0)
mean, sd = val.mean(0), val.std(0, ddof=1)
pd.DataFrame({"epoch": ep, "val_r2_mean": mean.round(6), "val_r2_sd": sd.round(6),
              "train_loss_mean": loss}).to_csv(K.RESULTS / f"epoch_sweep_{a.config}.csv", index=False)
b = int(np.argmax(mean))
sel = ep[int(np.argmax(mean >= mean[b] - sd[b]))]
print(f"best mean inner-val R2 {mean[b]:.4f} at {ep[b]} epochs (sd {sd[b]:.4f}); "
      f"smallest epoch count within 1 sd: {sel}")
