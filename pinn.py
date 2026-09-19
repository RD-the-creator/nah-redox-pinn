"""The physics-informed neural network (PINN), defined once.

Model (paper, 'PINN'): NAH(t) is represented by a network with NAH(0) = 0 by construction
and trained to satisfy

    dNAH/dt = k_eff * (Pinf - NAH)

where k_eff is a second-order Taylor expansion of a rate network K around the mean
condition (fuel, substrate, Rh-catalyst, MB), and Pinf is a plateau network of the
condition alone. Two additions bound and regularise the model:

  * mass balance:  0 <= Pinf <= S0 row by row, via lo + (hi - lo) * sigmoid(raw), with
    lo and hi the scaled images of 0 uM and of that row's own substrate S0 (the plateau
    is NAH formed from NA+, so it cannot exceed S0);
  * collocation:   the residual is also enforced, with zero data weight, at 100 random
    times on every measured condition and at 1600 random (condition, time) points drawn
    between the measured extremes of the training set.

All scaling (MinMax) and the Taylor centring are fitted on training rows only.

The reported configuration is FINAL = {**BASE, **M2} at EPOCHS = 1100. M0 (no mass
balance, no condition-space collocation) is the configuration the epoch count was
originally selected on (07_epoch_selection.py).

Software: Python 3.10, TensorFlow 2.10, SciANN 0.7.0.1 (see requirements.txt).
"""
import random

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import sciann as sn

import kinetics as K

_df = K.load_curves()
RAW = _df[K.COLS].values.astype(float)
CURVE_OF_ROW = _df["curve_id"].values
N = RAW.shape[0]
SUB_COL, T_COL, Y_COL = 1, 4, 5

EPOCHS = 1100
BASE = dict(arch=(40, 30, 20, 10), no_leak=True, lr_schedule=False, taylor_order=2)
M0 = dict(pinf_mode="tanh", n_colloc=100, n_cond=0)
M2 = dict(pinf_mode="mass_balance", n_colloc=100, n_cond=1600)
FINAL = {**BASE, **M2}


class Prep:
    """Train-only MinMax scaling plus the per-row mass-balance ceiling `hi`."""

    def __init__(self, train_idx, no_leak=True):
        fr = RAW[train_idx] if no_leak else RAW
        self.scaler = MinMaxScaler().fit(fr)
        self.scaled = self.scaler.transform(RAW)
        self.aver = self.scaler.transform(fr).mean(axis=0)
        self.deltas = self.scaled - self.aver
        self.labels = self.scaled[:, Y_COL]
        # scaled image of a NAH value in uM, under the same fitted scaler as the labels
        self._a = self.scaler.scale_[Y_COL]
        self._b = self.scaler.min_[Y_COL]
        self.lo = float(self.y_scaled(0.0))
        # condition ranges of the training rows, for condition-space collocation
        self.cond_lo = RAW[train_idx, :4].min(axis=0)
        self.cond_hi = RAW[train_idx, :4].max(axis=0)
        self.t_lo = self.scaled[train_idx, T_COL].min()
        self.t_hi = self.scaled[train_idx, T_COL].max()

    def y_scaled(self, v_uM):
        return np.asarray(v_uM, dtype=float) * self._a + self._b

    def y_native(self, v_scaled):
        return (np.asarray(v_scaled, dtype=float) - self._b) / self._a

    def hi_of_rows(self, idx):
        """Mass-balance ceiling for each row: scaled image of that row's own S0."""
        return self.y_scaled(RAW[np.asarray(idx), SUB_COL])

    def net_inputs(self, idx, with_hi):
        idx = np.asarray(idx)
        n = len(idx)
        x = ([self.scaled[idx, T_COL]]
             + [np.full(n, self.aver[j]) for j in range(4)]
             + [self.deltas[idx, j] for j in range(4)])
        return x + [self.hi_of_rows(idx)] if with_hi else x

    def colloc_time(self, train_idx, n_per, rng, with_hi):
        """Collocation on the measured conditions, at random times."""
        train_idx = np.asarray(train_idx)
        blk = []
        for cid in dict.fromkeys(CURVE_OF_ROW[train_idx]):
            rows = train_idx[CURVE_OF_ROW[train_idx] == cid]
            blk.append((rng.uniform(self.t_lo, self.t_hi, n_per),
                        np.tile(self.deltas[rows[0], :4], (n_per, 1)),
                        np.full(n_per, RAW[rows[0], SUB_COL])))
        t = np.concatenate([b[0] for b in blk])
        D4 = np.vstack([b[1] for b in blk])
        s0 = np.concatenate([b[2] for b in blk])
        return self._assemble(t, D4, s0, with_hi)

    def colloc_cond(self, n_pts, rng, with_hi):
        """Collocation across the condition space: fuel, substrate, Rh and MB drawn
        jointly and uniformly between the measured extremes, each at a random time."""
        c = rng.uniform(self.cond_lo, self.cond_hi, size=(n_pts, 4))
        full = np.column_stack([c, np.zeros(n_pts), np.zeros(n_pts)])
        D4 = (self.scaler.transform(full) - self.aver)[:, :4]
        t = rng.uniform(self.t_lo, self.t_hi, n_pts)
        return self._assemble(t, D4, c[:, SUB_COL], with_hi)

    def _assemble(self, t, D4, s0_uM, with_hi):
        m = len(t)
        x = ([t] + [np.full(m, self.aver[j]) for j in range(4)] + [D4[:, j] for j in range(4)])
        return x + [self.y_scaled(s0_uM)] if with_hi else x

    def invert(self, idx, v):
        sol = self.scaled[np.asarray(idx)].copy()
        sol[:, -1] = np.asarray(v).ravel()
        return self.scaler.inverse_transform(sol)[:, -1]


def _graph(arch, taylor_order, pinf_mode, lo):
    t = sn.Variable("t")
    F0 = sn.Variable("F0"); S0 = sn.Variable("S0"); RH0 = sn.Variable("RH0"); MB0 = sn.Variable("MB0")
    DF_ = sn.Variable("DF"); DS = sn.Variable("DS"); DRH = sn.Variable("DRH"); DMB = sn.Variable("DMB")
    ins = [t, F0, S0, RH0, MB0, DF_, DS, DRH, DMB]
    Kn = sn.Functional("K", ins, list(arch), "tanh")
    NAHraw = sn.Functional("NAHraw", ins, list(arch), "tanh")
    NAH = t * NAHraw                                  # NAH(t = 0) = 0 exactly
    Praw = sn.Functional("Pinf", [DF_, DS, DRH, DMB], list(arch), "tanh")
    if pinf_mode == "mass_balance":
        HI = sn.Variable("HI")                        # per-row scaled image of S0
        ins = ins + [HI]
        Pinf = lo + (HI - lo) * sn.math.sigmoid(Praw)  # 0 <= Pinf <= S0 (scaled units)
    elif pinf_mode == "tanh":
        Pinf = Praw
    else:
        raise ValueError(pinf_mode)
    NAH_t = sn.math.diff(NAH, t, order=1)
    d = {"DF": DF_, "DS": DS, "DRH": DRH, "DMB": DMB}
    k_eff = Kn
    if taylor_order >= 1:
        g = {k: sn.math.diff(Kn, v) for k, v in d.items()}
        for k, v in d.items():
            k_eff = k_eff + g[k] * v
    if taylor_order >= 2:
        for k, v in d.items():
            k_eff = k_eff + 0.5 * sn.math.diff(Kn, v, order=2) * v ** 2
        ks = list(d)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                k_eff = k_eff + sn.math.diff(g[ks[i]], d[ks[j]]) * d[ks[i]] * d[ks[j]]
    L = NAH_t - k_eff * (Pinf - NAH)                  # physics residual
    return ins, NAH, L, Pinf, k_eff


def training_arrays(prep, train_idx, n_colloc, n_cond, rng, with_hi):
    """Data rows (data weight 1) followed by collocation rows (data weight 0); the
    physics residual is weighted 1 everywhere."""
    x_data = prep.net_inputs(train_idx, with_hi)
    y_data = prep.labels[train_idx]
    n_d = len(train_idx)
    blocks = []
    if n_colloc > 0:
        blocks.append(prep.colloc_time(train_idx, n_colloc, rng, with_hi))
    if n_cond > 0:
        blocks.append(prep.colloc_cond(n_cond, rng, with_hi))
    if not blocks:
        return x_data, [y_data, "zero"], None
    xc = [np.concatenate(parts) for parts in zip(*blocks)]
    n_c = len(xc[0])
    x_all = [np.concatenate([a, b]) for a, b in zip(x_data, xc)]
    y_all = [np.concatenate([y_data, np.zeros(n_c)]), "zero"]
    sw = [np.concatenate([np.ones(n_d), np.zeros(n_c)]), np.ones(n_d + n_c)]
    return x_all, y_all, sw


def build(train_idx, seed, epochs=EPOCHS, arch=(40, 30, 20, 10), n_colloc=100, n_cond=0,
          no_leak=True, lr_schedule=False, taylor_order=2, pinf_mode="tanh", verbose=0,
          return_nets=False, callbacks=None):
    """Train one PINN on rows `train_idx` of data/curves.csv.

    Returns predict(idx) -> NAH (uM) for rows idx, or with return_nets=True a dict with
    'predict', 'predict_curve' (any condition and times), 'history', 'prep'.
    Optimiser: SciANN defaults (Adam, learning rate 1e-3, halved after epochs/10 epochs
    without improvement of the training loss), batch size 32, MSE loss."""
    train_idx = np.asarray(train_idx)
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    rng = np.random.default_rng(seed)
    prep = Prep(train_idx, no_leak=no_leak)
    with_hi = (pinf_mode == "mass_balance")
    ins, NAH, L, Pinf, k_eff = _graph(list(arch), taylor_order, pinf_mode, prep.lo)
    m = sn.SciModel(ins, [sn.Data(NAH), sn.Data(L)], loss_func="mse")

    x_all, y_all, sw = training_arrays(prep, train_idx, n_colloc, n_cond, rng, with_hi)
    lr = ([0, int(epochs * 0.6)], [1e-3, 1e-4]) if lr_schedule else 1e-3
    kw = dict(x_true=x_all, y_true=y_all, learning_rate=lr, batch_size=32,
              epochs=epochs, verbose=verbose)
    if sw is not None:
        kw["weights"] = sw
    if callbacks:
        kw["callbacks"] = callbacks
    h = m.train(**kw)

    def predict(idx):
        # NAH does not depend on HI, so it takes the 9 network inputs
        idx = np.asarray(idx)
        return prep.invert(idx, NAH.eval(prep.net_inputs(idx, with_hi)[:9]))

    if not return_nets:
        return predict

    def predict_curve(cond, times):
        """NAH (uM) at any condition [fuel_mM, sub_uM, rh_uM, mb_uM] and times (min);
        used for the held-out MB = 10 uM curve, which is not a row of curves.csv."""
        times = np.atleast_1d(times).astype(float)
        n = len(times)
        X = np.zeros((n, 6))
        X[:, :4] = np.asarray(cond, dtype=float)
        X[:, T_COL] = times
        Xs = prep.scaler.transform(X)
        dl = Xs - prep.aver
        xin = ([Xs[:, T_COL]] + [np.full(n, prep.aver[j]) for j in range(4)]
               + [dl[:, j] for j in range(4)])
        sol = Xs.copy()
        sol[:, -1] = np.asarray(NAH.eval(xin)).ravel()
        return prep.scaler.inverse_transform(sol)[:, -1]

    return {"predict": predict, "predict_curve": predict_curve, "prep": prep,
            "history": getattr(h, "history", h), "NAH": NAH}


def outer_split(split_seed=K.SPLIT_SEED):
    """The 80/20 split over row indices (181 train / 46 test for split_seed = 0)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(N), test_size=K.TEST_SIZE, random_state=split_seed)


def set_threads():
    """Optional: fix the TensorFlow thread count (env PINN_THREADS). The thread count
    changes floating-point summation order and moves results by ~0.001-0.005 in R2."""
    import os
    n = os.environ.get("PINN_THREADS")
    if n:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(n))
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass
