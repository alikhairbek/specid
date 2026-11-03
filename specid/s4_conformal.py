#!/usr/bin/env python3
"""
s4_conformal.py -- Step 4: learned retrieval models + distribution-free (conformal) uncertainty, compared with the Cramer-Rao
bounds of Step 3b and validated from synthetic spectra to the measured laboratory mixtures (sim-to-real).

Per observation mode (Step-3b definitions: lab = absolute REFF 0.45-2.45 um / 5 nm, SNR 1000 + model error; tel50 = normalised
0.45-2.45 um / 10 nm, SNR 50 + model error; dawnfc = Dawn FC 7 bands, normalised, SNR 100 + model error):
  1. synthetic population: states sampled over the olivine-opx-plagioclase box (olivine share of the mafic part 0.05-0.95,
     plagioclase 0-0.9, Fo 50-95, Fs 5-50, d 5-300 um log-uniform, weathering s = 0 with probability 0.3 else U(0, 2),
     level nuisances N(0, 0.5^2) clipped at 1.5); spectra from the Step-3a model; noise drawn from the full Step-3b covariance
     (instrument + shrunk empirical model-error covariance), one draw per state.
     Library mismatch ("nuisance-aware simulation"): the natural end-members that are NOT used by the measured test mixtures are
     fitted with the model (chemistry from Step 3a, rule size, free level); the residual fields delta_f(lambda) = ln w_meas - ln w_model
     form, per family, an empirical pool of end-member mismatch; each synthetic component is multiplied by exp(a * delta) with a random
     field from the pool, a random sign and amplitude U(0.6, 1.4) (probability p_aug per component). The same pools give a
     mismatch covariance that is propagated through the model for an "augmented" Cramer-Rao bound (library mismatch + mixing error).
     An ablation without the mismatch augmentation (lab_noaug) quantifies what the augmentation does to the sim-to-real gap.
  2. inverse models trained on the noisy spectra: gradient boosting (one model per target) and an MLP; targets
     (ilr1, ilr2, Fo, Fs_opx, ln d, s); gradient-boosting scale models on log|residual| (heteroscedastic normalisation)
  3. split conformal prediction (calibration set disjoint from the training and scale sets): per-target normalised intervals for
     M_ol, M_opx, M_plag, Fo, Fs_opx, ln d, and a joint compositional region = Aitchison (ilr) ball around the predicted composition
  4. evaluation: coverage and width on a synthetic test set; conformal 68 % half-width vs the Cramer-Rao sigma at the true state;
     the physics retrieval (whitened least squares, Step 3b) as a reference estimator; then the measured tier-A silicate mixtures
     (sim-to-real): coverage with the synthetic calibration, and after leave-one-series-out recalibration on measured mixtures
"""
from __future__ import annotations
import json, math, os, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_forward as S2
import forward_param as FP
import s3b_identify as S3B

VERSION = 's4-conformal/1.0'
GRID = S2.GRID; WONG = S2.WONG; INK = S2.INK
SCN = 'ol-opx-plag'; FAMS = ['ol', 'opx', 'plag']
TARGETS = ['ilr1', 'ilr2', 'Fo', 'Fs_opx', 'logd', 's']
REPORT_TARGETS = ['M_ol', 'M_opx', 'M_plag', 'Fo', 'Fs_opx', 'logd']
ALPHAS = (0.32, 0.10, 0.05)
POP = dict(ol=(0.05, 0.95), plag=(0.0, 0.9), Fo=(50.0, 95.0), Fs=(5.0, 50.0), d=(5.0, 300.0), s=(0.0, 2.0), p_s0=0.3, level_sd=0.5, level_clip=1.5, p_aug=0.8, aug_amp=(0.6, 1.4))
EVAL = S2.EVAL; MU0, MU = S2.mus(30.0, 0.0)

# ----------------------------------------------------------------------------- library mismatch (held-out natural end-members)
def d_rule(lo, hi, rule):
    lo, hi = S2._f(lo), S2._f(hi)
    if hi <= 0: return 60.0 * rule['bounded']
    if lo <= 0: return hi * rule['unbounded']
    return 0.5 * (lo + hi) * rule['bounded']

class LibraryMismatch:
    """per family, the pool of mismatch fields delta(lambda) = ln(1 - w_meas) - ln(1 - w_model) (model at the Step-3a chemistry, rule size,
    best level; 1 - w is the absorption-like quantity, proportional to k d for weak absorption, so delta is a log-k-like mismatch of level
    and shape) of the natural end-members not excluded (the end-members of the measured test mixtures are excluded -> strict sim-to-real)"""
    def __init__(self, model, qc, Xspec, spec_ids, em, nat, rule, exclude_codes=(), rms_max=0.35):
        from scipy.optimize import minimize_scalar
        famap = {'olivine': 'ol', 'opx': 'opx', 'plagioclase': 'plag'}; pos = {s: i for i, s in enumerate(spec_ids)}; qci = qc.set_index('spectrum_id')
        natq = nat[nat['model'] == 'quad'].set_index('code') if 'model' in nat else nat.set_index('code'); self.pools = {f: [] for f in FAMS}; self.table = []
        for _, e in em[em['family'].isin(famap)].iterrows():
            sid = e.get('primary_spectrum'); code = e['code']; fam = famap[e['family']]
            if not isinstance(sid, str) or sid not in pos or sid not in qci.index or qci.loc[sid, 'status'] != 'ok': continue
            chem = np.nan
            if fam in ('ol', 'opx'):
                if code not in natq.index: continue
                q = natq.loc[code]; q = q.iloc[0] if isinstance(q, pd.DataFrame) else q; chem = S2._f(q.get('Fo_fit' if fam == 'ol' else 'Fs_fit', np.nan), np.nan)
                if not np.isfinite(chem): continue
            r = qci.loc[sid]; mu0, mu = S2.mus(r['incidence_deg'], r['emission_deg']); w_meas = FP.w_from_reff(Xspec[pos[sid]], mu0, mu); d0 = d_rule(e['size_lo_um'], e['size_hi_um'], rule)
            th = {'Fo': float(chem) if fam == 'ol' else 90.0, 'Fs_opx': float(chem) if fam == 'opx' else 20.0, 'l_ol': 0.0, 'l_opx': 0.0, 'l_plag': 0.0, 'cpx_natural': True}
            ok = np.isfinite(w_meas) & EVAL & (w_meas > 1e-4) & (w_meas < 1 - 1e-4)
            if ok.sum() < 100: continue
            a_meas = np.log(np.clip(1.0 - np.where(ok, w_meas, 0.5), 1e-6, None))
            def rms(l):
                th2 = dict(th); th2['l_' + fam] = float(l); wm = model.w_component(fam, th2, d0); return float(np.sqrt(np.mean((a_meas[ok] - np.log(np.clip(1.0 - wm[ok], 1e-6, None))) ** 2)))
            res = minimize_scalar(rms, bounds=(-2.5, 2.5), method='bounded'); th2 = dict(th); th2['l_' + fam] = float(res.x); wm = model.w_component(fam, th2, d0)
            delta = np.where(ok & EVAL, a_meas - np.log(np.clip(1.0 - wm, 1e-6, None)), 0.0)
            row = {'code': code, 'family': fam, 'name': e.get('name', ''), 'd0_um': d0, 'chem': chem, 'l_fit': float(res.x), 'rms_ln1mw': float(res.fun), 'excluded': code in exclude_codes, 'used': (code not in exclude_codes) and (res.fun < rms_max)}
            self.table.append(row)
            if row['used']: self.pools[fam].append(delta)
        self.pools = {f: (np.vstack(v) if v else np.zeros((0, len(GRID)))) for f, v in self.pools.items()}; self.table = pd.DataFrame(self.table)
        a0, a1 = POP['aug_amp']; self.Ea2 = (a0 * a0 + a0 * a1 + a1 * a1) / 3.0
        self.C = {f: (self.Ea2 * (P.T @ P) / len(P) if len(P) else np.zeros((len(GRID), len(GRID)))) for f, P in self.pools.items()}
    def summary(self):
        return {f: {'n': int(len(P)), 'rms_ln1mw_median': float(np.median(np.sqrt(np.mean(P[:, EVAL] ** 2, axis=1)))) if len(P) else float('nan')} for f, P in self.pools.items()}
    def draw(self, rng, fams, p=None):
        """multiplicative factors on (1 - w) per family (absent family = no perturbation)"""
        p = POP['p_aug'] if p is None else p; out = {}
        for f in fams:
            P = self.pools.get(f)
            if P is None or len(P) == 0 or rng.uniform() > p: continue
            a = rng.uniform(*POP['aug_amp']) * (1.0 if rng.uniform() < 0.5 else -1.0); out[f] = np.exp(a * P[rng.integers(len(P))])
        return out

def spectrum_aug(model, theta, factors=None):
    """REFF at (30, 0) with per-family multiplicative perturbations of 1 - w (library mismatch augmentation): w_f' = 1 - (1 - w_f) * factor"""
    M = theta['M']; d = theta.get('d', 60.0); fams = [f for f in FP._FAMS if M.get(f, 0) > 0]
    F = np.array([M[f] / (FP.DENSITY[f] * d) for f in fams]); F = F / F.sum(); w = np.zeros(len(model.grid))
    for Fi, f in zip(F, fams):
        wf = model.w_component(f, theta, d)
        if factors and f in factors: wf = np.clip(1.0 - (1.0 - wf) * factors[f], 1e-6, 1 - 1e-6)
        w = w + Fi * wf
    s = theta.get('s', 0.0)
    if s: w = np.clip(w * model.T ** s, 1e-6, 1 - 1e-6)
    return S2.reff_from_w(np.clip(w, 1e-6, 1 - 1e-6), MU0, MU)

class AugObserver(S3B.Observer):
    """Observer whose covariance adds the library-mismatch term: Sigma_lib = sum_f (g_f g_f^T) o C_f, g_f = dR/d ln(1 - w_f), propagated to
    the mode's channels (hyperspectral: channel subset / normalisation; bands: filter integration)"""
    def __init__(self, model, mode, resid_npz, lib: LibraryMismatch, eps=0.05):
        super().__init__(model, mode, resid_npz); self.lib = lib; self.eps = eps
    def cov(self, theta):
        S = super().cov(theta); R = self.model.spectrum(theta, mode='reff'); fams = [f for f in FAMS if theta['M'].get(f, 0) > 0]; SR = np.zeros((len(R), len(R)))
        for f in fams:
            if len(self.lib.pools.get(f, [])) == 0: continue
            g = (spectrum_aug(self.model, theta, {f: np.exp(self.eps)}) - R) / self.eps; SR += np.outer(g, g) * self.lib.C[f]
        if self.bands is None:
            Sl = SR[np.ix_(self.idx, self.idx)]
            if self.cfg['norm'] is not None: Sl = Sl / np.interp(self.cfg['norm'], self.grid, R) ** 2
        else:
            Sl = self.B @ SR @ self.B.T; yb = self.B @ R
            if self.cfg['norm'] is not None: Sl = Sl / yb[self.iref] ** 2
        return S + Sl

# ----------------------------------------------------------------------------- synthetic population
def sample_states(n, rng, pop=POP):
    ol = rng.uniform(*pop['ol'], n); plag = rng.uniform(*pop['plag'], n); Fo = rng.uniform(*pop['Fo'], n); Fs = rng.uniform(*pop['Fs'], n)
    d = np.exp(rng.uniform(np.log(pop['d'][0]), np.log(pop['d'][1]), n)); s = np.where(rng.uniform(0, 1, n) < pop['p_s0'], 0.0, rng.uniform(*pop['s'], n))
    L = np.clip(rng.normal(0, pop['level_sd'], (n, 3)), -pop['level_clip'], pop['level_clip']); states = []
    for i in range(n):
        th = S3B.base_theta(SCN, ol=float(ol[i]), Fo=float(Fo[i]), Fs=float(Fs[i]), d=float(d[i]), s=float(s[i]), plag=float(max(plag[i], 1e-3)))
        th['l_ol'], th['l_opx'], th['l_plag'] = (float(x) for x in L[i]); states.append(th)
    return states

def targets_of(th):
    z = FP.ilr([th['M']['ol'], th['M']['opx'], th['M']['plag']]); return np.array([z[0], z[1], th['Fo'], th['Fs_opx'], math.log(th['d']), th['s']])

def y_from_R(obs: S3B.Observer, R):
    """the mode's observation vector from a REFF spectrum on the common grid (same operator as Observer.y)"""
    if obs.bands is None:
        y = R[obs.idx]; return y / np.interp(obs.cfg['norm'], obs.grid, R) if obs.cfg['norm'] is not None else y
    yb = obs.B @ R; return yb / yb[obs.iref] if obs.cfg['norm'] is not None else yb

def cov_from_R(obs: S3B.Observer, R):
    """full covariance of y from the REFF spectrum (same as Observer.cov, without recomputing the forward model)"""
    y = y_from_R(obs, R); S = np.diag((y / obs.cfg['snr']) ** 2)
    if obs.Crel is not None:
        if obs.bands is None: S = S + obs.Crel[np.ix_(obs.idx, obs.idx)] * np.outer(y, y)
        else:
            Sa = obs.B @ (obs.Crel * np.outer(R, R)) @ obs.B.T; yb = obs.B @ R; S = S + (Sa / yb[obs.iref] ** 2 if obs.cfg['norm'] is not None else Sa)
    return S + np.eye(obs.n) * (1e-8 * np.mean(np.diag(S)) + 1e-14)

def simulate(model, observers: dict, states, rng, noise=True, lib: LibraryMismatch = None):
    """forward model once per state (with a library-mismatch draw when lib is given); per mode: observation vector + one noise draw from
    the full covariance -> {mode: Y}, T"""
    out = {m: np.zeros((len(states), o.n)) for m, o in observers.items()}; T = np.zeros((len(states), len(TARGETS)))
    for i, th in enumerate(states):
        fac = lib.draw(rng, [f for f in FAMS if th['M'].get(f, 0) > 0]) if lib is not None else None
        R = spectrum_aug(model, th, fac) if fac else model.spectrum(th, mode='reff'); T[i] = targets_of(th)
        for m, o in observers.items():
            y = y_from_R(o, R)
            if noise: y = y + np.linalg.cholesky(cov_from_R(o, R)) @ rng.normal(0, 1, o.n)
            out[m][i] = y
    return out, T

def M_from_targets(T): return np.vstack([FP.ilr_inv(t[:2]) for t in T])

def report_space(T):
    """(ilr1, ilr2, Fo, Fs, logd, s) -> (M_ol, M_opx, M_plag, Fo, Fs, logd)"""
    M = M_from_targets(T); return np.column_stack([M[:, 0], M[:, 1], M[:, 2], T[:, 2], T[:, 3], T[:, 4]])

# ----------------------------------------------------------------------------- inverse models
class Inverse:
    """point model: gradient boosting per target ('hgb') or one MLP ('mlp'); scale models: gradient boosting on log|residual| per
    reported target and on log||ilr residual|| (for the compositional ball)"""
    def __init__(self, kind='hgb', seed=0, quick=False):
        self.kind = kind; self.seed = seed; self.quick = quick; self.models = {}; self.scale_models = {}; self.scaler = None; self.tscaler = None; self.mlp = None
    def fit(self, X, T):
        if self.kind == 'hgb':
            from sklearn.ensemble import HistGradientBoostingRegressor
            for j, nm in enumerate(TARGETS):
                m = HistGradientBoostingRegressor(max_iter=100 if self.quick else 500, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.1,
                                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=self.seed)
                m.fit(X, T[:, j]); self.models[nm] = m
        else:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler().fit(X); self.tscaler = StandardScaler().fit(T)
            self.mlp = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=40 if self.quick else 400,
                                    early_stopping=True, n_iter_no_change=20, validation_fraction=0.1, random_state=self.seed)
            self.mlp.fit(self.scaler.transform(X), self.tscaler.transform(T))
        return self
    def predict(self, X):
        if self.kind == 'hgb': return np.column_stack([self.models[nm].predict(X) for nm in TARGETS])
        return self.tscaler.inverse_transform(self.mlp.predict(self.scaler.transform(X)))
    def fit_scale(self, X, T):
        from sklearn.ensemble import HistGradientBoostingRegressor
        P = self.predict(X); R = report_space(T) - report_space(P); dz = np.linalg.norm(T[:, :2] - P[:, :2], axis=1)
        for j, nm in enumerate(REPORT_TARGETS + ['ilr_ball']):
            r = dz if nm == 'ilr_ball' else np.abs(R[:, j])
            m = HistGradientBoostingRegressor(max_iter=60 if self.quick else 200, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
            m.fit(X, np.log(r + 1e-6)); self.scale_models[nm] = m
        return self
    def scale(self, X):
        return np.column_stack([np.exp(self.scale_models[nm].predict(X)) for nm in REPORT_TARGETS + ['ilr_ball']])

# ----------------------------------------------------------------------------- conformal calibration
def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, float); scores = scores[np.isfinite(scores)]; n = len(scores)
    if n == 0: return float('nan')
    k = min(n, int(np.ceil((n + 1) * (1 - alpha)))); return float(np.sort(scores)[k - 1])

def scores_of(inv: Inverse, X, T):
    """normalised nonconformity scores: |residual| / scale for the reported targets, ||ilr residual|| / scale for the ball"""
    P = inv.predict(X); C = report_space(P); Rt = report_space(T); S = inv.scale(X); dz = np.linalg.norm(T[:, :2] - P[:, :2], axis=1)
    return np.column_stack([np.abs(Rt - C) / S[:, :-1], dz / S[:, -1]]), C, S, P

class Conformal:
    """split conformal with normalised scores; q[alpha] holds one quantile per reported target + the ilr ball"""
    def __init__(self, inv: Inverse): self.inv = inv; self.q = {}
    def calibrate(self, X, T, alphas=ALPHAS):
        sc, _, _, _ = scores_of(self.inv, X, T); self.cal_scores = sc
        for a in alphas: self.q[a] = np.array([conformal_quantile(sc[:, j], a) for j in range(sc.shape[1])])
        return self
    def predict(self, X, alpha):
        """centres (report space), half-widths per reported target, ilr-ball radius, raw target predictions"""
        P = self.inv.predict(X); C = report_space(P); S = self.inv.scale(X); q = self.q[alpha]
        return C, S[:, :-1] * q[:-1], S[:, -1] * q[-1], P

def evaluate(conf: Conformal, X, T, label, alphas=ALPHAS):
    Rt = report_space(T); rows = []
    for a in alphas:
        C, hw, rad, P = conf.predict(X, a); dz = np.linalg.norm(T[:, :2] - P[:, :2], axis=1)
        for j, nm in enumerate(REPORT_TARGETS):
            m = np.isfinite(Rt[:, j]); cov = np.abs(Rt[m, j] - C[m, j]) <= hw[m, j]
            rows.append({'set': label, 'alpha': a, 'target': nm, 'coverage': float(cov.mean()) if m.any() else np.nan, 'mean_halfwidth': float(hw[m, j].mean()), 'median_halfwidth': float(np.median(hw[m, j])),
                         'mae': float(np.mean(np.abs(Rt[m, j] - C[m, j]))) if m.any() else np.nan, 'n': int(m.sum())})
        rows.append({'set': label, 'alpha': a, 'target': 'ilr_ball', 'coverage': float((dz <= rad).mean()), 'mean_halfwidth': float(rad.mean()), 'median_halfwidth': float(np.median(rad)), 'mae': float(dz.mean()), 'n': len(X)})
    return pd.DataFrame(rows)

def loso_recalibration(conf: Conformal, meta, Xm, Tm, alphas=ALPHAS, min_train=10):
    """leave-one-series-out on the measured mixtures: quantiles recalibrated on the other series, tested on the held-out series"""
    sc, C, S, P = scores_of(conf.inv, Xm, Tm); Rt = report_space(Tm); dz = np.linalg.norm(Tm[:, :2] - P[:, :2], axis=1); rows = []; per_mix = {}
    for a in alphas:
        cov = np.full(sc.shape, np.nan); hw = np.full(sc.shape, np.nan)
        for ser in meta['series'].unique():
            te = (meta['series'] == ser).values; tr = ~te
            if tr.sum() < min_train: continue
            q = np.array([conformal_quantile(sc[tr, j], a) for j in range(sc.shape[1])]); hw[te] = S[te] * q
            cov[te, :-1] = np.abs(Rt[te] - C[te]) <= hw[te, :-1]; cov[te, -1] = dz[te] <= hw[te, -1]
        for j, nm in enumerate(REPORT_TARGETS + ['ilr_ball']):
            m = np.isfinite(hw[:, j]) & (np.isfinite(Rt[:, j]) if j < len(REPORT_TARGETS) else np.ones(len(Rt), bool))
            err = np.abs(Rt[m, j] - C[m, j]) if j < len(REPORT_TARGETS) else dz[m]
            rows.append({'set': 'measured_LOSO_recal', 'alpha': a, 'target': nm, 'coverage': float(np.nanmean(cov[m, j])) if m.any() else np.nan, 'mean_halfwidth': float(np.nanmean(hw[m, j])) if m.any() else np.nan,
                         'median_halfwidth': float(np.nanmedian(hw[m, j])) if m.any() else np.nan, 'mae': float(np.mean(err)) if m.any() else np.nan, 'n': int(m.sum())})
        if a == 0.10: per_mix = {'hw90_loso_M_ol': hw[:, 0], 'covered90_loso_M_ol': cov[:, 0], 'hw90_loso_Fo': hw[:, 3], 'covered90_loso_Fo': cov[:, 3]}
    return pd.DataFrame(rows), per_mix

# ----------------------------------------------------------------------------- physics retrieval (reference estimator) and CRB
def physics_retrieve(obs: S3B.Observer, Yobs, n_starts=2, seed=0, max_nfev=300):
    """whitened least-squares retrieval of each observed vector (Step-3b estimator): coarse grid start on a diagonal misfit, covariance
    evaluated at the coarse-start state (the truth is unknown), nuisance priors as in Step 3b; returns report-space estimates + chi2"""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed); th0 = S3B.base_theta(SCN, ol=0.5, Fo=75, Fs=25, d=60, s=0.5, plag=0.25); pv = S3B.make_pv(SCN, th0); lo, hi = pv.bounds(); names = pv.names()
    P = S3B.prior_precision(pv); pw = np.sqrt(np.diag(P)) * np.array([0.0 if (nm.startswith('ilr') or nm in ('Fo', 'Fs_opx', 'logd')) else 1.0 for nm in names])
    grid = [(ol, plag, Fo, Fs) for ol in (0.15, 0.4, 0.65, 0.9) for plag in (0.05, 0.25, 0.45) for Fo in (60, 78, 90) for Fs in (10, 25, 40)]
    Yg = np.array([obs.y(S3B.base_theta(SCN, ol=ol, Fo=Fo, Fs=Fs, d=60, s=0.5, plag=plag)) for ol, plag, Fo, Fs in grid]); out = []
    for y in Yobs:
        c = np.sum(((Yg - y) / np.maximum(np.abs(y), 1e-3)) ** 2, axis=1); ol, plag, Fo, Fs = grid[int(np.argmin(c))]
        th_c = S3B.base_theta(SCN, ol=ol, Fo=Fo, Fs=Fs, d=60, s=0.5, plag=plag); L = obs.whitener(th_c); v0 = pv.pack(th_c)
        def resid(v): return np.concatenate([solve_triangular(L, obs.y(pv.unpack(v)) - y, lower=True), pw * v])
        best = None
        for j in range(n_starts):
            vs = v0 if j == 0 else np.clip(v0 + rng.normal(0, 1, len(v0)) * np.array([0.3 if nm.startswith('ilr') else {'Fo': 8, 'Fs_opx': 8, 'logd': 0.4, 's': 0.3}.get(nm, 0.2) for nm in names]), lo + 1e-6, hi - 1e-6)
            r = least_squares(resid, vs, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=max_nfev)
            if best is None or r.cost < best.cost: best = r
        th = pv.unpack(best.x); out.append([th['M']['ol'], th['M']['opx'], th['M']['plag'], th['Fo'], th['Fs_opx'], math.log(th['d']), 2 * best.cost])
    return np.array(out)

def crb_at(obs: S3B.Observer, states):
    rows = []
    for th in states:
        pv = S3B.make_pv(SCN, th); r = S3B.analyse(obs, pv, th); rows.append([r['sigma_M_ol'], r['sigma_M_opx'], r['sigma_M_plag'], r['sigma_Fo'], r['sigma_Fs_opx'], r['sigma_logd']])
    return np.array(rows)

# ----------------------------------------------------------------------------- measured mixtures (sim-to-real)
def measured_set(obs: S3B.Observer, qc, Xspec, spec_ids, mix, nat, rule, exclude_series=()):
    """observation vectors of the tier-A olivine-opx(-plagioclase) mixtures with 'ok' primary spectra, in the mode's geometry; targets =
    documented mass fractions (ilr; absent plagioclase -> 0.005), model-consistent end-member chemistry (Fo/Fs fitted on the pure
    end-member spectra in Step 3a), ln d_eff from the grain-size rule; weathering 0"""
    famap = {'olivine': 'ol', 'opx': 'opx', 'plagioclase': 'plag'}; pos = {s: i for i, s in enumerate(spec_ids)}; okset = set(qc.loc[qc['status'] == 'ok', 'spectrum_id'])
    natq = nat[nat['model'] == 'quad'].set_index('code') if 'model' in nat else nat.set_index('code'); rows, Y, T = [], [], []
    for _, r in mix[(mix['tier'] == 'A') & ~mix['series'].isin(exclude_series)].iterrows():
        comps = json.loads(r['components']); fams = [famap.get(c['fam']) for c in comps]
        if None in fams or len(set(fams)) < len(fams) or not ({'ol', 'opx'} <= set(fams)) or r['primary_spectrum'] not in okset or r['primary_spectrum'] not in pos: continue
        M = {f: 0.0 for f in FAMS}; M.update({f: float(c['frac']) for f, c in zip(fams, comps)}); tot = sum(M.values()); M = {k: v / tot for k, v in M.items()}
        d = float(np.mean([S2._f(c['size_hi']) * rule['unbounded'] if S2._f(c['size_lo']) <= 0 else 0.5 * (S2._f(c['size_lo']) + S2._f(c['size_hi'])) * rule['bounded'] for c in comps]))
        Fo, Fs = np.nan, np.nan
        for f, c in zip(fams, comps):
            if c['code'] in natq.index:
                q = natq.loc[c['code']]; q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
                if f == 'ol': Fo = S2._f(q.get('Fo_fit', np.nan), np.nan)
                if f == 'opx': Fs = S2._f(q.get('Fs_fit', np.nan), np.nan)
        y = y_from_R(obs, Xspec[pos[r['primary_spectrum']]])
        if not np.all(np.isfinite(y)): continue
        Mv = np.array([max(M['ol'], 0.005), max(M['opx'], 0.005), max(M['plag'], 0.005)]); Mv = Mv / Mv.sum(); z = FP.ilr(Mv)
        T.append([z[0], z[1], Fo, Fs, math.log(d), 0.0]); Y.append(y)
        rows.append({'mix_id': r['mix_id'], 'series': r['series'], 'n_components': len(fams), 'M_ol': M['ol'], 'M_opx': M['opx'], 'M_plag': M['plag'], 'Fo_ref': Fo, 'Fs_ref': Fs, 'd_eff_um': d, 'spectrum': r['primary_spectrum']})
    return pd.DataFrame(rows), np.array(Y), np.array(T)

# ----------------------------------------------------------------------------- figures (Wong palette, no black elements)
def plot_reliability(ev, out: Path):
    plt = S2.style(); modes = list(dict.fromkeys(ev['mode'])); fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    sets = (('synthetic_test', '-', 'synthetic test'), ('measured_syncal', '--', 'measured, synthetic calibration'), ('measured_LOSO_recal', ':', 'measured, LOSO recalibration'))
    for k, mode in enumerate(modes):
        for sett, ls, lab in sets:
            g = ev[(ev['mode'] == mode) & (ev['set'] == sett) & (ev['target'] == 'M_ol')].sort_values('alpha', ascending=False)
            if len(g): axes[0].plot(1 - g['alpha'], g['coverage'], color=WONG[k % 7], ls=ls, marker='o', ms=4, lw=1.3, label=f'{mode}: {lab}')
    axes[0].plot([0.6, 1], [0.6, 1], color='#999999', lw=0.8); axes[0].set_xlim(0.6, 1.0); axes[0].set_ylim(0.3, 1.02); axes[0].set_xlabel('nominal coverage'); axes[0].set_ylabel('empirical coverage, olivine mass fraction'); axes[0].legend(fontsize=5.5, frameon=False, loc='upper left')
    w = 0.8 / max(len(modes), 1)
    for k, mode in enumerate(modes):
        g = ev[(ev['mode'] == mode) & (ev['set'] == 'synthetic_test') & (ev['alpha'] == 0.10)]
        axes[1].bar(np.arange(len(REPORT_TARGETS) + 1) + w * k, [float(g[g['target'] == t]['coverage'].iloc[0]) if (g['target'] == t).any() else 0 for t in REPORT_TARGETS + ['ilr_ball']], width=w, color=WONG[k % 7], label=mode)
    axes[1].axhline(0.9, color='#999999', lw=0.8, ls='--'); axes[1].set_xticks(np.arange(len(REPORT_TARGETS) + 1) + w * (len(modes) - 1) / 2); axes[1].set_xticklabels(REPORT_TARGETS + ['ilr ball'], fontsize=8, rotation=20); axes[1].set_ylim(0.5, 1.0); axes[1].set_ylabel('coverage at nominal 90 % (synthetic test)'); axes[1].legend(fontsize=7, frameon=False)
    for k, mode in enumerate(modes):
        for i, (sett, al) in enumerate((('synthetic_test', 0.95), ('measured_syncal', 0.6), ('measured_LOSO_recal', 0.35))):
            g = ev[(ev['mode'] == mode) & (ev['set'] == sett) & (ev['alpha'] == 0.10) & (ev['target'] == 'M_ol')]
            axes[2].bar(k + (i - 1) * 0.27, float(g['median_halfwidth'].iloc[0]) if len(g) else 0, width=0.27, color=WONG[k % 7], alpha=al, label=sets[i][2] if k == 0 else None)
    axes[2].set_xticks(range(len(modes))); axes[2].set_xticklabels(modes, fontsize=8); axes[2].set_ylabel('median 90 % half-width, olivine mass fraction'); axes[2].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S4_F1_reliability.png'); fig.savefig(out / 'figures' / 'S4_F1_reliability.pdf'); plt.close(fig)

def plot_width_vs_crb(cmp, out: Path):
    plt = S2.style(); modes = list(dict.fromkeys(cmp['mode'])); fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    for row, (prefix, lab_row) in enumerate((('crb', 'Cramér–Rao σ, mixing error only (Step 3b)'), ('crbaug', 'Cramér–Rao σ, mixing error + library mismatch'))):
        for ax, (t, lab, lim) in zip(axes[row], (('M_ol', 'olivine mass fraction', 0.25), ('Fo', 'Fo', 25), ('logd', 'ln d', 1.4))):
            for k, mode in enumerate(modes):
                g = cmp[cmp['mode'] == mode]; ax.scatter(g[f'{prefix}_{t}'], g[f'hw68_{t}'], s=9, color=WONG[k % 7], marker='os^D'[k % 4], alpha=0.55, linewidths=0, label=mode)
            ax.plot([0, lim], [0, lim], color='#999999', lw=0.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel(f'{lab_row} ({lab})', fontsize=8); ax.set_ylabel('conformal 68 % half-width', fontsize=8)
            if row == 0: ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S4_F2_width_vs_crb.png'); fig.savefig(out / 'figures' / 'S4_F2_width_vs_crb.pdf'); plt.close(fig)

def plot_measured(meas, out: Path):
    plt = S2.style(); modes = list(dict.fromkeys(meas['mode'])); fig, axes = plt.subplots(2, len(modes), figsize=(4.9 * len(modes), 8.2), squeeze=False)
    for k, mode in enumerate(modes):
        g = meas[meas['mode'] == mode]
        for row, (xc, yc, hc, lab, lim) in enumerate((('M_ol', 'pred_M_ol', 'hw90_M_ol', 'olivine mass fraction', (0, 1)), ('Fo_ref', 'pred_Fo', 'hw90_Fo', 'Fo (end-member reference)', (45, 100)))):
            ax = axes[row][k]
            for j, (ser, gg) in enumerate(g.groupby('series')):
                ax.errorbar(gg[xc], gg[yc], yerr=gg[hc], fmt='os^Dv<>'[j % 7], ms=4, color=WONG[j % 7], ecolor=WONG[j % 7], elinewidth=0.6, capsize=0, alpha=0.8, label=ser)
            ax.plot(lim, lim, color='#999999', lw=0.8); ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel(f'documented {lab}'); ax.set_ylabel('predicted ± 90 % conformal (synthetic calibration)'); ax.set_title(f'{mode}: measured mixtures', fontsize=9)
            if k == 0 and row == 0: ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S4_F3_measured_mixtures.png'); fig.savefig(out / 'figures' / 'S4_F3_measured_mixtures.pdf'); plt.close(fig)

def plot_models(mc, out: Path):
    plt = S2.style(); modes = list(dict.fromkeys(mc['mode'])); fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    labels = {'hgb': 'gradient boosting', 'mlp': 'MLP', 'physics_LSQ': 'physics (whitened LSQ)'}
    for row, sett in enumerate(('synthetic_test', 'measured')):
        for ax, (t, lab) in zip(axes[row], (('M_ol', 'olivine mass fraction'), ('Fo', 'Fo'), ('logd', 'ln d'))):
            models = ['hgb', 'mlp', 'physics_LSQ']; w = 0.8 / len(models)
            for i, mdl in enumerate(models):
                sel = (mc['set'] == sett) & (mc['model'] == mdl) & (mc['target'] == t)
                vals = [float(mc[sel & (mc['mode'] == m)]['mae_test'].iloc[0]) if (sel & (mc['mode'] == m)).any() else np.nan for m in modes]
                ax.bar(np.arange(len(modes)) + w * i, vals, width=w, color=WONG[i], label=labels[mdl])
            ax.set_xticks(np.arange(len(modes)) + w); ax.set_xticklabels(modes, fontsize=8); ax.set_ylabel(f'MAE, {sett.replace("_", " ")} ({lab})', fontsize=8)
            if row == 0 and t == 'M_ol': ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S4_F4_model_comparison.png'); fig.savefig(out / 'figures' / 'S4_F4_model_comparison.pdf'); plt.close(fig)

def plot_mismatch(lib: LibraryMismatch, out: Path):
    plt = S2.style(); fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, f, lab in zip(axes, FAMS, ('olivine', 'orthopyroxene', 'plagioclase')):
        P = lib.pools[f]
        for i, d in enumerate(P): ax.plot(GRID[EVAL], d[EVAL], color=WONG[i % 7], lw=0.9, alpha=0.8)
        ax.axhline(0, color='#999999', lw=0.7); ax.set_title(f'{lab}: {len(P)} held-out natural end-members', fontsize=9); ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('ln(1 − w) mismatch, level removed'); ax.set_ylim(-1.2, 1.2)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S4_F0_library_mismatch.png'); fig.savefig(out / 'figures' / 'S4_F0_library_mismatch.pdf'); plt.close(fig)

# ----------------------------------------------------------------------------- driver
MODE_SETS = {'lab': ('lab', True), 'tel50': ('tel50', True), 'dawnfc': ('dawnfc', True), 'lab_noaug': ('lab', False)}   # tag -> (observation mode, mismatch augmentation)

def _f(x, default=float('nan')):
    try:
        v = float(x); return v if np.isfinite(v) else default
    except Exception: return default

def test_codes(mix):
    """end-member codes used by the measured test mixtures (excluded from the mismatch pools)"""
    famap = {'olivine': 'ol', 'opx': 'opx', 'plagioclase': 'plag'}; used = set()
    for _, r in mix[mix['tier'] == 'A'].iterrows():
        comps = json.loads(r['components']); fams = [famap.get(c['fam']) for c in comps]
        if None in fams or len(set(fams)) < len(fams) or not ({'ol', 'opx'} <= set(fams)): continue
        used |= {c['code'] for c in comps}
    return used

def run_all(model, out: Path, proj: Path, resid_npz, qc, Xspec, spec_ids, mix, em, nat, rule, tags=('lab', 'tel50', 'dawnfc', 'lab_noaug'), n_train=20000, n_scale=4000, n_cal=4000, n_test=2000, n_phys=120, n_crb=500, quick=False, seed=0):
    t0 = time.time(); (out / 'figures').mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    if quick: n_train, n_scale, n_cal, n_test, n_phys, n_crb = 1500, 400, 400, 300, 6, 40
    lib = LibraryMismatch(model, qc, Xspec, spec_ids, em, nat, rule, exclude_codes=test_codes(mix)); lib.table.to_csv(out / 'library_mismatch_endmembers.csv', index=False); plot_mismatch(lib, out)
    print('[S4] library mismatch pools (held-out natural end-members):', lib.summary())
    obs = {m: S3B.Observer(model, m, resid_npz) for m in dict.fromkeys(MODE_SETS[t][0] for t in tags)}; obs_aug = {m: AugObserver(model, m, resid_npz, lib) for m in obs}
    st = {k: sample_states(n, rng) for k, n in (('tr', n_train), ('sc', n_scale), ('cal', n_cal), ('te', n_test))}
    pd.DataFrame([targets_of(th) for th in st['te']], columns=TARGETS).to_csv(out / 'synthetic_test_states.csv', index=False)
    data = {}
    for aug in sorted({MODE_SETS[t][1] for t in tags}, reverse=True):
        ob = {m: obs[m] for m in dict.fromkeys(MODE_SETS[t][0] for t in tags if MODE_SETS[t][1] == aug)}
        print(f'[S4] simulating {sum(len(v) for v in st.values())} states for modes {list(ob)} (mismatch augmentation {aug}) ...')
        data[aug] = {k: simulate(model, ob, st[k], rng, lib=lib if aug else None) for k in st}
        print(f'[S4] simulation done, {time.time() - t0:.0f}s')
    ev_all, cmp_all, meas_all, mc_rows, meas_cache = [], [], [], [], {}
    for tag in tags:
        mode, aug = MODE_SETS[tag]; o = obs[mode]; oa = obs_aug[mode]; D = data[aug]; Xtr, Ttr = D['tr'][0][mode], D['tr'][1]; Xsc, Tsc = D['sc'][0][mode], D['sc'][1]; Xcal, Tcal = D['cal'][0][mode], D['cal'][1]; Xte, Tte = D['te'][0][mode], D['te'][1]
        hgb = Inverse('hgb', seed, quick).fit(Xtr, Ttr).fit_scale(Xsc, Tsc); print(f'[{tag}] gradient boosting trained, {time.time() - t0:.0f}s')
        mlp = Inverse('mlp', seed, quick).fit(Xtr, Ttr).fit_scale(Xsc, Tsc); print(f'[{tag}] MLP trained, {time.time() - t0:.0f}s')
        Rt = report_space(Tte); invs = {'hgb': hgb, 'mlp': mlp}; confs = {}
        for nm, inv in invs.items():
            Pp = report_space(inv.predict(Xte))
            for j, t in enumerate(REPORT_TARGETS): mc_rows.append({'mode': tag, 'model': nm, 'set': 'synthetic_test', 'target': t, 'mae_test': float(np.mean(np.abs(Rt[:, j] - Pp[:, j]))), 'rmse_test': float(np.sqrt(np.mean((Rt[:, j] - Pp[:, j]) ** 2))), 'n': len(Xte)})
            confs[nm] = Conformal(inv).calibrate(Xcal, Tcal); ev = evaluate(confs[nm], Xte, Tte, 'synthetic_test'); ev['mode'] = tag; ev['model'] = nm; ev_all.append(ev)
        # primary model = lower calibration-set MAE of the olivine fraction (chosen on calibration data, not on the test set)
        cal_mae = {nm: float(np.mean(np.abs(report_space(Tcal)[:, 0] - report_space(inv.predict(Xcal))[:, 0]))) for nm, inv in invs.items()}; prim = min(cal_mae, key=cal_mae.get); conf = confs[prim]
        print(f'[{tag}] primary model: {prim} (calibration MAE olivine fraction {cal_mae})')
        # conformal 68 % half-width vs CRB (mixing error only / + library mismatch) at the true state
        idx = rng.choice(n_test, size=min(n_crb, n_test), replace=False); sub = [st['te'][i] for i in idx]; C = crb_at(o, sub); Ca = crb_at(oa, sub); c68, hw68, rad68, _ = conf.predict(Xte[idx], 0.32)
        cmp = pd.DataFrame({'mode': tag, 'model': prim, **{f'crb_{t}': C[:, j] for j, t in enumerate(REPORT_TARGETS)}, **{f'crbaug_{t}': Ca[:, j] for j, t in enumerate(REPORT_TARGETS)}, **{f'hw68_{t}': hw68[:, j] for j, t in enumerate(REPORT_TARGETS)},
                            **{f'err_{t}': np.abs(Rt[idx, j] - c68[:, j]) for j, t in enumerate(REPORT_TARGETS)}, 'true_M_ol': Rt[idx, 0], 'true_M_plag': Rt[idx, 2], 'true_Fo': Rt[idx, 3], 'true_d_um': np.exp(Rt[idx, 5]), 'true_s': Tte[idx, 5]}); cmp_all.append(cmp)
        print(f'[{tag}] conformal + CRB comparison done, {time.time() - t0:.0f}s; physics retrieval of {n_phys} test spectra ...')
        pidx = idx[:n_phys]; Pphys = physics_retrieve(o, Xte[pidx], seed=seed)
        for j, t in enumerate(REPORT_TARGETS): mc_rows.append({'mode': tag, 'model': 'physics_LSQ', 'set': 'synthetic_test', 'target': t, 'mae_test': float(np.mean(np.abs(Rt[pidx, j] - Pphys[:, j]))), 'rmse_test': float(np.sqrt(np.mean((Rt[pidx, j] - Pphys[:, j]) ** 2))), 'n': len(pidx)})
        pd.DataFrame({'idx': pidx, **{f'phys_{t}': Pphys[:, j] for j, t in enumerate(REPORT_TARGETS)}, 'phys_chi2': Pphys[:, 6], **{f'{prim}_{t}': c68[:len(pidx), j] for j, t in enumerate(REPORT_TARGETS)}, **{f'true_{t}': Rt[pidx, j] for j, t in enumerate(REPORT_TARGETS)}}).to_csv(out / f'physics_vs_learned_{tag}.csv', index=False)
        print(f'[{tag}] physics retrieval done, {time.time() - t0:.0f}s; measured mixtures ...')
        meta, Ym, Tm = measured_set(o, qc, Xspec, spec_ids, mix, nat, rule)
        if len(meta):
            Rm = report_space(Tm)
            for nm, cf in confs.items():
                evm = evaluate(cf, Ym, Tm, 'measured_syncal'); evm['mode'] = tag; evm['model'] = nm; ev_all.append(evm)
                evl, per_mix = loso_recalibration(cf, meta, Ym, Tm); evl['mode'] = tag; evl['model'] = nm; ev_all.append(evl)
                Cm, _, _, _ = cf.predict(Ym, 0.10)
                for j, t in enumerate(REPORT_TARGETS):
                    m = np.isfinite(Rm[:, j]); mc_rows.append({'mode': tag, 'model': nm, 'set': 'measured', 'target': t, 'mae_test': float(np.mean(np.abs(Rm[m, j] - Cm[m, j]))), 'rmse_test': float(np.sqrt(np.mean((Rm[m, j] - Cm[m, j]) ** 2))), 'n': int(m.sum())})
                if nm == prim: prim_per_mix = per_mix
            C90, hw90, rad90, P90 = conf.predict(Ym, 0.10); mm = meta.copy(); mm['mode'] = tag; mm['model'] = prim
            for j, t in enumerate(REPORT_TARGETS): mm[f'pred_{t}'] = C90[:, j]; mm[f'hw90_{t}'] = hw90[:, j]
            mm['pred_s'] = P90[:, 5]; mm['ilr_dist'] = np.linalg.norm(Tm[:, :2] - P90[:, :2], axis=1); mm['ball_r90'] = rad90
            mm['covered90_M_ol'] = np.abs(mm['M_ol'] - mm['pred_M_ol']) <= mm['hw90_M_ol']; mm['covered90_Fo'] = np.abs(mm['Fo_ref'] - mm['pred_Fo']) <= mm['hw90_Fo']
            for k, v in prim_per_mix.items(): mm[k] = v
            # physics retrieval of the measured mixtures: once per observation mode (independent of the augmentation)
            if mode not in meas_cache: n_pm = min(n_phys, len(Ym)); meas_cache[mode] = (n_pm, physics_retrieve(o, Ym[:n_pm], seed=seed))
            n_pm, pm = meas_cache[mode]
            for j, t in enumerate(REPORT_TARGETS): mm.loc[: n_pm - 1, f'phys_{t}'] = pm[:, j]
            mm.loc[: n_pm - 1, 'phys_chi2'] = pm[:, 6]
            for j, t in enumerate(REPORT_TARGETS):
                m = np.isfinite(Rm[:n_pm, j]); mc_rows.append({'mode': tag, 'model': 'physics_LSQ', 'set': 'measured', 'target': t, 'mae_test': float(np.mean(np.abs(Rm[:n_pm, j][m] - pm[m, j]))), 'rmse_test': float(np.sqrt(np.mean((Rm[:n_pm, j][m] - pm[m, j]) ** 2))), 'n': int(m.sum())})
            meas_all.append(mm)
        print(f'[{tag}] done, {time.time() - t0:.0f}s')
    ev_all = pd.concat(ev_all, ignore_index=True); cmp_all = pd.concat(cmp_all, ignore_index=True); meas_all = pd.concat(meas_all, ignore_index=True) if meas_all else pd.DataFrame(); mc = pd.DataFrame(mc_rows)
    ev_all.to_csv(out / 'conformal_evaluation.csv', index=False); cmp_all.to_csv(out / 'width_vs_crb.csv', index=False); meas_all.to_csv(out / 'measured_predictions.csv', index=False); mc.to_csv(out / 'model_comparison.csv', index=False)
    json.dump({'version': VERSION, 'population': POP, 'tags': list(tags), 'mode_sets': MODE_SETS, 'n_train': n_train, 'n_scale': n_scale, 'n_cal': n_cal, 'n_test': n_test, 'n_phys': n_phys, 'n_crb': n_crb, 'alphas': list(ALPHAS), 'seed': seed, 'shrink_alpha': S3B.SHRINK['alpha'], 'mismatch_pools': lib.summary()}, open(out / 'run_config.json', 'w'), indent=1)
    prim_ev = pd.concat([ev_all[(ev_all['mode'] == tag) & (ev_all['model'] == cmp_all[cmp_all['mode'] == tag]['model'].iloc[0])] for tag in tags], ignore_index=True)
    plot_reliability(prim_ev, out); plot_width_vs_crb(cmp_all, out); plot_models(mc, out)
    if len(meas_all): plot_measured(meas_all, out)
    zpath = stage_report(ev_all, cmp_all, meas_all, mc, lib, out, proj, time.time() - t0); return ev_all, cmp_all, meas_all, mc, zpath

def stage_report(ev_all, cmp_all, meas_all, mc, lib, out: Path, proj: Path, elapsed=0.0):
    prim = {tag: g['model'].iloc[0] for tag, g in cmp_all.groupby('mode', sort=False)}
    L = [f"# S4 report -- learned retrieval + conformal prediction regions ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')}, {elapsed / 60:.0f} min)", '',
         f"Synthetic population: olivine share {POP['ol']}, plagioclase {POP['plag']}, Fo {POP['Fo']}, Fs {POP['Fs']}, d {POP['d']} um (log-uniform), s = 0 w.p. {POP['p_s0']} else U{POP['s']}, level nuisances N(0, {POP['level_sd']}^2).",
         f"Library-mismatch augmentation: per component w.p. {POP['p_aug']}, ln(1 - w) perturbed by a random held-out mismatch field (amplitude U{POP['aug_amp']}, random sign); pools: {lib.summary()}.",
         'Noise: one draw per state from the Step-3b covariance (instrument + shrunk model-error covariance). Fo/Fs references for the measured mixtures are the model-consistent values fitted on the pure end-member spectra (Step 3a).',
         f"Primary model per mode (lower calibration MAE of the olivine fraction): {prim}. Tags: lab/tel50/dawnfc = augmented population; lab_noaug = ablation without library mismatch.", '',
         '## Point accuracy (MAE / RMSE)', '', '| mode | model | set | target | MAE | RMSE | n |', '|---|---|---|---|---|---|---|']
    for _, r in mc.iterrows(): L.append(f"| {r['mode']} | {r['model']} | {r['set']} | {r['target']} | {r['mae_test']:.3g} | {r['rmse_test']:.3g} | {int(r['n'])} |")
    L += ['', '## Conformal coverage and width (primary model; normalised split conformal; Aitchison ball for the composition)', '',
          '| mode | model | set | nominal | target | coverage | mean half-width | median half-width | MAE | n |', '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in ev_all.iterrows():
        if r['model'] == prim.get(r['mode']): L.append(f"| {r['mode']} | {r['model']} | {r['set']} | {1 - r['alpha']:.2f} | {r['target']} | {r['coverage']:.3f} | {r['mean_halfwidth']:.3g} | {r['median_halfwidth']:.3g} | {r['mae']:.3g} | {int(r['n'])} |")
    L += ['', '## Secondary model, nominal 0.90 (coverage / median half-width)', '', '| mode | model | set | target | coverage | median half-width |', '|---|---|---|---|---|---|']
    for _, r in ev_all[ev_all['alpha'] == 0.10].iterrows():
        if r['model'] != prim.get(r['mode']): L.append(f"| {r['mode']} | {r['model']} | {r['set']} | {r['target']} | {r['coverage']:.3f} | {r['median_halfwidth']:.3g} |")
    L += ['', '## Conformal 68 % half-width vs Cramér–Rao σ at the true state (synthetic test subset): mixing error only (CRB) and with library mismatch (CRB+lib)', '',
          '| mode | target | median CRB | median CRB+lib | median half-width | median width/CRB | median width/(CRB+lib) | Spearman(width, CRB+lib) | Spearman(abs err, CRB+lib) | frac abs err ≤ CRB+lib | frac abs err ≤ 2 (CRB+lib) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for mode, g in cmp_all.groupby('mode', sort=False):
        for t in ('M_ol', 'M_plag', 'Fo', 'Fs_opx', 'logd'):
            c, ca, w, e = g[f'crb_{t}'], g[f'crbaug_{t}'], g[f'hw68_{t}'], g[f'err_{t}']
            L.append(f"| {mode} | {t} | {c.median():.3g} | {ca.median():.3g} | {w.median():.3g} | {np.median(w / np.maximum(c, 1e-9)):.2f} | {np.median(w / np.maximum(ca, 1e-9)):.2f} | {w.corr(ca, method='spearman'):.2f} | {e.corr(ca, method='spearman'):.2f} | {(e <= ca).mean():.2f} | {(e <= 2 * ca).mean():.2f} |")
    if len(meas_all):
        L += ['', '## Measured mixtures, 90 % intervals of the primary model: per series (synthetic calibration / LOSO recalibration)', '', '| mode | series | n | cov M_ol syn | cov M_ol LOSO | median hw M_ol syn | median hw M_ol LOSO | MAE M_ol (learned) | MAE M_ol (physics) | cov Fo syn | cov Fo LOSO | MAE Fo |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for (mode, ser), g in meas_all.groupby(['mode', 'series'], sort=False):
            mf = g['Fo_ref'].notna(); gl = g['covered90_loso_M_ol'] if 'covered90_loso_M_ol' in g else pd.Series(np.nan, index=g.index); gf = g['covered90_loso_Fo'] if 'covered90_loso_Fo' in g else pd.Series(np.nan, index=g.index)
            L.append(f"| {mode} | {ser} | {len(g)} | {g['covered90_M_ol'].mean():.2f} | {_f(gl.astype(float).mean()):.2f} | {g['hw90_M_ol'].median():.3f} | {_f(g['hw90_loso_M_ol'].median()) if 'hw90_loso_M_ol' in g else float('nan'):.3f} | {np.mean(np.abs(g['M_ol'] - g['pred_M_ol'])):.3f} | {_f(np.nanmean(np.abs(g['M_ol'] - g['phys_M_ol']))) if 'phys_M_ol' in g else float('nan'):.3f} | {_f(g.loc[mf, 'covered90_Fo'].astype(float).mean()):.2f} | {_f(gf[mf].astype(float).mean()):.2f} | {_f(np.nanmean(np.abs(g.loc[mf, 'Fo_ref'] - g.loc[mf, 'pred_Fo']))):.1f} |")
    L += ['', '## Library mismatch of the natural end-members (ln(1 - w) RMS after level fit; excluded = used by the test mixtures)', '', '| code | family | d0 (um) | chemistry | level | RMS | excluded | in pool |', '|---|---|---|---|---|---|---|---|']
    for _, r in lib.table.iterrows(): L.append(f"| {r['code']} | {r['family']} | {r['d0_um']:.1f} | {_f(r['chem']):.1f} | {r['l_fit']:+.2f} | {r['rms_ln1mw']:.3f} | {r['excluded']} | {r['used']} |")
    L += ['', '## Gate checklist (S4)', '',
          '- [ ] synthetic coverage within +/- 0.03 of nominal for all reported targets (validity of the split-conformal construction)',
          '- [ ] measured mixtures: coverage with the synthetic calibration documented per mode (sim-to-real gap) and compared with the ablation without mismatch augmentation; LOSO recalibration restores coverage >= nominal - 0.1',
          '- [ ] conformal 68 % width vs CRB and CRB+lib documented (median ratio, rank correlation); learned estimator vs physics estimator documented',
          '- [ ] widths at the telescope and multispectral modes consistent with the Step-3b profile/twin findings']
    (out / 'S4_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    zpath = proj / 'S4_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in {'.csv', '.json', '.md', '.npz', '.png', '.pdf'}: zf.write(p, p.relative_to(proj))
    print(f"[R] report: {out / 'S4_REPORT.md'}\n[R] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath
