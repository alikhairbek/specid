#!/usr/bin/env python3
"""
s3b_identify.py -- Step 3b: the identifiability engine on the continuous forward model of Step 3a.

Fisher information with the full observation covariance (instrument noise + the empirical, correlated model-error covariance of
Step 2), Cramer-Rao bounds with nuisance priors, averaging kernels / degrees of freedom for signal, profile likelihoods and
spectral twins, for several observation modes; validation against synthetic retrievals and against the measured mixtures.

Observation modes
  lab      : absolute reflectance factor, 0.45-2.45 um at 5 nm, SNR 1000  (+ model-error covariance)
  tel100 / tel50 / tel20 : ground-based vis+NIR spectrum (0.45-2.45 um, 10 nm) normalised at 0.55 um, SNR 100 / 50 / 20
  onct     : Hayabusa2 ONC-T filters (b v Na w x p; ul excluded), normalised at v, SNR 100 per band
  dawnfc   : Dawn Framing Camera colour filters (F8 F2 F7 F3 F6 F4 F5), normalised at F2, SNR 100 per band
  mapcam   : OSIRIS-REx MapCam (b' v w x), normalised at v, SNR 100 per band
  *nf      : same without the model-error covariance (instrument noise only)
  Filter centres / widths are approximate (Gaussian responses) and are verified against the instrument papers in Step 5.

Model-error covariance: Sigma_model = floor * diag(y) C_rel diag(y), with C_rel the second-moment matrix of the relative residuals
(measured - Hapke)/measured of the tier-A silicate mixtures of Step 2 (n = 205; effective rank ~1.5: the model error is mostly a
correlated level error, which carries little compositional information).

State vector (scenario-dependent): ilr coordinates of the mass fractions, Fo (olivine), Fs_opx, ln d (um), s (weathering exposure,
pulse-laser doses), and the level nuisances l_ol, l_opx, l_cpx, l_plag (log k factors; Gaussian priors from the natural spread).
Cramer-Rao with priors: C = (J^T Sigma^-1 J + P)^-1 ; averaging kernel A = C J^T Sigma^-1 J ; DOFS = trace(A).
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

VERSION = 's3b-identify/1.1'
GRID = S2.GRID; WONG = S2.WONG; INK = S2.INK; CMAP = 'cividis'

FILTERS = {   # (centre um, FWHM um)  -- approximate, to be verified in Step 5
    'onct': [(0.48, 0.03), (0.55, 0.03), (0.59, 0.01), (0.70, 0.03), (0.86, 0.04), (0.95, 0.06)],
    'dawnfc': [(0.438, 0.040), (0.555, 0.043), (0.653, 0.042), (0.749, 0.044), (0.829, 0.033), (0.917, 0.045), (0.965, 0.085)],
    'mapcam': [(0.473, 0.09), (0.550, 0.09), (0.698, 0.08), (0.847, 0.10)]}
MODES = {
    'lab': dict(kind='hyper', wl=(0.45, 2.45), step=0.005, norm=None, snr=1000.0, floor=1.0, label='laboratory, absolute REFF, SNR 1000 + model error'),
    'labnf': dict(kind='hyper', wl=(0.45, 2.45), step=0.005, norm=None, snr=1000.0, floor=0.0, label='laboratory SNR 1000, noise only'),
    'tel100': dict(kind='hyper', wl=(0.45, 2.45), step=0.010, norm=0.55, snr=100.0, floor=1.0, label='telescope vis+NIR normalised, SNR 100'),
    'tel50': dict(kind='hyper', wl=(0.45, 2.45), step=0.010, norm=0.55, snr=50.0, floor=1.0, label='telescope vis+NIR normalised, SNR 50'),
    'tel20': dict(kind='hyper', wl=(0.45, 2.45), step=0.010, norm=0.55, snr=20.0, floor=1.0, label='telescope vis+NIR normalised, SNR 20'),
    'tel50nf': dict(kind='hyper', wl=(0.45, 2.45), step=0.010, norm=0.55, snr=50.0, floor=0.0, label='telescope SNR 50, noise only'),
    'onct': dict(kind='bands', filters='onct', norm=0.55, snr=100.0, floor=1.0, label='Hayabusa2 ONC-T (6 bands) normalised, SNR 100'),
    'dawnfc': dict(kind='bands', filters='dawnfc', norm=0.555, snr=100.0, floor=1.0, label='Dawn FC (7 bands) normalised, SNR 100'),
    'mapcam': dict(kind='bands', filters='mapcam', norm=0.55, snr=100.0, floor=1.0, label='OSIRIS-REx MapCam (4 bands) normalised, SNR 100')}
SHRINK = {'alpha': 0.1}            # shrinkage of the empirical model-error covariance towards its diagonal; set by calibrate_shrinkage()
PRIOR_SIGMA = {'l_ol': 0.5, 'l_opx': 0.5, 'l_cpx': 0.7, 'l_plag': 0.5, 's': 0.7, 'Fo': 60.0, 'Fs_opx': 60.0, 'logd': 1.5, 'ilr': 4.0}   # wide = effectively flat
PRIOR_VARIANTS = {'default': {}, 'known_nuisance': {'l_ol': 0.05, 'l_opx': 0.05, 'l_cpx': 0.05, 'l_plag': 0.05, 's': 0.05, 'logd': 0.05}, 'known_size_weathering': {'s': 0.05, 'logd': 0.05}}
STEPS = {'ilr': 0.02, 'Fo': 5.0, 'Fs_opx': 2.5, 'logd': 0.02, 's': 0.05, 'l_ol': 0.02, 'l_opx': 0.02, 'l_cpx': 0.02, 'l_plag': 0.02}   # Fo/Fs steps span the library nodes (sensitivity at the 5-Fo scale)
SCENARIOS = {
    'ol-opx': dict(fams=['ol', 'opx'], free=['ilr', 'Fo', 'Fs_opx', 'logd', 's', 'l_ol', 'l_opx']),
    'ol-opx-plag': dict(fams=['ol', 'opx', 'plag'], free=['ilr', 'Fo', 'Fs_opx', 'logd', 's', 'l_ol', 'l_opx', 'l_plag']),
    'ol-opx-cpx-plag': dict(fams=['ol', 'opx', 'cpx', 'plag'], free=['ilr', 'Fo', 'Fs_opx', 'logd', 's', 'l_ol', 'l_opx', 'l_cpx', 'l_plag'])}

# ----------------------------------------------------------------------------- observation operator + covariance
class Observer:
    def __init__(self, model: FP.ParamModel, mode: str, resid_npz=None, floor_scale=None):
        self.model = model; self.mode = mode; self.cfg = dict(MODES[mode]); self.grid = model.grid
        if floor_scale is not None: self.cfg['floor'] = floor_scale
        self.Crel = None
        if self.cfg['floor'] > 0 and resid_npz is not None and Path(resid_npz).exists():
            z = np.load(resid_npz, allow_pickle=True); R = np.asarray(z['rel_resid'], float); g = z['grid_um']
            if len(g) != len(self.grid) or np.max(np.abs(g - self.grid)) > 1e-6: R = np.vstack([np.interp(self.grid, g, r) for r in R])
            C = (R.T @ R) / len(R); a = SHRINK['alpha']; self.Crel = ((1 - a) * C + a * np.diag(np.diag(C))) * self.cfg['floor']     # shrinkage towards the diagonal (calibrated on the measured mixtures)
        if self.cfg['kind'] == 'hyper':
            wl = np.round(np.arange(self.cfg['wl'][0], self.cfg['wl'][1] + 1e-9, self.cfg['step']), 4); self.idx = np.array([int(np.argmin(abs(self.grid - w))) for w in wl]); self.wl = self.grid[self.idx]
            self.bands = None; self.n = len(self.idx); self.B = None
        else:
            self.bands = FILTERS[self.cfg['filters']]; self.wl = np.array([c for c, _ in self.bands]); self.n = len(self.bands); self.idx = None
            B = np.vstack([np.exp(-0.5 * ((self.grid - c) / (fw / 2.3548)) ** 2) for c, fw in self.bands]); self.B = B / B.sum(axis=1, keepdims=True)
            self.iref = int(np.argmin(abs(self.wl - self.cfg['norm']))) if self.cfg['norm'] is not None else None
    def y(self, theta):
        R = self.model.spectrum(theta, mode='reff')
        if self.bands is None:
            y = R[self.idx]
            return y / np.interp(self.cfg['norm'], self.grid, R) if self.cfg['norm'] is not None else y
        yb = self.B @ R
        return yb / yb[self.iref] if self.cfg['norm'] is not None else yb
    def cov(self, theta):
        """full covariance of y: instrument noise (y/SNR, independent) + model error (relative covariance C_rel propagated)"""
        R = self.model.spectrum(theta, mode='reff'); y = self.y(theta); S = np.diag((y / self.cfg['snr']) ** 2)
        if self.Crel is not None:
            if self.bands is None:
                S = S + self.Crel[np.ix_(self.idx, self.idx)] * np.outer(y, y)
            else:
                Sa = self.B @ (self.Crel * np.outer(R, R)) @ self.B.T; yb = self.B @ R
                S = S + (Sa / yb[self.iref] ** 2 if self.cfg['norm'] is not None else Sa)
        S = S + np.eye(self.n) * (1e-8 * np.mean(np.diag(S)) + 1e-14)
        return S
    def whitener(self, theta):
        """lower-triangular L with Sigma = L L^T; whitened residual = solve(L, r)"""
        return np.linalg.cholesky(self.cov(theta))

# ----------------------------------------------------------------------------- Fisher / CRB
def make_pv(scn, base_theta):
    sc = SCENARIOS[scn]; return FP.ParamVector(sc['fams'], sc['free'], base_theta)

def prior_precision(pv, variant='default'):
    names = pv.names(); P = np.zeros(len(names)); ov = PRIOR_VARIANTS.get(variant, {})
    for i, nm in enumerate(names):
        key = 'ilr' if nm.startswith('ilr') else nm; P[i] = 1.0 / ov.get(key, PRIOR_SIGMA.get(key, 1e3)) ** 2
    return np.diag(P)

def jacobian(obs: Observer, pv: FP.ParamVector, v):
    names = pv.names(); J = np.zeros((obs.n, len(v)))
    for i, nm in enumerate(names):
        h = STEPS['ilr' if nm.startswith('ilr') else nm]; vp = v.copy(); vm = v.copy(); vp[i] += h; vm[i] -= h
        J[:, i] = (obs.y(pv.unpack(vp)) - obs.y(pv.unpack(vm))) / (2 * h)
    return J

def dM_dz(pv, v):
    names = pv.names(); iz = [i for i, nm in enumerate(names) if nm.startswith('ilr')]; th = pv.unpack(v); M0 = np.array([th['M'][f] for f in pv.fams]); G = np.zeros((len(M0), len(v)))
    for i in iz:
        vp = v.copy(); vp[i] += 1e-3; thp = pv.unpack(vp); G[:, i] = (np.array([thp['M'][f] for f in pv.fams]) - M0) / 1e-3
    return G

def analyse(obs: Observer, pv: FP.ParamVector, theta, variant='default', J=None, L=None):
    """CRB with priors at one state: sigmas (natural units), averaging-kernel diagonals, DOFS, condition number, weakest direction"""
    v = pv.pack(theta); J = jacobian(obs, pv, v) if J is None else J; L = obs.whitener(theta) if L is None else L
    Jw = solve_triangular(L, J, lower=True); F = Jw.T @ Jw; P = prior_precision(pv, variant); C = np.linalg.pinv(F + P); A = C @ F
    names = pv.names(); out = {'dofs': float(np.trace(A))}
    G = dM_dz(pv, v); CM = G @ C @ G.T
    for k, f in enumerate(pv.fams): out[f'sigma_M_{f}'] = float(np.sqrt(max(CM[k, k], 0)))
    for i, nm in enumerate(names):
        if not nm.startswith('ilr'): out[f'sigma_{nm}'] = float(np.sqrt(max(C[i, i], 0))); out[f'A_{nm}'] = float(A[i, i])
    iz = [i for i, nm in enumerate(names) if nm.startswith('ilr')]; out['A_abund'] = float(np.mean([A[i, i] for i in iz]))
    sc = np.array([PRIOR_SIGMA['ilr' if nm.startswith('ilr') else nm] for nm in names]); Fs_ = (F * sc[:, None]) * sc[None, :]; ev, evec = np.linalg.eigh(Fs_)
    out['cond_log10'] = float(np.log10(max(ev.max(), 1e-30) / max(ev.min(), 1e-30))); w = evec[:, 0]
    out['weak_dir'] = ' '.join(f'{names[i]}:{w[i]:+.2f}' for i in np.argsort(-np.abs(w))[:3])
    return out

# ----------------------------------------------------------------------------- states and maps
def base_theta(scn, ol=0.6, Fo=80.0, Fs=20.0, d=60.0, s=0.0, plag=0.0, cpx=0.0):
    fams = SCENARIOS[scn]['fams']; M = {}; rest = 1.0 - plag - cpx; M['ol'] = ol * rest; M['opx'] = (1 - ol) * rest
    if 'plag' in fams: M['plag'] = plag
    if 'cpx' in fams: M['cpx'] = cpx
    return {'M': {k: float(v) for k, v in M.items()}, 'Fo': float(Fo), 'Fs_opx': float(Fs), 'd': float(d), 's': float(s), 'l_ol': 0.0, 'l_opx': 0.0, 'l_cpx': 0.0, 'l_plag': 0.0, 'cpx_natural': True}

def _map(model, scn, modes, out, resid_npz, states, tag):
    rows = []; t0 = time.time()
    for mode in modes:
        obs = Observer(model, mode, resid_npz)
        for st in states:
            th = base_theta(scn, **st); pv = make_pv(scn, th); J = jacobian(obs, pv, pv.pack(th)); L = obs.whitener(th)
            for variant in ('default', 'known_nuisance'):
                r = analyse(obs, pv, th, variant, J, L); rows.append({'scenario': scn, 'mode': mode, 'variant': variant, 'd_um': st.get('d', 60.0), 'Fo': st.get('Fo', 80.0), 'Fs': st.get('Fs', 20.0),
                                                                       'ol_frac': st.get('ol', 0.6), 'plag': st.get('plag', 0.0), 's': st.get('s', 0.0), **r})
        print(f'      map {tag} {scn} {mode}: {len(states)} states, {time.time() - t0:.0f}s')
    df = pd.DataFrame(rows); df.to_csv(out / f'map_{tag}_{scn}.csv', index=False); return df

def map_olFo(model, scn, modes, out, resid_npz, ol_vals, Fo_vals, d=60.0, Fs=20.0, plag=0.0, tag='olFo'):
    return _map(model, scn, modes, out, resid_npz, [dict(ol=ol, Fo=Fo, Fs=Fs, d=d, plag=plag) for Fo in Fo_vals for ol in ol_vals], tag)

def map_old(model, scn, modes, out, resid_npz, ol_vals, d_vals, Fo=80.0, Fs=20.0, plag=0.0, tag='old'):
    return _map(model, scn, modes, out, resid_npz, [dict(ol=ol, Fo=Fo, Fs=Fs, d=d, plag=plag) for d in d_vals for ol in ol_vals], tag)

def map_FoFs(model, scn, modes, out, resid_npz, Fo_vals, Fs_vals, ol=0.5, d=60.0, plag=0.0, tag='FoFs'):
    return _map(model, scn, modes, out, resid_npz, [dict(ol=ol, Fo=Fo, Fs=Fs, d=d, plag=plag) for Fs in Fs_vals for Fo in Fo_vals], tag)

def plot_maps(df, modes, out: Path, xcol, ycol, tag, scn, variant='default',
              metrics=(('sigma_M_ol', 'σ(olivine mass fraction)', 0.3), ('sigma_Fo', 'σ(Fo)', 25), ('sigma_Fs_opx', 'σ(Fs)', 25), ('sigma_logd', 'σ(ln d)', 1.2), ('dofs', 'DOFS', None))):
    plt = S2.style(); xs = np.sort(df[xcol].unique()); ys = np.sort(df[ycol].unique()); nm = len(modes); nc = len(metrics)
    fig, axes = plt.subplots(nm, nc, figsize=(3.4 * nc, 2.9 * nm), squeeze=False)
    for i, mode in enumerate(modes):
        g = df[(df['mode'] == mode) & (df['variant'] == variant)]
        for j, (met, lab, vmax) in enumerate(metrics):
            Z = np.full((len(ys), len(xs)), np.nan)
            for _, r in g.iterrows(): Z[np.searchsorted(ys, r[ycol]), np.searchsorted(xs, r[xcol])] = r[met]
            ax = axes[i][j]; im = ax.pcolormesh(xs, ys, Z, cmap=CMAP, vmin=0, vmax=vmax if vmax else None, shading='nearest')
            if ycol == 'd_um': ax.set_yscale('log')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if i == 0: ax.set_title(lab, fontsize=9)
            if j == 0: ax.set_ylabel(f"{MODES[mode]['label'].split(',')[0]}\n{ {'d_um': 'd (µm)', 'Fo': 'Fo', 'Fs': 'Fs'}.get(ycol, ycol)}", fontsize=8)
            if i == nm - 1: ax.set_xlabel({'ol_frac': 'olivine / (olivine + opx), mass', 'Fo': 'Fo'}.get(xcol, xcol), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'Cramér–Rao bounds — scenario {scn} ({tag}; priors: {variant})', fontsize=10); fig.tight_layout(); fig.savefig(out / 'figures' / f'S3B_map_{tag}_{scn}_{variant}.png'); plt.close(fig)

# ----------------------------------------------------------------------------- profile likelihood + twins
def _whitened_resid_fn(obs, pv, theta_true, variant='default'):
    """whitened data residual + the nuisance prior terms (same priors as the CRB), so that profiles and twins are posterior Delta chi2"""
    y0 = obs.y(theta_true); L = obs.whitener(theta_true); names = pv.names()
    pw = np.sqrt(np.diag(prior_precision(pv, variant))) * np.array([0.0 if (nm.startswith('ilr') or nm in ('Fo', 'Fs_opx', 'logd')) else 1.0 for nm in names])
    def resid(v): return np.concatenate([solve_triangular(L, obs.y(pv.unpack(v)) - y0, lower=True), pw * v])
    return resid, y0, L

def profile(obs: Observer, scn, theta_true, par, values, n_starts=2, seed=0):
    """profile chi2 along one parameter: fix it, optimise the rest (bounded, whitened least squares); Delta chi2 vs value"""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed); pv = make_pv(scn, theta_true); resid_all, y0, L = _whitened_resid_fn(obs, pv, theta_true)
    names = pv.names(); ip = names.index(par); free = [i for i in range(len(names)) if i != ip]; lo, hi = pv.bounds(); v_true = pv.pack(theta_true); rows = []; v_prev = v_true.copy()
    for val in values:
        best = None
        for k in range(n_starts):
            v0 = v_prev.copy() if k == 0 else np.clip(v_true + rng.normal(0, 1, len(v_true)) * np.array([STEPS['ilr' if n.startswith('ilr') else n] * 10 for n in names]), lo + 1e-6, hi - 1e-6)
            def resid(u):
                v = v0.copy(); v[free] = u; v[ip] = val; return resid_all(v)
            r = least_squares(resid, np.clip(v0[free], lo[free] + 1e-6, hi[free] - 1e-6), bounds=(lo[free], hi[free]), method='trf', x_scale='jac', max_nfev=300)
            if best is None or r.cost < best.cost: best = r
        v_prev = v_true.copy(); v_prev[free] = best.x; v_prev[ip] = val
        rows.append({'par': par, 'value': val, 'chi2': 2 * best.cost})
    df = pd.DataFrame(rows); df['dchi2'] = df['chi2'] - df['chi2'].min(); return df

def twins(obs: Observer, scn, theta_true, n_random=2000, n_refine=15, seed=0, conf=0.95):
    """spectral twins: states statistically indistinguishable from the true one (Delta chi2 <= chi2_conf(k free parameters)) but far away"""
    from scipy.optimize import least_squares; from scipy.stats import chi2 as chi2dist
    rng = np.random.default_rng(seed); pv = make_pv(scn, theta_true); resid, y0, L = _whitened_resid_fn(obs, pv, theta_true); lo, hi = pv.bounds(); names = pv.names()
    crit = float(chi2dist.ppf(conf, len(names)))
    V = np.zeros((n_random, len(names)))
    for i, nm in enumerate(names):
        key = 'ilr' if nm.startswith('ilr') else nm
        if key in ('l_ol', 'l_opx', 'l_cpx', 'l_plag'): V[:, i] = np.clip(rng.normal(0, PRIOR_SIGMA[key], n_random), lo[i], hi[i])
        elif key == 's': V[:, i] = rng.uniform(0, 2, n_random)
        elif key == 'ilr': V[:, i] = rng.uniform(-3, 3, n_random)
        else: V[:, i] = rng.uniform(lo[i], hi[i], n_random)
    c2 = np.array([float(np.sum(resid(V[k]) ** 2)) for k in range(n_random)]); order = np.argsort(c2)
    tol = theta_true['M']['ol'] / (theta_true['M']['ol'] + theta_true['M']['opx'])
    def dist(v):
        th = pv.unpack(v); M = th['M']; return {'d_ol': abs(M['ol'] / (M['ol'] + M['opx']) - tol), 'd_Fo': abs(th['Fo'] - theta_true['Fo']), 'd_Fs': abs(th['Fs_opx'] - theta_true['Fs_opx']),
                                               'd_logd': abs(math.log(th['d']) - math.log(theta_true['d'])), 'd_s': abs(th['s'] - theta_true['s'])}
    sols = []
    for k in order[:n_refine]:
        r = least_squares(resid, V[k], bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=300); v = r.x; c = 2 * r.cost
        if c <= crit: sols.append({'chi2': c, **dist(v), **{nm: float(v[i]) for i, nm in enumerate(names)}})
    sols = pd.DataFrame(sols)
    return (sols.sort_values('d_ol', ascending=False) if len(sols) else sols), crit

# ----------------------------------------------------------------------------- validation
def _coarse_start(obs, pv, scn, resid, plag):
    """coarse grid over (olivine fraction, Fo, Fs) at d = 60, s = 0.5, levels 0 -> best start for the refinement"""
    best = None
    for ol in (0.15, 0.35, 0.55, 0.75, 0.9):
        for Fo in (60, 75, 88):
            for Fs in (10, 25, 40):
                th = base_theta(scn, ol=ol, Fo=Fo, Fs=Fs, d=60, s=0.5, plag=plag); v = pv.pack(th); c = float(np.sum(resid(v) ** 2))
                if best is None or c < best[0]: best = (c, v)
    return best[1]

def validate_synthetic(model, scn, mode, out: Path, resid_npz, n=120, seed=1, n_starts=2):
    """simulate noisy observations (full covariance) at random states, retrieve by whitened least squares (coarse grid start + refinement), compare with the CRB"""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed); obs = Observer(model, mode, resid_npz); rows = []; t0 = time.time()
    for k in range(n):
        plag = float(rng.uniform(0.05, 0.5)) if 'plag' in SCENARIOS[scn]['fams'] else 0.0
        th = base_theta(scn, ol=rng.uniform(0.1, 0.9), Fo=rng.uniform(55, 92), Fs=rng.uniform(8, 45), d=float(np.exp(rng.uniform(np.log(15), np.log(200)))), s=rng.uniform(0, 1.5), plag=plag)
        for key in ('l_ol', 'l_opx', 'l_plag'): th[key] = float(rng.normal(0, 0.3))
        pv = make_pv(scn, th); lo, hi = pv.bounds(); y0 = obs.y(th); L = obs.whitener(th); y = y0 + L @ rng.normal(0, 1, obs.n); cr = analyse(obs, pv, th, 'default', None, L)
        P = prior_precision(pv); names = pv.names(); pw = np.sqrt(np.diag(P)) * np.array([0.0 if (nm.startswith('ilr') or nm in ('Fo', 'Fs_opx', 'logd')) else 1.0 for nm in names])
        def resid(v): return np.concatenate([solve_triangular(L, obs.y(pv.unpack(v)) - y, lower=True), pw * v])
        v0 = _coarse_start(obs, pv, scn, resid, plag if plag else 0.0); best = None
        for j in range(n_starts):
            vs = v0 if j == 0 else np.clip(v0 + rng.normal(0, 1, len(v0)) * np.array([0.3 if nm.startswith('ilr') else {'Fo': 8, 'Fs_opx': 8, 'logd': 0.4, 's': 0.3}.get(nm, 0.2) for nm in names]), lo + 1e-6, hi - 1e-6)
            r = least_squares(resid, vs, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=400)
            if best is None or r.cost < best.cost: best = r
        tf = pv.unpack(best.x)
        rows.append({'scenario': scn, 'mode': mode, 'true_ol': th['M']['ol'] / (th['M']['ol'] + th['M']['opx']), 'true_Fo': th['Fo'], 'true_Fs': th['Fs_opx'], 'true_d': th['d'], 'true_s': th['s'], 'true_plag': plag,
                     'err_M_ol': tf['M']['ol'] - th['M']['ol'], 'err_Fo': tf['Fo'] - th['Fo'], 'err_Fs': tf['Fs_opx'] - th['Fs_opx'], 'err_logd': math.log(tf['d'] / th['d']), 'err_s': tf['s'] - th['s'],
                     'crb_M_ol': cr['sigma_M_ol'], 'crb_Fo': cr['sigma_Fo'], 'crb_Fs': cr['sigma_Fs_opx'], 'crb_logd': cr['sigma_logd'], 'crb_s': cr['sigma_s'], 'chi2_fit': 2 * best.cost, 'n_obs': obs.n})
        if (k + 1) % 40 == 0: print(f'      synthetic validation {scn} {mode}: {k + 1}/{n}, {time.time() - t0:.0f}s')
    df = pd.DataFrame(rows); df.to_csv(out / f'validation_synthetic_{scn}_{mode}.csv', index=False); summ = {}
    for p in ('M_ol', 'Fo', 'Fs', 'logd', 's'):
        e = df[f'err_{p}'].values; c = df[f'crb_{p}'].values
        summ[p] = {'rmse': float(np.sqrt(np.mean(e ** 2))), 'median_abs_err': float(np.median(np.abs(e))), 'median_crb': float(np.median(c)), 'cover_1sigma': float(np.mean(np.abs(e) <= c)), 'cover_2sigma': float(np.mean(np.abs(e) <= 2 * c)),
                   'spearman_abs_err_vs_crb': float(pd.Series(np.abs(e)).corr(pd.Series(c), method='spearman'))}
    return df, summ

def validate_measured(model, out: Path, resid_npz, s2_results, s3a_results, mixtures, nat, rule):
    """CRB at the documented state of the measured tier-A silicate mixtures (lab mode) vs the actual errors of Step 2 (NNLS with measured
    end-members -> abundance-only CRB) and of Step 3a mode (a) (chemistry and levels free -> full CRB)"""
    famap = {'olivine': 'ol', 'opx': 'opx', 'cpx': 'cpx', 'plagioclase': 'plag'}; mixi = mixtures.set_index('mix_id')
    natq = nat[nat['model'] == 'quad'].set_index('code') if 'model' in nat else nat.set_index('code'); obs = Observer(model, 'lab', resid_npz); rows = []
    s3 = s3a_results.set_index('mix_id') if s3a_results is not None else None
    for _, r in s2_results.iterrows():
        if r['status'] != 'ok' or r['tier'] != 'A' or r['mix_id'] not in mixi.index: continue
        comps = json.loads(mixi.loc[r['mix_id'], 'components']); fams = [famap.get(c['fam']) for c in comps]
        if None in fams or len(set(fams)) < len(fams) or 'ol' not in fams or 'opx' not in fams: continue
        scn = {frozenset(['ol', 'opx']): 'ol-opx', frozenset(['ol', 'opx', 'plag']): 'ol-opx-plag', frozenset(['ol', 'opx', 'cpx', 'plag']): 'ol-opx-cpx-plag'}.get(frozenset(fams))
        if scn is None: continue
        M = {f: c['frac'] for f, c in zip(fams, comps)}; d = float(np.mean([S2._f(c['size_hi']) * rule['unbounded'] if S2._f(c['size_lo']) <= 0 else 0.5 * (S2._f(c['size_lo']) + S2._f(c['size_hi'])) * rule['bounded'] for c in comps]))
        Fo, Fs = 85.0, 20.0
        for f, c in zip(fams, comps):
            if c['code'] in natq.index:
                q = natq.loc[c['code']]; q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
                if f == 'ol' and np.isfinite(S2._f(q.get('Fo_fit', np.nan), np.nan)): Fo = float(q['Fo_fit'])
                if f == 'opx' and np.isfinite(S2._f(q.get('Fs_fit', np.nan), np.nan)): Fs = float(q['Fs_fit'])
        th = {'M': M, 'Fo': float(min(Fo, 95.0)), 'Fs_opx': float(max(Fs, 5.0)), 'd': d, 's': 0.0, 'l_ol': 0.0, 'l_opx': 0.0, 'l_cpx': 0.0, 'l_plag': 0.0, 'cpx_natural': True}
        L = obs.whitener(th); pv_ab = FP.ParamVector(SCENARIOS[scn]['fams'], ['ilr'], th); cr_ab = analyse(obs, pv_ab, th, 'default', None, L); pv_full = make_pv(scn, th); cr_full = analyse(obs, pv_full, th, 'default', None, L)
        row = {'mix_id': r['mix_id'], 'series': r['series'], 'scenario': scn, 'd_um': d, 'Fo': th['Fo'], 'Fs': th['Fs_opx'], 'ol_frac': M['ol'] / (M['ol'] + M['opx']), 's2_mae': r['mae_frac_hapke'],
               'crb_ab_M_ol': cr_ab['sigma_M_ol'], 'crb_full_M_ol': cr_full['sigma_M_ol'], 'crb_full_Fo': cr_full['sigma_Fo'], 'crb_full_Fs': cr_full['sigma_Fs_opx'], 'dofs_full': cr_full['dofs'], 's2_err_ol': np.nan}
        for i in range(1, 6):
            if r.get(f'fam{i}') == 'olivine': row['s2_err_ol'] = float(abs(r[f'hap{i}'] - r[f'true{i}']))
        if s3 is not None and r['mix_id'] in s3.index:
            q = s3.loc[r['mix_id']]
            row['s3a_mae_chemfree'] = q.get('mae_frac_fixed_d', np.nan); row['s3a_err_ol_chemfree'] = abs(q.get('fit_ol', np.nan) - q.get('true_ol', np.nan)) if 'fit_ol' in q else np.nan
            row['s3a_err_Fo'] = abs(q.get('fit_Fo', np.nan) - q.get('ref_Fo', np.nan)) if 'fit_Fo' in q else np.nan; row['s3a_err_Fs'] = abs(q.get('fit_Fs_opx', np.nan) - q.get('ref_Fs_opx', np.nan)) if 'fit_Fs_opx' in q else np.nan
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(out / 'validation_measured.csv', index=False); summ = {}
    if len(df):
        for a, b in (('s2_err_ol', 'crb_ab_M_ol'), ('s3a_err_ol_chemfree', 'crb_full_M_ol'), ('s3a_err_Fo', 'crb_full_Fo'), ('s3a_err_Fs', 'crb_full_Fs')):
            if a in df and df[a].notna().sum() > 5:
                m = df[a].notna() & df[b].notna(); summ[a] = {'n': int(m.sum()), 'median_abs_err': float(df.loc[m, a].median()), 'median_crb': float(df.loc[m, b].median()), 'spearman': float(df.loc[m, a].corr(df.loc[m, b], method='spearman')),
                                                               'frac_within_2crb': float((df.loc[m, a] <= 2 * df.loc[m, b]).mean())}
    return df, summ

# ----------------------------------------------------------------------------- figures
def plot_profiles(profs, out: Path, tag):
    plt = S2.style(); pars = ['olivine fraction', 'Fo', 'd (µm)']; states = list(dict.fromkeys(s for (s, _, _) in profs)); modes = list(dict.fromkeys(m for (_, m, _) in profs))
    fig, axes = plt.subplots(len(states), len(pars), figsize=(4.2 * len(pars), 2.8 * len(states)), squeeze=False)
    for i, st in enumerate(states):
        for j, p in enumerate(pars):
            ax = axes[i][j]; true_val = None
            for k, m in enumerate(modes):
                if (st, m, p) in profs:
                    df, true_val = profs[(st, m, p)]; ax.plot(df['x'], df['dchi2'], color=WONG[k % 7], lw=1.4, marker='.', ms=3, label=MODES[m]['label'].split(',')[0])
            ax.axhline(3.84, color='#999999', lw=0.8, ls='--')
            if true_val is not None: ax.axvline(true_val, color=INK, lw=0.7, ls=':')
            if p == 'd (µm)': ax.set_xscale('log')
            ax.set_ylim(0, 30); ax.set_xlabel(p, fontsize=8); ax.set_ylabel('Δχ²', fontsize=8); ax.set_title(st, fontsize=7.5)
            if i == 0 and j == 0: ax.legend(fontsize=6, frameon=False)
    fig.suptitle('Profile likelihoods (dashed: Δχ² = 3.84, 95 % for one parameter; dotted: true value)', fontsize=9); fig.tight_layout(); fig.savefig(out / 'figures' / f'S3B_profiles_{tag}.png'); plt.close(fig)

def plot_validation(df_syn, df_meas, out: Path):
    plt = S2.style(); fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)); mk = 'os^Dv<>'
    if df_syn is not None and len(df_syn):
        for k, ((scn, mode), g) in enumerate(df_syn.groupby(['scenario', 'mode'])):
            axes[0].scatter(g['crb_M_ol'], np.abs(g['err_M_ol']), s=10, color=WONG[k % 7], marker=mk[k % 7], linewidths=0, alpha=0.7, label=f'{scn} / {mode}')
            axes[1].scatter(g['crb_Fo'], np.abs(g['err_Fo']), s=10, color=WONG[k % 7], marker=mk[k % 7], linewidths=0, alpha=0.7)
        lim = max(0.05, float(np.nanpercentile(np.abs(df_syn['err_M_ol']), 95))); axes[0].plot([0, lim], [0, lim], color='#999999', lw=0.8); axes[0].plot([0, lim], [0, 2 * lim], color='#bbbbbb', lw=0.6, ls='--')
        axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim); axes[0].set_xlabel('CRB σ (olivine mass fraction)'); axes[0].set_ylabel('|retrieval error| (synthetic)'); axes[0].set_title('synthetic: abundance error vs predicted σ', fontsize=9); axes[0].legend(fontsize=6, frameon=False)
        lim = max(5, float(np.nanpercentile(np.abs(df_syn['err_Fo']), 95))); axes[1].plot([0, lim], [0, lim], color='#999999', lw=0.8); axes[1].plot([0, lim], [0, 2 * lim], color='#bbbbbb', lw=0.6, ls='--')
        axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim); axes[1].set_xlabel('CRB σ(Fo)'); axes[1].set_ylabel('|error Fo| (synthetic)'); axes[1].set_title('synthetic: Fo', fontsize=9)
    if df_meas is not None and len(df_meas):
        for k, (s, g) in enumerate(df_meas.groupby('series')):
            axes[2].scatter(g['crb_ab_M_ol'], g['s2_err_ol'], s=14, color=WONG[k % 7], marker=mk[k % 7], linewidths=0, alpha=0.8, label=s)
        lim = max(0.05, float(np.nanpercentile(df_meas['s2_err_ol'].dropna(), 95))); axes[2].plot([0, lim], [0, lim], color='#999999', lw=0.8); axes[2].plot([0, lim], [0, 2 * lim], color='#bbbbbb', lw=0.6, ls='--')
        axes[2].set_xlim(0, lim); axes[2].set_ylim(0, lim); axes[2].set_xlabel('CRB σ (olivine fraction; abundance-only, lab)'); axes[2].set_ylabel('|Step-2 NNLS error|, measured mixtures'); axes[2].set_title('measured mixtures: error vs predicted σ', fontsize=9); axes[2].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S3B_validation.png'); plt.close(fig)

# ----------------------------------------------------------------------------- report
def stage_report(maps, profs_summary, twin_tables, syn_summ, meas_summ, out: Path, proj: Path, extra_lines=()):
    L = [f"# S3b report -- identifiability engine ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')})", '', '## Observation modes', '', '| mode | description | n channels/bands |', '|---|---|---|']
    L += [f"| {m} | {MODES[m]['label']} | {len(np.arange(MODES[m]['wl'][0], MODES[m]['wl'][1] + 1e-9, MODES[m]['step'])) if MODES[m]['kind'] == 'hyper' else len(FILTERS[MODES[m]['filters']])} |" for m in MODES]
    L += ['', f"Nuisance priors (1σ): {PRIOR_SIGMA}; 'known_nuisance' variant: {PRIOR_VARIANTS['known_nuisance']}", '',
          '## Cramér–Rao maps: median / 90th percentile over the mapped states', '', '| scenario | map | mode (priors) | σ(ol fraction) | σ(Fo) | σ(Fs) | σ(ln d) | DOFS | A_abund | A_Fo | A_Fs | A_ln d |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for (scn, tag), df in maps.items():
        for (mode, variant), g in df.groupby(['mode', 'variant'], sort=False):
            q = lambda c: f"{g[c].median():.3g} / {g[c].quantile(0.9):.3g}"
            L.append(f"| {scn} | {tag} | {mode} ({variant}) | {q('sigma_M_ol')} | {q('sigma_Fo')} | {q('sigma_Fs_opx')} | {q('sigma_logd')} | {q('dofs')} | {g['A_abund'].median():.2f} | {g['A_Fo'].median():.2f} | {g['A_Fs_opx'].median():.2f} | {g['A_logd'].median():.2f} |")
    L += ['', '## Profile likelihoods (95 % intervals, Δχ² ≤ 3.84) at representative states', '', '| state | mode | parameter | true | 95 % interval | width |', '|---|---|---|---|---|---|']
    for r in profs_summary: L.append(f"| {r['state']} | {r['mode']} | {r['par']} | {r['true']:.3g} | [{r['lo']:.3g}, {r['hi']:.3g}] | {r['hi'] - r['lo']:.3g} |")
    L += ['', '## Spectral twins (Δχ² ≤ χ²₉₅ for the number of free parameters; farthest indistinguishable state found)', '', '| state | mode | n twins | max Δ(ol fraction) | max ΔFo | max ΔFs | max Δln d |', '|---|---|---|---|---|---|---|']
    for r in twin_tables: L.append(f"| {r['state']} | {r['mode']} | {r['n']} | {r['d_ol']:.3f} | {r['d_Fo']:.1f} | {r['d_Fs']:.1f} | {r['d_logd']:.2f} |")
    L += ['', '## Validation of the bounds', '', '### synthetic retrievals (whitened least squares, coarse-grid start + refinement)', '', '| scenario | mode | parameter | RMSE | median abs error | median CRB | within 1σ | within 2σ | Spearman(|err|, CRB) |', '|---|---|---|---|---|---|---|---|---|']
    for (scn, mode), summ in syn_summ.items():
        for p, s in summ.items(): L.append(f"| {scn} | {mode} | {p} | {s['rmse']:.3g} | {s['median_abs_err']:.3g} | {s['median_crb']:.3g} | {s['cover_1sigma']:.2f} | {s['cover_2sigma']:.2f} | {s['spearman_abs_err_vs_crb']:.2f} |")
    L += ['', '### measured mixtures (Step 2 / Step 3a errors vs CRB at the documented state, lab mode)', '', '| comparison | n | median abs error | median CRB | Spearman | fraction ≤ 2 CRB |', '|---|---|---|---|---|---|']
    for a, s in meas_summ.items(): L.append(f"| {a} | {s['n']} | {s['median_abs_err']:.3g} | {s['median_crb']:.3g} | {s['spearman']:.2f} | {s['frac_within_2crb']:.2f} |")
    L += [''] + list(extra_lines)
    (out / 'S3B_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    zpath = proj / 'S3b_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in {'.csv', '.json', '.md', '.npz', '.png', '.pdf'}: zf.write(p, p.relative_to(proj))
    print(f"[R] report: {out / 'S3B_REPORT.md'}\n[R] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath

# ----------------------------------------------------------------------------- shrinkage calibration on the measured mixtures
def calibrate_shrinkage(model, resid_npz, s2_results, mixtures, nat, rule, alphas=(0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """choose the shrinkage alpha so that the abundance-only lab CRB reproduces the median NNLS error of Step 2 on the measured mixtures"""
    rows = []
    for a in alphas:
        SHRINK['alpha'] = a; df, summ = validate_measured(model, Path('/tmp'), resid_npz, s2_results, None, mixtures, nat, rule)
        s = summ.get('s2_err_ol', {}); rows.append({'alpha': a, 'median_crb': s.get('median_crb', np.nan), 'median_err': s.get('median_abs_err', np.nan), 'spearman': s.get('spearman', np.nan), 'frac_within_2crb': s.get('frac_within_2crb', np.nan)})
        print(f"      alpha {a:g}: median CRB {s.get('median_crb', np.nan):.4f} vs median |error| {s.get('median_abs_err', np.nan):.4f}, Spearman {s.get('spearman', np.nan):.2f}")
    tab = pd.DataFrame(rows); ok = tab[tab['median_crb'].notna()]
    best = ok.iloc[int(np.argmin(np.abs(np.log(ok['median_crb'] / ok['median_err']))))]['alpha'] if len(ok) else 0.1
    SHRINK['alpha'] = float(best); return float(best), tab

# ----------------------------------------------------------------------------- driver
REP_STATES = {'OC-like (ol 0.6, Fo 80, Fs 20, 60 µm)': dict(scn='ol-opx', ol=0.6, Fo=80, Fs=20, d=60), 'px-rich (ol 0.3, Fo 75, Fs 30, 30 µm)': dict(scn='ol-opx', ol=0.3, Fo=75, Fs=30, d=30),
              'ol-rich (ol 0.85, Fo 90, Fs 15, 120 µm)': dict(scn='ol-opx', ol=0.85, Fo=90, Fs=15, d=120), 'ternary (ol 0.5, plag 0.3, Fo 80, Fs 20, 60 µm)': dict(scn='ol-opx-plag', ol=0.5, Fo=80, Fs=20, d=60, plag=0.3)}

def run_all(model, out: Path, proj: Path, resid_npz, s2_results, s3a_results, mixtures, nat, rule, map_modes=('lab', 'tel50', 'tel20', 'onct', 'dawnfc', 'mapcam'), prof_modes=('lab', 'tel50', 'dawnfc'),
            n_syn=120, n_twin=2000, quick=False):
    t0 = time.time(); (out / 'figures').mkdir(exist_ok=True, parents=True)
    if quick: map_modes = ('lab', 'tel50', 'dawnfc'); prof_modes = ('lab', 'tel50'); n_syn = 12; n_twin = 300
    alpha, tab = calibrate_shrinkage(model, resid_npz, s2_results, mixtures, nat, rule); tab.to_csv(out / 'shrinkage_calibration.csv', index=False); print(f'[C] model-error covariance shrinkage alpha = {alpha:g}')
    ol_vals = np.round(np.linspace(0.05, 0.95, 19 if not quick else 7), 3); Fo_vals = np.arange(50, 96, 5 if not quick else 15); d_vals = np.round(np.exp(np.linspace(np.log(10), np.log(300), 12 if not quick else 4)), 1)
    maps = {('ol-opx', 'olFo'): map_olFo(model, 'ol-opx', map_modes, out, resid_npz, ol_vals, Fo_vals), ('ol-opx', 'old'): map_old(model, 'ol-opx', map_modes, out, resid_npz, ol_vals, d_vals),
            ('ol-opx', 'FoFs'): map_FoFs(model, 'ol-opx', map_modes, out, resid_npz, Fo_vals, np.arange(5, 51, 5 if not quick else 15)), ('ol-opx-plag', 'olFo'): map_olFo(model, 'ol-opx-plag', map_modes, out, resid_npz, ol_vals, Fo_vals, plag=0.3)}
    for (scn, tag), df in maps.items():
        for variant in ('default', 'known_nuisance'):
            plot_maps(df, list(map_modes), out, 'Fo' if tag == 'FoFs' else 'ol_frac', {'old': 'd_um', 'FoFs': 'Fs'}.get(tag, 'Fo'), tag, scn, variant)
    print(f'[M] maps done, {time.time() - t0:.0f}s')
    profs, profs_summary, twin_tables = {}, [], []
    for name, st in REP_STATES.items():
        scn = st['scn']; th = base_theta(scn, ol=st['ol'], Fo=st['Fo'], Fs=st['Fs'], d=st['d'], plag=st.get('plag', 0.0)); pv = make_pv(scn, th)
        for mode in prof_modes:
            obs = Observer(model, mode, resid_npz)
            for par in ('ilr1', 'Fo', 'logd'):
                if par == 'ilr1':
                    xs = np.unique(np.append(np.linspace(0.05, 0.95, 19 if not quick else 5), st['ol'])); rest = 1 - st.get('plag', 0.0)
                    vals = np.array([FP.ilr([f * rest, (1 - f) * rest] + ([st['plag']] if 'plag' in st else []))[0] for f in xs]); true_val = st['ol']; pname = 'olivine fraction'
                elif par == 'Fo': xs = np.unique(np.append(np.arange(40, 96, 5 if not quick else 15), st['Fo'])); vals = xs; true_val = st['Fo']; pname = 'Fo'
                else: xs = np.unique(np.append(np.round(np.exp(np.linspace(np.log(10), np.log(300), 12 if not quick else 4)), 2), st['d'])); vals = np.log(xs); true_val = st['d']; pname = 'd (µm)'
                df = profile(obs, scn, th, par, vals, n_starts=2 if not quick else 1); df['x'] = xs; df['dchi2'] = df['chi2'] - df['chi2'].min()
                profs[(name, mode, pname)] = (df, true_val); inside = df[df['dchi2'] <= 3.84]
                profs_summary.append({'state': name, 'mode': mode, 'par': pname, 'true': true_val, 'lo': float(inside['x'].min()), 'hi': float(inside['x'].max())})
                df.assign(state=name, mode=mode).to_csv(out / f"profile_{name.split(' ')[0]}_{mode}_{par}.csv", index=False)
            tw, crit = twins(obs, scn, th, n_random=n_twin, n_refine=15 if not quick else 5)
            twin_tables.append({'state': name, 'mode': mode, 'n': int(len(tw)), 'chi2_crit': crit, 'd_ol': float(tw['d_ol'].max()) if len(tw) else 0.0, 'd_Fo': float(tw['d_Fo'].max()) if len(tw) else 0.0,
                                'd_Fs': float(tw['d_Fs'].max()) if len(tw) else 0.0, 'd_logd': float(tw['d_logd'].max()) if len(tw) else 0.0})
            if len(tw): tw.assign(state=name, mode=mode).to_csv(out / f"twins_{name.split(' ')[0]}_{mode}.csv", index=False)
            print(f'      profiles+twins {name} / {mode}: {time.time() - t0:.0f}s')
    plot_profiles(profs, out, 'all'); pd.DataFrame(profs_summary).to_csv(out / 'profile_intervals.csv', index=False); pd.DataFrame(twin_tables).to_csv(out / 'twins_summary.csv', index=False)
    print(f'[P] profiles and twins done, {time.time() - t0:.0f}s')
    syn_summ, syn_dfs = {}, []
    for scn in ('ol-opx', 'ol-opx-plag'):
        for mode in (('lab', 'tel50', 'dawnfc') if not quick else ('lab',)):
            df, summ = validate_synthetic(model, scn, mode, out, resid_npz, n=n_syn, seed=1 + len(syn_dfs)); syn_summ[(scn, mode)] = summ; syn_dfs.append(df)
    syn = pd.concat(syn_dfs, ignore_index=True); meas, meas_summ = validate_measured(model, out, resid_npz, s2_results, s3a_results, mixtures, nat, rule)
    plot_validation(syn, meas, out); print(f'[V] validation done, {time.time() - t0:.0f}s')
    json.dump({'synthetic': {f'{k[0]}|{k[1]}': v for k, v in syn_summ.items()}, 'measured': meas_summ}, open(out / 'validation_summary.json', 'w'), indent=1, default=float)
    zpath = stage_report(maps, profs_summary, twin_tables, syn_summ, meas_summ, out, proj, extra_lines=[f"Model-error covariance: shrunk empirical covariance of the Step-2 relative residuals (alpha = {alpha:g}, calibrated so that the abundance-only lab bound reproduces the median Step-2 NNLS error)."])
    return maps, profs, twin_tables, syn, meas, zpath
