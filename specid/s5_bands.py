#!/usr/bin/env python3
"""
s5_bands.py -- Step 5: optimal band selection, mission filter benchmark and the near-Earth-asteroid application.

A. Verified mission filter sets (instrument papers; centre / FWHM in um):
   Hayabusa2 ONC-T  (Kameda et al. 2017, Space Sci. Rev. 208, 17; Tatsumi et al. 2019, Icarus 325, 153 -- Table 3.2 effective wavelengths)
   Dawn FC          (Sierks et al. 2011, Space Sci. Rev. 163, 263; Schroeder et al. 2013, Icarus 226, 1304 -- Table 3)
   OSIRIS-REx MapCam (Rizk et al. 2018, Space Sci. Rev. 214, 26; Golish et al. 2022, Space Sci. Rev. 218, 5 -- cut-on/off wavelengths)
B. Band design: Gaussian candidate bands (centres 0.45-2.45 um every 25 nm; FWHM 30/60/100 nm); the Fisher information of a band set is
   computed with the full observation covariance (instrument noise at the given SNR + the Step-3b mixing-error covariance + the Step-4
   library-mismatch covariance, both propagated through the filter responses) and a free overall scale (normalised photometry);
   criterion = D-optimality for the targets (ilr1, ilr2, Fo, Fs): minimise the mean (over a design population) log det of the
   posterior covariance block; greedy forward selection + exchange refinement; minimum centre separation.
   Baselines under equal budget (same k and SNR): uniform spacing, random sets, Jacobian-sensitivity peaks, PCA-loading peaks,
   and the mission sets; "best extra filter" for each mission set.
C. Validation: CRB+lib on an evaluation population for every set; the Step-4 learned retrieval + conformal regions trained on the
   band-integrated nuisance-aware synthetic population and tested on the measured mixtures (sim-to-real) for the optimal, uniform,
   random and mission sets.
D. NEA application: ground-based reflectance spectra of (433) Eros and (25143) Itokawa (downloaded at run time when possible, or
   supplied as text files), retrieved with the telescope-mode learned model + conformal regions (synthetic calibration and
   measured-mixture calibration) and the physics estimator; compared with the published remote and sample mineralogy.
"""
from __future__ import annotations
import io, json, math, os, re, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_forward as S2
import forward_param as FP
import s3b_identify as S3B
import s4_conformal as S4

VERSION = 's5-bands/1.2'
GRID = S2.GRID; WONG = S2.WONG; INK = S2.INK; EVAL = S2.EVAL
SCN = S4.SCN; FAMS = S4.FAMS; TARGETS = S4.TARGETS; REPORT_TARGETS = S4.REPORT_TARGETS

# ----------------------------------------------------------------------------- A. verified mission filters (centre um, FWHM um)
FILTERS_V = {
    'onct': {'bands': [(0.3975, 0.036), (0.4798, 0.0266), (0.5489, 0.0306), (0.5899, 0.0118), (0.7001, 0.0292), (0.8573, 0.0417), (0.9451, 0.0560)],
             'names': ['ul', 'b', 'v', 'Na', 'w', 'x', 'p'], 'label': 'Hayabusa2 ONC-T (7)', 'norm': 0.55,
             'ref': 'Tatsumi et al. 2019 (Table 3.2, solar effective wavelengths and bandwidths); Kameda et al. 2017'},
    'dawnfc': {'bands': [(0.438, 0.041), (0.555, 0.044), (0.653, 0.043), (0.749, 0.045), (0.829, 0.037), (0.917, 0.046), (0.965, 0.087)],
               'names': ['F8', 'F2', 'F7', 'F3', 'F6', 'F4', 'F5'], 'label': 'Dawn FC (7)', 'norm': 0.555,
               'ref': 'Schroeder et al. 2013 (Table 3, effective wavelengths and FWHM); Sierks et al. 2011'},
    'mapcam': {'bands': [(0.473, 0.061), (0.550, 0.057), (0.698, 0.060), (0.847, 0.078)],
               'names': ["b'", 'v', 'w', 'x'], 'label': 'OSIRIS-REx MapCam (4)', 'norm': 0.55,
               'ref': 'Golish et al. 2022 (effective wavelengths 0.473/0.550/0.698/0.847 um; cut-on/off 0.439-0.500, 0.521-0.578, 0.671-0.731, 0.815-0.893 um -> widths); Rizk et al. 2018'}}
NOTE_GRID = 'the ONC-T ul band (0.3975 um, FWHM 36 nm) is truncated by the 0.40-um lower edge of the common spectral grid; its response is renormalised over the covered part'

# ----------------------------------------------------------------------------- band observers (custom filter sets on the S3b/S4 machinery)
def band_matrix(bands, grid=GRID):
    B = np.vstack([np.exp(-0.5 * ((grid - c) / (fw / 2.3548)) ** 2) for c, fw in bands]); return B / B.sum(axis=1, keepdims=True)

def make_band_observer(model, bands, resid_npz, snr=100.0, norm='auto', lib=None):
    """an Observer (AugObserver when lib is given) for an arbitrary Gaussian filter set; norm: 'auto' = the band nearest 0.55 um, a
    wavelength, or None (absolute)"""
    obs = S4.AugObserver(model, 'dawnfc', resid_npz, lib) if lib is not None else S3B.Observer(model, 'dawnfc', resid_npz)
    wl = np.array([c for c, _ in bands]); nrm = float(wl[int(np.argmin(abs(wl - 0.55)))]) if norm == 'auto' else norm
    obs.cfg = dict(kind='bands', filters='custom', norm=nrm, snr=float(snr), floor=1.0, label=f'{len(bands)} bands, SNR {snr:.0f}')
    obs.bands = list(bands); obs.wl = wl; obs.n = len(bands); obs.idx = None; obs.B = band_matrix(bands, obs.grid)
    obs.iref = int(np.argmin(abs(wl - nrm))) if nrm is not None else None
    return obs

def make_hyper_observer(model, wl_range, step, norm, snr, resid_npz, lib=None):
    """a hyperspectral Observer for an arbitrary wavelength range / step / normalisation (for the NEA spectra)"""
    obs = S4.AugObserver(model, 'tel50', resid_npz, lib) if lib is not None else S3B.Observer(model, 'tel50', resid_npz)
    obs.cfg = dict(kind='hyper', wl=tuple(wl_range), step=step, norm=norm, snr=float(snr), floor=1.0, label=f'{wl_range[0]}-{wl_range[1]} um / {step * 1e3:.0f} nm, SNR {snr:.0f}')
    wl = np.round(np.arange(wl_range[0], wl_range[1] + 1e-9, step), 4); obs.idx = np.array([int(np.argmin(abs(obs.grid - w))) for w in wl]); obs.wl = obs.grid[obs.idx]
    obs.bands = None; obs.n = len(obs.idx); obs.B = None; obs.iref = None
    return obs

# ----------------------------------------------------------------------------- B. band design (Fisher information of band sets)
class BandDesign:
    """precomputes, for a design population, the full-grid REFF, its Jacobian w.r.t. the state (ol-opx-plag scenario), and the
    model-error (+ library mismatch) covariance on the grid; evaluates the D-criterion of any Gaussian band set quickly"""
    TARGET_NAMES = ('ilr1', 'ilr2', 'Fo', 'Fs_opx')
    def __init__(self, model, states, resid_npz, lib=None, snr=100.0, centres=None, fwhms=(0.03, 0.06, 0.10), min_sep_factor=0.5):
        self.model = model; self.snr = float(snr); self.states = states; obs0 = S3B.Observer(model, 'lab', resid_npz); self.Crel = obs0.Crel
        self.centres = np.round(np.arange(0.45, 2.4501, 0.025), 4) if centres is None else np.asarray(centres); self.fwhms = tuple(fwhms)
        self.cand = [(float(c), float(f)) for f in self.fwhms for c in self.centres]; self.Bc = band_matrix(self.cand); self.min_sep_factor = min_sep_factor
        self.R, self.J, self.SR, self.P, self.names = [], [], [], None, None
        for th in states:
            pv = S3B.make_pv(SCN, th); v = pv.pack(th); names = pv.names(); R = model.spectrum(th, mode='reff'); J = np.zeros((len(R), len(v)))
            for i, nm in enumerate(names):
                h = S3B.STEPS['ilr' if nm.startswith('ilr') else nm]; vp = v.copy(); vm = v.copy(); vp[i] += h; vm[i] -= h
                J[:, i] = (model.spectrum(pv.unpack(vp), mode='reff') - model.spectrum(pv.unpack(vm), mode='reff')) / (2 * h)
            SR = (self.Crel * np.outer(R, R)) if self.Crel is not None else np.zeros((len(R), len(R)))
            if lib is not None:
                for f in [f for f in FAMS if th['M'].get(f, 0) > 0]:
                    if len(lib.pools.get(f, [])) == 0: continue
                    g = (S4.spectrum_aug(model, th, {f: np.exp(0.05)}) - R) / 0.05; SR = SR + np.outer(g, g) * lib.C[f]
            self.R.append(R); self.J.append(J); self.SR.append(SR)
            if self.P is None: self.P = S3B.prior_precision(pv); self.names = names; self.it = [names.index(n) for n in self.TARGET_NAMES]
        # per state: candidate responses of R, J and Sigma (Bc @ ...), so that a set S is a row/column subset
        self.yc = [self.Bc @ R for R in self.R]; self.Jc = [self.Bc @ J for J in self.J]; self.Sc = [self.Bc @ SR @ self.Bc.T for SR in self.SR]
    def criterion(self, S, return_sigmas=False, free_scale=True):
        """mean over the design states of log det of the posterior covariance of the targets (lower = better); with a free overall scale
        (normalised photometry) unless free_scale=False (absolute photometry)"""
        S = list(S); k = len(S); tot = 0.0; sig = []
        for yc, Jc, Sc in zip(self.yc, self.Jc, self.Sc):
            y = yc[S]; J = np.column_stack([Jc[S], y]) if free_scale else Jc[S]; Sig = Sc[np.ix_(S, S)] + np.diag((y / self.snr) ** 2) + np.eye(k) * 1e-12
            P = np.diag(np.append(np.diag(self.P), 1e-6)) if free_scale else self.P; L = np.linalg.cholesky(Sig); Jw = solve_triangular(L, J, lower=True); F = Jw.T @ Jw + P
            C = np.linalg.pinv(F); CT = C[np.ix_(self.it, self.it)]; sgn, ld = np.linalg.slogdet(CT); tot += ld if sgn > 0 else 50.0
            if return_sigmas: sig.append(np.sqrt(np.maximum(np.diag(C), 0)))
        return (tot / len(self.yc), np.array(sig)) if return_sigmas else tot / len(self.yc)
    def _ok(self, S, j):
        cj, fj = self.cand[j]
        return all(abs(cj - self.cand[i][0]) >= self.min_sep_factor * max(fj, self.cand[i][1]) + 0.01 for i in S)
    def greedy(self, k, S0=(), exchange_passes=3, verbose=True):
        S = list(S0); t0 = time.time()
        while len(S) < k:
            best = None
            for j in range(len(self.cand)):
                if j in S or not self._ok(S, j): continue
                c = self.criterion(S + [j])
                if best is None or c < best[0]: best = (c, j)
            S.append(best[1])
            if verbose: print(f'      greedy {len(S)}/{k}: + {self.cand[best[1]]} -> {best[0]:.3f} ({time.time() - t0:.0f}s)')
        cur = self.criterion(S)
        for _ in range(exchange_passes):
            improved = False
            for pos in range(len(S)):
                for j in range(len(self.cand)):
                    if j in S or not self._ok([s for s in S if s != S[pos]], j): continue
                    T = S.copy(); T[pos] = j; c = self.criterion(T)
                    if c < cur - 1e-9: S, cur, improved = T, c, True
            if not improved: break
        if verbose: print(f'      exchange -> {cur:.3f}; set: {[self.cand[j] for j in sorted(S, key=lambda j: self.cand[j][0])]}')
        return sorted(S, key=lambda j: self.cand[j][0]), cur
    def set_from_bands(self, bands):
        """evaluate an arbitrary band list (e.g. a mission set) by temporarily adding its responses as candidates"""
        B = band_matrix(bands); yc = [B @ R for R in self.R]; Jc = [B @ J for J in self.J]; Sc = [B @ SR @ B.T for SR in self.SR]
        saved = (self.yc, self.Jc, self.Sc); self.yc, self.Jc, self.Sc = yc, Jc, Sc
        try: return self.criterion(list(range(len(bands))), return_sigmas=True)
        finally: self.yc, self.Jc, self.Sc = saved
    def best_extra(self, bands, n_extra=1, max_centre=None):
        """which candidate band(s) added to a given set improve the criterion most (optionally restricted to centres <= max_centre, e.g. 1.0 um for a silicon detector)"""
        out = []; cur = list(bands)
        for _ in range(n_extra):
            best = None
            for j, (c, f) in enumerate(self.cand):
                if max_centre is not None and c > max_centre: continue
                if any(abs(c - c0) < self.min_sep_factor * max(f, f0) + 0.01 for c0, f0 in cur): continue
                v, _ = self.set_from_bands(cur + [(c, f)])
                if best is None or v < best[0]: best = (v, (c, f))
            cur = cur + [best[1]]; out.append((best[1], best[0]))
        return out

def uniform_set(k, fwhm=0.06, lo=0.5, hi=2.4): return [(float(c), fwhm) for c in np.linspace(lo, hi, k)]

def sensitivity_set(design: BandDesign, k, fwhm=0.06):
    """Jacobian-sensitivity heuristic: band centres at the k largest peaks of the mean |dR/dtheta_targets| (normalised per target)"""
    s = np.zeros(len(GRID))
    for J, R in zip(design.J, design.R):
        for i in design.it: s += np.abs(J[:, i]) / (np.abs(J[:, i]).max() + 1e-12)
    return peaks_to_bands(s, k, fwhm)

def pca_set(R_matrix, k, fwhm=0.06):
    """PCA-loading heuristic: band centres at the k largest |loading| peaks of the first principal components (normalised spectra)"""
    X = R_matrix / np.interp(0.55, GRID, np.ones(len(GRID)))[None, :] if False else R_matrix / R_matrix[:, int(np.argmin(abs(GRID - 0.55)))][:, None]
    Xc = X - X.mean(axis=0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False); w = np.zeros(len(GRID))
    for i in range(min(4, len(s))): w += (s[i] / s[0]) * np.abs(Vt[i])
    return peaks_to_bands(w, k, fwhm)

def peaks_to_bands(score, k, fwhm, min_sep=0.06):
    score = np.where(EVAL, score, 0.0); order = np.argsort(-score); chosen = []
    for i in order:
        c = float(GRID[i])
        if all(abs(c - c0) >= max(min_sep, fwhm) for c0, _ in chosen): chosen.append((c, fwhm))
        if len(chosen) == k: break
    return sorted(chosen)

def random_sets(k, n, rng, fwhm=0.06, lo=0.5, hi=2.4, min_sep=0.06):
    out = []
    while len(out) < n:
        c = np.sort(rng.uniform(lo, hi, k))
        if np.all(np.diff(c) >= min_sep): out.append([(float(x), fwhm) for x in c])
    return out

# ----------------------------------------------------------------------------- C. learned retrieval on band sets (S4 machinery)
def simulate_R(model, states, rng, lib=None):
    """noise-free REFF spectra of the population (with library-mismatch draws) -> (n, 421)"""
    R = np.zeros((len(states), len(GRID))); T = np.zeros((len(states), len(TARGETS)))
    for i, th in enumerate(states):
        fac = lib.draw(rng, [f for f in FAMS if th['M'].get(f, 0) > 0]) if lib is not None else None
        R[i] = S4.spectrum_aug(model, th, fac) if fac else model.spectrum(th, mode='reff'); T[i] = S4.targets_of(th)
    return R, T

def observe_R(obs, Rm, rng, noise=True):
    """apply a band/hyper observer to a matrix of REFF spectra, with one noise draw per spectrum from the full covariance"""
    Y = np.zeros((len(Rm), obs.n))
    for i, R in enumerate(Rm):
        y = S4.y_from_R(obs, R)
        if noise: y = y + np.linalg.cholesky(S4.cov_from_R(obs, R)) @ rng.normal(0, 1, obs.n)
        Y[i] = y
    return Y

def learned_eval_set(model, obs, data, qc, Xspec, spec_ids, mix, nat, rule, rng, tag, quick=False, seed=0, with_loso=True):
    """train HGB + MLP on the band-integrated population, calibrate conformal, evaluate on synthetic test + measured mixtures"""
    Xtr = observe_R(obs, data['tr'][0], rng); Xsc = observe_R(obs, data['sc'][0], rng); Xcal = observe_R(obs, data['cal'][0], rng); Xte = observe_R(obs, data['te'][0], rng)
    Ttr, Tsc, Tcal, Tte = data['tr'][1], data['sc'][1], data['cal'][1], data['te'][1]
    invs = {nm: S4.Inverse(nm, seed, quick).fit(Xtr, Ttr).fit_scale(Xsc, Tsc) for nm in ('hgb', 'mlp')}
    cal_mae = {nm: float(np.mean(np.abs(S4.report_space(Tcal)[:, 0] - S4.report_space(inv.predict(Xcal))[:, 0]))) for nm, inv in invs.items()}; prim = min(cal_mae, key=cal_mae.get)
    conf = S4.Conformal(invs[prim]).calibrate(Xcal, Tcal); ev = S4.evaluate(conf, Xte, Tte, 'synthetic_test'); ev['set'] = 'synthetic_test'
    Rt = S4.report_space(Tte); Pp = S4.report_space(invs[prim].predict(Xte)); rows = [{'tag': tag, 'model': prim, 'set': 'synthetic_test', 'target': t, 'mae': float(np.mean(np.abs(Rt[:, j] - Pp[:, j]))), 'n': len(Xte)} for j, t in enumerate(REPORT_TARGETS)]
    meta, Ym, Tm = S4.measured_set(obs, qc, Xspec, spec_ids, mix, nat, rule); out_meas = None
    if len(meta):
        evm = S4.evaluate(conf, Ym, Tm, 'measured_syncal'); Rm = S4.report_space(Tm); C90, hw90, rad90, P90 = conf.predict(Ym, 0.10)
        rows += [{'tag': tag, 'model': prim, 'set': 'measured', 'target': t, 'mae': float(np.nanmean(np.abs(Rm[:, j] - C90[:, j]))), 'n': int(np.isfinite(Rm[:, j]).sum())} for j, t in enumerate(REPORT_TARGETS)]
        ev = pd.concat([ev, evm], ignore_index=True)
        if with_loso: evl, _ = S4.loso_recalibration(conf, meta, Ym, Tm); ev = pd.concat([ev, evl], ignore_index=True)
        out_meas = meta.copy(); out_meas['tag'] = tag
        for j, t in enumerate(REPORT_TARGETS): out_meas[f'pred_{t}'] = C90[:, j]; out_meas[f'hw90_{t}'] = hw90[:, j]
    ev['tag'] = tag; ev['model'] = prim
    return ev, pd.DataFrame(rows), out_meas, conf, invs

# ----------------------------------------------------------------------------- D. NEA spectra
NEA = {'eros': {'number': 433, 'name': '(433) Eros', 'type': 'S (L/LL-like)',
                'ref': {'ol_frac_vol': 0.576, 'ol_sd': 0.045, 'opx_vol': 0.289, 'cpx_vol': 0.135, 'Fo': 100 - 22.8, 'Fo_sd': 1.1, 'Fs': 21.9, 'Fs_sd': 3.6, 'source': 'Korda et al. 2023 (NEAR NIS, remote)'}},
       'itokawa': {'number': 25143, 'name': '(25143) Itokawa', 'type': 'S (LL-like; returned samples)',
                   'ref': {'ol_frac_vol': 0.752, 'ol_sd': 0.03, 'opx_vol': 0.215, 'cpx_vol': 0.033, 'Fo': 100 - 28.4, 'Fo_sd': 1.2, 'Fs': 23.4, 'Fs_sd': 2.0, 'source': 'Nakamura et al. 2011 returned particles (as compiled by Korda et al. 2023); remote NIRS: ol 69 +/- 1.5 vol%, Fa 23.1, Fs 22.8'}}}

# direct download from the MIT archives (the same files the 'classy' package indexes; no classy index needed)
MIT_SPEX = 'http://smass.mit.edu/data/spex'
MIT_SMASS = 'http://smass.mit.edu/data/smass'
CLASSY_OBSLOG = 'https://raw.githubusercontent.com/maxmahlke/classy/main/data/mithneos/obslog.csv'
NEA_SPEX = {   # MITHNEOS IRTF/SpeX near-infrared spectra listed in the classy MITHNEOS observation log (run / file, UT date); refreshed from the log at run time
    'eros': [('sp15/a000433.sp15.txt', '2002-06-17'), ('sp16/a000433.sp16.txt', '2002-08-17'), ('sp17/a000433.sp17.txt', '2002-09-16'), ('sp101/a000433.sp101.txt', '2011-08-22'),
             ('sp102/a000433.sp102.txt', '2011-09-25'), ('sp103/a000433.sp103.txt', '2011-10-24'), ('sp105/a000433.sp105.txt', '2011-12-28'), ('sp201/a000433.sp201n1.txt', '2014-09-01'),
             ('sp223/a000433.sp223.txt', '2016-10-28'), ('sp245/a000433.sp245n2.txt', '2018-08-15'), ('sp247/a000433.sp247n1.txt', '2018-11-07'), ('sp256/a000433.sp256n2.txt', '2019-05-23')],
    'itokawa': []}
SMASS_ARCHIVES = [('smassref5', 'Binzel et al. 2001 (MAPS 36, 1167; 1998 SF36 = Itokawa)'), ('smassneodata', 'Binzel et al. 2001 (Icarus 151, 139)'), ('smassref7', 'Binzel et al. 2004 (MAPS 39, 351)'),
                  ('smassref8', 'Binzel et al. 2004 (Icarus 170, 259)'), ('smassref6', 'Binzel et al. 2004 (PSS 52, 291)'), ('smass2data', 'Bus and Binzel 2002 (Icarus 158, 106)')]

_DEAD_HOSTS = set()

def _http_get(url, timeout=90, max_bytes=300_000_000):
    """bytes of a URL (urllib, browser user agent as in classy) or None; never raises; a host that fails at the connection level
    (DNS, refused, timeout) is not contacted again in this session, so an unreachable archive costs one timeout at most"""
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    host = url.split('/')[2] if '://' in url else url
    if host in _DEAD_HOSTS: return None
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) specid-s5'})
        with urlopen(req, timeout=timeout) as r:
            b = r.read(max_bytes + 1)
        if len(b) > max_bytes: print(f'[NEA]   {url}: larger than {max_bytes / 1e6:.0f} MB, skipped'); return None
        return b
    except HTTPError as e:
        print(f'[NEA]   {url}: HTTP {e.code}'); return None
    except Exception as e:
        _DEAD_HOSTS.add(host); print(f'[NEA]   {url}: {type(e).__name__}: {str(e)[:120]} -> {host} marked unreachable for this session'); return None

def _parse_spectrum_text(text):
    """robust reader for archive spectra: numeric rows (wavelength, reflectance[, error[, flag]]); '#', '%', '!' and text lines skipped;
    wavelength in A / nm / um -> um; rows with flag == 0 (MITHNEOS/SMASS convention: rejected point) and non-positive reflectance removed"""
    rows = []
    for line in text.splitlines():
        t = line.strip()
        if not t or t[0] in '#%!;/': continue
        parts = t.replace(',', ' ').split()
        try: vals = [float(x) for x in parts[:4]]
        except ValueError: continue
        if len(vals) >= 2: rows.append(vals + [np.nan] * (4 - len(vals)))
    if len(rows) < 10: return None
    a = np.array(rows, float); wl, rf, err, flag = a[:, 0].copy(), a[:, 1], a[:, 2], a[:, 3]
    mx = np.nanmax(wl)
    if mx > 1000: wl = wl / 10000.0
    elif mx > 100: wl = wl / 1000.0
    ok = np.isfinite(wl) & np.isfinite(rf) & (rf > 0) & (wl > 0.3) & (wl < 2.7) & ~(flag == 0)
    if ok.sum() < 10: return None
    wl, rf, err = wl[ok], rf[ok], err[ok]; o = np.argsort(wl, kind='stable'); wl, rf, err = wl[o], rf[o], err[o]
    keep = np.concatenate([[True], np.diff(wl) > 0]); return wl[keep], rf[keep], err[keep]

def _smass_id(fname):
    """6-digit asteroid number from an MIT archive file name (a000433.sp15.txt, a025143.5, a000433visir8.dat ...), None otherwise"""
    s = fname.split('.')[0].split('_')[0]
    for suf in ('visir8', 'ir8'):
        if s.endswith(suf): s = s[:-len(suf)]
    m = re.match(r'a(\d{6})', s); return m.group(1) if m else None

def fetch_nea_direct(raw_csv: Path, keys=('eros', 'itokawa')):
    """download the MITHNEOS SpeX spectra and the SMASS archives that contain the NEA spectra; writes the long-format table
    raw_csv (key, source, ref, file, date_obs, wave, refl, err) -- always with a header, even when nothing was found"""
    import tarfile
    numbers = {k: f"{NEA[k]['number']:06d}" for k in keys}; rows = []; t0 = time.time()
    spex = {k: [(f'{MIT_SPEX}/{f}', d) for f, d in NEA_SPEX.get(k, [])] for k in keys}
    b = _http_get(CLASSY_OBSLOG, timeout=60)
    if b is not None:
        try:
            log = pd.read_csv(io.BytesIO(b))
            for k in keys:
                g = log[log['number'] == NEA[k]['number']]
                if len(g): spex[k] = [(u, str(d)[:10]) for u, d in zip(g['url'], g['date_obs'].fillna(''))]
            print(f"[NEA] MITHNEOS observation log (classy): {', '.join(f'{k} {len(v)}' for k, v in spex.items())} spectra")
        except Exception as e: print(f'[NEA] obslog unreadable ({e}); using the embedded list')
    for k in keys:
        for url, date in spex[k]:
            b = _http_get(url, timeout=60)
            if b is None: continue
            r = _parse_spectrum_text(b.decode('latin-1'))
            if r is None: print(f'[NEA]   {url}: no usable rows'); continue
            wl, rf, er = r; rows += [{'key': k, 'source': 'MITHNEOS', 'ref': 'MITHNEOS (Binzel et al. 2019; Marsset et al. 2022)', 'file': url.split('/')[-1], 'date_obs': date, 'wave': w, 'refl': f_, 'err': e_} for w, f_, e_ in zip(wl, rf, er)]
    print(f'[NEA] MITHNEOS done, {time.time() - t0:.0f}s')
    for arch, ref in SMASS_ARCHIVES:
        b = _http_get(f'{MIT_SMASS}/{arch}.tar.gz', timeout=120)
        if b is None: continue
        try:
            with tarfile.open(fileobj=io.BytesIO(b), mode='r:gz') as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    name = m.name.split('/')[-1]; low = name.lower()
                    if any(x in low for x in ('model', 'spfit', 'readme', 'directory')) or name.startswith('._'): continue
                    sid = _smass_id(name); key = [k for k, n in numbers.items() if n == sid]
                    if not key: continue
                    r = _parse_spectrum_text(tf.extractfile(m).read().decode('latin-1'))
                    if r is None: print(f'[NEA]   {arch}/{name}: no usable rows'); continue
                    wl, rf, er = r; rows += [{'key': key[0], 'source': f'SMASS/{arch}', 'ref': ref, 'file': name, 'date_obs': '', 'wave': w, 'refl': f_, 'err': e_} for w, f_, e_ in zip(wl, rf, er)]
                    print(f'[NEA]   {arch}/{name}: {len(wl)} points, {wl.min():.3f}-{wl.max():.3f} um')
        except Exception as e: print(f'[NEA]   {arch}: unreadable archive ({e})')
    df = pd.DataFrame(rows, columns=['key', 'source', 'ref', 'file', 'date_obs', 'wave', 'refl', 'err']); df.to_csv(raw_csv, index=False)
    print(f'[NEA] archive download finished in {time.time() - t0:.0f}s: ' + ', '.join(f"{k} {df[df['key'] == k]['file'].nunique()} spectra" for k in keys))
    return df

def _read_raw(raw: Path):
    try:
        if not raw.exists() or raw.stat().st_size < 30: return None
        df = pd.read_csv(raw); return df if {'key', 'wave', 'refl'} <= set(df.columns) else None
    except Exception as e: print(f'[NEA] {raw.name} unreadable ({e})'); return None

def _splice(vis, nir, min_overlap=0.04):
    """visible + NIR composite: the NIR spectrum is scaled to the visible one by the median ratio in their overlap and joined at its middle"""
    lo, hi = max(vis['wl'].min(), nir['wl'].min()), min(vis['wl'].max(), nir['wl'].max())
    if hi - lo < min_overlap: return None
    g = np.linspace(lo, hi, 25); sc = float(np.median(np.interp(g, vis['wl'], vis['refl']) / np.interp(g, nir['wl'], nir['refl']))); cut = 0.5 * (lo + hi)
    a = vis['wl'] < cut; b = nir['wl'] >= cut
    return {'wl': np.concatenate([vis['wl'][a], nir['wl'][b]]), 'refl': np.concatenate([vis['refl'][a], nir['refl'][b] * sc]), 'label': f"{nir['label']} + {vis['label']}",
            'date_obs': nir['date_obs'], 'nir': nir['label'], 'vis': vis['label'], 'splice_um': round(cut, 3), 'scale': round(sc, 4)}

def load_nea_spectra(inp_dir: Path, out: Path, allow_download=True):
    """returns {key: [composite spectra]}; each composite = dict(wl um, refl, label, date_obs, nir, vis). Sources, in order: (1) text files <key>*.txt / .csv /
    .dat in inp_dir (wavelength, reflectance[, error]); (2) nea_raw.csv in inp_dir or out (archive table written by fetch_nea_direct); (3) the direct
    download (needs Internet). Visible-only and NIR-only spectra are spliced in their overlap (each NIR epoch with the visible spectrum that reaches
    farthest into the NIR); combined visible+NIR files are used as they are."""
    singles = {}
    for key in NEA:
        for pth in sorted(list(inp_dir.glob(f'{key}*.txt')) + list(inp_dir.glob(f'{key}*.csv')) + list(inp_dir.glob(f'{key}*.dat'))):
            r = _parse_spectrum_text(pth.read_text(encoding='latin-1'))
            if r is None: print(f'[NEA] {pth.name}: no usable rows'); continue
            singles.setdefault(key, []).append({'wl': r[0], 'refl': r[1], 'label': f'file {pth.name}', 'date_obs': '', 'source': 'file'})
    missing = [k for k in NEA if k not in singles]
    if missing:
        raw = inp_dir / 'nea_raw.csv' if (inp_dir / 'nea_raw.csv').exists() else out / 'nea_raw.csv'
        df = _read_raw(raw)
        if (df is None or df.empty) and allow_download: df = fetch_nea_direct(out / 'nea_raw.csv')
        if df is not None and not df.empty:
            for key in missing:
                for (src, fname), g in df[df['key'] == key].groupby(['source', 'file'], sort=False):
                    g = g.sort_values('wave'); d = str(g['date_obs'].iloc[0]) if pd.notna(g['date_obs'].iloc[0]) else ''
                    singles.setdefault(key, []).append({'wl': g['wave'].values.astype(float), 'refl': g['refl'].values.astype(float), 'label': f'{src}:{fname}', 'date_obs': d, 'source': src})
    out_comp = {}; rows = []
    for key, lst in singles.items():
        full = [s for s in lst if s['wl'].min() < 0.55 and s['wl'].max() > 1.6]; vis = [s for s in lst if s['wl'].min() < 0.55 and s['wl'].max() <= 1.6]; nir = [s for s in lst if s['wl'].min() >= 0.55 and s['wl'].max() > 1.6]
        comps = [dict(s, nir=s['label'], vis='', splice_um=np.nan, scale=1.0) for s in full]
        best_vis = max(vis, key=lambda s: (s['wl'].max(), len(s['wl']))) if vis else None
        for n in nir:
            c = _splice(best_vis, n) if best_vis is not None else None
            comps.append(c if c is not None else dict(n, nir=n['label'], vis='', splice_um=np.nan, scale=1.0))
        if not comps and vis: print(f"[NEA] {NEA[key]['name']}: only visible spectra found ({len(vis)}); the 1- and 2-um bands are not covered -> skipped")
        for c in comps:
            c['wl'] = np.asarray(c['wl'], float); c['refl'] = np.asarray(c['refl'], float); o = np.argsort(c['wl']); c['wl'], c['refl'] = c['wl'][o], c['refl'][o]
            rows += [{'key': key, 'label': c['label'], 'date_obs': c.get('date_obs', ''), 'wave': w, 'refl': f_} for w, f_ in zip(c['wl'], c['refl'])]
        if comps:
            out_comp[key] = comps; print(f"[NEA] {NEA[key]['name']}: {len(comps)} composite spectra ({len(full)} combined, {len(nir)} NIR epochs, visible: {best_vis['label'] if best_vis is not None else 'none'}); coverage {min(c['wl'].min() for c in comps):.2f}-{max(c['wl'].max() for c in comps):.2f} um")
    if rows: pd.DataFrame(rows).to_csv(out / 'nea_composites.csv', index=False)
    return out_comp

def nea_observation(obs, wl, rf):
    """resample a spectrum on the observer's channels (hyper mode) and normalise as the observer does; returns y and a coverage mask"""
    y = np.interp(obs.wl, wl, rf, left=np.nan, right=np.nan)
    if obs.cfg['norm'] is not None: y = y / np.interp(obs.cfg['norm'], wl, rf)
    return y

def physics_fit_full(obs, y, seed=0, n_starts=2, max_nfev=300):
    """Step-3b physics estimator (coarse grid start, whitened least squares with nuisance priors) returning the full state and chi2"""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed); th0 = S3B.base_theta(SCN, ol=0.5, Fo=75, Fs=25, d=60, s=0.5, plag=0.25); pv = S3B.make_pv(SCN, th0); lo, hi = pv.bounds(); names = pv.names()
    P = S3B.prior_precision(pv); pw = np.sqrt(np.diag(P)) * np.array([0.0 if (nm.startswith('ilr') or nm in ('Fo', 'Fs_opx', 'logd')) else 1.0 for nm in names])
    grid = [(ol, plag, Fo, Fs) for ol in (0.15, 0.4, 0.65, 0.9) for plag in (0.05, 0.25, 0.45) for Fo in (60, 78, 90) for Fs in (10, 25, 40)]
    Yg = np.array([obs.y(S3B.base_theta(SCN, ol=ol, Fo=Fo, Fs=Fs, d=60, s=0.5, plag=plag)) for ol, plag, Fo, Fs in grid])
    c = np.sum(((Yg - y) / np.maximum(np.abs(y), 1e-3)) ** 2, axis=1); ol, plag, Fo, Fs = grid[int(np.argmin(c))]
    th_c = S3B.base_theta(SCN, ol=ol, Fo=Fo, Fs=Fs, d=60, s=0.5, plag=plag); L = obs.whitener(th_c); v0 = pv.pack(th_c)
    def resid(v): return np.concatenate([solve_triangular(L, obs.y(pv.unpack(v)) - y, lower=True), pw * v])
    best = None
    for j in range(n_starts):
        vs = v0 if j == 0 else np.clip(v0 + rng.normal(0, 1, len(v0)) * np.array([0.3 if nm.startswith('ilr') else {'Fo': 8, 'Fs_opx': 8, 'logd': 0.4, 's': 0.3}.get(nm, 0.2) for nm in names]), lo + 1e-6, hi - 1e-6)
        r = least_squares(resid, vs, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=max_nfev)
        if best is None or r.cost < best.cost: best = r
    th = pv.unpack(best.x); chi2_data = float(np.sum(solve_triangular(L, obs.y(th) - y, lower=True) ** 2))
    return th, chi2_data

def _olpx(Tz):
    M = S4.M_from_targets(Tz); return M[:, 0] / np.maximum(M[:, 0] + M[:, 1], 1e-9)

def ratio_quantile(conf, X, T, alpha):
    """split-conformal quantile for r = M_ol / (M_ol + M_px), normalised by the olivine-fraction scale model; NaN-safe"""
    P = conf.inv.predict(X); S = conf.inv.scale(X)[:, 0]; sc = np.abs(_olpx(T) - _olpx(P)) / S
    return S4.conformal_quantile(sc, alpha)

BP_COLS = ['BI', 'depthI', 'areaI', 'BII', 'depthII', 'areaII', 'BAR']
BP_FEATURES = ['BI', 'BII', 'BAR']                      # weathering-robust features: band centres and the band-area ratio
BP_NOISE = {'BI': 0.02, 'BII': 0.04, 'BAR': 0.12}       # feature scatter (about the epoch-to-epoch scatter of the Eros spectra)
BP_SLOPE = (-0.2, 0.8)                                  # continuum-slope augmentation exp(a (lambda - 0.55)), a in um^-1

def band_params(R, grid=GRID):
    """continuum-removed band parameters of one REFF spectrum on the common grid: BI centre (um), depth and area; BII centre, depth and
    area; BAR = area II / area I. Continuum = straight lines between the local maxima of the 0.62-0.86 um shoulder, the 1.35-1.72 um
    inter-band maximum and the 2.25-2.50 um end; centres from a parabola through the 9 points around the minimum."""
    R = np.asarray(R, float); ok = np.isfinite(R)
    if ok.sum() < 50: return [np.nan] * 7
    Rf = np.interp(grid, grid[ok], R[ok])
    def argmax_in(lo, hi): m = (grid >= lo) & (grid <= hi); return np.where(m)[0][np.argmax(Rf[m])]
    i0, i1, i2 = argmax_in(0.62, 0.86), argmax_in(1.35, 1.72), argmax_in(2.25, 2.50); out = []
    for (a, b, lo, hi) in ((i0, i1, 0.85, 1.20), (i1, i2, 1.75, 2.35)):
        cont = Rf[a] + (Rf[b] - Rf[a]) * (grid - grid[a]) / (grid[b] - grid[a]); r = Rf / cont; m = (grid >= lo) & (grid <= hi); idx = np.where(m)[0]; j = idx[np.argmin(r[idx])]
        jj = np.arange(max(j - 4, idx[0]), min(j + 5, idx[-1] + 1)); c = float(grid[j])
        if len(jj) >= 5:
            pp = np.polyfit(grid[jj], r[jj], 2)
            if pp[0] > 0 and lo <= -pp[1] / (2 * pp[0]) <= hi: c = float(-pp[1] / (2 * pp[0]))
        band = np.arange(a, b + 1); out += [c, float(1 - r[j]), float(np.trapezoid(np.clip(1 - r[band], 0, None), grid[band]))]
    out.append(out[5] / max(out[2], 1e-6)); return out

def band_param_matrix(Rm, rng=None, slope=None, noise=None):
    """band parameters of a matrix of REFF spectra, optionally with a random continuum slope before extraction and feature noise after"""
    Rm = np.asarray(Rm, float); n = len(Rm); a = rng.uniform(*slope, n) if (rng is not None and slope is not None) else np.zeros(n)
    B = np.array([band_params(R * np.exp(ai * (GRID - 0.55))) for R, ai in zip(Rm, a)])
    X = B[:, [BP_COLS.index(f) for f in BP_FEATURES]]
    if rng is not None and noise is not None: X = X + rng.normal(0, 1, X.shape) * np.array([noise[f] for f in BP_FEATURES])
    return X, B

def band_param_stage(model, out: Path, data, qc, Xspec, spec_ids, mix, nat, rule, rng, comps, quick=False, seed=0):
    """Level 2 of the NEA application: retrieval from the band parameters (BI, BII, BAR) with the Step-4 learned model + split conformal,
    trained on the nuisance-aware population with slope augmentation and feature noise; validated on the measured mixtures; applied to
    every NEA composite spectrum. Returns (per-spectrum rows, measured-mixture check, reference band parameters)."""
    t0 = time.time(); X = {}
    for k_ in ('tr', 'sc', 'cal', 'te'): X[k_], _ = band_param_matrix(data[k_][0], rng, BP_SLOPE, BP_NOISE)
    ok = {k_: np.all(np.isfinite(X[k_]), axis=1) for k_ in X}; T = {k_: data[k_][1][ok[k_]] for k_ in X}; X = {k_: X[k_][ok[k_]] for k_ in X}
    invs = {nm: S4.Inverse(nm, seed, quick).fit(X['tr'], T['tr']).fit_scale(X['sc'], T['sc']) for nm in ('hgb', 'mlp')}
    cal_mae = {nm: float(np.mean(np.abs(S4.report_space(T['cal'])[:, 0] - S4.report_space(inv.predict(X['cal']))[:, 0]))) for nm, inv in invs.items()}; prim = min(cal_mae, key=cal_mae.get)
    conf = S4.Conformal(invs[prim]).calibrate(X['cal'], T['cal']); qr = ratio_quantile(conf, X['cal'], T['cal'], 0.10); ev = S4.evaluate(conf, X['te'], T['te'], 'synthetic_test')
    Rt = S4.report_space(T['te']); Pp = S4.report_space(invs[prim].predict(X['te'])); dz = np.abs(_olpx(T['te']) - _olpx(invs[prim].predict(X['te'])))
    print(f"[NEA-BP] band-parameter model ({prim}) trained on {len(X['tr'])} states, {time.time() - t0:.0f}s: synthetic MAE ol/(ol+px) {dz.mean():.3f}, Fo {np.mean(np.abs(Rt[:, 3] - Pp[:, 3])):.1f}, Fs {np.mean(np.abs(Rt[:, 4] - Pp[:, 4])):.1f}")
    # measured mixtures: band parameters of the laboratory spectra (no slope augmentation), synthetic calibration
    obs_lab = S3B.Observer(model, 'lab', None); meta, Ym, Tm = S4.measured_set(obs_lab, qc, Xspec, spec_ids, mix, nat, rule); pos = {s_: i for i, s_ in enumerate(spec_ids)}
    Xm, Bm = band_param_matrix(np.array([Xspec[pos[sp]] for sp in meta['spectrum']])); okm = np.all(np.isfinite(Xm), axis=1); Xm, Tm, meta, Bm = Xm[okm], Tm[okm], meta[okm].reset_index(drop=True), Bm[okm]
    P = invs[prim].predict(Xm); S0 = invs[prim].scale(Xm)[:, 0]; C, hw, rad, _ = conf.predict(Xm, 0.10); r_true = _olpx(Tm); r_pred = _olpx(P); Rm = S4.report_space(Tm)
    meas_rows = meta.copy()
    for j, c_ in enumerate(BP_COLS): meas_rows[c_] = Bm[:, j]
    meas_rows['olpx_true'] = r_true; meas_rows['olpx_pred'] = r_pred; meas_rows['olpx_hw90'] = S0 * qr; meas_rows['Fo_pred'] = C[:, 3]; meas_rows['Fo_hw90'] = hw[:, 3]; meas_rows['Fs_pred'] = C[:, 4]; meas_rows['Fs_hw90'] = hw[:, 4]
    mFo = np.isfinite(Rm[:, 3]); mFs = np.isfinite(Rm[:, 4])
    check = {'model': prim, 'n_measured': int(len(Xm)), 'mae_olpx': float(np.mean(np.abs(r_true - r_pred))), 'cov90_olpx': float(np.mean(np.abs(r_true - r_pred) <= S0 * qr)), 'hw90_olpx_median': float(np.median(S0 * qr)),
             'mae_Fo': float(np.mean(np.abs(Rm[mFo, 3] - C[mFo, 3]))), 'cov90_Fo': float(np.mean(np.abs(Rm[mFo, 3] - C[mFo, 3]) <= hw[mFo, 3])), 'mae_Fs': float(np.mean(np.abs(Rm[mFs, 4] - C[mFs, 4]))), 'cov90_Fs': float(np.mean(np.abs(Rm[mFs, 4] - C[mFs, 4]) <= hw[mFs, 4])),
             'cov90_olpx_synthetic': float(np.mean(dz <= invs[prim].scale(X['te'])[:, 0] * qr)), 'cov90_M_ol_synthetic': float(ev[(ev['alpha'] == 0.10) & (ev['target'] == 'M_ol')]['coverage'].iloc[0]), 'spearman_BAR_olpx_measured': float(pd.Series(Bm[:, 6]).corr(pd.Series(r_true), method='spearman'))}
    print(f"[NEA-BP]   measured mixtures: MAE ol/(ol+px) {check['mae_olpx']:.3f}, coverage {check['cov90_olpx']:.2f} (hw {check['hw90_olpx_median']:.2f}); Fo MAE {check['mae_Fo']:.1f} cov {check['cov90_Fo']:.2f}; Fs MAE {check['mae_Fs']:.1f} cov {check['cov90_Fs']:.2f}")
    # the asteroids: every composite spectrum, resampled on the grid
    rows = []
    for key, lst in comps.items():
        ref = NEA[key]['ref']; ref_olpx = ref['ol_frac_vol'] / (ref['ol_frac_vol'] + ref['opx_vol'] + ref['cpx_vol'])
        for c in lst:
            R = np.interp(GRID, c['wl'], c['refl'], left=np.nan, right=np.nan)
            if np.isfinite(R).sum() < 150 or np.nanmin(np.where(np.isfinite(R), GRID, 9)) > 0.86 or np.nanmax(np.where(np.isfinite(R), GRID, 0)) < 2.3: print(f"[NEA-BP]   {c['label']}: insufficient coverage for the band parameters"); continue
            bp = band_params(R); x = np.array([[bp[BP_COLS.index(f)] for f in BP_FEATURES]])
            if not np.all(np.isfinite(x)): continue
            P = invs[prim].predict(x); S0 = float(invs[prim].scale(x)[0, 0]); C, hw, rad, _ = conf.predict(x, 0.10); r = float(_olpx(P)[0])
            rows.append({'key': key, 'asteroid': NEA[key]['name'], 'spectrum': c['label'], 'date_obs': c.get('date_obs', ''), **dict(zip(BP_COLS, bp)), 'olpx': r, 'olpx_lo90': max(0.0, r - S0 * qr), 'olpx_hi90': min(1.0, r + S0 * qr),
                         'Fo': C[0, 3], 'hw90_Fo': hw[0, 3], 'Fs': C[0, 4], 'hw90_Fs': hw[0, 4], 'M_plag': C[0, 2], 'hw90_M_plag': hw[0, 2], 'ref_olpx': ref_olpx, 'ref_Fo': ref['Fo'], 'ref_Fs': ref['Fs']})
            q = rows[-1]; print(f"[NEA-BP]   {c['label'][:52]:52s} BI {bp[0]:.3f} BII {bp[3]:.3f} BAR {bp[6]:.2f} depthI {bp[1]:.2f} -> ol/(ol+px) {r:.2f} [{q['olpx_lo90']:.2f}, {q['olpx_hi90']:.2f}]  Fo {C[0, 3]:.0f} ± {hw[0, 3]:.0f}  Fs {C[0, 4]:.0f} ± {hw[0, 4]:.0f}")
    # model band parameters at the reference compositions (unweathered, d = 30/60/150 um): consistency of positions and BAR, excess of depth
    refs = []
    for key in comps:
        ref = NEA[key]['ref']; ro = ref['ol_frac_vol'] / (ref['ol_frac_vol'] + ref['opx_vol'] + ref['cpx_vol'])
        for d in (30, 60, 150):
            th = {'M': {'ol': ro * 0.9, 'opx': (1 - ro) * 0.9, 'plag': 0.1}, 'Fo': ref['Fo'], 'Fs_opx': ref['Fs'], 'd': float(d), 'cpx_natural': True}; bp = band_params(model.spectrum(th, mode='reff'))
            refs.append({'key': key, 'asteroid': NEA[key]['name'], 'composition': f"ol/(ol+px) {ro:.2f}, Fo {ref['Fo']:.0f}, Fs {ref['Fs']:.0f}, plag 0.10", 'd_um': d, **dict(zip(BP_COLS, bp))})
    rows = pd.DataFrame(rows); refs = pd.DataFrame(refs); rows.to_csv(out / 'nea_bandparam_retrievals.csv', index=False); meas_rows.to_csv(out / 'nea_bandparam_measured.csv', index=False); refs.to_csv(out / 'nea_bandparam_reference_model.csv', index=False)
    return rows, check, refs, meas_rows, (invs[prim], conf, qr, X['te'], T['te'])

def nea_stage(model, out: Path, data, qc, Xspec, spec_ids, mix, nat, rule, rng, inp_dir: Path, resid_npz, quick=False, seed=0, allow_download=True, snr=50.0):
    """telescope-mode retrieval of the NEA composite spectra: per asteroid an observation mode matching the spectra's coverage (10-nm channels,
    normalised at 0.55 um, SNR + mixing error), the Step-4 learned model + 90 % conformal regions (synthetic and measured-mixture calibration;
    ratio ol/(ol+px) with its own conformal quantile), the physics estimator, and the reference mineralogy"""
    comps = load_nea_spectra(inp_dir, out, allow_download=allow_download)
    if not comps: print('[NEA] no spectra available: the NEA stage is skipped'); return {}
    models = {}; per, checks = [], []
    def mode_of(lst):
        lo = max(0.45, max(float(c['wl'].min()) for c in lst) + 0.005); hi = min(2.45, min(float(c['wl'].max()) for c in lst) - 0.005)
        lo = float(np.ceil(lo * 100) / 100); hi = float(np.floor(hi * 100) / 100); return lo, hi, (0.55 if lo <= 0.55 else 1.25)
    glo, ghi, gnorm = mode_of([c for lst in comps.values() for c in lst]); common = (glo <= 0.55 and ghi >= 2.3)   # one shared mode (one model) when every spectrum covers it
    for key, lst in comps.items():
        lo, hi, norm = (glo, ghi, gnorm) if common else mode_of(lst)
        if hi - lo < 1.0 or hi < 2.1: print(f"[NEA] {NEA[key]['name']}: coverage {lo}-{hi} um too short for the 1- and 2-um bands -> skipped"); continue
        mk = (lo, hi, norm)
        if mk not in models:
            obs = make_hyper_observer(model, (lo, hi), 0.01, norm, snr, resid_npz, lib=None); print(f'[NEA] training the telescope-mode model for {lo}-{hi} um / 10 nm, normalised at {norm} um, SNR {snr:.0f} + mixing error ...')
            ev, mc, meas, conf, invs = learned_eval_set(model, obs, data, qc, Xspec, spec_ids, mix, nat, rule, rng, f'nea_{lo}-{hi}', quick=quick, seed=seed, with_loso=False)
            meta, Ym, Tm = S4.measured_set(obs, qc, Xspec, spec_ids, mix, nat, rule); conf_meas = S4.Conformal(conf.inv).calibrate(Ym, Tm) if len(meta) else None
            Xcal = observe_R(obs, data['cal'][0], rng); Tcal = data['cal'][1]; qr = ratio_quantile(conf, Xcal, Tcal, 0.10); qr_meas = ratio_quantile(conf, Ym, Tm, 0.10) if len(meta) else np.nan
            # sim-to-real check of this exact mode on the 70 measured mixtures (synthetic calibration)
            P = conf.inv.predict(Ym); S = conf.inv.scale(Ym)[:, 0]; r_cov = float(np.mean(np.abs(_olpx(Tm) - _olpx(P)) <= S * qr)) if len(meta) else np.nan
            g = ev[(ev['set'] == 'measured_syncal') & (ev['alpha'] == 0.10)]
            chk = {'mode': f'{lo}-{hi} um, norm {norm}', 'model': ev['model'].iloc[0], 'cov90_M_ol_measured': float(g[g['target'] == 'M_ol']['coverage'].iloc[0]) if len(g) else np.nan, 'cov90_Fo_measured': float(g[g['target'] == 'Fo']['coverage'].iloc[0]) if len(g) else np.nan,
                   'cov90_olpx_measured': r_cov, 'mae_M_ol_measured': float(mc[(mc['set'] == 'measured') & (mc['target'] == 'M_ol')]['mae'].iloc[0]) if len(meta) else np.nan, 'mae_Fo_measured': float(mc[(mc['set'] == 'measured') & (mc['target'] == 'Fo')]['mae'].iloc[0]) if len(meta) else np.nan,
                   'cov90_M_ol_synthetic': float(ev[(ev['set'] == 'synthetic_test') & (ev['alpha'] == 0.10) & (ev['target'] == 'M_ol')]['coverage'].iloc[0])}
            checks.append(chk); models[mk] = (obs, conf, conf_meas, qr, qr_meas); print(f"[NEA]   mode check on the measured mixtures: coverage(M_ol) {chk['cov90_M_ol_measured']:.2f}, coverage(ol/(ol+px)) {r_cov:.2f}, MAE(M_ol) {chk['mae_M_ol_measured']:.3f}")
        obs, conf, conf_meas, qr, qr_meas = models[mk]; ref = NEA[key]['ref']; ref_olpx = ref['ol_frac_vol'] / (ref['ol_frac_vol'] + ref['opx_vol'] + ref['cpx_vol'])
        for c in lst:
            y = nea_observation(obs, c['wl'], c['refl'])
            if not np.all(np.isfinite(y)): print(f"[NEA]   {c['label']}: does not cover the mode, skipped"); continue
            C, hw, rad, P = conf.predict(y[None, :], 0.10); S0 = float(conf.inv.scale(y[None, :])[0, 0]); r = float(_olpx(P[:, :2])[0])
            hwm = conf_meas.predict(y[None, :], 0.10)[1] if conf_meas is not None else hw * np.nan
            try: th, chi2 = physics_fit_full(obs, y, seed=seed)
            except Exception as e: print(f"[NEA]   {c['label']}: physics fit failed ({e})"); th, chi2 = {'M': {'ol': np.nan, 'opx': np.nan, 'plag': np.nan}, 'Fo': np.nan, 'Fs_opx': np.nan, 'd': np.nan, 's': np.nan, 'l_ol': np.nan, 'l_opx': np.nan, 'l_plag': np.nan}, np.nan
            rp = th['M']['ol'] / (th['M']['ol'] + th['M']['opx'])
            per.append({'key': key, 'asteroid': NEA[key]['name'], 'spectrum': c['label'], 'date_obs': c.get('date_obs', ''), 'mode': obs.cfg['label'], 'model': conf.inv.kind,
                        'olpx': r, 'olpx_lo90': max(0.0, r - S0 * qr), 'olpx_hi90': min(1.0, r + S0 * qr), 'olpx_lo90_meascal': max(0.0, r - S0 * qr_meas), 'olpx_hi90_meascal': min(1.0, r + S0 * qr_meas),
                        'M_ol': C[0, 0], 'M_opx': C[0, 1], 'M_plag': C[0, 2], 'hw90_M_plag': hw[0, 2], 'Fo': C[0, 3], 'hw90_Fo': hw[0, 3], 'hw90_Fo_meascal': hwm[0, 3], 'Fs': C[0, 4], 'hw90_Fs': hw[0, 4], 'hw90_Fs_meascal': hwm[0, 4],
                        'd_um': math.exp(C[0, 5]), 'hw90_logd': hw[0, 5], 's': P[0, 5],
                        'phys_olpx': rp, 'phys_M_plag': th['M']['plag'], 'phys_Fo': th['Fo'], 'phys_Fs': th['Fs_opx'], 'phys_d_um': th['d'], 'phys_s': th['s'], 'phys_l_ol': th['l_ol'], 'phys_l_opx': th['l_opx'], 'phys_l_plag': th['l_plag'], 'phys_chi2_per_channel': chi2 / obs.n,
                        'ref_olpx': ref_olpx, 'ref_Fo': ref['Fo'], 'ref_Fo_sd': ref['Fo_sd'], 'ref_Fs': ref['Fs'], 'ref_Fs_sd': ref['Fs_sd'], 'ref_source': ref['source']})
            q = per[-1]; print(f"[NEA]   {c['label'][:60]:60s} ol/(ol+px) {r:.2f} [{q['olpx_lo90']:.2f}, {q['olpx_hi90']:.2f}]  Fo {C[0, 3]:.0f} ± {hw[0, 3]:.0f}  Fs {C[0, 4]:.0f} ± {hw[0, 4]:.0f}  s {P[0, 5]:.2f} | physics {rp:.2f}, Fo {th['Fo']:.0f}, Fs {th['Fs_opx']:.0f}, χ²/n {chi2 / obs.n:.2f}")
    if not per: return {}
    rows = pd.DataFrame(per); rows['flag_chi2'] = rows['phys_chi2_per_channel'] > 1.5; rows['flag_bounds'] = (rows['Fo'] > 95) | (rows['Fo'] < 0) | (rows['phys_Fo'] >= 94.9) | (rows['phys_Fs'] <= 5.1) | (rows['phys_M_plag'] > 0.6)
    rows['adequate'] = ~(rows['flag_chi2'] | rows['flag_bounds']); rows.to_csv(out / 'nea_retrievals.csv', index=False); pd.DataFrame(checks).to_csv(out / 'nea_mode_checks.csv', index=False)
    summ = []
    for key, g in rows.groupby('key', sort=False):
        i = int(np.argsort(g['olpx'].values)[len(g) // 2]); gm = g.iloc[i]
        summ.append({'key': key, 'asteroid': gm['asteroid'], 'n_spectra': len(g), 'olpx_median': float(g['olpx'].median()), 'olpx_min': float(g['olpx'].min()), 'olpx_max': float(g['olpx'].max()), 'olpx_lo90': gm['olpx_lo90'], 'olpx_hi90': gm['olpx_hi90'],
                     'olpx_lo90_meascal': gm['olpx_lo90_meascal'], 'olpx_hi90_meascal': gm['olpx_hi90_meascal'], 'Fo_median': float(g['Fo'].median()), 'Fo_hw90': float(g['hw90_Fo'].median()), 'Fo_hw90_meascal': float(g['hw90_Fo_meascal'].median()),
                     'Fs_median': float(g['Fs'].median()), 'Fs_hw90': float(g['hw90_Fs'].median()), 'M_plag_median': float(g['M_plag'].median()), 's_median': float(g['s'].median()), 'd_um_median': float(g['d_um'].median()),
                     'phys_olpx_median': float(g['phys_olpx'].median()), 'phys_Fo_median': float(g['phys_Fo'].median()), 'phys_Fs_median': float(g['phys_Fs'].median()), 'phys_chi2_median': float(g['phys_chi2_per_channel'].median()),
                     'ref_olpx': gm['ref_olpx'], 'ref_Fo': gm['ref_Fo'], 'ref_Fs': gm['ref_Fs'], 'ref_source': gm['ref_source'],
                     'olpx_ref_inside_90': bool(gm['olpx_lo90'] <= gm['ref_olpx'] <= gm['olpx_hi90']), 'Fo_ref_inside_90': bool(abs(float(g['Fo'].median()) - gm['ref_Fo']) <= float(g['hw90_Fo'].median()))})
    summ = pd.DataFrame(summ); summ.to_csv(out / 'nea_summary.csv', index=False)
    try: plot_nea(rows, comps, models, out)
    except Exception as e: print(f'[NEA] figure failed: {e}')
    res = {'rows': rows, 'summary': summ, 'checks': pd.DataFrame(checks)}
    try:
        bp_rows, bp_check, bp_refs, bp_meas, bp_model = band_param_stage(model, out, data, qc, Xspec, spec_ids, mix, nat, rule, rng, comps, quick=quick, seed=seed)
        res.update({'bp_rows': bp_rows, 'bp_check': bp_check, 'bp_refs': bp_refs, 'bp_meas': bp_meas})
        try: plot_nea_bandparams(bp_rows, bp_meas, bp_refs, bp_model, data, rng, out)
        except Exception as e: print(f'[NEA-BP] figure failed: {e}')
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'[NEA-BP] band-parameter stage failed ({type(e).__name__}: {e})')
    return res

# ----------------------------------------------------------------------------- figures
def plot_bands(design: BandDesign, sets: dict, mean_R, out: Path, info_abs=None, info_add=None, kbase=4):
    plt = S2.style(); fig, axes = plt.subplots(3, 1, figsize=(10, 9.6), sharex=True, gridspec_kw={'height_ratios': [1.3, 1, 1]})
    ax = axes[0]; ax.plot(GRID, mean_R, color='#999999', lw=1.0, label='mean synthetic spectrum')
    keys = [k for k in sets if k.startswith('opt')]; keys = keys + [k for k in ('dawnfc', 'onct', 'mapcam') if k in sets]
    for i, key in enumerate(keys):
        bands = sets[key]['bands']; y0 = 0.95 - 0.07 * i
        for c, fw in bands: ax.plot([c - fw / 2, c + fw / 2], [y0, y0], color=WONG[i % 7], lw=3, solid_capstyle='butt'); ax.plot([c], [y0], marker='|', color=WONG[i % 7], ms=8)
        ax.text(2.52, y0, sets[key]['label'], fontsize=7, color=WONG[i % 7], va='center', ha='left')
    ax.set_ylim(0, 1.05); ax.set_xlim(0.4, 3.0); ax.set_ylabel('REFF / band positions'); ax.legend(fontsize=7, frameon=False, loc='lower left')
    cc = np.array([c for c, _ in design.cand])
    for ax, info, lab in ((axes[1], info_abs, 'one band alone, absolute photometry'), (axes[2], info_add, f'one band added to the optimal-{kbase} set (free scale)')):
        if info is None: continue
        for i, fw in enumerate(design.fwhms):
            m = np.array([f == fw for _, f in design.cand]); ax.plot(cc[m], info[m], color=WONG[i], lw=1.2, label=f'FWHM {fw * 1e3:.0f} nm')
        ax.set_ylabel(f'D-criterion: {lab}\n(lower = more information)', fontsize=8); ax.legend(fontsize=7, frameon=False)
    axes[2].set_xlabel('wavelength (µm)'); fig.tight_layout(); fig.savefig(out / 'figures' / 'S5_F1_bands.png'); fig.savefig(out / 'figures' / 'S5_F1_bands.pdf'); plt.close(fig)

def plot_crb_vs_k(tab, out: Path):
    plt = S2.style(); fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    fams = {'opt': ('optimal (D)', WONG[0], 'o'), 'uniform': ('uniform', WONG[1], 's'), 'random': ('random (median of 30)', WONG[2], '^'), 'sens': ('sensitivity peaks', WONG[3], 'D'), 'pca': ('PCA peaks', WONG[4], 'v')}
    for ax, col, lab in zip(axes, ('sigma_M_ol', 'sigma_Fo', 'sigma_Fs_opx'), ('σ(olivine mass fraction)', 'σ(Fo)', 'σ(Fs)')):
        for fam, (name, col_, mk) in fams.items():
            g = tab[tab['family'] == fam].sort_values('k')
            if len(g): ax.plot(g['k'], g[col], color=col_, marker=mk, ms=4, lw=1.2, label=name)
        for key, mk in (('dawnfc', '*'), ('onct', 'P'), ('mapcam', 'X')):
            g = tab[tab['family'] == key]
            if len(g): ax.scatter(g['k'], g[col], color=WONG[5 if key == 'dawnfc' else (6 if key == 'onct' else 3)], marker=mk, s=70, zorder=5, label=g['label'].iloc[0])
        hy = tab[tab['family'] == 'hyper']
        if len(hy): ax.axhline(float(hy[col].iloc[0]), color='#999999', lw=0.8, ls='--', label='hyperspectral 0.45–2.45 µm / 10 nm, SNR 50')
        ax.set_xlabel('number of bands k'); ax.set_ylabel(f'median {lab} (CRB+lib, SNR 100 per band)'); ax.set_yscale('log')
        if col == 'sigma_M_ol': ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S5_F2_crb_vs_k.png'); fig.savefig(out / 'figures' / 'S5_F2_crb_vs_k.pdf'); plt.close(fig)

def plot_measured_sets(mc, ev, out: Path):
    plt = S2.style(); tags = list(dict.fromkeys(mc['tag'])); fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (t, lab) in zip(axes[:2], (('M_ol', 'olivine mass fraction'), ('Fo', 'Fo'))):
        for i, sett in enumerate(('synthetic_test', 'measured')):
            vals = [float(mc[(mc['tag'] == tg) & (mc['set'] == sett) & (mc['target'] == t)]['mae'].iloc[0]) if ((mc['tag'] == tg) & (mc['set'] == sett) & (mc['target'] == t)).any() else np.nan for tg in tags]
            ax.bar(np.arange(len(tags)) + 0.4 * i, vals, width=0.4, color=WONG[i], label=sett.replace('_', ' '))
        ax.set_xticks(np.arange(len(tags)) + 0.2); ax.set_xticklabels(tags, rotation=35, ha='right', fontsize=7); ax.set_ylabel(f'MAE ({lab}), primary model'); ax.legend(fontsize=7, frameon=False)
    ax = axes[2]
    for i, sett in enumerate(('measured_syncal', 'measured_LOSO_recal')):
        g = ev[(ev['set'] == sett) & (ev['alpha'] == 0.10) & (ev['target'] == 'M_ol')]
        ax.bar(np.arange(len(tags)) + 0.4 * i, [float(g[g['tag'] == tg]['coverage'].iloc[0]) if (g['tag'] == tg).any() else np.nan for tg in tags], width=0.4, color=WONG[i + 2], label=sett.replace('_', ' '))
    ax.axhline(0.9, color='#999999', lw=0.8, ls='--'); ax.set_xticks(np.arange(len(tags)) + 0.2); ax.set_xticklabels(tags, rotation=35, ha='right', fontsize=7); ax.set_ylabel('coverage at nominal 90 % (olivine fraction), measured mixtures'); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S5_F3_measured_sets.png'); fig.savefig(out / 'figures' / 'S5_F3_measured_sets.pdf'); plt.close(fig)

def plot_nea(rows, comps, models, out: Path):
    plt = S2.style(); keys = list(dict.fromkeys(rows['key'])); fig, axes = plt.subplots(len(keys), 3, figsize=(15, 4.3 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        g = rows[rows['key'] == key].reset_index(drop=True); lst = {c['label']: c for c in comps[key]}; mk = [m for m in models if models[m][0].cfg['label'] == g['mode'].iloc[0]][0]; obs = models[mk][0]
        ax = axes[i][0]; j_med = int(np.argsort(g['olpx'].values)[len(g) // 2])
        for j, r in g.iterrows():
            c = lst[r['spectrum']]; y = nea_observation(obs, c['wl'], c['refl']); ax.plot(obs.wl, y, color=WONG[0] if j == j_med else '#9ec7e0', lw=1.5 if j == j_med else 0.7, alpha=1.0 if j == j_med else 0.7, label='median epoch' if j == j_med else None)
        r = g.iloc[j_med]
        c = lst[r['spectrum']]; y = nea_observation(obs, c['wl'], c['refl']); th, _ = physics_fit_full(obs, y)
        ax.plot(obs.wl, obs.y(th), color=WONG[1], lw=1.2, ls='--', label=f"physics fit: ol/(ol+px) {r['phys_olpx']:.2f}, Fo {r['phys_Fo']:.0f}, Fs {r['phys_Fs']:.0f}, s {r['phys_s']:.2f}")
        ax.set_xlabel('wavelength (µm)'); ax.set_ylabel(f"reflectance / R({obs.cfg['norm']} µm)"); ax.set_title(f"{NEA[key]['name']}: {len(g)} spectra", fontsize=9); ax.legend(fontsize=6.5, frameon=False)
        for ax, (col, lo, hi, lo2, hi2, ref, lab) in zip(axes[i][1:], (('olpx', 'olpx_lo90', 'olpx_hi90', 'olpx_lo90_meascal', 'olpx_hi90_meascal', 'ref_olpx', 'olivine / (olivine + pyroxene)'), ('Fo', None, None, None, None, 'ref_Fo', 'Fo'))):
            xx = np.arange(len(g))
            if lo is not None:
                ax.errorbar(xx - 0.1, g[col], yerr=[g[col] - g[lo], g[hi] - g[col]], fmt='o', ms=4, color=WONG[0], ecolor=WONG[0], elinewidth=1.0, capsize=0, label='learned, 90 % (synthetic calibration)')
                ax.errorbar(xx + 0.1, g[col], yerr=[g[col] - g[lo2], g[hi2] - g[col]], fmt='none', ecolor=WONG[2], elinewidth=1.0, capsize=0, label='90 % (measured-mixture calibration)')
                ax.scatter(xx, g['phys_olpx'], marker='s', s=16, color=WONG[1], label='physics estimator')
            else:
                ax.errorbar(xx - 0.1, g['Fo'], yerr=g['hw90_Fo'], fmt='o', ms=4, color=WONG[0], ecolor=WONG[0], elinewidth=1.0, capsize=0, label='learned, 90 % (synthetic calibration)')
                ax.errorbar(xx + 0.1, g['Fo'], yerr=g['hw90_Fo_meascal'], fmt='none', ecolor=WONG[2], elinewidth=1.0, capsize=0, label='90 % (measured-mixture calibration)')
                ax.scatter(xx, g['phys_Fo'], marker='s', s=16, color=WONG[1], label='physics estimator')
            rv = float(g[ref].iloc[0]); ax.axhline(rv, color='#999999', lw=1.0, ls='--', label=f'reference {rv:.2f}' if col == 'olpx' else f'reference Fo {rv:.0f}')
            ax.set_xlim(-0.6, len(g) - 0.4); ax.set_xticks(xx); ax.set_xticklabels([str(d)[:7] if str(d) not in ('', 'nan') else f'#{k + 1}' for k, d in enumerate(g['date_obs'])], rotation=45, ha='right', fontsize=6.5); ax.set_ylabel(lab); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S5_F4_nea.png'); fig.savefig(out / 'figures' / 'S5_F4_nea.pdf'); plt.close(fig)

def plot_nea_bandparams(bp_rows, bp_meas, bp_refs, bp_model, data, rng, out: Path):
    plt = S2.style(); fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    inv, conf, qr, Xte, Tte = bp_model; Xs, Bs = band_param_matrix(data['te'][0][:1500]); okx = np.all(np.isfinite(Xs), axis=1); r_syn = _olpx(data['te'][1][:1500])[okx]
    for ax, (xc, yc, xl, yl) in zip(axes[:2], (('BI', 'BAR', 'band I centre (µm)', 'band area ratio'), ('BI', 'BII', 'band I centre (µm)', 'band II centre (µm)'))):
        jx, jy = BP_COLS.index(xc), BP_COLS.index(yc); sc = ax.scatter(Bs[okx, jx], Bs[okx, jy], c=r_syn, cmap='cividis', s=5, alpha=0.35, linewidths=0, vmin=0, vmax=1, label='synthetic population')
        ax.scatter(bp_meas[xc], bp_meas[yc], c=bp_meas['olpx_true'], cmap='cividis', s=22, marker='s', edgecolors='#555555', linewidths=0.4, vmin=0, vmax=1, label='measured mixtures')
        for k, (key, g) in enumerate(bp_rows.groupby('key', sort=False)): ax.scatter(g[xc], g[yc], color=WONG[1 + k], marker='*', s=90, edgecolors='#333333', linewidths=0.5, zorder=5, label=NEA[key]['name'])
        for k, (key, g) in enumerate(bp_refs.groupby('key', sort=False)): ax.plot(g[xc], g[yc], color=WONG[1 + k], lw=1.0, ls='--', marker='o', ms=3, label=f"model at the {NEA[key]['name']} reference (d 30–150 µm)")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=5.5, frameon=False)
        if xc == 'BI' and yc == 'BAR': ax.set_ylim(0, 3); cb = plt.colorbar(sc, ax=ax, pad=0.02); cb.set_label('olivine / (olivine + pyroxene)', fontsize=7)
    ax = axes[2]; keys = list(dict.fromkeys(bp_rows['key'])); x0 = 0
    for k, key in enumerate(keys):
        g = bp_rows[bp_rows['key'] == key].reset_index(drop=True); xx = x0 + np.arange(len(g))
        ax.errorbar(xx, g['olpx'], yerr=[g['olpx'] - g['olpx_lo90'], g['olpx_hi90'] - g['olpx']], fmt='o', ms=4, color=WONG[1 + k], ecolor=WONG[1 + k], elinewidth=1.0, capsize=0, label=f"{NEA[key]['name']}: band-parameter retrieval, 90 %")
        ax.plot([x0 - 0.4, x0 + len(g) - 0.6], [g['ref_olpx'].iloc[0]] * 2, color=WONG[1 + k], lw=1.0, ls='--', label=f'reference {g["ref_olpx"].iloc[0]:.2f}'); x0 += len(g) + 1
    ax.set_ylim(0, 1); ax.set_ylabel('olivine / (olivine + pyroxene)'); ax.set_xlabel('spectrum'); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S5_F5_nea_bandparams.png'); fig.savefig(out / 'figures' / 'S5_F5_nea_bandparams.pdf'); plt.close(fig)

# ----------------------------------------------------------------------------- driver
def run_all(model, out: Path, proj: Path, resid_npz, qc, Xspec, spec_ids, mix, em, nat, rule, inp_dir: Path, ks=(4, 5, 6, 7, 8, 10, 12), snr_band=100.0, allow_download=True, n_design=48, n_eval=200, n_train=20000, n_scale=4000, n_cal=4000, n_test=2000, n_random=30, quick=False, seed=0):
    t0 = time.time(); (out / 'figures').mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    if quick: ks = (4, 7, 12); n_design, n_eval, n_train, n_scale, n_cal, n_test, n_random = 10, 30, 1500, 400, 400, 300, 5
    lib = S4.LibraryMismatch(model, qc, Xspec, spec_ids, em, nat, rule, exclude_codes=S4.test_codes(mix)); print('[S5] mismatch pools:', lib.summary())
    # ---- B. design
    st_design = S4.sample_states(n_design, rng); st_eval = S4.sample_states(n_eval, rng)
    print(f'[S5] band design: {n_design} design states, candidates = centres 0.45-2.45 um / 25 nm x FWHM {[30, 60, 100]} nm, SNR {snr_band:.0f} per band, free scale ...')
    design = BandDesign(model, st_design, resid_npz, lib=lib, snr=snr_band); print(f'[S5] design precomputed, {time.time() - t0:.0f}s')
    info_abs = np.array([design.criterion([j], free_scale=False) for j in range(len(design.cand))])
    sets = {}
    for k in ks:
        S, c = design.greedy(k, verbose=not quick); sets[f'opt{k}'] = {'bands': [design.cand[j] for j in S], 'label': f'optimal {k} (D-criterion)', 'family': 'opt', 'k': k, 'criterion_design': c}
        print(f'[S5] optimal {k}: {[(round(c_, 3), round(f, 3)) for c_, f in sets[f"opt{k}"]["bands"]]}  ({time.time() - t0:.0f}s)')
    for k in ks:
        sets[f'uniform{k}'] = {'bands': uniform_set(k), 'label': f'uniform {k}', 'family': 'uniform', 'k': k}
        sets[f'sens{k}'] = {'bands': sensitivity_set(design, k), 'label': f'sensitivity peaks {k}', 'family': 'sens', 'k': k}
    Rm_design = np.vstack(design.R);
    for k in ks: sets[f'pca{k}'] = {'bands': pca_set(Rm_design, k), 'label': f'PCA peaks {k}', 'family': 'pca', 'k': k}
    for key, F in FILTERS_V.items(): sets[key] = {'bands': F['bands'], 'label': F['label'], 'family': key, 'k': len(F['bands']), 'names': F['names'], 'ref': F['ref']}
    # ---- evaluation population: CRB+lib sigmas for every set (band observers), plus random sets and the hyperspectral reference
    print(f'[S5] evaluating sets on {n_eval} states ...'); rows = []
    def eval_set(bands, label, family, k, extra=None, states=None):
        obs = make_band_observer(model, bands, resid_npz, snr=snr_band, norm='auto', lib=lib); sig = []; ld = []
        for th in (st_eval if states is None else states):
            pv = S3B.make_pv(SCN, th); v = pv.pack(th); J = S3B.jacobian(obs, pv, v); L = obs.whitener(th); r = S3B.analyse(obs, pv, th, 'default', J, L)
            sig.append([r['sigma_M_ol'], r['sigma_M_plag'], r['sigma_Fo'], r['sigma_Fs_opx'], r['sigma_logd'], r['dofs']])
            Jw = solve_triangular(L, J, lower=True); C = np.linalg.pinv(Jw.T @ Jw + S3B.prior_precision(pv)); it = [pv.names().index(n) for n in BandDesign.TARGET_NAMES]; sgn, l_ = np.linalg.slogdet(C[np.ix_(it, it)]); ld.append(l_ if sgn > 0 else 50.0)
        sig = np.array(sig); crit, _ = design.set_from_bands(bands)
        row = {'tag': label, 'label': label, 'family': family, 'k': k, 'criterion_design': crit, 'criterion_eval': float(np.mean(ld)), 'sigma_M_ol': float(np.median(sig[:, 0])), 'sigma_M_plag': float(np.median(sig[:, 1])), 'sigma_Fo': float(np.median(sig[:, 2])), 'sigma_Fs_opx': float(np.median(sig[:, 3])), 'sigma_logd': float(np.median(sig[:, 4])), 'dofs': float(np.median(sig[:, 5])),
               'sigma_M_ol_p90': float(np.percentile(sig[:, 0], 90)), 'sigma_Fo_p90': float(np.percentile(sig[:, 2], 90)), 'bands': json.dumps([[round(c, 4), round(f, 4)] for c, f in bands])}
        if extra: row.update(extra)
        return row
    for key, s in sets.items(): rows.append(eval_set(s['bands'], key, s['family'], s['k'], {'ref': s.get('ref', '')}));
    for k in ks:
        rs = random_sets(k, n_random, rng); vals = [eval_set(b, f'random{k}_{i}', 'random_draw', k, states=st_eval[:max(20, n_eval // 4)]) for i, b in enumerate(rs)]; df = pd.DataFrame(vals)
        rows.append({'tag': f'random{k}', 'label': f'random {k} (median of {n_random})', 'family': 'random', 'k': k, 'criterion_design': float(df['criterion_design'].median()), 'criterion_eval': float(df['criterion_eval'].median()), 'sigma_M_ol': float(df['sigma_M_ol'].median()), 'sigma_M_plag': float(df['sigma_M_plag'].median()), 'sigma_Fo': float(df['sigma_Fo'].median()), 'sigma_Fs_opx': float(df['sigma_Fs_opx'].median()), 'sigma_logd': float(df['sigma_logd'].median()), 'dofs': float(df['dofs'].median()), 'sigma_M_ol_p90': float(df['sigma_M_ol_p90'].median()), 'sigma_Fo_p90': float(df['sigma_Fo_p90'].median()), 'bands': 'random', 'random_best_sigma_M_ol': float(df['sigma_M_ol'].min())})
        sets[f'random{k}'] = {'bands': rs[int(np.argsort(df['sigma_M_ol'].values)[len(rs) // 2])], 'label': f'random {k} (median draw)', 'family': 'random', 'k': k}
    hy = S4.AugObserver(model, 'tel50', resid_npz, lib); sig = []
    for th in st_eval:
        pv = S3B.make_pv(SCN, th); r = S3B.analyse(hy, pv, th); sig.append([r['sigma_M_ol'], r['sigma_M_plag'], r['sigma_Fo'], r['sigma_Fs_opx'], r['sigma_logd'], r['dofs']])
    sig = np.array(sig); rows.append({'tag': 'tel50', 'label': 'hyperspectral tel50', 'family': 'hyper', 'k': hy.n, 'criterion_eval': float('nan'), 'sigma_M_ol': float(np.median(sig[:, 0])), 'sigma_M_plag': float(np.median(sig[:, 1])), 'sigma_Fo': float(np.median(sig[:, 2])), 'sigma_Fs_opx': float(np.median(sig[:, 3])), 'sigma_logd': float(np.median(sig[:, 4])), 'dofs': float(np.median(sig[:, 5])), 'sigma_M_ol_p90': float(np.percentile(sig[:, 0], 90)), 'sigma_Fo_p90': float(np.percentile(sig[:, 2], 90)), 'bands': 'hyper'})
    tab = pd.DataFrame(rows); tab.to_csv(out / 'band_sets_crb.csv', index=False); print(f'[S5] CRB evaluation done, {time.time() - t0:.0f}s')
    # best extra filter for each mission set
    extra_rows = []
    for key in FILTERS_V:
        base, _ = design.set_from_bands(FILTERS_V[key]['bands'])
        for constraint, mc_ in (('any', None), ('silicon <= 1.0 um', 1.0)):
            for i, ((c, f), v) in enumerate(design.best_extra(FILTERS_V[key]['bands'], n_extra=2, max_centre=mc_)): extra_rows.append({'mission': key, 'constraint': constraint, 'order': i + 1, 'added_centre_um': c, 'added_fwhm_um': f, 'criterion_before': base, 'criterion_after': v})
    pd.DataFrame(extra_rows).to_csv(out / 'mission_best_extra_filter.csv', index=False)
    base4 = sets[f'opt{min(ks)}']['bands']; info_add = np.array([design.set_from_bands(base4 + [cf])[0] if all(abs(cf[0] - c0) >= design.min_sep_factor * max(cf[1], f0) + 0.01 for c0, f0 in base4) else np.nan for cf in design.cand])
    plot_bands(design, sets, Rm_design.mean(axis=0), out, info_abs, info_add, min(ks)); plot_crb_vs_k(tab, out)
    pd.DataFrame({'centre_um': [c for c, _ in design.cand], 'fwhm_um': [f for _, f in design.cand], 'criterion_single_absolute': info_abs, f'criterion_added_to_opt{min(ks)}': info_add}).to_csv(out / 'single_band_information.csv', index=False)
    # ---- C. learned retrieval + conformal on selected sets, measured validation
    print(f'[S5] simulating the nuisance-aware population ({n_train + n_scale + n_cal + n_test} states) ...')
    st = {k_: S4.sample_states(n, rng) for k_, n in (('tr', n_train), ('sc', n_scale), ('cal', n_cal), ('te', n_test))}; data = {k_: simulate_R(model, v, rng, lib) for k_, v in st.items()}
    print(f'[S5] simulated, {time.time() - t0:.0f}s')
    learn_tags = [f'opt{k}' for k in ks if k in (4, 7, 12)] + [f'uniform{k}' for k in ks if k in (4, 7, 12)] + [f'random{k}' for k in ks if k in (7,)] + list(FILTERS_V)
    learn_tags = [t for t in learn_tags if t in sets]; ev_all, mc_all, meas_all = [], [], []
    for tag in learn_tags:
        obs = make_band_observer(model, sets[tag]['bands'], resid_npz, snr=snr_band, norm='auto', lib=None)
        ev, mc, meas, conf, invs = learned_eval_set(model, obs, data, qc, Xspec, spec_ids, mix, nat, rule, rng, tag, quick=quick, seed=seed); ev_all.append(ev); mc_all.append(mc)
        if meas is not None: meas_all.append(meas)
        print(f"[{tag}] learned retrieval done ({ev['model'].iloc[0]}): measured MAE M_ol {float(mc[(mc['set'] == 'measured') & (mc['target'] == 'M_ol')]['mae'].iloc[0]):.3f}, {time.time() - t0:.0f}s")
    ev_all = pd.concat(ev_all, ignore_index=True); mc_all = pd.concat(mc_all, ignore_index=True); meas_all = pd.concat(meas_all, ignore_index=True) if meas_all else pd.DataFrame()
    ev_all.to_csv(out / 'learned_sets_conformal.csv', index=False); mc_all.to_csv(out / 'learned_sets_accuracy.csv', index=False); meas_all.to_csv(out / 'learned_sets_measured.csv', index=False); plot_measured_sets(mc_all, ev_all, out)
    # ---- D. NEA application (telescope mode)
    try: nea_res = nea_stage(model, out, data, qc, Xspec, spec_ids, mix, nat, rule, rng, inp_dir, resid_npz, quick=quick, seed=seed, allow_download=allow_download)
    except Exception as e:   # the NEA stage must never cost the band-design results
        import traceback; traceback.print_exc(); print(f'[NEA] stage failed ({type(e).__name__}: {e}); the report is written without it'); nea_res = {}
    json.dump({'version': VERSION, 'filters_verified': {k: {kk: vv for kk, vv in v.items()} for k, v in FILTERS_V.items()}, 'note_grid': NOTE_GRID, 'ks': list(ks), 'snr_band': snr_band, 'n_design': n_design, 'n_eval': n_eval, 'n_train': n_train, 'n_cal': n_cal, 'n_test': n_test, 'seed': seed, 'mismatch_pools': lib.summary()}, open(out / 'run_config.json', 'w'), indent=1)
    zpath = stage_report(tab, pd.DataFrame(extra_rows), ev_all, mc_all, meas_all, nea_res, sets, out, proj, time.time() - t0)
    return tab, ev_all, mc_all, meas_all, nea_res, zpath

def stage_report(tab, extra, ev_all, mc_all, meas_all, nea_res, sets, out: Path, proj: Path, elapsed=0.0):
    L = [f"# S5 report -- optimal bands, mission filter benchmark and NEA application ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')}, {elapsed / 60:.0f} min)", '',
         '## Verified mission filter sets (centre / FWHM, um)', '']
    for key, F in FILTERS_V.items(): L.append(f"- **{F['label']}**: " + ', '.join(f'{n} {c:.4f}/{f:.4f}' for n, (c, f) in zip(F['names'], F['bands'])) + f" — {F['ref']}")
    L += [f'- Note: {NOTE_GRID}', '', '## Optimal band sets (D-criterion on ilr1, ilr2, Fo, Fs; SNR 100 per band; free scale; mixing-error + library-mismatch covariance)', '', '| k | centres / FWHM (um) | design criterion |', '|---|---|---|']
    for key, s in sets.items():
        if s['family'] == 'opt': L.append(f"| {s['k']} | {', '.join(f'{c:.3f}/{f:.2f}' for c, f in s['bands'])} | {s['criterion_design']:.3f} |")
    L += ['', '## CRB+lib on the evaluation population (median over states; σ(M_ol) 90th percentile in brackets; D-criterion = mean log det of the target covariance, lower = better)', '', '| set | family | k | D (design) | D (evaluation) | σ(M_ol) | σ(M_plag) | σ(Fo) | σ(Fs) | σ(ln d) | DOFS |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in tab.sort_values(['k', 'family']).iterrows():
        L.append(f"| {r['label']} | {r['family']} | {int(r['k'])} | {S4._f(r.get('criterion_design', np.nan)):.3f} | {S4._f(r.get('criterion_eval', np.nan)):.3f} | {r['sigma_M_ol']:.3f} [{r['sigma_M_ol_p90']:.3f}] | {r['sigma_M_plag']:.3f} | {r['sigma_Fo']:.1f} | {r['sigma_Fs_opx']:.1f} | {r['sigma_logd']:.2f} | {r['dofs']:.2f} |")
    L += ['', '## Best extra filter(s) for each mission set (D-criterion before/after adding; sequential; unconstrained and silicon-detector-limited)', '', '| mission | constraint | order | added centre (um) | added FWHM (um) | criterion before | criterion after |', '|---|---|---|---|---|---|---|']
    for _, r in extra.iterrows(): L.append(f"| {r['mission']} | {r['constraint']} | {int(r['order'])} | {r['added_centre_um']:.3f} | {r['added_fwhm_um']:.2f} | {r['criterion_before']:.3f} | {r['criterion_after']:.3f} |")
    L += ['', '## Learned retrieval (primary model) on band sets: MAE on the synthetic test set and on the measured mixtures', '', '| set | model | target | MAE synthetic | MAE measured |', '|---|---|---|---|---|']
    for tag in dict.fromkeys(mc_all['tag']):
        g = mc_all[mc_all['tag'] == tag]
        for t in ('M_ol', 'M_plag', 'Fo', 'Fs_opx', 'logd'):
            a = g[(g['set'] == 'synthetic_test') & (g['target'] == t)]['mae']; b = g[(g['set'] == 'measured') & (g['target'] == t)]['mae']
            L.append(f"| {tag} | {g['model'].iloc[0]} | {t} | {float(a.iloc[0]) if len(a) else float('nan'):.3g} | {float(b.iloc[0]) if len(b) else float('nan'):.3g} |")
    L += ['', '## Conformal 90 % intervals on band sets (olivine mass fraction and Fo): coverage / median half-width', '', '| set | synthetic cov | synthetic hw | measured cov (syn. cal.) | measured hw | measured cov (LOSO) | measured hw (LOSO) | Fo measured cov (syn.) | Fo hw |', '|---|---|---|---|---|---|---|---|---|']
    for tag in dict.fromkeys(ev_all['tag']):
        g = ev_all[(ev_all['tag'] == tag) & (ev_all['alpha'] == 0.10)]
        def pick(sett, t, col):
            x = g[(g['set'] == sett) & (g['target'] == t)][col]; return float(x.iloc[0]) if len(x) else float('nan')
        L.append(f"| {tag} | {pick('synthetic_test', 'M_ol', 'coverage'):.2f} | {pick('synthetic_test', 'M_ol', 'median_halfwidth'):.3f} | {pick('measured_syncal', 'M_ol', 'coverage'):.2f} | {pick('measured_syncal', 'M_ol', 'median_halfwidth'):.3f} | {pick('measured_LOSO_recal', 'M_ol', 'coverage'):.2f} | {pick('measured_LOSO_recal', 'M_ol', 'median_halfwidth'):.3f} | {pick('measured_syncal', 'Fo', 'coverage'):.2f} | {pick('measured_syncal', 'Fo', 'median_halfwidth'):.1f} |")
    if nea_res: L += nea_report_lines(nea_res)
    L += ['', '## Gate checklist (S5)', '',
          '- [ ] optimal sets beat uniform / random / sensitivity / PCA sets under equal k and SNR (criterion and σ(M_ol), σ(Fo) on the evaluation population)',
          '- [ ] mission sets placed on the k-curve; best extra filter identified for each',
          '- [ ] learned retrieval on the optimal sets: measured MAE and conformal coverage at least as good as uniform/random at equal k',
          '- [ ] NEA retrievals consistent with the published mineralogy within the conformal intervals (or the discrepancy explained)']
    (out / 'S5_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    zpath = proj / 'S5_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in {'.csv', '.json', '.md', '.npz', '.png', '.pdf'}: zf.write(p, p.relative_to(proj))
    print(f"[R] report: {out / 'S5_REPORT.md'}\n[R] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath

def nea_report_lines(nea_res):
    rows, summ, chk = nea_res['rows'], nea_res['summary'], nea_res['checks']
    L = ['', '## NEA application (telescope mode; learned model + split-conformal 90 % regions; physics estimator; reference mineralogy)', '',
         'Observation mode per asteroid matched to the spectra (10-nm channels, SNR 50 + mixing-error covariance); the learned model is trained on the nuisance-aware synthetic population of Step 4. Regions with the synthetic calibration and, in brackets, with the 70 measured mixtures as the calibration set (the sim-to-real-aware choice). ol/(ol+px) compared with ol/(ol+opx+cpx) of the references (the model has no separate clinopyroxene).', '',
         '### Sim-to-real check of the NEA observation modes on the measured mixtures (synthetic calibration, nominal 0.90)', '', '| mode | model | coverage M_ol | coverage ol/(ol+px) | coverage Fo | MAE M_ol | MAE Fo | coverage M_ol (synthetic test) |', '|---|---|---|---|---|---|---|---|']
    for _, r in chk.iterrows(): L.append(f"| {r['mode']} | {r['model']} | {r['cov90_M_ol_measured']:.2f} | {r['cov90_olpx_measured']:.2f} | {r['cov90_Fo_measured']:.2f} | {r['mae_M_ol_measured']:.3f} | {r['mae_Fo_measured']:.1f} | {r['cov90_M_ol_synthetic']:.2f} |")
    L += ['', '### Summary per asteroid (median over spectra; the 90 % region is that of the median spectrum)', '', '| asteroid | n | ol/(ol+px) [90 %] (meas. cal.) | range over epochs | ref | Fo ± hw (meas. cal.) | ref Fo | Fs ± hw | ref Fs | plag | s | d (µm) | physics ol/(ol+px), Fo, Fs | χ²/n |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in summ.iterrows():
        L.append(f"| {r['asteroid']} | {int(r['n_spectra'])} | {r['olpx_median']:.2f} [{r['olpx_lo90']:.2f}, {r['olpx_hi90']:.2f}] ([{r['olpx_lo90_meascal']:.2f}, {r['olpx_hi90_meascal']:.2f}]) | {r['olpx_min']:.2f}–{r['olpx_max']:.2f} | {r['ref_olpx']:.2f} | {r['Fo_median']:.0f} ± {r['Fo_hw90']:.0f} ({r['Fo_hw90_meascal']:.0f}) | {r['ref_Fo']:.0f} | {r['Fs_median']:.0f} ± {r['Fs_hw90']:.0f} | {r['ref_Fs']:.0f} | {r['M_plag_median']:.2f} | {r['s_median']:.2f} | {r['d_um_median']:.0f} | {r['phys_olpx_median']:.2f}, {r['phys_Fo_median']:.0f}, {r['phys_Fs_median']:.0f} | {r['phys_chi2_median']:.2f} |")
    L += ['', '### Per spectrum', '', '| asteroid | spectrum | date | ol/(ol+px) [90 %] | Fo ± hw | Fs ± hw | plag | s | physics ol/(ol+px), Fo, Fs, s | χ²/n |', '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in rows.iterrows():
        L.append(f"| {r['asteroid']} | {r['spectrum']} | {r['date_obs']} | {r['olpx']:.2f} [{r['olpx_lo90']:.2f}, {r['olpx_hi90']:.2f}] | {r['Fo']:.0f} ± {r['hw90_Fo']:.0f} | {r['Fs']:.0f} ± {r['hw90_Fs']:.0f} | {r['M_plag']:.2f} | {r['s']:.2f} | {r['phys_olpx']:.2f}, {r['phys_Fo']:.0f}, {r['phys_Fs']:.0f}, {r['phys_s']:.2f} | {r['phys_chi2_per_channel']:.2f} |")
    nad = rows['adequate'] if 'adequate' in rows else pd.Series(False, index=rows.index)
    L += ['', f"Model adequacy (level 1): {int(nad.sum())} of {len(rows)} spectra pass the adequacy test (physics χ²/n ≤ 1.5 and no parameter at a bound); the full-spectrum regions are reported for completeness but are not applicable where the test fails."]
    if 'bp_rows' in nea_res and len(nea_res['bp_rows']):
        bp, ck, refs = nea_res['bp_rows'], nea_res['bp_check'], nea_res['bp_refs']
        L += ['', '### Level 2: retrieval from the band parameters (BI, BII, BAR) — the weathering-robust information', '',
              f"Learned model ({ck['model']}) + split conformal on the band parameters of the nuisance-aware population (continuum-slope augmentation U{BP_SLOPE} µm⁻¹, feature noise {BP_NOISE}); synthetic 90 % coverage of ol/(ol+px) {ck['cov90_olpx_synthetic']:.2f}.",
              f"Measured mixtures ({ck['n_measured']}): MAE ol/(ol+px) {ck['mae_olpx']:.3f}, coverage {ck['cov90_olpx']:.2f} (median half-width {ck['hw90_olpx_median']:.2f}); Fo MAE {ck['mae_Fo']:.1f}, coverage {ck['cov90_Fo']:.2f}; Fs MAE {ck['mae_Fs']:.1f}, coverage {ck['cov90_Fs']:.2f}; Spearman(BAR, ol/(ol+px)) = {ck['spearman_BAR_olpx_measured']:.2f}.", '',
              '| asteroid | spectrum | date | BI (µm) | BII (µm) | BAR | depth I | ol/(ol+px) [90 %] | ref | Fo ± hw | ref Fo | Fs ± hw | ref Fs |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in bp.iterrows(): L.append(f"| {r['asteroid']} | {r['spectrum'][:45]} | {r['date_obs']} | {r['BI']:.3f} | {r['BII']:.3f} | {r['BAR']:.2f} | {r['depthI']:.2f} | {r['olpx']:.2f} [{r['olpx_lo90']:.2f}, {r['olpx_hi90']:.2f}] | {r['ref_olpx']:.2f} | {r['Fo']:.0f} ± {r['hw90_Fo']:.0f} | {r['ref_Fo']:.0f} | {r['Fs']:.0f} ± {r['hw90_Fs']:.0f} | {r['ref_Fs']:.0f} |")
        L += ['', '| asteroid | n | ol/(ol+px) median [90 % of the median spectrum] | epoch range | ref | ref inside | Fo median ± hw | ref Fo | Fs median ± hw | ref Fs |', '|---|---|---|---|---|---|---|---|---|---|']
        for key, g in bp.groupby('key', sort=False):
            i = int(np.argsort(g['olpx'].values)[len(g) // 2]); gm = g.iloc[i]
            L.append(f"| {gm['asteroid']} | {len(g)} | {g['olpx'].median():.2f} [{gm['olpx_lo90']:.2f}, {gm['olpx_hi90']:.2f}] | {g['olpx'].min():.2f}–{g['olpx'].max():.2f} | {gm['ref_olpx']:.2f} | {'yes' if gm['olpx_lo90'] <= gm['ref_olpx'] <= gm['olpx_hi90'] else 'no'} | {g['Fo'].median():.0f} ± {g['hw90_Fo'].median():.0f} | {gm['ref_Fo']:.0f} | {g['Fs'].median():.0f} ± {g['hw90_Fs'].median():.0f} | {gm['ref_Fs']:.0f} |")
        L += ['', '### Band parameters of the model at the reference compositions (unweathered; depth excess = weathering/dilution not in the model)', '', '| asteroid | composition | d (µm) | BI | depth I | BII | depth II | BAR |', '|---|---|---|---|---|---|---|---|']
        for _, r in refs.iterrows(): L.append(f"| {r['asteroid']} | {r['composition']} | {int(r['d_um'])} | {r['BI']:.3f} | {r['depthI']:.2f} | {r['BII']:.3f} | {r['depthII']:.2f} | {r['BAR']:.2f} |")
    L += ['', 'References: ' + '; '.join(sorted(set(summ['ref_source']))) + '. Spectra: MITHNEOS (IRTF/SpeX; Binzel et al. 2019, Marsset et al. 2022) and SMASS (Binzel et al. 2001, 2004; Bus and Binzel 2002) archives at smass.mit.edu, as indexed by the classy package (Mahlke et al. 2022).']
    return L

def nea_application(model, out: Path, proj: Path, resid_npz, qc, Xspec, spec_ids, mix, em, nat, rule, inp_dir: Path, n_train=20000, n_scale=4000, n_cal=4000, n_test=2000, quick=False, seed=0, allow_download=True):
    """standalone NEA step (S5b): mismatch pools -> nuisance-aware population -> nea_stage -> S5b_NEA_REPORT.md + S5b_results.zip"""
    t0 = time.time(); (out / 'figures').mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed + 11)
    if quick: n_train, n_scale, n_cal, n_test = 1500, 400, 400, 300
    lib = S4.LibraryMismatch(model, qc, Xspec, spec_ids, em, nat, rule, exclude_codes=S4.test_codes(mix)); print('[S5b] mismatch pools:', lib.summary())
    print(f'[S5b] simulating the nuisance-aware population ({n_train + n_scale + n_cal + n_test} states) ...')
    st = {k_: S4.sample_states(n, rng) for k_, n in (('tr', n_train), ('sc', n_scale), ('cal', n_cal), ('te', n_test))}; data = {k_: simulate_R(model, v, rng, lib) for k_, v in st.items()}
    print(f'[S5b] simulated, {time.time() - t0:.0f}s')
    nea_res = nea_stage(model, out, data, qc, Xspec, spec_ids, mix, nat, rule, rng, inp_dir, resid_npz, quick=quick, seed=seed, allow_download=allow_download)
    L = [f"# S5b report -- NEA application ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')}, {(time.time() - t0) / 60:.0f} min)"]
    L += nea_report_lines(nea_res) if nea_res else ['', 'No NEA spectra could be obtained (see the log): turn Internet ON, or add eros*.txt / itokawa*.txt (wavelength, reflectance) to the inputs dataset.']
    L += ['', '## Gate item (S5)', '', '- [ ] NEA retrievals consistent with the published mineralogy within the conformal regions (or the discrepancy explained)']
    (out / 'S5b_NEA_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    json.dump({'version': VERSION, 'n_train': n_train, 'n_scale': n_scale, 'n_cal': n_cal, 'n_test': n_test, 'seed': seed, 'mismatch_pools': lib.summary(), 'nea_refs': {k: v['ref'] for k, v in NEA.items()}}, open(out / 'run_config_s5b.json', 'w'), indent=1)
    zpath = proj / 'S5b_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p_ in sorted(out.rglob('*')):
            if p_.is_file() and p_.suffix in {'.csv', '.json', '.md', '.png', '.pdf'}: zf.write(p_, p_.relative_to(proj))
    print(f"[R] report: {out / 'S5b_NEA_REPORT.md'}\n[R] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return nea_res, zpath
