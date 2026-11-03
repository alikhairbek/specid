#!/usr/bin/env python3
"""
forward_param.py -- continuous (parametric) forward model for olivine-pyroxene-plagioclase assemblages.

State theta (dict):
  M      : mass fractions {'ol', 'opx', 'cpx', 'plag'} (sum 1; components absent -> 0)
  Fo     : olivine forsterite number (0-100)
  Fs_opx : orthopyroxene ferrosilite number (0-100; Wo fixed at WO_OPX)
  Wo_cpx, Fs_cpx : clinopyroxene composition (quadrilateral coordinates; used only when cpx_natural is False)
  l_plag, l_cpx (l_ol, l_opx) : log-level factors on k of the plagioclase / natural-augite (olivine / opx) components -- nuisance axes
                   for the spread of absorption strength among natural samples (0 = library median)
  d      : effective grain diameter (um), common to all components, or dict per component
  s      : space-weathering exposure (0 = fresh; 1 = one pulse-laser dose of the Hiroi pellets; T(lambda)^s)
Observation: REFF at geometry (i, e) through the isotropic Hapke model (reff_from_w), optionally normalised at a wavelength,
optionally integrated over instrument bands (list of (centre, FWHM) in um; Gaussian response).

Optical constants k(lambda; composition) come from s3a_library.npz (Step 3a): olivine Fo-series (Dyar et al. 2009 synthetic
olivines), pyroxene quadrilateral (Klima et al. 2007, 2011 synthetic pyroxenes), natural anorthite, natural diopside/augite;
grain size enters through the Hapke (1993) slab approximation of the single-scattering albedo; the weathering transfer
function T(lambda) is the pulse-laser response measured on olivine-bronzite pellets (RELAB XT-TXH series).
"""
from __future__ import annotations
import json, math
import numpy as np

WO_OPX = 2.0                                   # Wo of the orthopyroxene component
OL_SMOOTH_DEG = 0                              # extra smoothing of the olivine node library at load time (0 = none; the v1.1 library nodes are already smoothed in Step 3a)
DENSITY = {'ol': 3.32, 'opx': 3.35, 'cpx': 3.30, 'plag': 2.72}
_FAMS = ('ol', 'opx', 'cpx', 'plag')

# ----------------------------------------------------------------------------- Hapke: bidirectional reflectance + slab albedo
def H2002(w, x):
    g = np.sqrt(np.clip(1.0 - w, 1e-12, 1.0)); r0 = (1 - g) / (1 + g)
    return 1.0 / (1.0 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log((1 + x) / x)))

def reff_from_w(w, mu0, mu):
    return (w / 4.0) * H2002(w, mu0) * H2002(w, mu) / (mu0 + mu)

def w_from_reff(R, mu0, mu, iters=50):
    R = np.asarray(R, float); ok = np.isfinite(R); Rc = np.where(ok, np.clip(R, 1e-6, None), 0.5)
    rmax = reff_from_w(np.full_like(Rc, 1 - 1e-9), mu0, mu); Rc = np.minimum(Rc, rmax * 0.999999)
    lo = np.zeros_like(Rc); hi = np.ones_like(Rc) - 1e-9
    for _ in range(iters):
        mid = 0.5 * (lo + hi); f = reff_from_w(mid, mu0, mu); lo = np.where(f < Rc, mid, lo); hi = np.where(f >= Rc, mid, hi)
    return np.where(ok, 0.5 * (lo + hi), np.nan)

def mus(inc_deg, emi_deg):
    return math.cos(math.radians(float(inc_deg))), math.cos(math.radians(float(emi_deg)))

def slab_w(k, n, d_um, lam_um):
    """Hapke (1993, eq. 6.20) single-scattering albedo of an equant particle of diameter d with optical constants n, k (no internal scatterers)"""
    k = np.asarray(k, float); Se = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2) + 0.05
    Si = 1.0 - 4.0 / (n * (n + 1) ** 2)
    D = (2.0 / 3.0) * (n ** 2 - (1.0 / n) * (n ** 2 - 1) ** 1.5) * d_um
    Th = np.exp(-4 * np.pi * k / lam_um * D)
    return Se + (1 - Se) * (1 - Si) * Th / (1 - Si * Th)

def slab_k(w, n, d_um, lam_um, iters=3):
    """invert slab_w for k (iterating the weak k-dependence of Se); w clipped to the physical range"""
    w = np.asarray(w, float); Si = 1.0 - 4.0 / (n * (n + 1) ** 2); D = (2.0 / 3.0) * (n ** 2 - (1.0 / n) * (n ** 2 - 1) ** 1.5) * d_um
    k = np.zeros_like(w)
    for _ in range(iters):
        Se = ((n - 1) ** 2 + k ** 2) / ((n + 1) ** 2 + k ** 2) + 0.05
        wc = np.clip(w, Se + 1e-6, 1 - 1e-6)
        Th = (wc - Se) / ((1 - Se) * (1 - Si) + Si * (wc - Se)); Th = np.clip(Th, 1e-12, 1 - 1e-12)
        k = -lam_um * np.log(Th) / (4 * np.pi * D)
    return k

def n_of(fam, chem=None):
    """real refractive index (weak composition dependence): olivine 1.635 (Fo100) -> 1.86 (Fo0); pyroxene 1.65 (En) -> 1.77 (Fs); cpx 1.70; anorthite 1.58"""
    if fam == 'ol': return 1.635 + 0.00225 * (100.0 - (chem if chem is not None else 90.0))
    if fam == 'opx': return 1.650 + 0.0012 * (chem if chem is not None else 15.0)
    if fam == 'cpx': return 1.70
    if fam == 'plag': return 1.58
    return 1.65

# ----------------------------------------------------------------------------- simplex coordinates (ILR) for the abundance block
def ilr(M):
    M = np.asarray(M, float); M = np.clip(M, 1e-6, None); M = M / M.sum(); D = len(M); y = np.log(M)
    return np.array([math.sqrt(j / (j + 1)) * (y[:j].mean() - y[j]) for j in range(1, D)])

def ilr_inv(z):
    z = np.asarray(z, float); D = len(z) + 1; y = np.zeros(D)
    for j in range(1, D):
        c = math.sqrt(j / (j + 1)); y[:j] += c * z[j - 1] / j; y[j] -= c * z[j - 1]
    e = np.exp(y - y.max()); return e / e.sum()

# ----------------------------------------------------------------------------- the model
class ParamModel:
    def __init__(self, lib_path):
        z = np.load(lib_path, allow_pickle=True); self.grid = z['grid_um']; self.meta = json.loads(str(z['meta']))
        self.ol_B = z['ol_B']; self.ol_offset = z['ol_offset']                      # (3, n_wl) quadratic-in-Fe coefficients + level offset (fallback model)
        self.ol_nodes_Fo = z['ol_nodes_Fo'] if 'ol_nodes_Fo' in z else None; self.ol_nodes_logk = z['ol_nodes_logk'] if 'ol_nodes_logk' in z else None
        self.ol_poly = None
        if self.ol_nodes_Fo is not None and OL_SMOOTH_DEG > 0:           # smooth the node library: weighted polynomial in Fo per wavelength (nodes <= 95; the natural Fo-90 node weighted x3)
            sel = self.ol_nodes_Fo <= 95.0; w = np.sqrt(np.where(self.ol_nodes_Fo[sel] == 90.0, 3.0, 1.0))[:, None]
            V = np.vander(self.ol_nodes_Fo[sel] / 100.0, OL_SMOOTH_DEG + 1, increasing=True); self.ol_poly, *_ = np.linalg.lstsq(V * w, self.ol_nodes_logk[sel] * w, rcond=None)
        self.px_WoFs = z['px_WoFs']; self.px_logk = z['px_logk']; self.px_off = {'opx': z['px_offset_opx'], 'cpx': z['px_offset_cpx']}
        self.plag_k = z['plag_k']; self.cpx_nat_k = z['cpx_nat_k']; self.T = z['T_weather']
        self.T = np.where(np.isfinite(self.T), self.T, 1.0)
        from scipy.interpolate import RBFInterpolator
        self._rbf = RBFInterpolator(self.px_WoFs / 100.0, self.px_logk, kernel='thin_plate_spline', smoothing=float(self.meta.get('px_smoothing', 1e-1)))
        self.cache = {}

    # --- optical constants
    def k_olivine(self, Fo):
        if self.ol_poly is not None:                                             # smoothed node library (v1.1): cubic in Fo per wavelength, valid Fo 0-95
            return np.exp(np.vander(np.array([float(np.clip(Fo, 0.0, 95.0)) / 100.0]), self.ol_poly.shape[0], increasing=True) @ self.ol_poly)[0]
        if self.ol_nodes_Fo is not None:                                         # node library (v1.1): piecewise-linear log k in Fo, natural Fo-90 node
            nF, nK = self.ol_nodes_Fo, self.ol_nodes_logk; Fo = float(np.clip(Fo, nF.min(), nF.max())); j = min(max(int(np.searchsorted(nF, Fo)), 1), len(nF) - 1)
            tt = (Fo - nF[j - 1]) / (nF[j] - nF[j - 1]); return np.exp((1 - tt) * nK[j - 1] + tt * nK[j])
        x = (100.0 - float(np.clip(Fo, 0.0, 95.0))) / 100.0
        return np.exp(self.ol_B[0] + self.ol_B[1] * x + self.ol_B[2] * x ** 2 + self.ol_offset)

    def k_pyroxene(self, Wo, Fs, fam='opx'):
        key = ('px', fam, float(Wo), float(Fs))                     # exact keys: finite-difference steps must not hit the cache
        v = self.cache.get(key)
        if v is None:
            v = np.exp(self._rbf(np.array([[np.clip(Wo, 0, 55) / 100.0, np.clip(Fs, 5, 100) / 100.0]]))[0] + self.px_off[fam])
            if len(self.cache) > 5000: self.cache.clear()
            self.cache[key] = v
        return v

    # --- component single-scattering albedos
    def w_component(self, fam, theta, d):
        if fam == 'ol': return slab_w(self.k_olivine(theta.get('Fo', 90.0)) * math.exp(theta.get('l_ol', 0.0)), n_of('ol', theta.get('Fo', 90.0)), d, self.grid)
        if fam == 'opx': return slab_w(self.k_pyroxene(WO_OPX, theta.get('Fs_opx', 15.0), 'opx') * math.exp(theta.get('l_opx', 0.0)), n_of('opx', theta.get('Fs_opx', 15.0)), d, self.grid)
        if fam == 'cpx':
            k = self.cpx_nat_k if theta.get('cpx_natural', True) else self.k_pyroxene(theta.get('Wo_cpx', 45.0), theta.get('Fs_cpx', 15.0), 'cpx')
            return slab_w(k * math.exp(theta.get('l_cpx', 0.0)), n_of('cpx'), d, self.grid)
        if fam == 'plag': return slab_w(self.plag_k * math.exp(theta.get('l_plag', 0.0)), n_of('plag'), d, self.grid)
        raise KeyError(fam)

    def w_mix(self, theta):
        M = theta['M']; d = theta.get('d', 60.0); fams = [f for f in _FAMS if M.get(f, 0) > 0]
        dd = {f: (d[f] if isinstance(d, dict) else d) for f in fams}
        F = np.array([M[f] / (DENSITY[f] * dd[f]) for f in fams]); F = F / F.sum()
        w = sum(Fi * self.w_component(f, theta, dd[f]) for Fi, f in zip(F, fams))
        s = theta.get('s', 0.0)
        if s: w = np.clip(w * self.T ** s, 1e-6, 1 - 1e-6)
        return w

    def spectrum(self, theta, inc=30.0, emi=0.0, mode='reff', norm_wl=0.55, bands=None):
        """REFF on the grid ('reff'), normalised at norm_wl ('norm'); bands -> list of (centre, FWHM) um -> band-integrated values"""
        mu0, mu = mus(inc, emi); R = reff_from_w(self.w_mix(theta), mu0, mu)
        if bands is not None:
            out = []
            for c, fw in bands:
                g = np.exp(-0.5 * ((self.grid - c) / (fw / 2.3548)) ** 2); out.append(float(np.sum(g * R) / np.sum(g)))
            R = np.array(out)
            if mode == 'norm': R = R / R[int(np.argmin([abs(c - norm_wl) for c, _ in bands]))]
            return R
        if mode == 'norm': R = R / np.interp(norm_wl, self.grid, R)
        return R

# ----------------------------------------------------------------------------- vector parameterisation for fitting / Fisher analysis
class ParamVector:
    """maps a free-parameter list to/from a vector: 'ilr' (abundance block, D-1 coordinates), 'Fo', 'Fs_opx', 'Wo_cpx', 'Fs_cpx', 'logd', 's'"""
    def __init__(self, fams, free, base):
        self.fams = list(fams); self.free = list(free); self.base = dict(base)
    def pack(self, theta):
        v = []
        for p in self.free:
            if p == 'ilr': v += list(ilr([theta['M'].get(f, 0.0) for f in self.fams]))
            elif p == 'logd': v.append(math.log(theta['d'] if not isinstance(theta['d'], dict) else np.mean(list(theta['d'].values()))))
            else: v.append(float(theta[p]))
        return np.array(v)
    def unpack(self, v):
        theta = json.loads(json.dumps(self.base, default=float)); i = 0
        for p in self.free:
            if p == 'ilr':
                D = len(self.fams); M = ilr_inv(v[i:i + D - 1]); theta['M'] = {f: float(m) for f, m in zip(self.fams, M)}; i += D - 1
            elif p == 'logd': theta['d'] = float(math.exp(v[i])); i += 1
            else: theta[p] = float(v[i]); i += 1
        return theta
    def bounds(self):
        lo, hi = [], []
        for p in self.free:
            if p == 'ilr': lo += [-6.0] * (len(self.fams) - 1); hi += [6.0] * (len(self.fams) - 1)
            elif p == 'logd': lo.append(math.log(3.0)); hi.append(math.log(1000.0))
            elif p in ('Fo',): lo.append(0.0); hi.append(95.0)
            elif p in ('Fs_opx', 'Fs_cpx'): lo.append(5.0); hi.append(100.0)
            elif p == 'Wo_cpx': lo.append(0.0); hi.append(55.0)
            elif p == 's': lo.append(0.0); hi.append(3.0)
            elif p in ('l_plag', 'l_cpx', 'l_ol', 'l_opx'): lo.append(-2.0); hi.append(2.0)
        return np.array(lo), np.array(hi)
    def names(self):
        out = []
        for p in self.free:
            if p == 'ilr': out += [f'ilr{j}' for j in range(1, len(self.fams))]
            else: out.append(p)
        return out

def fit_spectrum(model, y, pv: ParamVector, v0, inc=30.0, emi=0.0, mode='reff', norm_wl=0.55, bands=None, mask=None, sigma=None):
    """weighted least-squares fit of the parametric model to a measured spectrum y (on the model grid or on bands)"""
    from scipy.optimize import least_squares
    lo, hi = pv.bounds(); v0 = np.clip(v0, lo + 1e-6, hi - 1e-6)
    m = np.ones_like(y, bool) if mask is None else mask; sg = np.ones_like(y) if sigma is None else sigma
    def resid(v):
        yy = model.spectrum(pv.unpack(v), inc, emi, mode, norm_wl, bands); return ((yy - y) / sg)[m]
    r = least_squares(resid, v0, bounds=(lo, hi), method='trf', x_scale='jac', max_nfev=400)
    return pv.unpack(r.x), float(np.sqrt(np.mean(r.fun ** 2))), r
