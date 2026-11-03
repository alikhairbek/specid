#!/usr/bin/env python3
"""
s2_forward.py -- Step 2 (M1) of the spectral-identifiability project:
laboratory-mixture benchmark assembly + forward-model (mixing-model) validation.

Self-contained (no import of s1_audit.py).  Stages, all resumable, outputs under <PROJ>/s2/:
  A  tables      : read the curated benchmark tables (mixtures, end-members, spectra) and print the inventory
  B  fetch       : download the benchmark .tab spectra from the PDS RELAB bundle (MD5-verified) unless present locally
  C  load        : common grid 0.40-2.50 um (5 nm), QC table, gallery PDF (one page per series)
  D/E forward    : linear (areal) and Hapke (intimate) mixing from the documented fractions -> spectral errors;
                   NNLS abundance retrieval in reflectance space (linear) and single-scattering-albedo space (Hapke)
  F  bands       : Band I centre, band depths, Band Area Ratio vs olivine fraction (Cloutis-1986-type calibration)
  H  photometric : single-scattering-albedo invariance with viewing geometry + Hapke mixing vs phase angle
  I  report      : S2_REPORT.md (+ gate checklist) and S2_results.zip

Physics.  RELAB bidirectional spectra are reflectance factors REFF(i, e) = pi * r / mu0 (ratio to a perfect
Lambertian surface at the same geometry).  With Hapke's isotropic-scatterer bidirectional reflectance
r = (w / 4 pi) * mu0 / (mu0 + mu) * H(mu0) H(mu)  (no opposition surge, isotropic phase function; the form used by
Mustard & Pieters 1989 for i = 30 deg, e = 0 deg), REFF = (w / 4) H(mu0) H(mu) / (mu0 + mu), with H the Hapke (2002)
approximation.  An intimate mixture has w_mix = sum_i F_i w_i with relative cross-section weights
F_i = (M_i / (rho_i d_i)) / sum_j (M_j / (rho_j d_j))  (Hapke 1981): M = mass fraction, rho = density, d = mean grain size.
Linear (areal) mixing uses the same weights on REFF.  Volume-basis mixtures use F_i ~ V_i / d_i.
"""
from __future__ import annotations
import hashlib, io, json, math, os, re, sys, time, zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

VERSION = 's2-forward/1.0'
PDS_RELAB_BASE = 'https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/'
RELAB_MANIFEST = 'urn-nasa-pds-relab.md5'
USER_AGENT = f'spectral-identifiability/{VERSION} (academic research pipeline; identifiability of mineral mixtures from VNIR reflectance)'
GRID = np.round(np.arange(0.40, 2.5001, 0.005), 4)        # common grid, um (421 channels)
EVAL = (GRID >= 0.45) & (GRID <= 2.45)                    # evaluation window for spectral errors
WONG = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']
INK = '#333333'                                           # dark grey for axes / text (no black plot elements)
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
DENSITY_DEFAULT = {'olivine': 3.32, 'opx': 3.35, 'cpx': 3.30, 'plagioclase': 2.72, 'magnetite': 5.18, 'ilmenite': 4.72, 'metal': 7.87,
                   'glass_basaltic': 2.80, 'chromite': 4.79, 'hematite': 5.26, 'troilite': 4.65, 'pyrrhotite': 4.62, 'graphite': 2.23,
                   'lampblack': 1.90, 'spinel': 3.64, 'quartz': 2.65, 'basalt': 2.90, 'palagonite': 2.30, 'diaspore': 3.38}
OPAQUE_FAMS = {'magnetite', 'ilmenite', 'metal', 'chromite', 'lampblack', 'graphite', 'troilite', 'pyrrhotite', 'hematite', 'coal_tar',
               'bitumen', 'wustite', 'cementite', 'sic', 'diamond'}

def style():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                         'axes.prop_cycle': matplotlib.cycler(color=WONG), 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 9, 'savefig.dpi': 300, 'figure.dpi': 100})
    return plt

# ----------------------------------------------------------------------------- HTTP (same as s1_audit)
def make_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(['GET', 'HEAD']))
    s.mount('http://', HTTPAdapter(max_retries=retry, pool_maxsize=16)); s.mount('https://', HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers.update({'User-Agent': USER_AGENT}); return s

def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def download_file(session, url: str, dest: Path, expected_md5=None, timeout: int = 120) -> str:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_md5:
            if md5_of(dest) == expected_md5.lower(): return 'skip_verified'
            dest.unlink()
        else:
            return 'skip_exists'
    tmp = dest.with_suffix(dest.suffix + '.part')
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200: return f'http_error:{r.status_code}'
            with open(tmp, 'wb') as fh:
                for chunk in r.iter_content(chunk_size=1 << 16): fh.write(chunk)
        if expected_md5 and md5_of(tmp) != expected_md5.lower():
            tmp.unlink(missing_ok=True); return 'md5_mismatch'
        tmp.replace(dest); return 'ok'
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True); return f'error:{type(exc).__name__}'

_MD5_LINE = re.compile(r'^([0-9a-fA-F]{32})\s+\*?(.+)$'); _MD5_BSD = re.compile(r'^MD5 \((.+)\) = ([0-9a-fA-F]{32})$')

def parse_md5_manifest(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        m = _MD5_LINE.match(line)
        if m: digest, path = m.group(1), m.group(2)
        else:
            m = _MD5_BSD.match(line)
            if not m: continue
            path, digest = m.group(1), m.group(2)
        out[path.replace('\\', '/').lstrip('./')] = digest.lower()
    return out

def get_manifest(session, raw: Path) -> dict:
    cache = raw / RELAB_MANIFEST
    if cache.exists() and cache.stat().st_size > 1000: return parse_md5_manifest(cache.read_text())
    try:
        r = session.get(PDS_RELAB_BASE + RELAB_MANIFEST, timeout=180); r.raise_for_status()
    except Exception as exc:
        print(f'[B] WARNING: could not fetch the PDS manifest ({type(exc).__name__}); continuing without MD5 verification'); return {}
    raw.mkdir(parents=True, exist_ok=True); cache.write_text(r.text); return parse_md5_manifest(r.text)

def fetch_many(session, rels, manifest, raw: Path, workers: int = 8, label: str = '') -> Counter:
    counts = Counter(); t0 = time.time()
    def task(rel):
        st = download_file(session, PDS_RELAB_BASE + rel, raw / rel, manifest.get(rel))
        if st == 'md5_mismatch': st = download_file(session, PDS_RELAB_BASE + rel, raw / rel, manifest.get(rel))
        return st
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(task, rel) for rel in rels]
        for i, f in enumerate(as_completed(futs), 1):
            counts[f.result()] += 1
            if i % 200 == 0 or i == len(rels): print(f'      {label} {i}/{len(rels)}  {time.time() - t0:.0f}s  {dict(counts)}', flush=True)
    return counts

def find_local_bundle(hint: str = '') -> Path | None:
    cands = []
    if hint: cands.append(Path(hint))
    for root in ['/kaggle/input', '/content', '/content/drive/MyDrive', '.']:
        rp = Path(root)
        if rp.exists():
            cands += [p.parent for p in rp.glob('**/data_reflectance/bdr2') if p.is_dir()][:20]
            cands += [p.parent for p in rp.glob('**/bdr2') if p.is_dir() and p.parent.name != 'data_reflectance'][:20]
    for c in cands:
        if (c / 'data_reflectance' / 'bdr2').is_dir(): return c / 'data_reflectance'
        if (c / 'bdr2').is_dir(): return c
    return None

def parse_tab(path: Path):
    wl, rf = [], []
    for line in path.read_text(errors='replace').splitlines():
        parts = line.replace(',', ' ').split()
        if len(parts) < 2: continue
        try: w, r = float(parts[0]), float(parts[1])
        except ValueError: continue
        wl.append(w); rf.append(r)
    if not wl: return None, None
    wl = np.asarray(wl); rf = np.asarray(rf)
    if np.nanmedian(wl) > 50: wl = wl / 1000.0
    keep = (rf > -0.05) & (rf < 2.0)                       # drops header-like lines ("400 2500") and fill values
    wl, rf = wl[keep], rf[keep]
    if len(wl) < 2: return None, None
    o = np.argsort(wl); return wl[o], rf[o]

# ----------------------------------------------------------------------------- band parameters (as in s1_audit)
def continuum_removed(wl, rf, lo, hi):
    def peak(a, b):
        m = (wl >= a) & (wl <= b) & np.isfinite(rf); i = np.where(m)[0]
        return i[np.argmax(rf[i])] if len(i) else None
    i0, i1 = peak(lo - 0.10, lo + 0.10), peak(hi - 0.15, hi + 0.15)
    if i0 is None or i1 is None or i1 <= i0 + 5: return None
    seg = slice(i0, i1 + 1); cont = rf[i0] + (rf[i1] - rf[i0]) * (wl[seg] - wl[i0]) / (wl[i1] - wl[i0])
    return wl[seg], rf[seg] / cont

def band_params(wl, rf):
    outp = {}; b1 = continuum_removed(wl, rf, 0.75, 1.50); b2 = continuum_removed(wl, rf, 1.55, 2.42)
    if b1 is not None:
        w, c = b1; outp['b1_center'] = float(w[np.argmin(c)]); outp['b1_depth'] = float(1 - c.min()); outp['b1_area'] = float(_trapz(1 - c, w))
    if b2 is not None:
        w, c = b2; outp['b2_center'] = float(w[np.argmin(c)]); outp['b2_depth'] = float(1 - c.min()); outp['b2_area'] = float(_trapz(1 - c, w))
    if 'b1_area' in outp and 'b2_area' in outp and outp['b1_area'] > 0: outp['BAR'] = outp['b2_area'] / outp['b1_area']
    return outp

# ----------------------------------------------------------------------------- Hapke
def H2002(w, x):
    g = np.sqrt(np.clip(1.0 - w, 1e-12, 1.0)); r0 = (1 - g) / (1 + g)
    return 1.0 / (1.0 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log((1 + x) / x)))

def reff_from_w(w, mu0, mu):
    return (w / 4.0) * H2002(w, mu0) * H2002(w, mu) / (mu0 + mu)

def w_from_reff(R, mu0, mu, iters=50):
    """invert REFF -> single-scattering albedo by bisection (REFF is monotonic in w); NaN in -> NaN out"""
    R = np.asarray(R, float); ok = np.isfinite(R); Rc = np.where(ok, np.clip(R, 1e-6, None), 0.5)
    rmax = reff_from_w(np.full_like(Rc, 1 - 1e-9), mu0, mu); Rc = np.minimum(Rc, rmax * 0.999999)
    lo = np.zeros_like(Rc); hi = np.ones_like(Rc) - 1e-9
    for _ in range(iters):
        mid = 0.5 * (lo + hi); f = reff_from_w(mid, mu0, mu); lo = np.where(f < Rc, mid, lo); hi = np.where(f >= Rc, mid, hi)
    w = 0.5 * (lo + hi); return np.where(ok, w, np.nan)

def mus(inc_deg, emi_deg):
    return math.cos(math.radians(float(inc_deg))), math.cos(math.radians(float(emi_deg)))

def _f(x, default=0.0):
    try:
        v = float(x); return default if not np.isfinite(v) else v
    except (TypeError, ValueError):
        return default

def mean_size(lo, hi):
    """mean grain diameter (um) of a sieve fraction; unstated -> 60 um (the 45-75 um fraction typical of this benchmark)"""
    lo, hi = _f(lo), _f(hi)
    if hi <= 0: return 60.0
    if lo <= 0: return hi / 2.0
    return 0.5 * (lo + hi)

def xsec_weights(comps, density, basis):
    """relative cross-section weights F_i (sum 1): mass basis -> M/(rho d); volume basis -> V/d"""
    F = []
    for c in comps:
        d = mean_size(c.get('size_lo'), c.get('size_hi')); rho = density.get(c['fam'], 3.0)
        F.append(c['frac'] / d / (1.0 if basis == 'vol' else rho))
    F = np.array(F, float); return F / F.sum()

def F_to_basis(F, comps, density, basis):
    """convert retrieved cross-section weights back to the documented basis (mass or volume fractions)"""
    out = np.array([f * mean_size(c.get('size_lo'), c.get('size_hi')) * (1.0 if basis == 'vol' else density.get(c['fam'], 3.0)) for f, c in zip(F, comps)])
    return out / out.sum() if out.sum() > 0 else out

def nnls(A, b):
    from scipy.optimize import nnls as _n
    x, _ = _n(A, b); return x

# ----------------------------------------------------------------------------- stage A: tables + inventory
def stage_tables(bench: Path, out: Path):
    mix = pd.read_csv(bench / 'benchmark_mixtures_v0.csv'); em = pd.read_csv(bench / 'benchmark_endmembers_v0.csv'); sp = pd.read_csv(bench / 'benchmark_spectra_v0.csv')
    mix['flags'] = mix['flags'].fillna(''); mix['basis'] = mix['basis'].replace({'wt_from_mass': 'wt'})
    density = dict(DENSITY_DEFAULT); density.update({r['family']: float(r['density_g_cm3']) for _, r in em.iterrows() if np.isfinite(_f(r['density_g_cm3'], np.nan))})
    rows = []
    for s, g in mix.groupby('series', sort=False):
        comps = [c for js in g['components'] for c in json.loads(js)]
        sizes = sorted({(round(_f(c.get('size_lo')), 1), round(_f(c.get('size_hi')), 1)) for c in comps})
        flags = sorted({f.split('(')[0] for fl in g['flags'] for f in fl.split(';') if f})
        rows.append({'series': s, 'reference': g['ref'].iloc[0], 'n_mixtures': len(g), 'tiers': ''.join(sorted(set(g['tier']))), 'n_components': '/'.join(str(x) for x in sorted(set(g['n_components']))),
                     'basis': '/'.join(sorted(set(g['basis']))), 'families': '; '.join(sorted({c['fam'] for c in comps})), 'end_members': '; '.join(sorted({c['code'] for c in comps})),
                     'size_fractions_um': '; '.join(f'{a:g}-{b:g}' for a, b in sizes), 'geometry_i_e': '; '.join(f'{int(a)}/{int(b)}' for a, b in sorted({(r.inc_deg, r.emi_deg) for r in g.itertuples()})),
                     'flags': '; '.join(flags), 'n_full_range': int((g['range_max_um'] >= 2.45).sum())})
    inv = pd.DataFrame(rows); inv.to_csv(out / 'benchmark_inventory.csv', index=False)
    print(f"[A] benchmark: {len(mix)} mixtures in {mix['series'].nunique()} series (tiers {mix['tier'].value_counts().to_dict()}); "
          f"{len(em)} end-members ({int((em['status'] != 'ok').sum())} without a RELAB spectrum); {len(sp)} spectra to load ({sp['role'].value_counts().to_dict()})")
    return mix, em, sp, density, inv

# ----------------------------------------------------------------------------- stage B: fetch
def stage_fetch(sp: pd.DataFrame, raw: Path, workers: int, local_hint: str = ''):
    local = find_local_bundle(local_hint); roots = [raw / 'data_reflectance'] + ([local] if local else [])
    def tab_path(r):
        for root in roots:
            p = root / r['collection'] / r['tab_file']
            if p.exists(): return p
        return raw / 'data_reflectance' / r['collection'] / r['tab_file']
    missing = [r for _, r in sp.iterrows() if not tab_path(r).exists()]
    if missing and os.environ.get('S2_OFFLINE'):
        print(f'[B] S2_OFFLINE set: {len(missing)} spectra not on disk will be skipped')
    elif missing:
        session = make_session(); manifest = get_manifest(session, raw)
        rels = [f"data_reflectance/{r['collection']}/{r['tab_file']}" for r in missing]
        if manifest:
            absent = [p for p in rels if p not in manifest]
            if absent: print(f'[B] {len(absent)} files are not in the PDS manifest (skipped): {absent[:5]} ...')
            rels = [p for p in rels if p in manifest]
        print(f'[B] downloading {len(rels)} .tab files from {PDS_RELAB_BASE} ...')
        counts = fetch_many(session, rels, manifest, raw, workers, 'spectra'); print('[B] download status:', dict(counts))
    else:
        print(f'[B] all {len(sp)} benchmark spectra already on disk' + (f' (local bundle {local})' if local else ''))
    return {r['spectrum_id']: tab_path(r) for _, r in sp.iterrows()}

# ----------------------------------------------------------------------------- stage C: load + QC + gallery
def stage_load(sp: pd.DataFrame, paths: dict, out: Path):
    rows, X = [], []
    for _, r in sp.iterrows():
        p = paths[r['spectrum_id']]; row = {'spectrum_id': r['spectrum_id']}
        if not p.exists():
            row['status'] = 'file_missing'; rows.append(row); X.append(np.full(len(GRID), np.nan)); continue
        wl, rf = parse_tab(p)
        if wl is None or len(wl) < 20:
            row['status'] = 'parse_failed'; rows.append(row); X.append(np.full(len(GRID), np.nan)); continue
        y = np.interp(GRID, wl, rf, left=np.nan, right=np.nan)
        full = wl.min() <= GRID[0] + 0.011 and wl.max() >= GRID[-1] - 0.011
        nbad = int(np.sum(~np.isfinite(y[EVAL])) + np.sum(y[EVAL] <= 0))
        row.update({'status': 'ok' if (full and nbad == 0) else ('partial' if not full else 'nonpositive'), 'wl_min': round(float(wl.min()), 3), 'wl_max': round(float(wl.max()), 3),
                    'n_points': int(len(wl)), 'step_nm': round(float(np.median(np.diff(wl))) * 1000, 2), 'r055': round(float(np.interp(0.55, wl, rf)), 4),
                    'r105': round(float(np.interp(1.05, wl, rf)), 4), 'r150': round(float(np.interp(1.50, wl, rf)), 4), 'r240': round(float(np.interp(2.40, wl, rf)), 4), 'n_bad': nbad})
        rows.append(row); X.append(y)
    qc = sp.merge(pd.DataFrame(rows), on='spectrum_id', how='left'); qc.to_csv(out / 'spectra_qc.csv', index=False)
    X = np.vstack(X); np.savez_compressed(out / 'benchmark_matrix.npz', grid_um=GRID, X=X, spectrum_id=np.array(list(sp['spectrum_id'])))
    print('[C] load status:', qc['status'].value_counts().to_dict())
    return qc, X

def stage_gallery(mix, em, qc, X, out: Path):
    plt = style(); from matplotlib.backends.backend_pdf import PdfPages
    pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; okset = set(qc.loc[qc['status'].isin(['ok', 'partial']), 'spectrum_id']); emmap = em.set_index('code')
    (out / 'figures').mkdir(exist_ok=True)
    with PdfPages(out / 'figures' / 'S2_series_gallery.pdf') as pdf:
        for series, grp in mix.groupby('series', sort=False):
            codes = sorted({c['code'] for s in grp['components'] for c in json.loads(s)})
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
            for k, code in enumerate(codes):
                if code in emmap.index and emmap.loc[code, 'primary_spectrum'] in okset:
                    axes[0].plot(GRID, X[pos[emmap.loc[code, 'primary_spectrum']]], color=WONG[k % 7], lw=1.5, ls=['-', '--', ':', '-.'][(k // 7) % 4], label=f"{code} ({emmap.loc[code, 'family']})")
            axes[0].set_title(f'{series}: end-members')
            if axes[0].get_legend_handles_labels()[0]: axes[0].legend(fontsize=6, frameon=False, ncol=1 + len(codes) // 8)
            n = 0
            for _, r in grp.iterrows():
                if r['primary_spectrum'] in okset: axes[1].plot(GRID, X[pos[r['primary_spectrum']]], color=WONG[n % 7], lw=0.7, alpha=0.8); n += 1
            axes[1].set_title(f"{series}: {n} mixture spectra (tier {','.join(sorted(set(grp['tier'])))}; {grp['ref'].iloc[0][:60]})", fontsize=8)
            for ax in axes: ax.set_xlabel('Wavelength (µm)'); ax.set_ylabel('Reflectance factor')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('[C] gallery ->', out / 'figures' / 'S2_series_gallery.pdf')

# ----------------------------------------------------------------------------- stage D/E: forward + retrieval
def _em_spectrum(code, inc, emi, em_idx, qc_geo, okset):
    """end-member spectrum measured at the mixture's geometry when available, else the curated primary"""
    r = em_idx.loc[code]
    for s in str(r['spectrum_ids']).split(';'):
        if s in okset and qc_geo.get(s) == (inc, emi): return s
    p = r['primary_spectrum']; return p if p in okset else None

def stage_forward(mix, em, qc, X, out: Path, density: dict):
    pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; okset = set(qc.loc[qc['status'] == 'ok', 'spectrum_id'])
    qc_geo = {r.spectrum_id: (float(r.incidence_deg), float(r.emission_deg)) for r in qc.itertuples()}
    em_idx = em.set_index('code'); res, preds = [], {}
    for _, r in mix.iterrows():
        comps = json.loads(r['components']); sid = r['primary_spectrum']
        if sid not in okset: res.append({'mix_id': r['mix_id'], 'series': r['series'], 'tier': r['tier'], 'status': 'mixture_spectrum_missing'}); continue
        inc, emi = qc_geo[sid]
        ems = [_em_spectrum(c['code'], inc, emi, em_idx, qc_geo, okset) if c['code'] in em_idx.index else None for c in comps]
        if any(e is None for e in ems): res.append({'mix_id': r['mix_id'], 'series': r['series'], 'tier': r['tier'], 'status': 'endmember_missing'}); continue
        Rm = X[pos[sid]]; mu0m, mum = mus(inc, emi)
        Re = np.vstack([X[pos[e]] for e in ems]); We = np.vstack([w_from_reff(X[pos[e]], *mus(*qc_geo[e])) for e in ems])
        F = xsec_weights(comps, density, r['basis']); lin = F @ Re; hap = reff_from_w(F @ We, mu0m, mum)
        Wm = w_from_reff(Rm, mu0m, mum)
        def rms(a, b): return float(np.sqrt(np.nanmean((a[EVAL] - b[EVAL]) ** 2)))
        def rms_shape(a, b):
            a2 = a / np.interp(0.75, GRID, a); b2 = b / np.interp(0.75, GRID, b); return float(np.sqrt(np.nanmean((a2[EVAL] - b2[EVAL]) ** 2)))
        Fl_raw = nnls(Re[:, EVAL].T, Rm[EVAL]); Fh_raw = nnls(We[:, EVAL].T, Wm[EVAL])
        Fl = Fl_raw / Fl_raw.sum() if Fl_raw.sum() > 0 else Fl_raw; Fh = Fh_raw / Fh_raw.sum() if Fh_raw.sum() > 0 else Fh_raw
        true = np.array([c['frac'] for c in comps], float); Ml = F_to_basis(Fl, comps, density, r['basis']); Mh = F_to_basis(Fh, comps, density, r['basis'])
        row = {'mix_id': r['mix_id'], 'series': r['series'], 'tier': r['tier'], 'status': 'ok', 'n_components': len(comps), 'families': r['families'], 'basis': r['basis'],
               'nano_additive': any(0 < _f(c.get('size_hi')) < 1.0 for c in comps), 'inc_deg': inc, 'emi_deg': emi, 'em_geometry_matched': all(qc_geo[e] == (inc, emi) for e in ems),
               'r_mix_105': round(float(Rm[np.argmin(abs(GRID - 1.05))]), 4), 'rms_linear': rms(Rm, lin), 'rms_hapke': rms(Rm, hap), 'rms_shape_linear': rms_shape(Rm, lin),
               'rms_shape_hapke': rms_shape(Rm, hap), 'rms_w_hapke': float(np.sqrt(np.nanmean((Wm[EVAL] - (F @ We)[EVAL]) ** 2))), 'nnls_sum_linear': float(Fl_raw.sum()),
               'nnls_sum_hapke': float(Fh_raw.sum()), 'mae_frac_linear': float(np.mean(np.abs(Ml - true))), 'mae_frac_hapke': float(np.mean(np.abs(Mh - true))),
               'maxerr_frac_linear': float(np.max(np.abs(Ml - true))), 'maxerr_frac_hapke': float(np.max(np.abs(Mh - true)))}
        for i, c in enumerate(comps):
            row[f'c{i + 1}'] = c['code']; row[f'fam{i + 1}'] = c['fam']; row[f'true{i + 1}'] = round(float(c['frac']), 4); row[f'lin{i + 1}'] = round(float(Ml[i]), 4); row[f'hap{i + 1}'] = round(float(Mh[i]), 4)
        res.append(row); preds[r['mix_id']] = (lin, hap)
    res = pd.DataFrame(res); res.to_csv(out / 'forward_results.csv', index=False)
    ok = res[res['status'] == 'ok']
    # residual spectra (measured - model) of the evaluated mixtures: the model-error floor used by the identifiability step
    if len(ok):
        mixi = mix.set_index('mix_id'); ids = list(ok['mix_id'])
        Rmeas = np.vstack([X[pos[mixi.loc[m, 'primary_spectrum']]] for m in ids]); Rlin = np.vstack([preds[m][0] for m in ids]); Rhap = np.vstack([preds[m][1] for m in ids])
        np.savez_compressed(out / 'forward_residuals.npz', grid_um=GRID, mix_id=np.array(ids), measured=Rmeas, linear=Rlin, hapke=Rhap)
        tA = (ok['tier'] == 'A').values & (~ok['nano_additive'].astype(bool)).values
        pd.DataFrame({'wavelength_um': GRID, 'rms_resid_hapke_tierA': np.sqrt(np.nanmean((Rmeas[tA] - Rhap[tA]) ** 2, axis=0)), 'rms_resid_linear_tierA': np.sqrt(np.nanmean((Rmeas[tA] - Rlin[tA]) ** 2, axis=0)),
                      'median_abs_resid_hapke_tierA': np.nanmedian(np.abs(Rmeas[tA] - Rhap[tA]), axis=0), 'median_rel_resid_hapke_tierA': np.nanmedian(np.abs(Rmeas[tA] - Rhap[tA]) / Rmeas[tA], axis=0)}
                     ).round(5).to_csv(out / 'forward_residual_vs_wavelength.csv', index=False)
    summ = ok.groupby('series').agg(n=('mix_id', 'size'), rms_lin=('rms_linear', 'median'), rms_hap=('rms_hapke', 'median'), shape_lin=('rms_shape_linear', 'median'),
                                     shape_hap=('rms_shape_hapke', 'median'), mae_lin_pct=('mae_frac_linear', lambda x: 100 * x.median()), mae_hap_pct=('mae_frac_hapke', lambda x: 100 * x.median()),
                                     maxerr_hap_pct=('maxerr_frac_hapke', lambda x: 100 * x.median())).round(4)
    summ.to_csv(out / 'forward_summary_by_series.csv')
    print('[D/E] status:', res['status'].value_counts().to_dict()); print(summ.to_string())
    return res, preds

def stage_forward_figures(mix, res, preds, qc, X, out: Path):
    plt = style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; ok = res[res['status'] == 'ok']; mixi = mix.set_index('mix_id')
    if not len(ok): print('[F] nothing to plot'); return
    ser = list(ok.groupby('series', sort=False)); n = len(ser); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), squeeze=False)
    for k, (s, g) in enumerate(ser):
        ax = axes[k // cols][k % cols]; g2 = g.sort_values('rms_hapke'); r = g2.iloc[len(g2) // 2]; sid = mixi.loc[r['mix_id'], 'primary_spectrum']; lin, hap = preds[r['mix_id']]
        ax.plot(GRID, X[pos[sid]], color=WONG[0], lw=1.8, label='measured'); ax.plot(GRID, lin, color=WONG[1], lw=1.1, ls='--', label=f"linear (rms {r['rms_linear']:.3f})")
        ax.plot(GRID, hap, color=WONG[2], lw=1.1, label=f"Hapke (rms {r['rms_hapke']:.3f})")
        ax.set_title(f"{s}: {r['mix_id']} ({str(mixi.loc[r['mix_id'], 'name'])[:38]})", fontsize=8); ax.legend(fontsize=6.5, frameon=False); ax.set_xlabel('µm', fontsize=8)
    for k in range(n, rows * cols): axes[k // cols][k % cols].axis('off')
    fig.suptitle('Median-error mixture of every series: measured vs linear vs Hapke forward model', fontsize=10); fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F1_measured_vs_modelled.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, key, title in ((axes[0], 'lin', 'linear mixing, NNLS on reflectance'), (axes[1], 'hap', 'Hapke mixing, NNLS on single-scattering albedo')):
        for k, (s, g) in enumerate(ser):
            t = np.concatenate([g[f'true{i}'].astype(float).values for i in range(1, 6) if f'true{i}' in g]); p = np.concatenate([g[f'{key}{i}'].astype(float).values for i in range(1, 6) if f'{key}{i}' in g])
            m = np.isfinite(t) & np.isfinite(p); ax.scatter(t[m], p[m], s=12, color=WONG[k % 7], marker='os^Dv<>'[k % 7], label=f'{s} (n={len(g)})', alpha=0.75, linewidths=0)
        ax.plot([0, 1], [0, 1], color='#999999', lw=0.8); ax.set_xlabel('documented fraction (wt or vol)'); ax.set_ylabel('retrieved fraction (same basis)'); ax.set_title(title, fontsize=9)
    axes[1].legend(fontsize=6, frameon=False, ncol=2, loc='upper left'); fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F2_retrieval_parity.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.4)); labels = [s for s, _ in ser]; x = np.arange(len(labels))
    for off, key, col in ((-0.18, 'mae_frac_linear', WONG[1]), (0.18, 'mae_frac_hapke', WONG[2])):
        ax.boxplot([g[key].values * 100 for _, g in ser], positions=x + off, widths=0.3, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.6, edgecolor=INK),
                   medianprops=dict(color=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK), flierprops=dict(marker='.', markeredgecolor=INK, markersize=3))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8); ax.set_ylabel('MAE of retrieved fraction (%)'); ax.set_title('Abundance retrieval error per series: linear (orange) vs Hapke (green)', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F3_retrieval_error_by_series.png'); plt.close(fig)
    z = np.load(out / 'forward_residuals.npz', allow_pickle=True); tA = (ok['tier'] == 'A').values & (~ok['nano_additive'].astype(bool)).values
    if tA.sum() > 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, key, col in ((axes[0], 'hapke', WONG[2]), (axes[1], 'linear', WONG[1])):
            D = z['measured'][tA] - z[key][tA]; q = np.nanpercentile(D, [5, 25, 50, 75, 95], axis=0)
            ax.fill_between(GRID, q[0], q[4], color=col, alpha=0.18, lw=0, label='5-95 %'); ax.fill_between(GRID, q[1], q[3], color=col, alpha=0.4, lw=0, label='25-75 %')
            ax.plot(GRID, q[2], color=col, lw=1.2, label='median'); ax.plot(GRID, np.sqrt(np.nanmean(D ** 2, axis=0)), color=INK, lw=0.9, ls='--', label='RMS')
            ax.axhline(0, color='#999999', lw=0.6); ax.set_xlabel('Wavelength (µm)'); ax.set_ylabel('measured - modelled (reflectance factor)'); ax.set_title(f'{key} model residuals, tier A (n={int(tA.sum())})', fontsize=9); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F6_residual_vs_wavelength.png'); plt.close(fig)
    print('[F] figures F1-F3, F6 ->', out / 'figures')

# ----------------------------------------------------------------------------- stage G: band parameters
def stage_bands(mix, qc, X, out: Path):
    plt = style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; okset = set(qc.loc[qc['status'].isin(['ok', 'partial']), 'spectrum_id']); rows = []
    for _, r in mix.iterrows():
        sid = r['primary_spectrum']
        if sid not in okset: continue
        comps = json.loads(r['components']); y = X[pos[sid]]; m = np.isfinite(y); bp = band_params(GRID[m], y[m]) if m.sum() > 50 else {}
        ol = sum(c['frac'] for c in comps if c['fam'] == 'olivine'); px = sum(c['frac'] for c in comps if c['fam'] in ('opx', 'cpx'))
        bp.update({'mix_id': r['mix_id'], 'series': r['series'], 'tier': r['tier'], 'basis': r['basis'], 'ol_frac': ol, 'px_frac': px, 'opx_frac': sum(c['frac'] for c in comps if c['fam'] == 'opx'),
                   'cpx_frac': sum(c['frac'] for c in comps if c['fam'] == 'cpx'), 'ol_over_olpx': ol / (ol + px) if ol + px > 0 else np.nan,
                   'plag_frac': sum(c['frac'] for c in comps if c['fam'] == 'plagioclase'), 'glass_frac': sum(c['frac'] for c in comps if c['fam'].startswith('glass')),
                   'opaque_frac': sum(c['frac'] for c in comps if c['fam'] in OPAQUE_FAMS), 'r_105': round(float(y[np.argmin(abs(GRID - 1.05))]), 4)})
        rows.append(bp)
    bp = pd.DataFrame(rows); bp.to_csv(out / 'band_parameters.csv', index=False)
    s = bp[(bp['tier'] == 'A') & bp['ol_over_olpx'].notna() & (bp['opaque_frac'] < 0.05) & (bp['plag_frac'] < 0.5) & (bp['glass_frac'] < 0.05) & bp['BAR'].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for k, (ser, g) in enumerate(s.groupby('series', sort=False)):
        kw = dict(s=16, color=WONG[k % 7], marker='os^Dv<>'[k % 7], linewidths=0, alpha=0.85)
        axes[0].scatter(g['ol_over_olpx'], g['BAR'], label=f'{ser} (n={len(g)})', **kw); axes[1].scatter(g['ol_over_olpx'], g['b1_center'], **kw); axes[2].scatter(g['b1_center'], g['BAR'], **kw)
    axes[0].set_xlabel('olivine / (olivine + pyroxene), documented'); axes[0].set_ylabel('Band Area Ratio (Band II / Band I)'); axes[0].legend(fontsize=6.5, frameon=False)
    axes[1].set_xlabel('olivine / (olivine + pyroxene), documented'); axes[1].set_ylabel('Band I centre (µm)'); axes[2].set_xlabel('Band I centre (µm)'); axes[2].set_ylabel('Band Area Ratio')
    fig.suptitle(f'Cloutis-1986-type calibration on the RELAB benchmark (tier A, <5 % opaques/glass, <50 % plagioclase; n={len(s)})', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F4_BAR_calibration.png'); plt.close(fig)
    print(f'[G] band parameters for {len(bp)} mixtures; BAR calibration subset: {len(s)}')
    return bp

# ----------------------------------------------------------------------------- stage H: photometric series
def stage_photometric(mix, em, qc, X, out: Path, density: dict):
    plt = style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; ok = qc[qc['status'] == 'ok']
    phot = ok[ok['role'] == 'photometric']
    if not len(phot): print('[H] no photometric spectra loaded'); return pd.DataFrame()
    band = (GRID >= 0.5) & (GRID <= 2.4); rows = []
    for spec, g in phot.groupby('specimen_id'):
        allg = ok[ok['specimen_id'] == spec]                                  # the i=30/e=0 reference is usually the curated std spectrum (role mixture/endmember)
        ref = allg[(allg['incidence_deg'] == 30) & (allg['emission_deg'] == 0)]
        if not len(ref): continue
        rid = ref.iloc[0]['spectrum_id']; wref = w_from_reff(X[pos[rid]], *mus(30, 0)); Rref = X[pos[rid]]
        for _, r in allg.iterrows():
            w = w_from_reff(X[pos[r['spectrum_id']]], *mus(r['incidence_deg'], r['emission_deg'])); ratio = w[band] / wref[band]; rr = X[pos[r['spectrum_id']]][band] / Rref[band]
            rows.append({'specimen_id': spec, 'spectrum_id': r['spectrum_id'], 'inc_deg': r['incidence_deg'], 'emi_deg': r['emission_deg'], 'phase_deg': r['phase_deg'],
                         'reff_ratio_median': float(np.nanmedian(rr)), 'w_ratio_median': float(np.nanmedian(ratio)), 'w_ratio_075': float(ratio[np.argmin(abs(GRID[band] - 0.75))]),
                         'w_ratio_150': float(ratio[np.argmin(abs(GRID[band] - 1.50))]), 'w_ratio_240': float(ratio[np.argmin(abs(GRID[band] - 2.40))]), 'w_ratio_spread': float(np.nanstd(ratio))})
    inv = pd.DataFrame(rows); inv.to_csv(out / 'photometric_invariance.csv', index=False)
    if not len(inv): print('[H] no photometric specimen has its i=30/e=0 reference loaded'); return inv
    # Hapke mixing of the photometric mixtures at every geometry where both end-members were measured at the same geometry
    em_idx = em.set_index('code'); geo = {r.spectrum_id: (float(r.incidence_deg), float(r.emission_deg), float(r.phase_deg)) for r in ok.itertuples()}
    mrows = []
    for _, r in mix[mix['mix_id'].isin(phot['specimen_id'].unique())].iterrows():
        comps = json.loads(r['components']); F = xsec_weights(comps, density, r['basis'])
        for _, s in phot[phot['specimen_id'] == r['mix_id']].iterrows():
            key = (float(s['incidence_deg']), float(s['emission_deg'])); ems = []
            for c in comps:
                cands = [x for x in str(em_idx.loc[c['code'], 'spectrum_ids']).split(';') if x in geo and geo[x][:2] == key] if c['code'] in em_idx.index else []
                if c.get('em'): cands += [x for x in ok[ok['specimen_id'] == c['em']]['spectrum_id'] if geo[x][:2] == key]
                ems.append(cands[0] if cands else None)
            if any(e is None for e in ems): continue
            Rm = X[pos[s['spectrum_id']]]; mu0, mu = mus(*key); Re = np.vstack([X[pos[e]] for e in ems]); We = np.vstack([w_from_reff(X[pos[e]], mu0, mu) for e in ems])
            lin = F @ Re; hap = reff_from_w(F @ We, mu0, mu)
            mrows.append({'mix_id': r['mix_id'], 'name': r['name'], 'spectrum_id': s['spectrum_id'], 'inc_deg': key[0], 'emi_deg': key[1], 'phase_deg': s['phase_deg'],
                          'rms_linear': float(np.sqrt(np.nanmean((Rm[EVAL] - lin[EVAL]) ** 2))), 'rms_hapke': float(np.sqrt(np.nanmean((Rm[EVAL] - hap[EVAL]) ** 2))),
                          'rel_rms_hapke': float(np.sqrt(np.nanmean(((Rm[EVAL] - hap[EVAL]) / Rm[EVAL]) ** 2)))})
    pm = pd.DataFrame(mrows); pm.to_csv(out / 'photometric_mixing.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k, (spec, g) in enumerate(inv.groupby('specimen_id')):
        g = g.sort_values('phase_deg'); axes[0].scatter(g['phase_deg'], g['w_ratio_median'], s=14, color=WONG[k % 7], marker='os^Dv'[k % 5], label=spec, linewidths=0)
    axes[0].axhline(1, color='#999999', lw=0.8); axes[0].set_xlabel('phase angle (deg)'); axes[0].set_ylabel('w(i,e) / w(30,0), median 0.5-2.4 µm'); axes[0].set_title('Single-scattering albedo invariance (isotropic Hapke, no opposition term)', fontsize=8.5); axes[0].legend(fontsize=7, frameon=False)
    for k, (mid, g) in enumerate(pm.groupby('mix_id') if len(pm) else []):
        g = g.sort_values('phase_deg'); axes[1].scatter(g['phase_deg'], g['rms_hapke'], s=16, color=WONG[k % 7], marker='os'[k % 2], label=f'{g["name"].iloc[0]}: Hapke', linewidths=0)
        axes[1].scatter(g['phase_deg'], g['rms_linear'], s=16, facecolors='none', edgecolors=WONG[k % 7], marker='os'[k % 2], label=f'{g["name"].iloc[0]}: linear')
    axes[1].set_xlabel('phase angle (deg)'); axes[1].set_ylabel('RMS (reflectance factor, 0.45-2.45 µm)'); axes[1].set_title('Olivine-magnetite mixtures: forward-model error vs geometry', fontsize=8.5)
    if len(pm): axes[1].legend(fontsize=6.5, frameon=False)
    else: axes[1].text(0.5, 0.5, 'no geometry-matched end-member spectra loaded', ha='center', va='center', transform=axes[1].transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S2_F5_photometric.png'); plt.close(fig)
    print(f'[H] photometric: {len(inv)} spectra of {inv["specimen_id"].nunique()} specimens; mixing test at {len(pm)} geometries')
    return inv

# ----------------------------------------------------------------------------- stage I: report + zip
def stage_report(mix, em, inv, qc, res, bp, out: Path, proj: Path, bench: Path):
    ok = res[res['status'] == 'ok'] if len(res) else res; A = ok[(ok['tier'] == 'A') & (~ok['nano_additive'].astype(bool))] if len(ok) else ok
    L = [f"# S2 report -- benchmark assembly and forward-model validation ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')})", '',
         f"- mixtures in the benchmark: **{len(mix)}** (tiers {mix['tier'].value_counts().to_dict()}); end-members: {len(em)} ({int((em['status'] != 'ok').sum())} without a RELAB spectrum)",
         f"- spectra loaded: {qc['status'].value_counts().to_dict()}", f"- forward model / retrieval evaluated on **{len(ok)}** mixtures ({res['status'].value_counts().to_dict() if len(res) else {}})", '',
         '## Benchmark inventory (Table 1 draft)', '', '| series | reference | n | tiers | components | basis | families | sizes (µm) | i/e | flags |', '|---|---|---|---|---|---|---|---|---|---|']
    for _, r in inv.iterrows():
        L.append(f"| {r['series']} | {r['reference']} | {r['n_mixtures']} | {r['tiers']} | {r['n_components']} | {r['basis']} | {r['families']} | {r['size_fractions_um']} | {r['geometry_i_e']} | {r['flags']} |")
    L += ['', '## Forward model per series: median RMS of the reflectance factor (0.45-2.45 µm) and median abundance MAE', '',
          '| series | n | RMS linear | RMS Hapke | shape RMS linear | shape RMS Hapke | MAE % linear | MAE % Hapke | max-err % Hapke |', '|---|---|---|---|---|---|---|---|---|']
    for s, g in ok.groupby('series', sort=False):
        L.append(f"| {s} | {len(g)} | {g['rms_linear'].median():.4f} | {g['rms_hapke'].median():.4f} | {g['rms_shape_linear'].median():.4f} | {g['rms_shape_hapke'].median():.4f} | "
                 f"{100 * g['mae_frac_linear'].median():.1f} | {100 * g['mae_frac_hapke'].median():.1f} | {100 * g['maxerr_frac_hapke'].median():.1f} |")
    if len(A):
        L += ['', f"**Headline (tier A, no nano-additives, n={len(A)}):** median abundance MAE Hapke **{100 * A['mae_frac_hapke'].median():.1f} %** (linear {100 * A['mae_frac_linear'].median():.1f} %); "
              f"75th percentile Hapke {100 * A['mae_frac_hapke'].quantile(0.75):.1f} %; median spectral RMS Hapke {A['rms_hapke'].median():.4f} (linear {A['rms_linear'].median():.4f}); "
              f"mixtures with Hapke MAE <= 5 %: {100 * (A['mae_frac_hapke'] <= 0.05).mean():.0f} %, <= 10 %: {100 * (A['mae_frac_hapke'] <= 0.10).mean():.0f} %"]
        isop = A['families'].apply(lambda s: bool(set(str(s).split(';')) & OPAQUE_FAMS)); sil = A[~isop]; opq = A[isop]
        if len(sil): L.append(f"**Silicate-only tier-A mixtures (n={len(sil)}):** median MAE Hapke {100 * sil['mae_frac_hapke'].median():.1f} %, linear {100 * sil['mae_frac_linear'].median():.1f} %")
        if len(opq): L.append(f"**Opaque-bearing tier-A mixtures (n={len(opq)}):** median MAE Hapke {100 * opq['mae_frac_hapke'].median():.1f} %, linear {100 * opq['mae_frac_linear'].median():.1f} %")
        L += ['', '## Worst 15 mixtures (Hapke abundance MAE)', '', '| mix | series | components | documented | Hapke | linear | RMS Hapke |', '|---|---|---|---|---|---|---|']
        for _, r in ok.sort_values('mae_frac_hapke', ascending=False).head(15).iterrows():
            cs = [r[f'c{i}'] for i in range(1, 6) if f'c{i}' in r and isinstance(r[f'c{i}'], str)]
            f = lambda k: '/'.join(f"{float(r[f'{k}{i}']):.2f}" for i in range(1, len(cs) + 1))
            L.append(f"| {r['mix_id']} | {r['series']} | {'/'.join(cs)} | {f('true')} | {f('hap')} | {f('lin')} | {r['rms_hapke']:.3f} |")
    bad = qc[qc['status'] != 'ok']
    L += ['', '## Spectra QC', '', f"- {len(bad)} spectra not fully usable: {bad['status'].value_counts().to_dict()}"] + [f"  - `{r['spectrum_id']}` {r['specimen_id']} ({r['role']}, {r['series']}) {r['status']}" for _, r in bad.head(40).iterrows()]
    nA = int(len(A)); g1 = nA >= 250; g2 = bool(len(A) and A['mae_frac_hapke'].median() <= 0.08); g3 = bool(len(A) and A['rms_hapke'].median() <= 0.03)
    L += ['', '## Gate checklist (S2 -> S3)', '', f"- [{'x' if g1 else ' '}] >= 250 tier-A mixtures with full-range spectrum and all end-members modelled (now {nA})",
          f"- [{'x' if g2 else ' '}] median Hapke abundance MAE <= 8 % on tier A (Mustard & Pieters 1989 level; now {100 * A['mae_frac_hapke'].median():.1f} %)" if len(A) else '- [ ] abundance MAE: not evaluated',
          f"- [{'x' if g3 else ' '}] median Hapke spectral RMS <= 0.03 on tier A (now {A['rms_hapke'].median():.4f})" if len(A) else '- [ ] spectral RMS: not evaluated',
          '- [ ] (manual) gallery and worst-15 table reviewed; outliers explained (basis assumptions, size mismatches, opaques, nano-additives)']
    (out / 'S2_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    zpath = proj / 'S2_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in list(out.rglob('*')) + list(bench.glob('*.csv')):
            if p.is_file() and p.suffix in {'.csv', '.json', '.md', '.npz', '.png', '.pdf'}: zf.write(p, p.relative_to(proj))
    print(f"[I] report: {out / 'S2_REPORT.md'}\n[I] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath
