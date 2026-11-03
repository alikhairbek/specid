#!/usr/bin/env python3
"""specid.pipeline -- the whole study as one sequence of stages, for the single notebook (Kaggle / Colab / local) and for the command line.

Stages (each writes its own folder under the project root and can be skipped when its outputs already exist):
  S1  audit of the RELAB spectra shipped with the dataset (and optional download of the missing ones from the PDS Geosciences Node)
  S2  benchmark tables, spectra on the common grid, Hapke vs linear forward model, NNLS retrieval, band parameters, report;
      + the relative-residual file of the silicate series (the empirical model-error covariance used by every later stage)
  S3a grain-size rule, olivine and pyroxene optical-constant libraries, natural anchors, weathering transfer, parametric fits, library file
  S3b identifiability engine: Cramér-Rao maps, profiles, spectral twins, validation of the bounds
  S4  nuisance-aware population with library mismatch, learned retrieval, split-conformal regions, sim-to-real evaluation
  S5  verified mission filters, D-optimal band design and benchmark, learned retrieval on band sets, NEA application (levels 1 and 2)
  S6  the figures of the paper and of the Supporting Information

Usage (notebook):  from specid import pipeline as P;  prj = P.Project.prepare();  P.run_s2(prj); ...  P.package(prj)
Usage (CLI):       python -m specid.pipeline --base /path/to/work --quick --stages s2,s3a,s3b,s4,s5,s6"""
from __future__ import annotations
import os, sys, json, glob, shutil, time, zipfile, hashlib, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

SILICATE_SERIES = ('MP89', 'XT-OBA', 'XT-OBA-fine', 'SP93-pyx', 'McCoy-4', 'AEF-fine', 'CP14-plag')
DATASET_MARK = 'specid_data_manifest.json'
STAGES = ('s1', 's2', 's3a', 's3b', 's4', 's5', 's6')

def _log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ----------------------------------------------------------------------------- project layout
class Project:
    """folders of one run: base/specid_run/{benchmark, inputs, s2, s3a, s3b, s4, s5, figures}; the dataset folder holds the inputs"""
    def __init__(self, base: Path, dataset: Path):
        self.base = Path(base); self.dataset = Path(dataset); self.proj = self.base / 'specid_run'
        self.bench = self.proj / 'benchmark'; self.inputs = self.proj / 'inputs'; self.raw = self.proj / 'data' / 'raw' / 'relab'
        self.s2 = self.proj / 's2'; self.s3a = self.proj / 's3a'; self.s3b = self.proj / 's3b'; self.s4 = self.proj / 's4'; self.s5 = self.proj / 's5'; self.figures = self.proj / 'figures'
        for d in (self.bench, self.inputs, self.raw, self.s2, self.s3a, self.s3b, self.s4, self.s5, self.figures): d.mkdir(parents=True, exist_ok=True)
        for d in (self.s2, self.s3a, self.s3b, self.s4, self.s5): (d / 'figures').mkdir(exist_ok=True)
        for f in (self.dataset / 'benchmark').glob('*.csv'): shutil.copy(f, self.bench / f.name)
        for f in (self.dataset / 'nea').glob('*'):
            if f.is_file(): shutil.copy(f, self.inputs / f.name)
        self.relab_local = str(self.dataset / 'relab')          # the raw .tab files shipped with the dataset (PDS folder structure)
        self.resid = self.s2 / 's2_relative_residuals_silicate.npz'; self.lib = self.s3a / 's3a_library.npz'

    @classmethod
    def prepare(cls, base=None, dataset=None):
        base = Path(base) if base else cls.default_base(); ds = find_dataset(base, dataset) if dataset else find_dataset(base)   # dataset: a folder, a zip, or a folder containing either
        if ds is None: raise SystemExit('specid dataset not found: add the specid-data dataset (Kaggle: Add Input), or upload specid_data.zip, or pass dataset=...')
        P = cls(base, ds); _log(f'project root {P.proj}; dataset {ds}'); return P

    @staticmethod
    def default_base():
        for c in ('/kaggle/working', '/content'):
            if os.path.exists(c): return Path(c)
        return Path.cwd()

def find_dataset(base: Path, hint=None) -> Path | None:
    if hint and str(hint).lower().endswith('.zip') and Path(hint).is_file():          # the dataset zip itself: extract it once next to the project
        dest = base / 'specid_data'; dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(hint) as zf: zf.extractall(dest)
        hint = dest
    roots = [Path(h) for h in ([hint] if hint else [])] + [Path(r) for r in ('/kaggle/input', '/content', str(base), os.getcwd(), str(Path.home())) if os.path.exists(r)]
    for r in roots:
        hits = sorted(glob.glob(f'{r}/**/{DATASET_MARK}', recursive=True))
        if hits: return Path(hits[0]).parent
    for r in roots:                                                 # a zipped dataset: extract once next to the project
        hits = sorted(glob.glob(f'{r}/**/specid_data.zip', recursive=True))
        if hits:
            dest = base / 'specid_data'; dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(hits[0]) as zf: zf.extractall(dest)
            m = sorted(glob.glob(f'{dest}/**/{DATASET_MARK}', recursive=True))
            if m: return Path(m[0]).parent
    return None

def _done(path: Path, *files): return all((path / f).exists() for f in files)

# ----------------------------------------------------------------------------- S1
def run_s1(P: Project, download=False, workers=8):
    """inventory of the benchmark spectra: on disk (dataset) / missing; optional download of the missing files from the PDS node"""
    import s2_forward as S2
    mix, em, sp, density, inv = S2.stage_tables(P.bench, P.s2)
    if not download: os.environ['S2_OFFLINE'] = '1'
    else: os.environ.pop('S2_OFFLINE', None)
    paths = S2.stage_fetch(sp, P.raw, workers, local_hint=P.relab_local)
    have = {k: p.exists() for k, p in paths.items()}; sp2 = sp.copy(); sp2['on_disk'] = sp2['spectrum_id'].map(have).fillna(False)
    sp2.to_csv(P.s2 / 'S1_spectra_inventory.csv', index=False); n = int(sp2['on_disk'].sum())
    _log(f'S1: {n} of {len(sp2)} benchmark spectra on disk ({len(sp2) - n} missing: ' + ', '.join(sp2[~sp2["on_disk"]].groupby("role").size().astype(str).to_dict().keys()) + ')')
    return sp2

# ----------------------------------------------------------------------------- S2
def make_relative_residuals(out_s2: Path, series=SILICATE_SERIES):
    """(measured - Hapke) / measured for the tier-A mixtures of the silicate series: the empirical model-error term of Step 3b onwards"""
    fr = pd.read_csv(out_s2 / 'forward_results.csv'); z = np.load(out_s2 / 'forward_residuals.npz', allow_pickle=True)
    ids = [str(s) for s in z['mix_id']]; pos = {m: i for i, m in enumerate(ids)}
    sel = fr[(fr['status'] == 'ok') & (fr['tier'] == 'A') & fr['series'].isin(series)]; keep = [m for m in sel['mix_id'] if m in pos]
    R = np.array([(z['measured'][pos[m]] - z['hapke'][pos[m]]) / z['measured'][pos[m]] for m in keep], dtype=np.float32)
    ser = sel.set_index('mix_id').loc[keep, 'series'].values
    np.savez_compressed(out_s2 / 's2_relative_residuals_silicate.npz', grid_um=z['grid_um'], rel_resid=R, mix_id=np.array(keep, dtype=object), series=np.array(ser, dtype=object))
    _log(f'S2: relative residuals of {len(keep)} silicate tier-A mixtures -> s2_relative_residuals_silicate.npz'); return len(keep)

def run_s2(P: Project, workers=8, gallery=True, download=False):
    import s2_forward as S2
    t0 = time.time(); mix, em, sp, density, inv = S2.stage_tables(P.bench, P.s2)
    if not download: os.environ['S2_OFFLINE'] = '1'
    paths = S2.stage_fetch(sp, P.raw, workers, local_hint=P.relab_local)
    qc, X = S2.stage_load(sp, paths, P.s2)
    if gallery:
        try: S2.stage_gallery(mix, em, qc, X, P.s2)
        except Exception as e: _log(f'S2: gallery skipped ({e})')
    res, preds = S2.stage_forward(mix, em, qc, X, P.s2, density); S2.stage_forward_figures(mix, res, preds, qc, X, P.s2)
    bp = S2.stage_bands(mix, qc, X, P.s2)
    try: S2.stage_photometric(mix, em, qc, X, P.s2, density)
    except Exception as e: _log(f'S2: photometric stage skipped ({e}); the phase-angle spectra are not in the shipped bundle unless S1 downloaded them')
    S2.stage_report(mix, em, inv, qc, res, bp, P.s2, P.proj, P.bench); make_relative_residuals(P.s2)
    _log(f'S2 done in {(time.time() - t0) / 60:.1f} min'); return res

# ----------------------------------------------------------------------------- S3a
def run_s3a(P: Project, workers=8, max_per_series=60):
    import s3a_library as S3
    t0 = time.time(); os.environ.setdefault('S2_OFFLINE', '1')
    lst, allsp, mix, em = S3.stage_tables(P.bench, P.s3a)
    qc, X = S3.stage_load(allsp, P.raw, workers, P.relab_local, P.s3a)
    cal, val, rule, curves = S3.stage_grainsize(lst, qc, X, P.s3a)
    ol = S3.stage_olivine(lst, qc, X, rule, P.s3a); px = S3.stage_pyroxene(lst, qc, X, rule, em, mix, P.s3a)
    nat, offs = S3.stage_natural(qc, X, rule, em, ol, px, P.s3a); T = S3.stage_weathering(qc, X, P.s3a)
    S3.write_library(ol, px, qc, X, rule, T, em, nat, offs, P.s3a)
    res = S3.stage_mixtures(mix, em, qc, X, rule, nat, P.s3a, max_per_series=max_per_series)
    S3.stage_report(lst, qc, cal, val, rule, ol, px, nat, T, res, P.s3a, P.proj, P.bench)
    if P.resid.exists(): shutil.copy(P.resid, P.s3a / P.resid.name)
    _log(f'S3a done in {(time.time() - t0) / 60:.1f} min'); return res

# ----------------------------------------------------------------------------- shared loaders
def load_model_and_inputs(P: Project):
    import forward_param as FP
    model = FP.ParamModel(str(P.lib)); z = np.load(P.s3a / 'benchmark_matrix.npz', allow_pickle=True); Xspec = z['X']; spec_ids = [str(s) for s in z['spectrum_id']]
    qc = pd.read_csv(P.s3a / 'spectra_qc.csv'); mix = pd.read_csv(P.bench / 'benchmark_mixtures_v0.csv'); em = pd.read_csv(P.bench / 'benchmark_endmembers_v0.csv')
    nat = pd.read_csv(P.s3a / 'natural_endmembers_fit.csv'); rule = json.load(open(P.s3a / 'grainsize_rule.json'))
    return model, Xspec, spec_ids, qc, mix, em, nat, rule

# ----------------------------------------------------------------------------- S3b
def run_s3b(P: Project, quick=False, n_syn=120, n_twin=2000):
    import s3b_identify as S3B
    t0 = time.time(); model, Xspec, spec_ids, qc, mix, em, nat, rule = load_model_and_inputs(P)
    s2 = pd.read_csv(P.s2 / 'forward_results.csv'); s3a = pd.read_csv(P.s3a / 'parametric_fit_results.csv')
    out = S3B.run_all(model, P.s3b, P.proj, str(P.resid), s2, s3a, mix, nat, rule, n_syn=n_syn, n_twin=n_twin, quick=quick)
    _log(f'S3b done in {(time.time() - t0) / 60:.1f} min'); return out

# ----------------------------------------------------------------------------- S4
def run_s4(P: Project, quick=False, tags=('lab', 'tel50', 'dawnfc', 'lab_noaug'), n_train=20000, n_scale=4000, n_cal=4000, n_test=2000, n_phys=120, n_crb=500, seed=0):
    import s4_conformal as S4
    t0 = time.time(); model, Xspec, spec_ids, qc, mix, em, nat, rule = load_model_and_inputs(P)
    out = S4.run_all(model, P.s4, P.proj, str(P.resid), qc, Xspec, spec_ids, mix, em, nat, rule, tags=tags, n_train=n_train, n_scale=n_scale, n_cal=n_cal, n_test=n_test, n_phys=n_phys, n_crb=n_crb, quick=quick, seed=seed)
    _log(f'S4 done in {(time.time() - t0) / 60:.1f} min'); return out

# ----------------------------------------------------------------------------- S5 (bands + missions + NEA levels 1 and 2)
def run_s5(P: Project, quick=False, allow_download=False, ks=(4, 5, 6, 7, 8, 10, 12), snr_band=100.0, n_design=48, n_eval=200, n_train=20000, n_scale=4000, n_cal=4000, n_test=2000, n_random=30, seed=0):
    import s5_bands as S5
    t0 = time.time(); model, Xspec, spec_ids, qc, mix, em, nat, rule = load_model_and_inputs(P)
    out = S5.run_all(model, P.s5, P.proj, str(P.resid), qc, Xspec, spec_ids, mix, em, nat, rule, P.inputs, ks=ks, snr_band=snr_band, allow_download=allow_download, n_design=n_design, n_eval=n_eval,
                     n_train=n_train, n_scale=n_scale, n_cal=n_cal, n_test=n_test, n_random=n_random, quick=quick, seed=seed)
    _log(f'S5 done in {(time.time() - t0) / 60:.1f} min'); return out

# ----------------------------------------------------------------------------- S6 figures
def layout(P: Project):
    return {'bench': P.bench, 's2': P.s2, 's3a_tables': P.s3a, 's3a_matrix': P.s3a, 'lib': P.s3a / 's3a_library.npz', 'rule': P.s3a / 'grainsize_rule.json', 's3b': P.s3b, 's4': P.s4, 's5': P.s5, 's5b': P.s5, 'out': P.figures}

def run_s6(P: Project, which_main=('1', '2', '3', '4', '5', '6', '7'), which_si=tuple(str(i) for i in range(1, 12))):
    import s6_figures as F6, s6_si_figures as F6S
    t0 = time.time(); F6.configure(layout(P)); D = F6.load()
    for w in which_main:
        try: getattr(F6, f'fig{w}')(D)
        except Exception as e: _log(f'S6: Figure {w} skipped ({type(e).__name__}: {e})')
    for w in which_si:
        try: getattr(F6S, f'figS{w}')(D)
        except Exception as e: _log(f'S6: Figure S{w} skipped ({type(e).__name__}: {e})')
    _log(f'S6 done in {(time.time() - t0) / 60:.1f} min')

# ----------------------------------------------------------------------------- packaging
def package(P: Project, name='SpecID_results.zip', exts=('.csv', '.json', '.md', '.png', '.pdf', '.npz', '.txt')):
    zpath = P.base / name
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for sub in (P.s2, P.s3a, P.s3b, P.s4, P.s5, P.figures):
            for p in sorted(sub.rglob('*')):
                if p.is_file() and p.suffix in exts and 'raw' not in p.parts: zf.write(p, p.relative_to(P.proj))
    _log(f'results packed: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)'); return zpath

def run(P: Project, stages=STAGES, quick=False, download=False, allow_nea_download=False):
    for s in stages:
        if s == 's1': run_s1(P, download=download)
        elif s == 's2': run_s2(P, download=download)
        elif s == 's3a': run_s3a(P)
        elif s == 's3b': run_s3b(P, quick=quick)
        elif s == 's4': run_s4(P, quick=quick)
        elif s == 's5': run_s5(P, quick=quick, allow_download=allow_nea_download)
        elif s == 's6': run_s6(P)
    return package(P)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='run the SpecID pipeline'); ap.add_argument('--base', default=None); ap.add_argument('--dataset', default=None)
    ap.add_argument('--stages', default=','.join(STAGES)); ap.add_argument('--quick', action='store_true'); ap.add_argument('--download', action='store_true', help='download missing RELAB spectra (S1/S2)')
    ap.add_argument('--nea-download', action='store_true', help='allow the NEA spectra download from smass.mit.edu'); a = ap.parse_args()
    P = Project.prepare(a.base, a.dataset); run(P, stages=tuple(a.stages.split(',')), quick=a.quick, download=a.download, allow_nea_download=a.nea_download)
