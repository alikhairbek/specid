#!/usr/bin/env python3
"""build the single pipeline notebook notebooks/SpecID_pipeline.ipynb (no embedded code or data: the code comes from the dataset's
code/ snapshot or from the cloned repository, the data from the specid-data dataset)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def md(t): return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n').splitlines(keepends=True)}
def code(t): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t.strip('\n').splitlines(keepends=True)}

cells = [md("""
# SpecID — how much mineralogy does a reflectance spectrum hold?
**One notebook for the whole study** (olivine–pyroxene mixtures: forward model → continuous library → identifiability →
learned retrieval with distribution-free regions → optimal bands and mission filters → (433) Eros and (25143) Itokawa → figures).

**Inputs.** The `specid-data` dataset (Kaggle: *Add Input*; Colab/local: upload `specid_data.zip`): the curated benchmark tables,
the RELAB reflectance spectra (`.tab`, PDS Geosciences Node), the archive spectra of Eros and Itokawa (`nea_raw.csv`) and a snapshot
of the `specid` code. No Internet is needed; with Internet on, `DOWNLOAD_RELAB` fetches the phase-angle and legacy spectra that are
not in the bundle and `NEA_DOWNLOAD` refreshes the asteroid spectra from smass.mit.edu.

**Stages and times** (Kaggle CPU, full settings): S1 audit (seconds) · S2 forward model (≈ 3 min) · S3a library (≈ 10 min) ·
S3b identifiability (≈ 40 min) · S4 learned retrieval + conformal (≈ 25 min) · S5 bands + missions + asteroids (≈ 35 min) ·
S6 figures (≈ 3 min). `QUICK = True` runs everything in ≈ 10 min with reduced sizes (smoke test; the numbers are not those of the paper, and
Figure S4, whose observation modes are not computed in the reduced S3b, is skipped).

**Output.** `SpecID_results.zip` in the working directory (all tables, reports and figures). Download it from the Output panel.
"""),
code("""
# ============================ CONFIGURATION ============================
QUICK = False                         # True = smoke test with reduced sizes (~10 min)
STAGES = ['s1', 's2', 's3a', 's3b', 's4', 's5', 's6']   # remove stages to skip them (a stage needs the outputs of the previous ones)
DOWNLOAD_RELAB = False                # S1/S2: download the spectra missing from the bundle (needs Internet; not needed for the paper's results)
NEA_DOWNLOAD = False                  # S5: refresh the Eros/Itokawa spectra from smass.mit.edu (needs Internet); False uses the cached nea_raw.csv
MAX_MIXTURES_PER_SERIES = 60          # S3a parametric fits
WORKERS = 8
"""),
code("""
# ============================ CODE AND DATA ============================
import os, sys, glob, importlib
from pathlib import Path
BASE = Path('/kaggle/working') if os.path.exists('/kaggle/working') else (Path('/content') if os.path.exists('/content') else Path.cwd())
roots = [r for r in ('/kaggle/input', '/content', str(BASE), os.getcwd()) if os.path.exists(r)]
def locate_code():
    for r in roots:                                               # 1) the code snapshot inside the dataset; 2) a cloned repository
        hits = sorted(glob.glob(f'{r}/**/code/specid/pipeline.py', recursive=True)) or sorted(glob.glob(f'{r}/**/specid/pipeline.py', recursive=True))
        if hits: return str(Path(hits[0]).parents[1])
code_dir = locate_code()
if code_dir is None:                                              # 3) a zipped dataset (Colab / local): extract it once next to the notebook
    import zipfile
    zips = [z for r in roots for z in glob.glob(f'{r}/**/specid_data.zip', recursive=True)]
    if zips:
        dest = BASE / 'specid_data'; dest.mkdir(parents=True, exist_ok=True); zipfile.ZipFile(zips[0]).extractall(dest); print('dataset extracted to', dest); code_dir = locate_code()
if code_dir is None: raise SystemExit('specid code not found: add the specid-data dataset (or put specid_data.zip next to this notebook) or clone the repository')
sys.path.insert(0, code_dir); print('code from', code_dir)
import specid; from specid import pipeline as P; importlib.reload(P)
print('specid', specid.__version__)
"""),
code("%pip install -q pandas numpy scipy matplotlib scikit-learn requests"),
code("""
prj = P.Project.prepare(BASE)           # creates BASE/specid_run/{benchmark, inputs, s2, s3a, s3b, s4, s5, figures}
"""),
md("### S1 — inventory of the benchmark spectra (and optional download of the missing ones from the PDS node)"),
code("""
if 's1' in STAGES: inv = P.run_s1(prj, download=DOWNLOAD_RELAB, workers=WORKERS); print(inv.groupby(['role', 'on_disk']).size())
"""),
md("### S2 — benchmark tables, spectra on the common grid, Hapke vs linear forward model, NNLS retrieval, band parameters, model-error residuals"),
code("""
if 's2' in STAGES:
    res2 = P.run_s2(prj, workers=WORKERS, download=DOWNLOAD_RELAB)
    print(open(prj.s2 / 'S2_REPORT.md', encoding='utf-8').read()[:3000])
"""),
md("### S3a — grain-size rule, olivine (Fo) and pyroxene (Wo, Fs) optical-constant libraries, natural anchors, weathering transfer, parametric fits"),
code("""
if 's3a' in STAGES:
    res3a = P.run_s3a(prj, workers=WORKERS, max_per_series=MAX_MIXTURES_PER_SERIES)
    print(open(prj.s3a / 'S3A_REPORT.md', encoding='utf-8').read()[:3000])
"""),
md("### S3b — identifiability engine: Cramér–Rao maps, profile likelihoods, spectral twins, validation of the bounds"),
code("""
if 's3b' in STAGES:
    out3b = P.run_s3b(prj, quick=QUICK)
    print(open(prj.s3b / 'S3B_REPORT.md', encoding='utf-8').read()[:3000])
"""),
md("### S4 — nuisance-aware population with library mismatch, learned retrieval, split-conformal regions, sim-to-real on the measured mixtures"),
code("""
if 's4' in STAGES:
    out4 = P.run_s4(prj, quick=QUICK)
    print(open(prj.s4 / 'S4_REPORT.md', encoding='utf-8').read()[:3000])
"""),
md("### S5 — verified mission filters, D-optimal band design, baselines, learned retrieval on band sets, (433) Eros and (25143) Itokawa (levels 1 and 2)"),
code("""
if 's5' in STAGES:
    out5 = P.run_s5(prj, quick=QUICK, allow_download=NEA_DOWNLOAD)
    print(open(prj.s5 / 'S5_REPORT.md', encoding='utf-8').read()[:3000])
"""),
md("### S6 — the figures of the paper (Figures 1–7) and of the Supporting Information (S1–S11)"),
code("""
if 's6' in STAGES:
    P.run_s6(prj)
    try: from IPython.display import Image, display as disp
    except ImportError: Image = lambda **k: None; disp = lambda *a, **k: None
    for f in ('Figure1.png', 'Figure3.png', 'Figure5.png', 'Figure7.png'):
        p = prj.figures / f
        if p.exists(): print(f); disp(Image(filename=str(p), width=900))
"""),
md("### Results archive — download `SpecID_results.zip` from the Output panel"),
code("""
zpath = P.package(prj); print(zpath)
""")]
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks' / 'SpecID_pipeline.ipynb'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(out, f'{out.stat().st_size / 1e3:.1f} kB')
