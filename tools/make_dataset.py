#!/usr/bin/env python3
"""make_dataset.py -- assemble specid_data.zip (benchmark tables, RELAB raw spectra, NEA archive cache, code snapshot, manifest with MD5s).
Upload the zip as a Kaggle dataset (Kaggle extracts it) or to /content on Colab; the notebook finds it by its manifest."""
import hashlib, json, sys, zipfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / 'data'; CODE = ROOT / 'specid'; OUT = ROOT / 'dist'; OUT.mkdir(exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def main():
    files = []
    for p in sorted(DATA.rglob('*')):
        if p.is_file() and p.name != 'specid_data_manifest.json': files.append(('data', p, p.relative_to(DATA)))
    for p in sorted(CODE.glob('*.py')): files.append(('code', p, Path('code') / 'specid' / p.name))
    manifest = {'name': 'specid-data', 'version': '1.0.0', 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'description': 'Inputs of the SpecID pipeline: curated benchmark tables (RELAB mixtures and end-members), RELAB reflectance spectra (.tab, PDS Geosciences Node, public domain), '
                               'archive spectra of (433) Eros and (25143) Itokawa (SMASS / MITHNEOS, cached as nea_raw.csv), and a snapshot of the specid code.',
                'files': [{'path': str(rel).replace('\\', '/'), 'bytes': p.stat().st_size, 'md5': md5(p)} for _, p, rel in files]}
    (DATA / 'specid_data_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    zpath = OUT / 'specid_data.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(DATA / 'specid_data_manifest.json', 'specid_data_manifest.json')
        for _, p, rel in files: zf.write(p, str(rel))
    print(f'{zpath}: {len(files)} files, {zpath.stat().st_size / 1e6:.1f} MB')

if __name__ == '__main__': main()
