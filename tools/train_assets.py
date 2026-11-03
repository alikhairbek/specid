#!/usr/bin/env python3
"""train_assets.py -- build the assets of SpecID Studio from the deposit (library, error model, mismatch pools, trained retrieval models
for the shipped observation modes, band-parameter model, paper results). Run once; the application only loads the result.

  python train_assets.py --deposit .. --out SpecID_Studio/assets [--quick]"""
from __future__ import annotations
import argparse, json, pickle, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--deposit', default=str(HERE.parent)); ap.add_argument('--out', default=str(HERE / 'SpecID_Studio' / 'assets')); ap.add_argument('--quick', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dep = Path(a.deposit); out = Path(a.out); (out / 'models').mkdir(parents=True, exist_ok=True); (out / 'results').mkdir(exist_ok=True)
    sys.path.insert(0, str(dep)); import specid; import forward_param as FP, s3b_identify as S3B, s4_conformal as S4, s5_bands as S5
    res = dep / 'results'; t0 = time.time(); rng = np.random.default_rng(a.seed)
    # ---- static assets
    for src, dst in ((res / 's3a' / 's3a_library.npz', 's3a_library.npz'), (res / 's3a' / 's2_relative_residuals_silicate.npz', 's2_relative_residuals_silicate.npz'), (res / 's3a' / 'grainsize_rule.json', 'grainsize_rule.json'), (res / 's3a' / 'natural_endmembers_fit.csv', 'natural_endmembers_fit.csv')):
        shutil.copy(src, out / dst)
    model = FP.ParamModel(str(out / 's3a_library.npz')); resid = str(out / 's2_relative_residuals_silicate.npz')
    z = np.load(res / 's3a' / 'benchmark_matrix.npz', allow_pickle=True); Xspec = z['X']; spec_ids = [str(s) for s in z['spectrum_id']]
    qc = pd.read_csv(res / 's3a' / 'spectra_qc.csv'); mix = pd.read_csv(dep / 'data' / 'benchmark' / 'benchmark_mixtures_v0.csv'); em = pd.read_csv(dep / 'data' / 'benchmark' / 'benchmark_endmembers_v0.csv')
    nat = pd.read_csv(out / 'natural_endmembers_fit.csv'); rule = json.load(open(out / 'grainsize_rule.json'))
    # ---- library mismatch pools (held-out natural end-members, as in the paper)
    lib = S4.LibraryMismatch(model, qc, Xspec, spec_ids, em, nat, rule, exclude_codes=S4.test_codes(mix)); print('[assets] mismatch pools:', lib.summary())
    pickle.dump({'pools': lib.pools, 'C': lib.C, 'Ea2': lib.Ea2, 'table': lib.table.to_dict('records')}, open(out / 'mismatch_pools.pkl', 'wb'))   # records, not a DataFrame: the pickle must not depend on pandas/pyarrow
    # ---- the nuisance-aware population (noise-free REFF + targets), simulated once
    n = dict(tr=20000, sc=4000, cal=4000, te=2000) if not a.quick else dict(tr=1500, sc=400, cal=400, te=300)
    st = {k: S4.sample_states(v, rng) for k, v in n.items()}; data = {k: S5.simulate_R(model, v, rng, lib) for k, v in st.items()}
    print(f'[assets] population simulated ({sum(n.values())} states), {time.time() - t0:.0f}s')
    # ---- observation modes shipped with the application
    modes = {'lab': dict(kind='hyper', label='laboratory: absolute reflectance 0.45–2.45 µm, 5 nm, SNR 1000', build=lambda: S3B.Observer(model, 'lab', resid)),
             'tel': dict(kind='hyper', label='telescope: normalised at 0.55 µm, 0.45–2.44 µm, 10 nm, SNR 50', build=lambda: S5.make_hyper_observer(model, (0.45, 2.44), 0.01, 0.55, 50.0, resid)),
             'nir': dict(kind='hyper', label='near-infrared only: normalised at 1.25 µm, 0.56–2.45 µm, 10 nm, SNR 50', build=lambda: S5.make_hyper_observer(model, (0.56, 2.45), 0.01, 1.25, 50.0, resid)),
             'dawnfc': dict(kind='bands', label='Dawn FC (7 filters), normalised at F2, SNR 100', build=lambda: S5.make_band_observer(model, S5.FILTERS_V['dawnfc']['bands'], resid, snr=100.0, norm=S5.FILTERS_V['dawnfc']['norm'])),
             'onct': dict(kind='bands', label='Hayabusa2 ONC-T (7 filters), normalised at v, SNR 100', build=lambda: S5.make_band_observer(model, S5.FILTERS_V['onct']['bands'], resid, snr=100.0, norm=S5.FILTERS_V['onct']['norm'])),
             'mapcam': dict(kind='bands', label='OSIRIS-REx MapCam (4 filters), normalised at v, SNR 100', build=lambda: S5.make_band_observer(model, S5.FILTERS_V['mapcam']['bands'], resid, snr=100.0, norm=S5.FILTERS_V['mapcam']['norm']))}
    index = {}
    for key, m in modes.items():
        t1 = time.time(); obs = m['build'](); print(f'[assets] mode {key}: {obs.cfg["label"]} ({obs.n} channels) — observing and training ...', flush=True)
        X = {k: S5.observe_R(obs, data[k][0], rng) for k in data}; T = {k: data[k][1] for k in data}
        invs = {nm: S4.Inverse(nm, a.seed, a.quick).fit(X['tr'], T['tr']).fit_scale(X['sc'], T['sc']) for nm in ('hgb', 'mlp')}      # primary = lower calibration MAE of M_ol (as in the paper)
        cal_mae = {nm: float(np.mean(np.abs(S4.report_space(T['cal'])[:, 0] - S4.report_space(iv.predict(X['cal']))[:, 0]))) for nm, iv in invs.items()}; prim = min(cal_mae, key=cal_mae.get); inv = invs[prim]
        conf = S4.Conformal(inv).calibrate(X['cal'], T['cal']); qr = S5.ratio_quantile(conf, X['cal'], T['cal'], 0.10)
        ev = S4.evaluate(conf, X['te'], T['te'], 'synthetic_test'); meta, Ym, Tm = S4.measured_set(obs, qc, Xspec, spec_ids, mix, nat, rule)
        conf_meas = S4.Conformal(inv).calibrate(Ym, Tm) if len(meta) else None; qr_meas = S5.ratio_quantile(conf, Ym, Tm, 0.10) if len(meta) else float('nan')
        evm = S4.evaluate(conf, Ym, Tm, 'measured_syncal') if len(meta) else None
        rec = {'key': key, 'kind': m['kind'], 'label': obs.cfg['label'], 'model': prim, 'cal_mae_M_ol': cal_mae, 'cfg': {k_: (list(v) if isinstance(v, tuple) else v) for k_, v in obs.cfg.items()}, 'bands': obs.bands, 'wl': obs.wl.tolist(), 'n': int(obs.n),
               'inv': inv, 'q': conf.q, 'q_meas': conf_meas.q if conf_meas is not None else None, 'qr': float(qr), 'qr_meas': float(qr_meas), 'alphas': list(S4.ALPHAS),
               'eval_synthetic': ev.to_dict('records'), 'eval_measured': evm.to_dict('records') if evm is not None else None, 'n_measured': int(len(meta))}
        pickle.dump(rec, open(out / 'models' / f'{key}.pkl', 'wb')); g = ev[(ev['alpha'] == 0.10) & (ev['target'] == 'M_ol')].iloc[0]
        gm = evm[(evm['alpha'] == 0.10) & (evm['target'] == 'M_ol')].iloc[0] if evm is not None else None
        index[key] = {'label': obs.cfg['label'], 'kind': m['kind'], 'model': prim, 'n': int(obs.n), 'wl_min': float(np.min(obs.wl)), 'wl_max': float(np.max(obs.wl)), 'norm': obs.cfg['norm'], 'snr': obs.cfg['snr'],
                      'synthetic_cov90_M_ol': float(g['coverage']), 'synthetic_hw90_M_ol': float(g['median_halfwidth']), 'measured_cov90_M_ol': float(gm['coverage']) if gm is not None else None, 'measured_mae_M_ol': float(gm['mae']) if gm is not None else None}
        print(f"[assets]   {key}: synthetic coverage {g['coverage']:.2f} (hw {g['median_halfwidth']:.3f}); measured coverage {index[key]['measured_cov90_M_ol']}, MAE {index[key]['measured_mae_M_ol']}; {time.time() - t1:.0f}s", flush=True)
    # ---- the band-parameter (level 2) model
    t1 = time.time(); Xb = {}; Tb = {}
    for k in data:
        xb, _ = S5.band_param_matrix(data[k][0], rng, S5.BP_SLOPE, S5.BP_NOISE); ok = np.all(np.isfinite(xb), axis=1); Xb[k] = xb[ok]; Tb[k] = data[k][1][ok]
    invs = {nm: S4.Inverse(nm, a.seed, a.quick).fit(Xb['tr'], Tb['tr']).fit_scale(Xb['sc'], Tb['sc']) for nm in ('hgb', 'mlp')}
    cal_mae = {nm: float(np.mean(np.abs(S4.report_space(Tb['cal'])[:, 0] - S4.report_space(inv.predict(Xb['cal']))[:, 0]))) for nm, inv in invs.items()}; prim = min(cal_mae, key=cal_mae.get)
    conf = S4.Conformal(invs[prim]).calibrate(Xb['cal'], Tb['cal']); qr = S5.ratio_quantile(conf, Xb['cal'], Tb['cal'], 0.10)
    obs_lab = S3B.Observer(model, 'lab', None); meta, Ym, Tm = S4.measured_set(obs_lab, qc, Xspec, spec_ids, mix, nat, rule); pos = {s: i for i, s in enumerate(spec_ids)}
    Xm, Bm = S5.band_param_matrix(np.array([Xspec[pos[sp]] for sp in meta['spectrum']])); okm = np.all(np.isfinite(Xm), axis=1); Xm, Tm, meta, Bm = Xm[okm], Tm[okm], meta[okm].reset_index(drop=True), Bm[okm]
    P = invs[prim].predict(Xm); S0 = invs[prim].scale(Xm)[:, 0]; r_true = S5._olpx(Tm); r_pred = S5._olpx(P)
    check = {'model': prim, 'n_measured': int(len(Xm)), 'mae_olpx': float(np.mean(np.abs(r_true - r_pred))), 'cov90_olpx': float(np.mean(np.abs(r_true - r_pred) <= S0 * qr)), 'hw90_olpx_median': float(np.median(S0 * qr))}
    meas = meta.copy()
    for j, c_ in enumerate(S5.BP_COLS): meas[c_] = Bm[:, j]
    meas['olpx_true'] = r_true; meas['olpx_pred'] = r_pred
    pickle.dump({'inv': invs[prim], 'q': conf.q, 'qr': float(qr), 'features': S5.BP_FEATURES, 'noise': S5.BP_NOISE, 'slope': S5.BP_SLOPE, 'check': check, 'measured': meas.to_dict('records')}, open(out / 'models' / 'bandparam.pkl', 'wb'))
    print(f'[assets] band-parameter model ({prim}): measured MAE ol/(ol+px) {check["mae_olpx"]:.3f}, coverage {check["cov90_olpx"]:.2f}; {time.time() - t1:.0f}s')
    # ---- paper results for the browser
    for sub in ('s2', 's3a', 's3b', 's4', 's5', 's5b'):
        (out / 'results' / sub).mkdir(exist_ok=True)
        for p in (res / sub).glob('*'):
            if p.suffix in ('.csv', '.md', '.json') and p.stat().st_size < 3_000_000: shutil.copy(p, out / 'results' / sub / p.name)
    figs = out / 'results' / 'figures'; figs.mkdir(exist_ok=True)
    for p in (dep / 'manuscript' / 'figures').glob('*.png'): shutil.copy(p, figs / p.name)
    json.dump({'version': specid.__version__, 'created': time.strftime('%Y-%m-%d'), 'modes': index, 'bandparam': check, 'population': n, 'quick': a.quick, 'mismatch_pools': lib.summary()}, open(out / 'index.json', 'w'), indent=1)
    print(f'[assets] done in {(time.time() - t0) / 60:.1f} min -> {out}')

if __name__ == '__main__': main()
