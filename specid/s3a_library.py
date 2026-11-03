#!/usr/bin/env python3
"""
s3a_library.py -- Step 3a: end-member optical-constant library and the continuous forward model.

Stages (outputs under <PROJ>/s3a/):
  A  tables     : spectra list (size series, Dyar olivine Fo series, Klima pyroxene quadrilateral, natural cpx, laser pellets)
                  + the S2 benchmark tables (for the mixture validation)
  B  fetch/load : download the .tab spectra (resume-safe) and load them on the 0.40-2.50 um grid
  C  grain size : derive k from the 45-75 um fraction of each size series; fit the effective Hapke diameter of every other
                  sieve class -> global rule d_eff(class); validate the slab model across sizes
  D  olivine    : k(lambda; Fo) from the Dyar synthetic olivines (<45 um) + leave-one-out validation
  E  pyroxene   : k(lambda; Wo, Fs) from the Klima synthetic pyroxenes (<45 / <100 um) by thin-plate-spline interpolation
                  on the quadrilateral + leave-one-out validation; natural end-members -> effective compositions
  F  weathering : transfer function T(lambda) = w(irradiated)/w(fresh) from the XT-TXH pellets (0, 1x, 2x pulse-laser)
  G  mixtures   : the parametric model fitted to the tier-A silicate benchmark mixtures (abundance + chemistry [+ size])
  H  report     : S3A_REPORT.md, s3a_library.npz, S3a_results.zip
"""
from __future__ import annotations
import json, math, os, re, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_forward as S2
import forward_param as FP

VERSION = 's3a-library/1.1'
GRID = S2.GRID; WONG = S2.WONG; INK = S2.INK
EVAL = (GRID >= 0.45) & (GRID <= 2.45)

SIZE_SERIES = {   # specimen base -> (family, {specimen: (lo, hi)}) ; reference class = the one containing 45-75 um (or 25-75)
    'PO-CMP-081': 'ol', 'PA-CMP-060': 'plag', 'PE-CMP-041': 'opx', 'PP-CMP-027': 'cpx', 'PP-CMP-02x': 'cpx', 'PE-CMP-034': 'opx', 'PE-CMP-03x': 'opx', 'PO-CMP-032': 'ol'}

def series_of(spec):
    if re.match(r'^PP-CMP-02[1-6]$', spec): return 'PP-CMP-02x'
    if re.match(r'^PE-CMP-03[012]$', spec): return 'PE-CMP-03x'
    m = re.match(r'^([A-Z]{2}-[A-Z]{3}-\d{3})', spec); return m.group(1) if m and m.group(1) in SIZE_SERIES else None

def nominal_size(lo, hi):
    lo, hi = S2._f(lo), S2._f(hi)
    if hi <= 0: return np.nan
    return hi / 2.0 if lo <= 0 else 0.5 * (lo + hi)

def d_eff_rule(lo, hi, rule):
    """effective Hapke diameter of a sieve class from the calibrated rule {'bounded': ratio, 'unbounded': ratio}"""
    lo, hi = S2._f(lo), S2._f(hi)
    if hi <= 0: return 60.0 * rule['bounded']
    if lo <= 0: return hi * rule['unbounded']
    return 0.5 * (lo + hi) * rule['bounded']

# ----------------------------------------------------------------------------- stage A/B
def stage_tables(bench: Path, out: Path):
    lst = pd.read_csv(bench / 's3a_spectra_list_v0.csv'); sp2 = pd.read_csv(bench / 'benchmark_spectra_v0.csv')
    mix = pd.read_csv(bench / 'benchmark_mixtures_v0.csv'); em = pd.read_csv(bench / 'benchmark_endmembers_v0.csv')
    mix['flags'] = mix['flags'].fillna(''); mix['basis'] = mix['basis'].replace({'wt_from_mass': 'wt'})
    need = sp2[sp2['role'].isin(['mixture', 'endmember'])][['spectrum_id', 'specimen_id', 'collection', 'tab_file', 'incidence_deg', 'emission_deg', 'min_size_um', 'max_size_um']].copy()
    need['role'] = 'benchmark'; need['specimen_name'] = ''; need['chem'] = '{}'; need['note'] = 'S2 benchmark'
    allsp = pd.concat([lst, need[~need['spectrum_id'].isin(lst['spectrum_id'])]], ignore_index=True)
    print(f"[A] spectra to load: {len(allsp)} ({lst['role'].value_counts().to_dict()} + {len(allsp) - len(lst)} benchmark)")
    return lst, allsp, mix, em

def stage_load(allsp: pd.DataFrame, raw: Path, workers: int, local_hint: str, out: Path):
    paths = S2.stage_fetch(allsp, raw, workers, local_hint); qc, X = S2.stage_load(allsp, paths, out)
    return qc, X

# ----------------------------------------------------------------------------- stage C: grain size
def stage_grainsize(lst, qc, X, out: Path):
    plt = S2.style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; ok = qc[(qc['status'] == 'ok') & (qc['role'] == 'size_series')].copy()
    ok['series'] = ok['specimen_id'].apply(series_of); ok = ok[ok['series'].notna()]
    rows, curves = [], {}
    for ser, g in ok.groupby('series'):
        fam = SIZE_SERIES[ser]; g = g.copy(); g['lo'] = g['min_size_um'].apply(S2._f); g['hi'] = g['max_size_um'].apply(S2._f); g['nom'] = [nominal_size(a, b) for a, b in zip(g['lo'], g['hi'])]
        ref = g[(g['lo'] == 45) & (g['hi'] == 75)]
        if not len(ref): ref = g[(g['lo'] == 25) & (g['hi'] == 75)]
        if not len(ref): print(f'[C] {ser}: no reference class'); continue
        rr = ref.iloc[0]; mu0, mu = FP.mus(rr['incidence_deg'], rr['emission_deg']); n = FP.n_of(fam)
        w_ref = FP.w_from_reff(X[pos[rr['spectrum_id']]], mu0, mu); k_ref = FP.slab_k(w_ref, n, rr['nom'], GRID)
        curves[ser] = {'k_ref': k_ref, 'ref': rr['specimen_id']}
        dgrid = np.exp(np.linspace(np.log(2), np.log(3000), 400))
        for _, r in g.iterrows():
            w = FP.w_from_reff(X[pos[r['spectrum_id']]], *FP.mus(r['incidence_deg'], r['emission_deg']))
            err = [np.sqrt(np.nanmean((FP.slab_w(k_ref, n, d, GRID)[EVAL] - w[EVAL]) ** 2)) for d in dgrid]
            j = int(np.argmin(err)); d_eff = dgrid[j]
            rows.append({'series': ser, 'family': fam, 'specimen_id': r['specimen_id'], 'spectrum_id': r['spectrum_id'], 'lo': r['lo'], 'hi': r['hi'], 'nominal_um': r['nom'], 'is_reference': r['specimen_id'] == rr['specimen_id'],
                         'd_eff_um': float(d_eff), 'ratio_eff_nominal': float(d_eff / r['nom']) if r['nom'] > 0 else np.nan, 'rms_w_at_deff': float(err[j]),
                         'rms_w_at_nominal': float(np.sqrt(np.nanmean((FP.slab_w(k_ref, n, r['nom'], GRID)[EVAL] - w[EVAL]) ** 2))) if r['nom'] > 0 else np.nan, 'w_075': float(w[np.argmin(abs(GRID - 0.75))])})
    cal = pd.DataFrame(rows); cal.to_csv(out / 'grainsize_calibration.csv', index=False)
    b = cal[(~cal['is_reference']) & (cal['lo'] > 0) & (cal['hi'] <= 250)]; u = cal[(cal['lo'] <= 0) & (cal['hi'] > 0)]
    rule = {'bounded': float(b['ratio_eff_nominal'].median()) if len(b) else 1.0, 'unbounded': float((u['d_eff_um'] / u['hi']).median()) if len(u) else 0.5,
            'n_bounded': int(len(b)), 'n_unbounded': int(len(u)), 'bounded_iqr': [float(b['ratio_eff_nominal'].quantile(q)) for q in (0.25, 0.75)] if len(b) else None,
            'unbounded_iqr': [float((u['d_eff_um'] / u['hi']).quantile(q)) for q in (0.25, 0.75)] if len(u) else None, 'coarse_note': '>250 um classes excluded from the bounded rule (saturation of the slab model)'}
    # validation with the global rule: predicted REFF at each class vs measured
    vrows = []
    for _, r in cal.iterrows():
        n = FP.n_of(r['family']); d = d_eff_rule(r['lo'], r['hi'], rule); sp = qc[qc['spectrum_id'] == r['spectrum_id']].iloc[0]; mu0, mu = FP.mus(sp['incidence_deg'], sp['emission_deg'])
        Rp = FP.reff_from_w(FP.slab_w(curves[r['series']]['k_ref'], n, d, GRID), mu0, mu); Rm = X[pos[r['spectrum_id']]]
        vrows.append({**{k: r[k] for k in ('series', 'family', 'specimen_id', 'lo', 'hi', 'nominal_um', 'is_reference')}, 'd_rule_um': d, 'rms_reff_rule': float(np.sqrt(np.nanmean((Rp[EVAL] - Rm[EVAL]) ** 2))),
                      'r075_meas': float(Rm[np.argmin(abs(GRID - 0.75))]), 'r075_pred': float(Rp[np.argmin(abs(GRID - 0.75))])})
    val = pd.DataFrame(vrows); val.to_csv(out / 'grainsize_validation.csv', index=False)
    (out / 'grainsize_rule.json').write_text(json.dumps(rule, indent=1))
    # figure: per series measured vs predicted at the rule sizes
    sers = sorted(curves); fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False)
    for k, ser in enumerate(sers):
        ax = axes[k // 4][k % 4]; g = val[val['series'] == ser].sort_values('nominal_um')
        for j, (_, r) in enumerate(g.iterrows()):
            sp = qc[qc['specimen_id'] == r['specimen_id']].iloc[0]; mu0, mu = FP.mus(sp['incidence_deg'], sp['emission_deg']); col = WONG[j % 7]
            ax.plot(GRID, X[pos[sp['spectrum_id']]], color=col, lw=1.4, label=f"{int(r['lo'])}-{int(r['hi'])} µm (d_eff {r['d_rule_um']:.0f}; rms {r['rms_reff_rule']:.3f})")
            ax.plot(GRID, FP.reff_from_w(FP.slab_w(curves[ser]['k_ref'], FP.n_of(r['family']), r['d_rule_um'], GRID), mu0, mu), color=col, lw=0.9, ls='--')
        ax.set_title(f"{ser} ({SIZE_SERIES[ser]}): k from {curves[ser]['ref']}", fontsize=8); ax.legend(fontsize=5.5, frameon=False); ax.set_xlabel('µm', fontsize=8)
    for k in range(len(sers), 8): axes[k // 4][k % 4].axis('off')
    fig.suptitle(f"Hapke slab model across sieve classes (solid measured, dashed predicted); rule: d_eff = {rule['bounded']:.2f} x class mean, {rule['unbounded']:.2f} x upper bound for '<X' classes", fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S3A_F1_grainsize.png'); plt.close(fig)
    print(f"[C] grain-size rule: bounded ratio {rule['bounded']:.2f} (n={rule['n_bounded']}), unbounded {rule['unbounded']:.2f} (n={rule['n_unbounded']}); "
          f"median RMS(REFF) with the rule: {val[~val['is_reference']]['rms_reff_rule'].median():.4f} (classes <=250 um: {val[(~val['is_reference']) & (val['hi'] <= 250)]['rms_reff_rule'].median():.4f})")
    return cal, val, rule, curves

# ----------------------------------------------------------------------------- stage D: olivine Fo library
ANCHORS_OL = {'PO-17': 91.0, 'CK-SCOL': 90.0, 'OL-014': 90.0, 'AG-OL': 90.0, 'PO-81-C': 90.0}      # natural Fo~90 olivines used to anchor the k level

def _huber_fit(Xd, Y, iters=8, c=1.345):
    """robust (Huber, IRLS) multi-output linear regression: Y (n, m) ~ Xd (n, p); one weight per sample (from its median residual)"""
    W = np.ones(len(Y))
    for _ in range(iters):
        sw = np.sqrt(W)[:, None]; B, *_ = np.linalg.lstsq(Xd * sw, Y * sw, rcond=None); R = Y - Xd @ B
        r = np.median(np.abs(R), axis=1); s = 1.4826 * np.median(r) + 1e-9; u = r / s; W = np.where(u <= c, 1.0, c / u)
    return B

def _ol_design(Fo):
    x = (100.0 - np.asarray(Fo, float)) / 100.0; return np.column_stack([np.ones_like(x), x, x ** 2])

def _k_of_row(r, X, pos, fam, chem, rule):
    w = FP.w_from_reff(X[pos[r['spectrum_id']]], *FP.mus(r['incidence_deg'], r['emission_deg'])); d = d_eff_rule(r['min_size_um'], r['max_size_um'], rule)
    return FP.slab_k(w, FP.n_of(fam, chem), d, GRID), d, w

def stage_olivine(lst, qc, X, rule, out: Path):
    """k(lambda; Fo) from the Dyar synthetic suite as a node library: per-member log k, level-corrected against a robust quadratic
    trend in Fe, members whose Band I centre departs from the robust Band-I-centre-vs-Fo line by > 12 nm rejected, nodes = unique Fo
    (mean of the level-corrected members); the Fo 90 node is replaced by the natural Fo~90 anchors in stage_natural, where the level is
    also anchored.  Piecewise-linear interpolation of log k in Fo keeps the measured band shift."""
    plt = S2.style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}
    g = qc[(qc['status'] == 'ok') & (qc['role'] == 'olivine_Fo')].copy(); g['Fo'] = g['chem'].apply(lambda s: json.loads(s).get('Fo', np.nan)); g = g[g['Fo'].notna()].sort_values('Fo')
    if len(g) < 5: print('[D] olivine library: too few spectra'); return None
    K, rows = [], []
    for _, r in g.iterrows():
        k, d, w = _k_of_row(r, X, pos, 'ol', r['Fo'], rule); K.append(np.log(np.clip(k, 1e-8, None)))
        bp = S2.band_params(GRID, FP.reff_from_w(FP.slab_w(k, FP.n_of('ol', r['Fo']), 60.0, GRID), *FP.mus(30, 0)))
        rows.append({'specimen_id': r['specimen_id'], 'name': r['specimen_name'], 'Fo': r['Fo'], 'd_eff_um': d, 'k_105': float(k[np.argmin(abs(GRID - 1.05))]), 'b1_center_d60': bp.get('b1_center'), 'b1_depth_d60': bp.get('b1_depth')})
    K = np.vstack(K); lib = pd.DataFrame(rows); fe = lib['Fo'].values < 99.0
    bic = lib['b1_center_d60'].values.astype(float); ok = fe & np.isfinite(bic)
    A = np.column_stack([np.ones(ok.sum()), lib['Fo'].values[ok]]); coef = _huber_fit(A, bic[ok][:, None])[:, 0]
    resid = bic - (coef[0] + coef[1] * lib['Fo'].values); keep = ok & (np.abs(resid) <= 0.012)
    lib['bic_trend_resid_nm'] = resid * 1000; lib['kept'] = keep
    B = _huber_fit(_ol_design(lib['Fo'].values[keep]), K[keep]); c = np.array([np.median((K[i] - (_ol_design([lib['Fo'][i]]) @ B)[0])[EVAL]) for i in range(len(lib))]); lib['level_c'] = c
    Kc = K - c[:, None]
    nodes_Fo = np.unique(lib['Fo'].values[keep]); nodes = np.vstack([Kc[keep & (lib['Fo'].values == f)].mean(axis=0) for f in nodes_Fo])
    if (~fe).any(): nodes_Fo = np.append(nodes_Fo, 100.0); nodes = np.vstack([nodes, K[~fe].mean(axis=0)])        # Fe-free forsterite as the Fo 100 node (uncorrected level)
    def interp_nodes(nF, nK, Fo): return np.array([np.interp(Fo, nF, nK[:, c]) for c in range(nK.shape[1])])
    loo = []
    for i in np.where(keep)[0]:
        m = keep.copy(); m[i] = False; nF = np.unique(lib['Fo'].values[m])
        if not (nF.min() < lib['Fo'][i] < nF.max()): continue
        nK = np.vstack([Kc[m & (lib['Fo'].values == f)].mean(axis=0) for f in nF]); kp = np.exp(interp_nodes(nF, nK, lib['Fo'][i]) + c[i])
        n = FP.n_of('ol', lib['Fo'][i]); R_t = FP.reff_from_w(FP.slab_w(np.exp(K[i]), n, 60.0, GRID), *FP.mus(30, 0)); R_p = FP.reff_from_w(FP.slab_w(kp, n, 60.0, GRID), *FP.mus(30, 0))
        b_t = S2.band_params(GRID, R_t); b_p = S2.band_params(GRID, R_p)
        loo.append({'specimen_id': lib['specimen_id'][i], 'Fo': lib['Fo'][i], 'rms_reff_d60': float(np.sqrt(np.mean((R_t[EVAL] - R_p[EVAL]) ** 2))), 'd_b1_center': b_p.get('b1_center', np.nan) - b_t.get('b1_center', np.nan),
                    'd_b1_depth': b_p.get('b1_depth', np.nan) - b_t.get('b1_depth', np.nan), 'rms_logk': float(np.sqrt(np.mean((np.log(kp) - K[i])[EVAL] ** 2)))})
    loo = pd.DataFrame(loo); lib.to_csv(out / 'olivine_library.csv', index=False); loo.to_csv(out / 'olivine_loo.csv', index=False)
    Fo_grid = np.arange(0, 101, 5); bc = [S2.band_params(GRID, FP.reff_from_w(FP.slab_w(np.exp(interp_nodes(nodes_Fo, nodes, f)), FP.n_of('ol', f), 60.0, GRID), *FP.mus(30, 0))).get('b1_center', np.nan) for f in Fo_grid]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3)); cm = plt.get_cmap('viridis')
    for i in range(len(lib)): axes[0].plot(GRID, np.exp(K[i]), color=cm(lib['Fo'][i] / 100.0), lw=0.8, alpha=0.9 if keep[i] else 0.35, ls='-' if keep[i] else ':')
    axes[0].set_yscale('log'); axes[0].set_xlabel('µm'); axes[0].set_ylabel('k'); axes[0].set_title(f'olivine k(λ): Dyar synthetic members (dotted = rejected, {int((~keep & fe).sum())})', fontsize=8.5)
    for f in Fo_grid[::4]: axes[1].plot(GRID, FP.reff_from_w(FP.slab_w(np.exp(interp_nodes(nodes_Fo, nodes, f)), FP.n_of('ol', f), 60.0, GRID), *FP.mus(30, 0)), color=cm(f / 100), lw=1.2, label=f'Fo {f}')
    axes[1].set_xlabel('µm'); axes[1].set_ylabel('REFF (d = 60 µm, i=30/e=0)'); axes[1].set_title('node library (level-corrected, before anchoring)', fontsize=8.5); axes[1].legend(fontsize=6, frameon=False, ncol=2)
    axes[2].scatter(lib['Fo'][keep], lib['b1_center_d60'][keep], s=18, color=WONG[0], label='members kept'); axes[2].scatter(lib['Fo'][~keep & fe], lib['b1_center_d60'][~keep & fe], s=18, marker='x', color=WONG[3], label='rejected (>12 nm off the line)')
    axes[2].plot(Fo_grid, coef[0] + coef[1] * Fo_grid, color='#999999', lw=0.8, label=f'robust line: {1000 * coef[1]:.2f} nm per Fo'); axes[2].plot(Fo_grid, bc, color=WONG[2], lw=1.4, label='node library')
    axes[2].set_xlabel('Fo'); axes[2].set_ylabel('Band I centre (µm), d = 60 µm'); axes[2].legend(fontsize=6.5, frameon=False); axes[2].set_title('olivine Band I centre vs Fo', fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S3A_F2_olivine_library.png'); plt.close(fig)
    print(f"[D] olivine library: {len(lib)} members, {int(keep.sum())} kept ({int((~keep & fe).sum())} rejected by the Band-I line, slope {1000 * coef[1]:.2f} nm/Fo); nodes {len(nodes_Fo)}; "
          f"LOO median |ΔBand I centre| {loo['d_b1_center'].abs().median() * 1000:.1f} nm, RMS(log k) {loo['rms_logk'].median():.3f}; node-library Band I centre Fo 90 -> {bc[18]:.3f}, Fo 10 -> {bc[2]:.3f} µm")
    return {'B': B, 'nodes_Fo': nodes_Fo, 'nodes_logk': nodes, 'lib': lib, 'loo': loo, 'K_all': K, 'bc_trend': (Fo_grid, bc), 'bic_line': coef}

# ----------------------------------------------------------------------------- stage E: pyroxene quadrilateral library
def _px_chem(s):
    c = json.loads(s); Wo, En, Fs = c.get('Wo'), c.get('En'), c.get('Fs')
    if Fs is None and En is not None and Wo is not None: Fs = 100 - Wo - En
    if Wo is None and En is not None and Fs is not None: Wo = max(0.0, 100 - En - Fs)
    if En is None and Wo is not None and Fs is not None: En = 100 - Wo - Fs
    if Fs is None and En is not None and Wo is None: Wo, Fs = 0.0, 100 - En
    if Wo is None and Fs is not None and En is None: Wo, En = 0.0, 100 - Fs
    if Wo is None or Fs is None: return None
    return float(Wo), float(Fs)

def stage_pyroxene(lst, qc, X, rule, em, mix, out: Path):
    """k(lambda; Wo, Fs) by thin-plate-spline interpolation of log k on the quadrilateral (Klima synthetic suite); smoothing chosen by
    leave-one-out band-centre error; level anchored later on natural pyroxenes (stage_natural)."""
    plt = S2.style(); from scipy.interpolate import RBFInterpolator
    pos = {s: i for i, s in enumerate(qc['spectrum_id'])}
    g = qc[(qc['status'] == 'ok') & (qc['role'].isin(['pyroxene_Klima', 'pyroxene_synthetic']))].copy(); g['WoFs'] = g['chem'].apply(_px_chem); g = g[g['WoFs'].notna()]
    if len(g) < 10: print('[E] pyroxene library: too few spectra'); return None
    K, rows = [], []
    for _, r in g.iterrows():
        Wo, Fs = r['WoFs']; k, d, w = _k_of_row(r, X, pos, 'opx' if Wo < 20 else 'cpx', Fs, rule); K.append(np.log(np.clip(k, 1e-8, None)))
        bp = S2.band_params(GRID, FP.reff_from_w(FP.slab_w(k, FP.n_of('opx', Fs), 60.0, GRID), *FP.mus(30, 0)))
        rows.append({'specimen_id': r['specimen_id'], 'name': r['specimen_name'], 'Wo': Wo, 'Fs': Fs, 'En': 100 - Wo - Fs, 'd_eff_um': d, 'size_class': f"{S2._f(r['min_size_um']):g}-{S2._f(r['max_size_um']):g}",
                     'b1_center_d60': bp.get('b1_center'), 'b2_center_d60': bp.get('b2_center'), 'b1_depth_d60': bp.get('b1_depth')})
    K = np.vstack(K); lib = pd.DataFrame(rows); fe = lib['Fs'].values >= 2.0
    lib, K = lib[fe].reset_index(drop=True), K[fe]; P = lib[['Wo', 'Fs']].values / 100.0; fe = np.ones(len(lib), bool)          # Fe-free members excluded from the library
    trend = RBFInterpolator(P, K, kernel='thin_plate_spline', smoothing=3.0)(P); c = np.median((K - trend)[:, EVAL], axis=1); lib['level_c'] = c; K = K - c[:, None]   # level correction
    def loo_eval(sm):
        out = []
        for i in np.where(fe)[0]:
            o = np.array([j for j in range(len(lib)) if j != i]); kp = np.exp(RBFInterpolator(P[o], K[o], kernel='thin_plate_spline', smoothing=sm)(P[i:i + 1])[0] + c[i])
            n = FP.n_of('opx', lib['Fs'][i]); R_t = FP.reff_from_w(FP.slab_w(np.exp(K[i] + c[i]), n, 60.0, GRID), *FP.mus(30, 0)); R_p = FP.reff_from_w(FP.slab_w(kp, n, 60.0, GRID), *FP.mus(30, 0))
            b_t = S2.band_params(GRID, R_t); b_p = S2.band_params(GRID, R_p)
            out.append({'specimen_id': lib['specimen_id'][i], 'Wo': lib['Wo'][i], 'Fs': lib['Fs'][i], 'rms_reff_d60': float(np.sqrt(np.mean((R_t[EVAL] - R_p[EVAL]) ** 2))), 'rms_logk': float(np.sqrt(np.mean((np.log(kp) - K[i] - c[i])[EVAL] ** 2))),
                        'd_b1_center': b_p.get('b1_center', np.nan) - b_t.get('b1_center', np.nan), 'd_b2_center': b_p.get('b2_center', np.nan) - b_t.get('b2_center', np.nan)})
        return pd.DataFrame(out)
    best = None
    for sm in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        lo = loo_eval(sm); crit = float(lo['d_b1_center'].abs().median() + 0.5 * lo['d_b2_center'].abs().median() + 0.02 * lo['rms_logk'].median())
        print(f"[E] TPS smoothing {sm:g}: LOO median |ΔB1| {lo['d_b1_center'].abs().median() * 1000:.1f} nm, |ΔB2| {lo['d_b2_center'].abs().median() * 1000:.1f} nm, RMS(log k) {lo['rms_logk'].median():.3f}")
        if best is None or crit < best[1]: best = (sm, crit, lo)
    sm, _, loo = best; lib.to_csv(out / 'pyroxene_library.csv', index=False); loo.to_csv(out / 'pyroxene_loo.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    sc = axes[0].scatter(lib['Fs'], lib['Wo'], c=lib['b1_center_d60'], cmap='viridis', s=28); plt.colorbar(sc, ax=axes[0], label='Band I centre (µm), d = 60 µm')
    axes[0].set_xlabel('Fs'); axes[0].set_ylabel('Wo'); axes[0].set_title(f'Klima pyroxene quadrilateral: {len(lib)} members', fontsize=8.5)
    sc = axes[1].scatter(lib['Fs'], lib['Wo'], c=lib['b2_center_d60'], cmap='plasma', s=28); plt.colorbar(sc, ax=axes[1], label='Band II centre (µm)'); axes[1].set_xlabel('Fs'); axes[1].set_ylabel('Wo')
    axes[2].scatter(loo['d_b1_center'] * 1000, loo['d_b2_center'] * 1000, s=16, color=WONG[0]); axes[2].axhline(0, color='#999999', lw=0.6); axes[2].axvline(0, color='#999999', lw=0.6)
    axes[2].set_xlabel('LOO error, Band I centre (nm)'); axes[2].set_ylabel('LOO error, Band II centre (nm)'); axes[2].set_title(f'leave-one-out interpolation (TPS smoothing {sm:g})', fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S3A_F3_pyroxene_library.png'); plt.close(fig)
    print(f"[E] pyroxene library: {len(lib)} members; smoothing {sm:g}; LOO median |ΔBand I| {loo['d_b1_center'].abs().median() * 1000:.1f} nm, |ΔBand II| {loo['d_b2_center'].abs().median() * 1000:.1f} nm, RMS(REFF, d60) {loo['rms_reff_d60'].median():.4f}")
    return {'WoFs': lib[['Wo', 'Fs']].values, 'logk': K, 'lib': lib, 'loo': loo, 'smoothing': sm}

# ----------------------------------------------------------------------------- natural end-members: level anchoring + effective composition
def _lib_k_fns(ol, px, off_ol, off_opx, off_cpx, ol_nodes=None):
    from scipy.interpolate import RBFInterpolator
    rbf = RBFInterpolator(px['WoFs'] / 100.0, px['logk'], kernel='thin_plate_spline', smoothing=px['smoothing'])
    nF, nK = ol_nodes if ol_nodes is not None else (ol['nodes_Fo'], ol['nodes_logk'])
    def k_ol(Fo):
        Fo = float(np.clip(Fo, nF.min(), nF.max())); j = int(np.searchsorted(nF, Fo)); j = min(max(j, 1), len(nF) - 1); tt = (Fo - nF[j - 1]) / (nF[j] - nF[j - 1])
        return np.exp((1 - tt) * nK[j - 1] + tt * nK[j] + off_ol)
    def k_px(Wo, Fs, fam): return np.exp(rbf(np.array([[Wo / 100.0, np.clip(Fs, 5, 100) / 100.0]]))[0] + (off_opx if fam == 'opx' else off_cpx))
    return k_ol, k_px

def _fit_natural(fam, w, d0, k_ol, k_px):
    from scipy.optimize import minimize
    def cost(v):
        if fam == 'ol': wp = FP.slab_w(k_ol(v[0]), FP.n_of('ol', v[0]), math.exp(v[1]), GRID)
        elif fam == 'opx': wp = FP.slab_w(k_px(FP.WO_OPX, v[0], 'opx'), FP.n_of('opx', v[0]), math.exp(v[1]), GRID)
        else: wp = FP.slab_w(k_px(v[2], v[0], 'cpx'), FP.n_of('cpx'), math.exp(v[1]), GRID)
        return float(np.sqrt(np.mean((wp[EVAL] - w[EVAL]) ** 2)))
    starts = [(f, math.log(d0)) for f in (10, 30, 50, 70, 90)] if fam != 'cpx' else [(f, math.log(d0), wo) for f in (5, 20, 45) for wo in (25, 40, 50)]
    # size (= level) constrained to a factor 2 around the rule value so that the chemistry is read from band positions, not traded against the level
    bnds = [(0, 95), (math.log(d0 / 2), math.log(d0 * 2))] + ([(5, 55)] if fam == 'cpx' else [])
    best = None
    for s0 in starts:
        res = minimize(cost, np.array(s0, float), method='L-BFGS-B', bounds=bnds)
        if best is None or res.fun < best.fun: best = res
    return best

def stage_natural(qc, X, rule, em, ol, px, out: Path):
    """(1) fit composition + size of the natural benchmark end-members with the un-anchored libraries; (2) anchor the library level so
    that the natural samples are reproduced at the calibrated d_eff (olivine: Fo~90 anchors; pyroxene: chemistry from the band positions
    of step 1); (3) refit; hold-one-anchor-out check of the anchoring."""
    pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; okset = set(qc.loc[qc['status'] == 'ok', 'spectrum_id'])
    famap = {'olivine': 'ol', 'opx': 'opx', 'cpx': 'cpx', 'plagioclase': 'plag'}
    items = []
    for _, e in em[em['family'].isin(['olivine', 'opx', 'cpx'])].iterrows():
        sid = e['primary_spectrum']
        if sid not in okset: continue
        r = qc[qc['spectrum_id'] == sid].iloc[0]; w = FP.w_from_reff(X[pos[sid]], *FP.mus(r['incidence_deg'], r['emission_deg'])); d0 = d_eff_rule(e['size_lo_um'], e['size_hi_um'], rule)
        items.append({'code': e['code'], 'em_id': e['em_id'], 'name': e['name'], 'fam': famap[e['family']], 'w': w, 'd0': d0, 'is_ol_anchor': e['code'] in ANCHORS_OL})
    k_ol0, k_px0 = _lib_k_fns(ol, px, 0.0, 0.0, 0.0)
    for it in items:
        b = _fit_natural(it['fam'], it['w'], it['d0'], k_ol0, k_px0); it['fit0'] = b.x; it['rms0'] = float(b.fun)
    # level offsets: log k measured at d0 minus library log k at the fitted chemistry (olivine anchors at their nominal Fo)
    def meas_logk(it, chem): return np.log(np.clip(FP.slab_k(it['w'], FP.n_of(it['fam'], chem), it['d0'], GRID), 1e-9, None))
    def offset_from(subset):
        diffs = []
        for it in subset:
            if it['fam'] == 'ol': Fo = ANCHORS_OL.get(it['code'], it['fit0'][0]); diffs.append(meas_logk(it, Fo) - np.log(k_ol0(Fo)))
            elif it['fam'] == 'opx': diffs.append(meas_logk(it, it['fit0'][0]) - np.log(k_px0(FP.WO_OPX, it['fit0'][0], 'opx')))
            else: diffs.append(meas_logk(it, it['fit0'][0]) - np.log(k_px0(it['fit0'][2], it['fit0'][0], 'cpx')))
        return np.median(np.vstack(diffs), axis=0) if diffs else np.zeros(len(GRID))
    anchors_ol = [it for it in items if it['is_ol_anchor']]; nat_opx = [it for it in items if it['fam'] == 'opx' and it['rms0'] < 0.05]; nat_cpx = [it for it in items if it['fam'] == 'cpx' and it['rms0'] < 0.06]
    off_ol = offset_from(anchors_ol); off_opx = offset_from(nat_opx); off_cpx = offset_from(nat_cpx)
    # anchored olivine node library: synthetic nodes + offset, the Fo 90 node replaced by the median natural anchor (synthetic nodes within Fo 85-95 dropped)
    def anchored_nodes(anchor_items, smooth_deg=3):
        """synthetic nodes + level offset, Fo-90 node = median natural anchor; then smoothed by a weighted cubic in Fo per wavelength
        (natural node weighted x3) and resampled on Fo = 0, 5, ..., 95 (the Fe-free Fo-100 member is not part of the model)"""
        nF, nK = ol['nodes_Fo'], ol['nodes_logk'] + off_ol; keepn = (nF < 85) | (nF > 95)
        if anchor_items:
            nat_node = np.median(np.vstack([meas_logk(it, ANCHORS_OL.get(it['code'], 90.0)) for it in anchor_items]), axis=0)
            nF = np.append(nF[keepn], 90.0); nK = np.vstack([nK[keepn], nat_node]); o = np.argsort(nF); nF, nK = nF[o], nK[o]
        if smooth_deg > 0:
            sel = nF <= 95.0; w = np.sqrt(np.where(nF[sel] == 90.0, 3.0, 1.0))[:, None]; V = np.vander(nF[sel] / 100.0, smooth_deg + 1, increasing=True)
            C, *_ = np.linalg.lstsq(V * w, nK[sel] * w, rcond=None); nF = np.arange(0.0, 95.01, 5.0); nK = np.vander(nF / 100.0, smooth_deg + 1, increasing=True) @ C
        return nF, nK
    ol_nodes = anchored_nodes(anchors_ol); ol['nodes_Fo_anch'], ol['nodes_logk_anch'] = ol_nodes
    k_ol1, k_px1 = _lib_k_fns(ol, px, 0.0, off_opx, off_cpx, ol_nodes=ol_nodes)
    rows = []
    for it in items:
        b = _fit_natural(it['fam'], it['w'], it['d0'], k_ol1, k_px1)
        row = {'code': it['code'], 'em_id': it['em_id'], 'name': it['name'], 'family': it['fam'], 'model': 'quad', 'd0_um': it['d0'], 'd_fit0_um': math.exp(it['fit0'][1]), 'rms_w0': it['rms0'], 'd_fit_um': math.exp(b.x[1]), 'rms_w': float(b.fun),
               'l_fit': float(b.x[1] - math.log(it['d0'])), 'anchor': it['is_ol_anchor'] or (it['fam'] == 'opx' and it in nat_opx) or (it['fam'] == 'cpx' and it in nat_cpx)}
        if it['fam'] == 'ol': row['Fo_fit0'] = it['fit0'][0]; row['Fo_fit'] = b.x[0]
        else: row['Fs_fit0'] = it['fit0'][0]; row['Fs_fit'] = b.x[0]; row['Wo_fit'] = b.x[2] if it['fam'] == 'cpx' else FP.WO_OPX
        # hold-one-anchor-out: offset without this sample
        if row['anchor']:
            grp = anchors_ol if it['fam'] == 'ol' else (nat_opx if it['fam'] == 'opx' else nat_cpx); others = [x for x in grp if x is not it]
            if others:
                o = offset_from(others); fns = _lib_k_fns(ol, px, 0.0, o if it['fam'] == 'opx' else off_opx, o if it['fam'] == 'cpx' else off_cpx, ol_nodes=(anchored_nodes(others) if it['fam'] == 'ol' else ol_nodes))
                bh = _fit_natural(it['fam'], it['w'], it['d0'], *fns); row['d_fit_holdout_um'] = math.exp(bh.x[1]); row['chem_fit_holdout'] = bh.x[0]
        rows.append(row)
    # plagioclase and natural-cpx shape libraries: median log k over the natural samples at the rule size, with per-sample level factors
    shapes = {}
    for famname, fam in (('plagioclase', 'plag'), ('cpx', 'cpx')):
        logks, codes = [], []
        for _, e in em[em['family'] == famname].iterrows():
            sid = e['primary_spectrum']
            if sid not in okset: continue
            r = qc[qc['spectrum_id'] == sid].iloc[0]; w = FP.w_from_reff(X[pos[sid]], *FP.mus(r['incidence_deg'], r['emission_deg'])); d0 = d_eff_rule(e['size_lo_um'], e['size_hi_um'], rule)
            logks.append(np.log(np.clip(FP.slab_k(w, FP.n_of(fam), d0, GRID), 1e-9, None))); codes.append((e['code'], e['em_id'], e['name'], d0))
        if not logks: continue
        Lk = np.vstack(logks); med = np.median(Lk, axis=0); shapes[fam] = med
        for (code, emid, name, d0), lk in zip(codes, Lk):
            lvl = float(np.median((lk - med)[EVAL])); rms = float(np.sqrt(np.mean(((lk - med - lvl)[EVAL]) ** 2)))
            rows.append({'code': code, 'em_id': emid, 'name': name, 'family': fam, 'model': 'shape', 'd0_um': d0, 'l_fit': lvl, 'rms_logk_shape': rms, 'anchor': True})
    nat = pd.DataFrame(rows); nat.to_csv(out / 'natural_endmembers_fit.csv', index=False)
    offs = pd.DataFrame({'wavelength_um': GRID, 'offset_logk_olivine': off_ol, 'offset_logk_opx': off_opx, 'offset_logk_cpx': off_cpx}); offs.to_csv(out / 'library_level_offsets.csv', index=False)
    for fam in shapes: print(f"[E] {fam}: shape library from {int((nat['family'] == fam).sum())} natural samples; level spread (IQR of l) {nat[nat['family'] == fam]['l_fit'].quantile(0.25):+.2f} .. {nat[nat['family'] == fam]['l_fit'].quantile(0.75):+.2f}")
    print(f"[E] level anchoring: olivine offset (median over λ) {np.median(off_ol):+.2f} in log k from {len(anchors_ol)} Fo~90 anchors; opx {np.median(off_opx):+.2f} ({len(nat_opx)} samples); cpx {np.median(off_cpx):+.2f} ({len(nat_cpx)} samples)")
    print('[E] natural end-members after anchoring:'); print(nat.round(3).to_string())
    return nat, {'ol': off_ol, 'opx': off_opx, 'cpx': off_cpx, 'shapes': shapes}

# ----------------------------------------------------------------------------- stage F: weathering transfer function
def stage_weathering(qc, X, out: Path):
    plt = S2.style(); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; g = qc[(qc['status'] == 'ok') & (qc['role'] == 'weathering_pellets')]
    T1, T2, names = [], [], []
    for base in ('XT-TXH-030', 'XT-TXH-031', 'XT-TXH-032'):
        f = {suf: g[g['specimen_id'] == f'{base}-{suf}'] for suf in ('P', 'P1', 'P2')}
        if any(len(v) == 0 for v in f.values()): continue
        w = {suf: FP.w_from_reff(X[pos[v.iloc[0]['spectrum_id']]], *FP.mus(v.iloc[0]['incidence_deg'], v.iloc[0]['emission_deg'])) for suf, v in f.items()}
        T1.append(w['P1'] / w['P']); T2.append(w['P2'] / w['P']); names.append(base)
    if not T1: print('[F] no pellet triplets loaded'); return None
    T1 = np.vstack(T1); T2 = np.vstack(T2); T = np.exp(np.mean(np.log(np.clip(T1, 0.05, 5)), axis=0)); T = np.clip(T, 0.2, 1.5)
    pd.DataFrame({'wavelength_um': GRID, 'T_1dose': T, **{f'T1_{n}': t for n, t in zip(names, T1)}, **{f'T2_{n}': t for n, t in zip(names, T2)}}).to_csv(out / 'weathering_transfer.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, n in enumerate(names):
        axes[0].plot(GRID, T1[k], color=WONG[k], lw=1.2, label=f'{n}: 1 dose'); axes[0].plot(GRID, T2[k], color=WONG[k], lw=1.0, ls='--', label=f'{n}: 2 doses')
    axes[0].plot(GRID, T, color=INK, lw=1.8, label='T(λ) = geometric mean, 1 dose'); axes[0].plot(GRID, T ** 2, color=INK, lw=1.2, ls=':', label='T²'); axes[0].axhline(1, color='#999999', lw=0.6)
    axes[0].set_xlabel('µm'); axes[0].set_ylabel('w(irradiated) / w(fresh)'); axes[0].set_title('pulse-laser weathering transfer function (olivine-bronzite pellets)', fontsize=8.5); axes[0].legend(fontsize=6, frameon=False, ncol=2)
    for k, n in enumerate(names):
        for suf, ls in (('P', '-'), ('P1', '--'), ('P2', ':')):
            v = g[g['specimen_id'] == f'{n}-{suf}'].iloc[0]; axes[1].plot(GRID, X[pos[v['spectrum_id']]], color=WONG[k], lw=1.0, ls=ls)
    axes[1].set_xlabel('µm'); axes[1].set_ylabel('REFF'); axes[1].set_title('pellets: fresh (solid), 1 dose (dashed), 2 doses (dotted)', fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / 'figures' / 'S3A_F4_weathering.png'); plt.close(fig)
    cons = float(np.sqrt(np.nanmean(((T2 - T ** 2) / T ** 2)[:, EVAL] ** 2)))
    print(f'[F] weathering transfer function from {len(names)} pellet triplets; T(0.55)={T[np.argmin(abs(GRID - 0.55))]:.3f}, T(1.0)={T[np.argmin(abs(GRID - 1.0))]:.3f}, T(2.4)={T[np.argmin(abs(GRID - 2.4))]:.3f}; '
          f'2-dose vs T^2 relative RMS {cons:.3f}')
    return T

# ----------------------------------------------------------------------------- library file + stage G: mixtures with the parametric model
def write_library(ol, px, qc, X, rule, T, em, nat, offs, out: Path):
    pos = {s: i for i, s in enumerate(qc['spectrum_id'])}
    def k_of_em(code, fam):
        e = em.set_index('code').loc[code]; sid = e['primary_spectrum']; r = qc[qc['spectrum_id'] == sid].iloc[0]
        w = FP.w_from_reff(X[pos[sid]], *FP.mus(r['incidence_deg'], r['emission_deg'])); return FP.slab_k(w, FP.n_of(fam), d_eff_rule(e['size_lo_um'], e['size_hi_um'], rule), GRID)
    plag_k = np.exp(offs['shapes']['plag']) if 'plag' in offs.get('shapes', {}) else np.full(len(GRID), 1e-5); cpx_k = np.exp(offs['shapes']['cpx']) if 'cpx' in offs.get('shapes', {}) else np.full(len(GRID), np.nan)
    plag_code = 'median of natural plagioclases'; cpx_code = 'median of natural clinopyroxenes'
    meta = {'version': VERSION, 'grainsize_rule': rule, 'px_smoothing': px['smoothing'], 'plag_source': plag_code, 'cpx_natural_source': cpx_code, 'ol_model': 'node library: level-corrected synthetic nodes (Dyar suite) + natural Fo-90 node (median of the anchors), smoothed by a weighted cubic in Fo per wavelength and resampled every 5 Fo; valid Fo 0-95',
            'px_model': 'thin-plate spline of level-corrected log k on (Wo/100, Fs/100) (Fe-free members excluded) + family offset (opx / cpx), valid Fs 5-100, Wo 0-55',
            'n_rules': 'olivine 1.635+0.00225(100-Fo); opx 1.65+0.0012 Fs; cpx 1.70; plag 1.58', 'weathering': 'XT-TXH pellets, w ratio geometric mean of 3 compositions (1 pulse-laser dose)',
            'ol_anchors': ANCHORS_OL, 'natural_endmember_fits': nat.to_dict(orient='records') if nat is not None else []}
    np.savez_compressed(out / 's3a_library.npz', grid_um=GRID, ol_B=ol['B'], ol_offset=offs['ol'], ol_nodes_Fo=ol['nodes_Fo_anch'], ol_nodes_logk=ol['nodes_logk_anch'], px_WoFs=px['WoFs'], px_logk=px['logk'], px_offset_opx=offs['opx'], px_offset_cpx=offs['cpx'],
                        plag_k=plag_k, cpx_nat_k=cpx_k, T_weather=(T if T is not None else np.ones(len(GRID))), meta=json.dumps(meta, default=float))
    print('[G] library ->', out / 's3a_library.npz')

def stage_mixtures(mix, em, qc, X, rule, nat, out: Path, series=('MP89', 'XT-OBA', 'XT-OBA-fine', 'SP93-pyx', 'McCoy-4', 'AEF-fine', 'CP14-plag'), max_per_series=60, free_size=True):
    """three retrieval modes on measured mixtures: (c) chemistry/levels fixed at the end-members' library values (abundance only);
    (a) chemistry and levels free, size fixed; (b) chemistry, levels and size free"""
    plt = S2.style(); model = FP.ParamModel(out / 's3a_library.npz'); pos = {s: i for i, s in enumerate(qc['spectrum_id'])}; okset = set(qc.loc[qc['status'] == 'ok', 'spectrum_id'])
    famap = {'olivine': 'ol', 'opx': 'opx', 'cpx': 'cpx', 'plagioclase': 'plag'}
    natq = nat[nat['model'] == 'quad'].set_index('code') if nat is not None and len(nat) else None; nats = nat[nat['model'] == 'shape'].set_index('code') if nat is not None and len(nat) else None
    rows = []; t0 = time.time()
    sub = mix[(mix['tier'] == 'A') & mix['series'].isin(series)]; sub = sub[sub['families'].apply(lambda s: set(s.split(';')) <= set(famap))]
    sub = sub.groupby('series', group_keys=False).head(max_per_series)
    chem_par = {'ol': ['Fo', 'l_ol'], 'opx': ['Fs_opx', 'l_opx'], 'cpx': ['l_cpx'], 'plag': ['l_plag']}; ref_col = {'ol': ['Fo_fit', 'l_fit'], 'opx': ['Fs_fit', 'l_fit'], 'cpx': ['l_fit'], 'plag': ['l_fit']}
    for _, r in sub.iterrows():
        sid = r['primary_spectrum']
        if sid not in okset: continue
        comps = json.loads(r['components']); fams = [famap[c['fam']] for c in comps]
        if len(set(fams)) < len(fams): continue
        M_true = {f: c['frac'] for f, c in zip(fams, comps)}; d_mean = float(np.mean([d_eff_rule(c['size_lo'], c['size_hi'], rule) for c in comps]))
        q = qc[qc['spectrum_id'] == sid].iloc[0]; y = X[pos[sid]]
        ref = {}; n_ref = 0
        for f, c in zip(fams, comps):
            tab = natq if f in ('ol', 'opx') else nats
            if tab is not None and c['code'] in tab.index:
                rr = tab.loc[c['code']]; rr = rr.iloc[0] if isinstance(rr, pd.DataFrame) else rr
                vals = [S2._f(rr.get(col, np.nan), np.nan) for col in ref_col[f]]
                if all(np.isfinite(v) for v in vals): ref.update(dict(zip(chem_par[f], vals))); n_ref += 1
        base = {'M': {f: 1.0 / len(fams) for f in fams}, 'Fo': 80.0, 'Fs_opx': 20.0, 'l_ol': 0.0, 'l_opx': 0.0, 'l_cpx': 0.0, 'l_plag': 0.0, 'd': d_mean, 's': 0.0, 'cpx_natural': True}
        free_chem = [p for f in fams for p in chem_par[f]]
        row = {'mix_id': r['mix_id'], 'series': r['series'], 'n_components': len(fams), 'families': ';'.join(fams), 'd_eff_true_um': d_mean, 'n_ref_chem': n_ref}
        for f in fams: row[f'true_{f}'] = M_true[f]
        # (c) chemistry and levels fixed at the end-members' library values
        if n_ref == len(fams):
            pv = FP.ParamVector(fams, ['ilr'], base | ref); th, rms, _ = FP.fit_spectrum(model, y, pv, pv.pack(base), q['incidence_deg'], q['emission_deg'], mask=EVAL)
            row['rms_chemfixed'] = rms; row['mae_frac_chemfixed'] = float(np.mean([abs(th['M'][f] - M_true[f]) for f in fams]))
            for f in fams: row[f'fitc_{f}'] = th['M'][f]
        # (a) chemistry free, size fixed
        pv = FP.ParamVector(fams, ['ilr'] + free_chem, base); th_a, rms_a, _ = FP.fit_spectrum(model, y, pv, pv.pack(base), q['incidence_deg'], q['emission_deg'], mask=EVAL)
        row['rms_fixed_d'] = rms_a; row['mae_frac_fixed_d'] = float(np.mean([abs(th_a['M'][f] - M_true[f]) for f in fams]))
        for f in fams: row[f'fit_{f}'] = th_a['M'][f]
        for p in free_chem: row[f'fit_{p}'] = th_a[p]; row[f'ref_{p}'] = ref.get(p, np.nan)
        # (b) chemistry and size free
        if free_size:
            pv2 = FP.ParamVector(fams, ['ilr'] + free_chem + ['logd'], base); th_b, rms_b, _ = FP.fit_spectrum(model, y, pv2, pv2.pack(th_a | {'d': d_mean}), q['incidence_deg'], q['emission_deg'], mask=EVAL)
            row.update({'rms_free_d': rms_b, 'd_fit_um': th_b['d'], 'mae_frac_free_d': float(np.mean([abs(th_b['M'][f] - M_true[f]) for f in fams]))})
            for p in free_chem: row[f'fitb_{p}'] = th_b[p]
        rows.append(row)
        if len(rows) % 25 == 0: print(f'      {len(rows)} mixtures fitted, {time.time() - t0:.0f}s')
    res = pd.DataFrame(rows); res.to_csv(out / 'parametric_fit_results.csv', index=False)
    if not len(res): print('[G] no mixtures fitted'); return res
    agg = {'n': ('mix_id', 'size'), 'rms_chemfixed': ('rms_chemfixed', 'median'), 'mae_chemfixed_pct': ('mae_frac_chemfixed', lambda x: 100 * x.median()), 'rms_chemfree': ('rms_fixed_d', 'median'),
           'mae_chemfree_pct': ('mae_frac_fixed_d', lambda x: 100 * x.median())}
    if free_size: agg.update({'rms_chem_size_free': ('rms_free_d', 'median'), 'mae_chem_size_free_pct': ('mae_frac_free_d', lambda x: 100 * x.median())})
    summ = res.groupby('series').agg(**agg).round(4); summ.to_csv(out / 'parametric_fit_summary.csv'); print('[G] parametric model fitted to benchmark mixtures (abundance MAE ladder):'); print(summ.to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4)); mk = 'os^Dv<>'
    for k, (s, g) in enumerate(res.groupby('series')):
        for f in ('ol', 'opx', 'cpx', 'plag'):
            if f'fitc_{f}' in g: m = g[f'fitc_{f}'].notna(); axes[0].scatter(g.loc[m, f'true_{f}'], g.loc[m, f'fitc_{f}'], s=12, color=WONG[k % 7], marker=mk[k % 7], alpha=0.8, linewidths=0, label=s if f == fams_label(g) else None)
            if f'fit_{f}' in g: m = g[f'fit_{f}'].notna(); axes[1].scatter(g.loc[m, f'true_{f}'], g.loc[m, f'fit_{f}'], s=12, color=WONG[k % 7], marker=mk[k % 7], alpha=0.8, linewidths=0)
        if 'fit_Fo' in g: axes[2].scatter(g['ref_Fo'], g['fit_Fo'], s=14, color=WONG[k % 7], marker=mk[k % 7], linewidths=0, label=f'{s}: Fo')
        if 'fit_Fs_opx' in g: axes[2].scatter(g['ref_Fs_opx'], g['fit_Fs_opx'], s=14, color=WONG[k % 7], marker=mk[k % 7], linewidths=0, facecolors='none', edgecolors=WONG[k % 7], label=f'{s}: Fs')
    for ax, ttl in ((axes[0], '(c) chemistry fixed at the end-member values'), (axes[1], '(a) chemistry and levels free, size fixed')):
        ax.plot([0, 1], [0, 1], color='#999999', lw=0.8); ax.set_xlabel('documented mass fraction'); ax.set_ylabel('fitted mass fraction'); ax.set_title(ttl, fontsize=9)
    axes[0].legend(fontsize=6, frameon=False); axes[2].plot([0, 100], [0, 100], color='#999999', lw=0.8); axes[2].set_xlabel('end-member value (library fit)'); axes[2].set_ylabel('value fitted in the mixture (mode a)'); axes[2].legend(fontsize=5.5, frameon=False, ncol=2)
    axes[2].set_title('chemistry recovered in mixtures', fontsize=9)
    fig.suptitle('Parametric forward model on measured mixtures', fontsize=9); fig.tight_layout(); fig.savefig(out / 'figures' / 'S3A_F5_parametric_fits.png'); plt.close(fig)
    return res

def fams_label(g):
    for f in ('ol', 'opx', 'cpx', 'plag'):
        if f'fitc_{f}' in g and g[f'fitc_{f}'].notna().any(): return f
    return 'ol'

# ----------------------------------------------------------------------------- stage H: report + zip
def stage_report(lst, qc, cal, val, rule, ol, px, nat, T, res, out: Path, proj: Path, bench: Path):
    L = [f"# S3a report -- optical-constant library and continuous forward model ({VERSION}, {datetime.now(timezone.utc).isoformat(timespec='seconds')})", '',
         f"- spectra loaded: {qc['status'].value_counts().to_dict()}; roles: {qc[qc['status'] == 'ok']['role'].value_counts().to_dict()}", '',
         '## Grain size (Hapke slab model across sieve classes)', '',
         f"- effective-diameter rule: d_eff = **{rule['bounded']:.2f} x class mean** for bounded classes (IQR {rule['bounded_iqr']}, n={rule['n_bounded']}), **{rule['unbounded']:.2f} x upper bound** for '<X' classes (IQR {rule['unbounded_iqr']}, n={rule['n_unbounded']})",
         f"- median RMS(REFF) of the slab prediction with the rule, classes <= 250 µm: {val[(~val['is_reference']) & (val['hi'] <= 250)]['rms_reff_rule'].median():.4f}; all classes: {val[~val['is_reference']]['rms_reff_rule'].median():.4f}", '',
         '| series | family | class (µm) | nominal | d_eff fitted | ratio | RMS(w) at d_eff | RMS(REFF) with rule |', '|---|---|---|---|---|---|---|---|']
    for _, r in cal.merge(val[['specimen_id', 'rms_reff_rule', 'd_rule_um']], on='specimen_id').sort_values(['series', 'nominal_um']).iterrows():
        L.append(f"| {r['series']} | {r['family']} | {r['lo']:g}-{r['hi']:g} | {r['nominal_um']:.0f} | {r['d_eff_um']:.0f} | {r['ratio_eff_nominal']:.2f} | {r['rms_w_at_deff']:.4f} | {r['rms_reff_rule']:.4f} |")
    if ol is not None:
        L += ['', '## Olivine Fo library (Dyar synthetic olivines)', '', f"- members: {len(ol['lib'])} (Fo {ol['lib']['Fo'].min():.0f}-{ol['lib']['Fo'].max():.0f}), {int(ol['lib']['kept'].sum())} kept; node library (level-corrected), Fo 90 node = natural anchors {list(ANCHORS_OL)}; robust Band-I-centre line of the synthetic suite: {1000 * ol['bic_line'][1]:.2f} nm per Fo; node-library Band I centre Fo 90 -> {ol['bc_trend'][1][18]:.3f} µm, Fo 10 -> {ol['bc_trend'][1][2]:.3f} µm",
              f"- leave-one-out: median |ΔBand I centre| {ol['loo']['d_b1_center'].abs().median() * 1000:.1f} nm (90th pct {ol['loo']['d_b1_center'].abs().quantile(0.9) * 1000:.1f} nm); median RMS(REFF at 60 µm) {ol['loo']['rms_reff_d60'].median():.4f}"]
    if px is not None:
        L += ['', '## Pyroxene quadrilateral library (Klima synthetic pyroxenes)', '', f"- members: {len(px['lib'])}; thin-plate-spline smoothing {px['smoothing']:g}",
              f"- leave-one-out: median |ΔBand I| {px['loo']['d_b1_center'].abs().median() * 1000:.1f} nm, |ΔBand II| {px['loo']['d_b2_center'].abs().median() * 1000:.1f} nm; median RMS(REFF at 60 µm) {px['loo']['rms_reff_d60'].median():.4f}"]
    if nat is not None and len(nat):
        L += ['', '## Natural end-members: effective composition and size from the anchored library', '', '| code | family | d_eff rule (µm) | d fitted (µm) | d fitted, anchor held out | Fo / Fs (Wo) fitted | RMS(w) | anchor |', '|---|---|---|---|---|---|---|---|']
        for _, r in nat.iterrows():
            ch = f"Fo {r['Fo_fit']:.0f}" if np.isfinite(S2._f(r.get('Fo_fit', np.nan), np.nan)) else (f"Fs {r['Fs_fit']:.0f} (Wo {r['Wo_fit']:.0f})" if np.isfinite(S2._f(r.get('Fs_fit', np.nan), np.nan)) else '-')
            ho = S2._f(r.get('d_fit_holdout_um', np.nan), np.nan)
            L.append(f"| {r['code']} | {r['family']} | {r['d0_um']:.0f} | {r['d_fit_um']:.0f} | {ho:.0f} | {ch} | {r['rms_w']:.4f} | {'yes' if r['anchor'] else ''} |")
        a = nat[nat['anchor'] & nat['d_fit_holdout_um'].notna()] if 'd_fit_holdout_um' in nat else nat.iloc[0:0]
        if len(a): L.append(f"\n- hold-one-anchor-out: median |log(d_fit/d_rule)| = {np.median(np.abs(np.log(a['d_fit_holdout_um'] / a['d0_um']))):.2f} (factor {np.exp(np.median(np.abs(np.log(a['d_fit_holdout_um'] / a['d0_um'])))):.2f}) over {len(a)} natural end-members")
    if T is not None:
        L += ['', '## Weathering transfer function (pulse-laser pellets)', '', f"- T(0.55) = {T[np.argmin(abs(GRID - 0.55))]:.3f}, T(0.75) = {T[np.argmin(abs(GRID - 0.75))]:.3f}, T(1.0) = {T[np.argmin(abs(GRID - 1.0))]:.3f}, T(1.5) = {T[np.argmin(abs(GRID - 1.5))]:.3f}, T(2.4) = {T[np.argmin(abs(GRID - 2.4))]:.3f}"]
    if res is not None and len(res):
        L += ['', '## Parametric model fitted to tier-A silicate benchmark mixtures: abundance-error ladder', '',
              '| series | n | (c) chem fixed: RMS | MAE % | (a) chem free: RMS | MAE % | (b) chem + size free: RMS | MAE % |', '|---|---|---|---|---|---|---|---|']
        for s, g in res.groupby('series'):
            f = lambda c, sc=1: (f"{sc * g[c].median():.{1 if sc == 100 else 4}f}" if c in g and g[c].notna().any() else '-')
            L.append(f"| {s} | {len(g)} | {f('rms_chemfixed')} | {f('mae_frac_chemfixed', 100)} | {f('rms_fixed_d')} | {f('mae_frac_fixed_d', 100)} | {f('rms_free_d')} | {f('mae_frac_free_d', 100)} |")
        L.append(f"| **all** | {len(res)} | {res['rms_chemfixed'].median() if 'rms_chemfixed' in res else float('nan'):.4f} | {100 * res['mae_frac_chemfixed'].median() if 'mae_frac_chemfixed' in res else float('nan'):.1f} | {res['rms_fixed_d'].median():.4f} | {100 * res['mae_frac_fixed_d'].median():.1f} | "
                 f"{res['rms_free_d'].median() if 'rms_free_d' in res else float('nan'):.4f} | {100 * res['mae_frac_free_d'].median() if 'mae_frac_free_d' in res else float('nan'):.1f} |")
        L.append('\n(S2 with the measured end-member spectra: median MAE ~4 % on the same kind of mixtures.)')
        if 'fit_Fo' in res: L.append(f"- olivine Fo recovered in mixtures (mode a) vs end-member Fo: median |Δ| {np.nanmedian(np.abs(res['fit_Fo'] - res['ref_Fo'])):.1f}")
        if 'fit_Fs_opx' in res: L.append(f"- opx Fs recovered in mixtures (mode a) vs end-member Fs: median |Δ| {np.nanmedian(np.abs(res['fit_Fs_opx'] - res['ref_Fs_opx'])):.1f}")
        if 'd_fit_um' in res: L.append(f"- size recovered in mixtures (mode b): median |log(d_fit/d_rule)| {np.nanmedian(np.abs(np.log(res['d_fit_um'] / res['d_eff_true_um']))):.2f}")
    g1 = bool(len(val) and val[(~val['is_reference']) & (val['hi'] <= 250)]['rms_reff_rule'].median() <= 0.03); g2 = bool(ol is not None and ol['loo']['d_b1_center'].abs().median() <= 0.01)
    g3 = bool(px is not None and px['loo']['d_b1_center'].abs().median() <= 0.01 and px['loo']['d_b2_center'].abs().median() <= 0.02); g4 = bool(res is not None and len(res) and 'mae_frac_chemfixed' in res and res['mae_frac_chemfixed'].median() <= 0.08)
    L += ['', '## Gate checklist (S3a -> S3b)', '', f"- [{'x' if g1 else ' '}] slab model across sieve classes (<= 250 µm) with the global rule: median RMS(REFF) <= 0.03",
          f"- [{'x' if g2 else ' '}] olivine Fo interpolation: LOO median |ΔBand I centre| <= 10 nm", f"- [{'x' if g3 else ' '}] pyroxene (Wo, Fs) interpolation: LOO median |ΔBand I| <= 10 nm and |ΔBand II| <= 20 nm",
          f"- [{'x' if g4 else ' '}] parametric model on measured mixtures: median abundance MAE <= 8 % with chemistry/levels fixed at the end-member values (mode c)", '- [ ] (manual) figures reviewed; natural end-member effective compositions plausible']
    (out / 'S3A_REPORT.md').write_text('\n'.join(L), encoding='utf-8')
    zpath = proj / 'S3a_results.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in list(out.rglob('*')) + [proj / 'forward_param.py']:
            if p.is_file() and p.suffix in {'.csv', '.json', '.md', '.npz', '.png', '.pdf', '.py'}: zf.write(p, p.relative_to(proj))
    print(f"[H] report: {out / 'S3A_REPORT.md'}\n[H] ZIP to send back: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")
    return zpath
