#!/usr/bin/env python3
"""s6_si_figures.py -- Supporting-Information figures S1-S10 (same conventions and layout helpers as s6_figures.py)."""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
from s6_figures import load, style, new_fig, legend_in, panel, save, W, GREY, GRID, S2, FP, S3B, S5, LAYOUT, configure

MODE_LAB = {'lab': 'laboratory', 'tel50': 'telescope SNR 50', 'tel20': 'telescope SNR 20', 'dawnfc': 'Dawn FC', 'onct': 'Hayabusa2 ONC-T', 'mapcam': 'OSIRIS-REx MapCam', 'lab_noaug': 'laboratory, no augmentation'}
MODE_COL = {'lab': W[0], 'tel50': W[1], 'dawnfc': W[3], 'lab_noaug': W[4], 'tel20': W[2], 'onct': W[4], 'mapcam': W[5]}

def parity(ax, x, y, lim, label_band, col, mk='o', s=12, label=None, band=None):
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    hw = float(np.median(np.abs(y[ok] - x[ok]))) if band is None else band
    ax.fill_between(lim, [lim[0] - hw, lim[1] - hw], [lim[0] + hw, lim[1] + hw], color=GREY, alpha=0.18, lw=0, label=label_band.format(hw))
    ax.plot(lim, lim, color=GREY, lw=0.8); ax.scatter(x[ok], y[ok], s=s, color=col, marker=mk, linewidths=0, label=label); ax.set_xlim(lim); ax.set_ylim(lim); return hw

# ----------------------------------------------------------------------------- S1: benchmark overview and forward-model ingredients
def figS1(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[0.95, 0.95], wspace=0.34)
    # (a) band-parameter plane of the benchmark mixtures
    ax = A[0][0]; panel(ax, 'a'); bp = D['s2_bands']; bp = bp[bp['status'] == 'ok'] if 'status' in bp else bp
    g = bp[np.isfinite(bp['b1_center']) & np.isfinite(bp['BAR'])]; sil = g[(g['glass_frac'].fillna(0) == 0) & (g['opaque_frac'].fillna(0) == 0)]; oth = g.drop(sil.index)
    sc = ax.scatter(sil['b1_center'], sil['BAR'], c=sil['ol_over_olpx'], cmap='cividis', s=14, vmin=0, vmax=1, linewidths=0, label=f'silicate mixtures (n = {len(sil)}); colour: olivine / (olivine + pyroxene)')
    ax.scatter(oth['b1_center'], oth['BAR'], s=14, marker='x', color=W[3], linewidths=0.6, label=f'mixtures with glass or opaques (n = {len(oth)})')
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06); cb.set_label('olivine / (olivine + pyroxene)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel('band I centre (µm)'); ax.set_ylabel('band area ratio'); ax.set_xlim(0.85, 1.12); ax.set_ylim(0, 3.2); legend_in(L[0][0], ax=ax, ncol=1)
    # (b) grain-size rule validation
    ax = A[0][1]; panel(ax, 'b'); gv = D['grainsize_validation']; fam_col = {'ol': W[2], 'opx': W[3], 'cpx': W[0], 'plag': W[1]}; fam_lab = {'ol': 'olivine', 'opx': 'orthopyroxene', 'cpx': 'clinopyroxene', 'plag': 'plagioclase'}
    allerr = float(np.median(np.abs(gv['r075_pred'] - gv['r075_meas'])))
    ax.fill_between([0, 1], [-allerr, 1 - allerr], [allerr, 1 + allerr], color=GREY, alpha=0.18, lw=0, label=f'± median absolute error ({allerr:.3f})'); ax.plot([0, 1], [0, 1], color=GREY, lw=0.8)
    for k, (f, col) in enumerate(fam_col.items()):
        g = gv[gv['family'] == f]; ax.scatter(g['r075_meas'], g['r075_pred'], s=16, color=col, marker='os^D'[k], linewidths=0, label=f'{fam_lab[f]} (n = {len(g)})')
    ax.set_xlim(0.1, 1.0); ax.set_ylim(0.1, 1.0); ax.set_xlabel('measured reflectance at 0.75 µm'); ax.set_ylabel('predicted from the reference fraction + rule'); legend_in(L[0][1], ax=ax, ncol=3)
    # (c) library level offsets
    ax = A[1][0]; panel(ax, 'c'); lo = D['library_level_offsets']
    for col, c_, lab in (('offset_logk_olivine', W[2], 'olivine (Dyar synthetic suite)'), ('offset_logk_opx', W[3], 'orthopyroxene (Klima synthetic suite)'), ('offset_logk_cpx', W[0], 'clinopyroxene (Klima synthetic suite)')):
        ax.plot(lo['wavelength_um'], lo[col], color=c_, lw=1.1, label=lab)
    ax.axhline(0, color=GREY, lw=0.6, ls=':'); ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('level offset of ln k (natural − synthetic)'); ax.set_xlim(0.4, 2.5); legend_in(L[1][0], ax=ax, ncol=1)
    # (d) weathering transfer function
    ax = A[1][1]; panel(ax, 'd'); wt = D['weathering_transfer']
    ax.plot(wt['wavelength_um'], wt['T_1dose'], color=W[3], lw=1.4, label='adopted transfer T(λ) per unit dose (s = 1)')
    for c, ls, lab in (('T1_XT-TXH-030', '-', 'dose 1, 30 % olivine'), ('T1_XT-TXH-031', '--', 'dose 1, 50 % olivine'), ('T1_XT-TXH-032', ':', 'dose 1, 75 % olivine')): ax.plot(wt['wavelength_um'], wt[c], color=W[0], lw=0.9, ls=ls, label=lab)
    for c, ls, lab in (('T2_XT-TXH-030', '-', 'dose 2, 30 % olivine'), ('T2_XT-TXH-031', '--', 'dose 2, 50 % olivine'), ('T2_XT-TXH-032', ':', 'dose 2, 75 % olivine')): ax.plot(wt['wavelength_um'], wt[c], color=W[1], lw=0.9, ls=ls, label=lab)
    ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('w(irradiated) / w(fresh)'); ax.set_xlim(0.4, 2.5); legend_in(L[1][1], ax=ax, ncol=2)
    save(fig, 'FigureS1')

# ----------------------------------------------------------------------------- S2: library leave-one-out and natural end-member fits
def figS2(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[0.8, 0.8], wspace=0.34)
    ax = A[0][0]; panel(ax, 'a'); lo = D['olivine_loo']
    ax.scatter(lo['Fo'], 100 * lo['rms_reff_d60'], s=18, color=W[0], linewidths=0, label='RMS reflectance error of the held-out member (d = 60 µm)')
    ax.set_xlabel('Fo (mol %)'); ax.set_ylabel('leave-one-out RMS error (% reflectance)'); ax.set_ylim(0, 15); legend_in(L[0][0], ax=ax, ncol=1)
    ax = A[0][1]; panel(ax, 'b'); ax.scatter(lo['Fo'], 1000 * lo['d_b1_center'], s=18, color=W[3], linewidths=0, label='band I centre: held-out member − library prediction')
    ax.axhline(0, color=GREY, lw=0.6, ls=':'); ax.set_xlabel('Fo (mol %)'); ax.set_ylabel('band I centre difference (nm)'); legend_in(L[0][1], ax=ax, ncol=1)
    ax = A[1][0]; panel(ax, 'c'); lp = D['pyroxene_loo']; opx = lp[lp['Wo'] <= 5]; cpx = lp[lp['Wo'] > 5]
    ax.scatter(opx['Fs'], 100 * opx['rms_reff_d60'], s=16, color=W[3], linewidths=0, label=f'orthopyroxene members, Wo ≤ 5 (n = {len(opx)})')
    ax.scatter(cpx['Fs'], 100 * cpx['rms_reff_d60'], s=16, color=W[4], marker='^', linewidths=0, label=f'clinopyroxene members, Wo > 5 (n = {len(cpx)})')
    ax.set_xlabel('Fs (mol %)'); ax.set_ylabel('leave-one-out RMS error (% reflectance)'); ax.set_ylim(0, 8); legend_in(L[1][0], ax=ax, ncol=1)
    ax = A[1][1]; panel(ax, 'd'); nat = D['natural_endmembers_fit']; fam_lab = {'ol': 'olivine', 'opx': 'orthopyroxene', 'cpx': 'clinopyroxene', 'plag': 'plagioclase'}; fam_col = {'ol': W[2], 'opx': W[3], 'cpx': W[0], 'plag': W[1]}
    g = nat.sort_values('model').drop_duplicates('code') if 'model' in nat else nat; rng = np.random.default_rng(0)   # one row per end-member ('quad' first, 'shape' for plagioclase)
    for k, f in enumerate(['ol', 'opx', 'cpx', 'plag']):
        gg = g[g['family'] == f]; xx = k + rng.uniform(-0.18, 0.18, len(gg))
        ax.scatter(xx, 100 * gg['rms_w'], s=16, color=fam_col[f], linewidths=0, label=f'{fam_lab[f]} (n = {len(gg)})')
        anc = gg[gg['anchor'].fillna(False).astype(bool)] if 'anchor' in gg else gg.iloc[0:0]
        if len(anc): ax.scatter(xx[gg.index.get_indexer(anc.index)], 100 * anc['rms_w'], s=50, facecolors='none', edgecolors='#333333', linewidths=0.6, label='anchor specimen' if k == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels([fam_lab[f] for f in ['ol', 'opx', 'cpx', 'plag']]); ax.set_ylabel('RMS error of the end-member fit, ln(1 − w) (%)'); legend_in(L[1][1], ax=ax, ncol=3)
    save(fig, 'FigureS2')

# ----------------------------------------------------------------------------- S3: CRB maps for the other observation modes
def _maps(D, key, modes, quants, name, xcol='ol_frac', ycol='Fo', xl='olivine fraction of the mafic part', yl='Fo (mol %)', logy=False):
    style(); from matplotlib.colors import LogNorm; m = D[key]
    fig, axes = plt.subplots(len(quants), len(modes), figsize=(7.2, 2.2 * len(quants) + 0.3), sharex=True, sharey=True, layout='constrained', squeeze=False)
    for i, (q, ql, lim, lev, fmt) in enumerate(quants):
        for j, (mode, ml) in enumerate(modes):
            ax = axes[i][j]; g = m[(m['mode'] == mode) & (m['variant'] == 'default')]; piv = g.pivot_table(index=ycol, columns=xcol, values=q); Z = np.clip(piv.values, lim[0], lim[1])
            im = ax.pcolormesh(piv.columns.values, piv.index.values, Z, cmap='cividis_r', norm=LogNorm(vmin=lim[0], vmax=lim[1]), shading='nearest')
            cs = ax.contour(piv.columns.values, piv.index.values, piv.values, levels=lev, colors=['#FFFFFF'], linewidths=0.6); ax.clabel(cs, fontsize=5, fmt=fmt, colors='#FFFFFF')
            if logy: ax.set_yscale('log')
            if i == 0: ax.set_title(ml, fontsize=8)
            if j == 0: ax.set_ylabel(yl)
            if i == len(quants) - 1: ax.set_xlabel(xl)
            if j == len(modes) - 1: cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06); cb.set_label(ql, fontsize=7); cb.ax.tick_params(labelsize=6)
            panel(ax, 'abcdefghijkl'[i * len(modes) + j], x=-0.08 if j else -0.3, y=1.02)
    save(fig, name)

def figS4(D):
    modes = [('tel20', 'telescope (normalised, SNR 20)'), ('onct', 'Hayabusa2 ONC-T (7 bands, SNR 100)'), ('mapcam', 'OSIRIS-REx MapCam (4 bands, SNR 100)')]
    quants = [('sigma_M_ol', 'σ(olivine mass fraction)', (0.01, 0.5), [0.05, 0.1, 0.2, 0.4], '%.2f'), ('sigma_Fo', 'σ(Fo) (mol %)', (1, 40), [5, 10, 20], '%.0f'), ('sigma_Fs_opx', 'σ(Fs) (mol %)', (1, 40), [5, 10, 20], '%.0f')]
    _maps(D, 'map_olFo_ol-opx', modes, quants, 'FigureS4')

def figS5(D):
    modes = [('lab', 'laboratory (absolute, SNR 1000)'), ('tel50', 'telescope (normalised, SNR 50)'), ('dawnfc', 'Dawn FC (7 bands, SNR 100)')]
    _maps(D, 'map_old_ol-opx', modes, [('sigma_M_ol', 'σ(olivine mass fraction)', (0.01, 0.5), [0.02, 0.05, 0.1, 0.2], '%.2f'), ('sigma_logd', 'σ(ln d)', (0.05, 2), [0.1, 0.2, 0.5, 1.0], '%.1f')], 'FigureS5', xcol='ol_frac', ycol='d_um', yl='grain size d (µm)', logy=True)

def figS6(D):
    modes = [('lab', 'laboratory (absolute, SNR 1000)'), ('tel50', 'telescope (normalised, SNR 50)'), ('dawnfc', 'Dawn FC (7 bands, SNR 100)')]
    _maps(D, 'map_FoFs_ol-opx', modes, [('sigma_Fo', 'σ(Fo) (mol %)', (1, 40), [2, 5, 10, 20], '%.0f'), ('sigma_Fs_opx', 'σ(Fs) (mol %)', (1, 40), [2, 5, 10, 20], '%.0f')], 'FigureS6', xcol='Fs', ycol='Fo', xl='Fs of the orthopyroxene (mol %)', yl='Fo (mol %)')

# ----------------------------------------------------------------------------- S6: validation of the Cramér–Rao bounds
def figS7(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[1.2, 1.0], wspace=0.34); vs = D['val_syn']; vm = D['validation_measured']
    for ax, lg, q, cq, lab, lim, letter in ((A[0][0], L[0][0], 'err_M_ol', 'crb_M_ol', 'olivine mass fraction', (0.003, 0.6), 'a'), (A[0][1], L[0][1], 'err_Fo', 'crb_Fo', 'Fo (mol %)', (0.1, 60), 'b')):
        panel(ax, letter)
        for key, g in sorted(vs.items()):
            scn, mode = key.replace('validation_synthetic_', '').rsplit('_', 1); mk = 'o' if scn == 'ol-opx' else '^'
            ax.scatter(g[cq], np.abs(g[q]), s=9, color=MODE_COL[mode], marker=mk, alpha=0.7, linewidths=0)
        xx = np.array(lim); ax.plot(xx, xx, color=GREY, lw=0.8); ax.plot(xx, 2 * xx, color=GREY, lw=0.8, ls='--')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f'Cramér–Rao σ, {lab}'); ax.set_ylabel(f'|error| of the full fit, {lab}')
        h = [Line2D([], [], color=MODE_COL[m], marker='o', ls='', ms=4, label=MODE_LAB[m]) for m in ('lab', 'tel50', 'dawnfc')] + [Line2D([], [], color=GREY, marker='o', ls='', ms=4, label='two-component states'), Line2D([], [], color=GREY, marker='^', ls='', ms=4, label='states with plagioclase'), Line2D([], [], color=GREY, lw=0.8, label='|error| = σ_CRB'), Line2D([], [], color=GREY, lw=0.8, ls='--', label='|error| = 2 σ_CRB')]
        legend_in(lg, h, [x.get_label() for x in h], ncol=2)
    ax = A[1][0]; panel(ax, 'c')
    ax.scatter(vm['crb_ab_M_ol'], np.abs(vm['s2_err_ol']), s=12, color=W[0], linewidths=0, label='Hapke NNLS with measured end-members vs abundance-only CRB')
    ax.scatter(vm['crb_full_M_ol'], np.abs(vm['s3a_err_ol_chemfree']), s=12, color=W[3], marker='^', linewidths=0, label='parametric fit, chemistry free, vs full CRB')
    xx = np.array([0.003, 0.6]); ax.plot(xx, xx, color=GREY, lw=0.8, label='|error| = σ_CRB'); ax.plot(xx, 2 * xx, color=GREY, lw=0.8, ls='--', label='|error| = 2 σ_CRB')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.003, 0.6); ax.set_ylim(0.003, 0.6); ax.set_xlabel('Cramér–Rao σ, olivine mass fraction'); ax.set_ylabel('|error| on the measured mixtures'); legend_in(L[1][0], ax=ax, ncol=1)
    ax = A[1][1]; panel(ax, 'd')
    ax.scatter(vm['crb_full_Fo'], np.abs(vm['s3a_err_Fo']), s=12, color=W[0], linewidths=0, label='Fo: parametric fit (Step 3a) vs full CRB')
    ax.scatter(vm['crb_full_Fs'], np.abs(vm['s3a_err_Fs']), s=12, color=W[1], marker='s', linewidths=0, label='Fs: parametric fit (Step 3a) vs full CRB')
    xx = np.array([0.1, 60]); ax.plot(xx, xx, color=GREY, lw=0.8, label='|error| = σ_CRB'); ax.plot(xx, 2 * xx, color=GREY, lw=0.8, ls='--', label='|error| = 2 σ_CRB')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.1, 60); ax.set_ylim(0.1, 60); ax.set_xlabel('Cramér–Rao σ (mol %)'); ax.set_ylabel('|error| on the measured mixtures (mol %)'); legend_in(L[1][1], ax=ax, ncol=2)
    save(fig, 'FigureS7')

# ----------------------------------------------------------------------------- S7: library mismatch, learned models, physics vs learned
def figS3(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[1.0, 0.9], wspace=0.34)
    ax = A[0][0]; panel(ax, 'a'); lm = D['s4_library_mismatch_endmembers'].copy(); fam_col = {'ol': W[2], 'opx': W[3], 'plag': W[1]}; fam_lab = {'ol': 'olivine', 'opx': 'orthopyroxene', 'plag': 'plagioclase'}
    lm['fam_order'] = lm['family'].map({'ol': 0, 'opx': 1, 'plag': 2}); lm = lm.sort_values(['fam_order', 'rms_ln1mw']).reset_index(drop=True); xx = np.arange(len(lm))
    for f, col in fam_col.items():
        g = lm[lm['family'] == f]; ax.bar(g.index, 100 * g['rms_ln1mw'], color=col, width=0.8, alpha=[1.0 if u else 0.35 for u in g['used']] if False else 1.0, label=f'{fam_lab[f]} (n = {len(g)})')
    ex = lm[lm['excluded'].astype(bool)]; ax.scatter(ex.index, 100 * ex['rms_ln1mw'] + 1.5, marker='v', s=14, color='#555555', linewidths=0, label='excluded (end-member of a benchmark mixture or rms > 35 %)')
    ax.set_xticks(xx); ax.set_xticklabels(lm['code'], rotation=90, fontsize=5); ax.set_ylabel('RMS of the mismatch field δ (%)'); legend_in(L[0][0], ax=ax, ncol=2)
    ax = A[0][1]; panel(ax, 'b'); mc = D['s4_model_comparison']; mc = mc[mc['set'] == 'synthetic_test']; targets = [('M_ol', 'olivine fraction'), ('Fo', 'Fo'), ('Fs_opx', 'Fs'), ('logd', 'ln d')]; modes = ['lab', 'tel50', 'dawnfc']
    scale = {'M_ol': 1.0, 'Fo': 1 / 20.0, 'Fs_opx': 1 / 20.0, 'logd': 1.0}; xx = np.arange(len(targets)); w = 0.13
    for i, mode in enumerate(modes):
        for j, model in enumerate(('hgb', 'mlp')):
            vals = [float(mc[(mc['mode'] == mode) & (mc['model'] == model) & (mc['target'] == t)]['mae_test'].iloc[0]) * scale[t] for t, _ in targets]
            ax.bar(xx + (i * 2 + j - 2.5) * w, vals, width=w, color=MODE_COL[mode], alpha=1.0 if model == 'mlp' else 0.45, label=f"{MODE_LAB[mode]}, {'MLP' if model == 'mlp' else 'gradient boosting'}")
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in targets]); ax.set_ylabel('MAE, synthetic test set (scaled)'); legend_in(L[0][1], ax=ax, ncol=2)
    for ax, lg, q, lab, lim, letter in ((A[1][0], L[1][0], 'M_ol', 'olivine mass fraction', (0, 0.6), 'c'), (A[1][1], L[1][1], 'Fo', 'Fo (mol %)', (0, 45), 'd')):
        panel(ax, letter)
        for mode in modes:
            g = D['s4_pvl'][mode]; ax.scatter(np.abs(g[f'phys_{q}'] - g[f'true_{q}']), np.abs(g[f'mlp_{q}'] - g[f'true_{q}']), s=9, color=MODE_COL[mode], alpha=0.7, linewidths=0, label=f'{MODE_LAB[mode]} (n = {len(g)})')
        ax.plot(lim, lim, color=GREY, lw=0.8, label='1:1'); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(f'|error| of the physics fit, {lab}'); ax.set_ylabel(f'|error| of the learned model'); legend_in(lg, ax=ax, ncol=2)
    save(fig, 'FigureS3')

# ----------------------------------------------------------------------------- S8: conformal results for Fo and Fs
def figS8(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[1.2, 1.0], wspace=0.34); ev = D['s4_conformal_evaluation']; cmp = D['s4_width_vs_crb']; meas = D['s4_measured_predictions']
    modes = [('lab', 'laboratory', W[0]), ('tel50', 'telescope SNR 50', W[1]), ('dawnfc', 'Dawn FC', W[3]), ('lab_noaug', 'laboratory, no augmentation', W[4])]
    for ax, lg, target, lab, letter in ((A[0][0], L[0][0], 'Fo', 'Fo', 'a'), (A[0][1], L[0][1], 'Fs_opx', 'Fs', 'b')):
        panel(ax, letter)
        for mode, ml, col in modes:
            for sett, ls, mk in (('synthetic_test', '-', 'o'), ('measured_syncal', '--', 's'), ('measured_LOSO_recal', ':', '^')):
                g = ev[(ev['mode'] == mode) & (ev['model'] == 'mlp') & (ev['set'] == sett) & (ev['target'] == target)].sort_values('alpha', ascending=False)
                if len(g) and np.isfinite(g['coverage']).any(): ax.plot(1 - g['alpha'], g['coverage'], color=col, ls=ls, marker=mk, ms=3.5, lw=1.1)
        ax.plot([0.6, 1], [0.6, 1], color=GREY, lw=0.8); ax.set_xlim(0.62, 0.98); ax.set_ylim(0.25, 1.0); ax.set_xlabel('nominal coverage'); ax.set_ylabel(f'empirical coverage, {lab}')
        h = [Line2D([], [], color=col, lw=1.2, label=ml) for _, ml, col in modes] + [Line2D([], [], color=GREY, ls=ls, marker=mk, ms=3.5, lw=1.1, label=l_) for ls, mk, l_ in (('-', 'o', 'synthetic test set'), ('--', 's', 'measured, synthetic calibration'), (':', '^', 'measured, LOSO recalibration'))]
        legend_in(lg, h, [x.get_label() for x in h], ncol=2)
    ax = A[1][0]; panel(ax, 'c'); g = cmp[cmp['mode'] == 'lab']
    ax.scatter(g['crb_Fo'], g['hw68_Fo'], s=7, color=W[5], alpha=0.6, linewidths=0, label='vs CRB, mixing-error covariance only'); ax.scatter(g['crbaug_Fo'], g['hw68_Fo'], s=7, color=W[3], alpha=0.6, linewidths=0, label='vs CRB, mixing error + library mismatch')
    ax.plot([0, 30], [0, 30], color=GREY, lw=0.8, label='1:1'); ax.set_xlim(0, 30); ax.set_ylim(0, 30); ax.set_xlabel('Cramér–Rao σ(Fo) (mol %)'); ax.set_ylabel('conformal 68 % half-width of Fo'); legend_in(L[1][0], ax=ax, ncol=1)
    ax = A[1][1]; panel(ax, 'd'); g = meas[meas['mode'] == 'lab']; g = g[np.isfinite(g['Fo_ref'])]; hw = float(g['hw90_Fo'].median())
    ax.fill_between([40, 100], [40 - hw, 100 - hw], [40 + hw, 100 + hw], color=GREY, alpha=0.18, lw=0, label=f'± median 90 % half-width ({hw:.0f} mol %)'); ax.plot([40, 100], [40, 100], color=GREY, lw=0.8)
    for k, (ser, gg) in enumerate(g.groupby('series')): ax.scatter(gg['Fo_ref'], gg['pred_Fo'], s=14, color=W[k % 7], marker='os^Dv<>'[k % 7], linewidths=0, label=f'{ser} (n = {len(gg)})')
    ax.set_xlim(40, 100); ax.set_ylim(40, 100); ax.set_xlabel('reference Fo of the olivine end-member (mol %)'); ax.set_ylabel('learned retrieval of Fo, laboratory mode'); legend_in(L[1][1], ax=ax, ncol=3)
    save(fig, 'FigureS8')

# ----------------------------------------------------------------------------- S9: single-band information and mission extra filters
def figS9(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[0.75, 1.55], wspace=0.34); sb = D['s5_single_band_information']; mb = D['s5_mission_best_extra_filter']
    for ax, lg, col, lab, letter in ((A[0][0], L[0][0], 'criterion_single_absolute', 'D-criterion of a single band (absolute photometry)', 'a'), (A[0][1], L[0][1], 'criterion_added_to_opt4', 'D-criterion of optimal-4 + one band', 'b')):
        panel(ax, letter)
        for fw, c_ in ((0.03, W[0]), (0.06, W[1]), (0.10, W[3])):
            g = sb[np.isclose(sb['fwhm_um'], fw)].sort_values('centre_um'); ax.plot(g['centre_um'], g[col], color=c_, lw=1.1, label=f'FWHM {int(fw * 1000)} nm')
        if col == 'criterion_added_to_opt4':
            base = float(D['s5_band_sets_crb'][D['s5_band_sets_crb']['tag'] == 'opt4']['criterion_eval'].iloc[0]); ax.axhline(base, color=GREY, lw=0.8, ls='--', label='optimal-4 alone')
        ax.set_xlabel('band centre (µm)'); ax.set_ylabel(lab + '\n(lower = more information)'); ax.set_xlim(0.45, 2.45); legend_in(lg, ax=ax, ncol=4)
    ax = A[1][0]; panel(ax, 'c'); rows = mb.sort_values(['mission', 'constraint', 'order']).reset_index(drop=True); mis = {'onct': 'ONC-T', 'dawnfc': 'Dawn FC', 'mapcam': 'MapCam'}
    xx = np.arange(len(rows)); ax.bar(xx - 0.2, rows['criterion_before'], width=0.4, color=W[5], label='criterion before'); ax.bar(xx + 0.2, rows['criterion_after'], width=0.4, color=W[0], label='criterion after adding the band(s)')
    for i, r in rows.iterrows(): ax.text(i + 0.2, r['criterion_after'] + 0.15, f"{r['added_centre_um']:.2f} µm", rotation=90, fontsize=5, ha='center', va='bottom')
    ax.set_xticks(xx); ax.set_xticklabels([f"{mis[r['mission']]} / {'any λ' if r['constraint'] == 'any' else '≤ 1.0 µm'} / +{r['order']}" for _, r in rows.iterrows()], rotation=90, fontsize=5.5); ax.set_ylabel('D-criterion (CRB with library mismatch)'); ax.set_ylim(0, 19); legend_in(L[1][0], ax=ax, ncol=2)
    ax = A[1][1]; panel(ax, 'd'); tab = D['s5_band_sets_crb']
    for key, mk, c_, ml in (('dawnfc', '*', W[3], 'Dawn FC'), ('onct', 'P', W[6], 'Hayabusa2 ONC-T'), ('mapcam', 'X', W[5], 'OSIRIS-REx MapCam')):
        g = tab[tab['family'] == key]; ax.scatter(g['sigma_M_ol'], g['sigma_Fo'], color=c_, marker=mk, s=70, edgecolors='#555555', linewidths=0.5, label=ml + ' (as flown)')
    for fam, fl, c_, mk in (('opt', 'optimal sets k = 4…12', W[0], 'o'), ('uniform', 'uniform sets', W[1], 's')):
        g = tab[tab['family'] == fam].sort_values('k'); ax.plot(g['sigma_M_ol'], g['sigma_Fo'], color=c_, marker=mk, ms=3.5, lw=1.0, label=fl)
        for _, r in g.iterrows(): ax.annotate(str(int(r['k'])), (r['sigma_M_ol'], r['sigma_Fo']), xytext=(3, 3), textcoords='offset points', fontsize=5, color=c_)
    from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator
    ax.set_xscale('log'); ax.set_yscale('log'); ax.xaxis.set_major_locator(FixedLocator([0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])); ax.yaxis.set_major_locator(FixedLocator([20, 25, 30, 40, 50, 60]))
    for a_ in (ax.xaxis, ax.yaxis): a_.set_major_formatter(ScalarFormatter()); a_.set_minor_formatter(NullFormatter())
    ax.set_xlabel('median σ(olivine mass fraction)'); ax.set_ylabel('median σ(Fo) (mol %)'); legend_in(L[1][1], ax=ax, ncol=2)
    save(fig, 'FigureS9')

# ----------------------------------------------------------------------------- S10: learned retrieval with the band sets
def figS10(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[1.0, 1.0], wspace=0.34); acc = D['s5_learned_sets_accuracy']; conf = D['s5_learned_sets_conformal']; tags = ['opt4', 'uniform4', 'mapcam', 'opt7', 'uniform7', 'random7', 'onct', 'dawnfc', 'opt12', 'uniform12']
    tl = {'opt4': 'optimal 4', 'uniform4': 'uniform 4', 'mapcam': 'MapCam (4)', 'opt7': 'optimal 7', 'uniform7': 'uniform 7', 'random7': 'random 7', 'onct': 'ONC-T (7)', 'dawnfc': 'Dawn FC (7)', 'opt12': 'optimal 12', 'uniform12': 'uniform 12'}
    xx = np.arange(len(tags)); w = 0.26
    ax = A[0][0]; panel(ax, 'a')
    for j, (t, lab, c_, sc) in enumerate((('M_ol', 'olivine fraction', W[0], 1.0), ('Fo', 'Fo / 20', W[1], 1 / 20), ('Fs_opx', 'Fs / 20', W[3], 1 / 20))):
        vals = [float(acc[(acc['tag'] == tg) & (acc['set'] == 'synthetic_test') & (acc['target'] == t)]['mae'].iloc[0]) * sc for tg in tags]; ax.bar(xx + (j - 1) * w, vals, width=w, color=c_, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([tl[t] for t in tags], rotation=45, ha='right', fontsize=6); ax.set_ylabel('MAE on the synthetic test set'); legend_in(L[0][0], ax=ax, ncol=3)
    ax = A[0][1]; panel(ax, 'b')
    for j, (t, lab, c_, sc) in enumerate((('M_ol', 'olivine fraction', W[0], 1.0), ('Fo', 'Fo / 20', W[1], 1 / 20), ('Fs_opx', 'Fs / 20', W[3], 1 / 20))):
        vals = [float(acc[(acc['tag'] == tg) & (acc['set'] == 'measured') & (acc['target'] == t)]['mae'].iloc[0]) * sc for tg in tags]; ax.bar(xx + (j - 1) * w, vals, width=w, color=c_, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([tl[t] for t in tags], rotation=45, ha='right', fontsize=6); ax.set_ylabel('MAE on the 70 measured mixtures'); legend_in(L[0][1], ax=ax, ncol=3)
    ax = A[1][0]; panel(ax, 'c')
    for j, (sett, lab, c_) in enumerate((('synthetic_test', 'synthetic test set', W[0]), ('measured_syncal', 'measured, synthetic calibration', W[1]), ('measured_LOSO_recal', 'measured, LOSO recalibration', W[2]))):
        vals = [float(conf[(conf['tag'] == tg) & (conf['set'] == sett) & (conf['target'] == 'M_ol') & (np.isclose(conf['alpha'], 0.1))]['coverage'].iloc[0]) for tg in tags]; ax.bar(xx + (j - 1) * w, vals, width=w, color=c_, label=lab)
    ax.axhline(0.9, color=GREY, lw=0.8, ls='--', label='nominal 0.90'); ax.set_xticks(xx); ax.set_xticklabels([tl[t] for t in tags], rotation=45, ha='right', fontsize=6); ax.set_ylim(0, 1.05); ax.set_ylabel('coverage at nominal 90 %, olivine fraction'); legend_in(L[1][0], ax=ax, ncol=2)
    ax = A[1][1]; panel(ax, 'd')
    for j, (sett, lab, c_) in enumerate((('synthetic_test', 'synthetic test set', W[0]), ('measured_syncal', 'measured, synthetic calibration', W[1]))):
        vals = [float(conf[(conf['tag'] == tg) & (conf['set'] == sett) & (conf['target'] == 'M_ol') & (np.isclose(conf['alpha'], 0.1))]['median_halfwidth'].iloc[0]) for tg in tags]; ax.bar(xx + (j - 0.5) * w, vals, width=w, color=c_, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([tl[t] for t in tags], rotation=45, ha='right', fontsize=6); ax.set_ylabel('median 90 % half-width, olivine fraction'); legend_in(L[1][1], ax=ax, ncol=2)
    save(fig, 'FigureS10')

# ----------------------------------------------------------------------------- S11: NEA level-1 diagnostics and the level-2 validation on the measured mixtures
def figS11(D):
    style(); fig, A, L = new_fig(7.2, 2, 3, strips=[1.3, 0.85], wspace=0.5, p_in=2.0); l1 = D['s5b_nea_retrievals'].copy(); bm = D['s5b_nea_bandparam_measured']; bp = D['s5b_nea_bandparam_retrievals']
    l1['sort'] = l1['date_obs'].fillna('9999'); l1 = l1.sort_values(['key', 'sort'], ascending=[False, True]).reset_index(drop=True)   # itokawa first? keep eros first
    l1 = pd.concat([l1[l1['key'] == 'eros'], l1[l1['key'] == 'itokawa']]).reset_index(drop=True); xx = np.arange(len(l1)); lab = [('Eros ' + str(d)[:7]) if k == 'eros' else 'Itokawa' for k, d in zip(l1['key'], l1['date_obs'])]
    ax = A[0][0]; panel(ax, 'a'); ax.bar(xx, l1['phys_chi2_per_channel'], color=[W[0] if k == 'eros' else W[2] for k in l1['key']], width=0.7, label='physics fit χ²/n per spectrum')
    ax.axhline(1.5, color=GREY, lw=0.8, ls='--', label='adequacy threshold χ²/n = 1.5'); ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=90, fontsize=5.5); ax.set_ylabel('χ² per channel of the full-spectrum fit'); legend_in(L[0][0], ax=ax, ncol=1)
    ax = A[0][1]; panel(ax, 'b'); ax.bar(xx - 0.2, l1['phys_M_plag'], width=0.4, color=W[1], label='plagioclase mass fraction (bound 1)'); ax.bar(xx + 0.2, l1['phys_s'] / 3.0, width=0.4, color=W[4], label='weathering dose s / 3 (bound 3)')
    ax.scatter(xx, l1['phys_d_um'] / 30.0, marker='v', s=14, color=W[3], linewidths=0, label='grain size d / 30 µm (bound 3 µm → 0.1)'); ax.set_ylim(0, 1.05); ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=90, fontsize=5.5); ax.set_ylabel('fitted nuisance parameters (scaled)'); legend_in(L[0][1], ax=ax, ncol=1)
    ax = A[0][2]; panel(ax, 'c'); ax.scatter(xx, l1['Fo'], s=14, color=W[0], linewidths=0, label='learned Fo (full spectrum)'); ax.scatter(xx, l1['Fs'], s=14, marker='s', color=W[1], linewidths=0, label='learned Fs (full spectrum)')
    ax.axhline(95, color=W[0], lw=0.8, ls=':', label='Fo range of the population (50–95)'); ax.axhline(50, color=W[0], lw=0.8, ls=':'); ax.axhline(100, color=GREY, lw=0.6, ls='--', label='physical limit 100 mol %')
    ax.set_ylim(0, 115); ax.set_xticks(xx); ax.set_xticklabels(lab, rotation=90, fontsize=5.5); ax.set_ylabel('Fo, Fs (mol %)'); legend_in(L[0][2], ax=ax, ncol=1)
    ax = A[1][0]; panel(ax, 'd'); hw = float(bm['olpx_hw90'].median())
    parity(ax, bm['olpx_true'], bm['olpx_pred'], (0, 1), '± median 90 % half-width ({:.2f})', W[0], label=f'measured mixtures (n = {len(bm)})', band=hw)
    ax.set_xlabel('documented olivine / (olivine + pyroxene)'); ax.set_ylabel('band-parameter retrieval'); legend_in(L[1][0], ax=ax, ncol=1)
    ax = A[1][1]; panel(ax, 'e'); g = bm[np.isfinite(bm['Fo_ref'])]; hw = float(g['Fo_hw90'].median())
    parity(ax, g['Fo_ref'], g['Fo_pred'], (40, 100), '± median 90 % half-width ({:.0f})', W[1], mk='s', label=f'measured mixtures (n = {len(g)})', band=hw)
    ax.set_xlabel('reference Fo of the end-member (mol %)'); ax.set_ylabel('band-parameter retrieval of Fo'); legend_in(L[1][1], ax=ax, ncol=1)
    ax = A[1][2]; panel(ax, 'f'); g = bm[np.isfinite(bm['Fs_ref'])]; hw = float(g['Fs_hw90'].median())
    parity(ax, g['Fs_ref'], g['Fs_pred'], (0, 60), '± median 90 % half-width ({:.0f})', W[3], mk='^', label=f'measured mixtures (n = {len(g)})', band=hw)
    ax.set_xlabel('reference Fs of the end-member (mol %)'); ax.set_ylabel('band-parameter retrieval of Fs'); legend_in(L[1][2], ax=ax, ncol=1)
    save(fig, 'FigureS11')

if __name__ == '__main__':
    D = load(); which = sys.argv[1:] or [str(i) for i in range(1, 12)]
    for w in which: globals()[f'figS{w}'](D)
