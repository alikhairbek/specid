#!/usr/bin/env python3
"""s6_figures.py -- publication figures for the manuscript (Wong palette; no black plot elements; text/labels black; PNG + PDF, 300 dpi).
All panels are drawn from the returned result files of Steps 1-5 (no re-computation of science results; small derived quantities only).
Layout rule: every legend sits in its own strip BELOW the panel (never over the data); constrained layout keeps tick labels, axis labels and legends apart."""
from __future__ import annotations
import json, math, sys, glob, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import s2_forward as S2, forward_param as FP, s3b_identify as S3B, s5_bands as S5

# result layout: configure() with the folders of a pipeline run (specid.pipeline.layout) or set SPECID_ROOT for a pipeline tree
_root = Path(os.environ.get('SPECID_ROOT', Path.cwd() / 'specid_run'))
LAYOUT = {'bench': _root / 'benchmark', 's2': _root / 's2', 's3a_tables': _root / 's3a', 's3a_matrix': _root / 's3a', 'lib': _root / 's3a' / 's3a_library.npz', 'rule': _root / 's3a' / 'grainsize_rule.json',
          's3b': _root / 's3b', 's4': _root / 's4', 's5': _root / 's5', 's5b': _root / 's5', 'out': _root / 'figures'}
def configure(d: dict):
    LAYOUT.update({k: Path(v) for k, v in d.items()}); Path(LAYOUT['out']).mkdir(parents=True, exist_ok=True)
W = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']   # Wong palette (no black)
GREY = '#8C8C8C'; GRID = S2.GRID

def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.edgecolor': '#000000', 'axes.labelcolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000', 'text.color': '#000000', 'axes.linewidth': 0.7,
                         'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False, 'lines.linewidth': 1.2})

def panel(ax, letter, x=-0.16, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')

def save(fig, name):
    out = Path(LAYOUT['out']); out.mkdir(parents=True, exist_ok=True); fig.savefig(out / f'{name}.png'); fig.savefig(out / f'{name}.pdf'); plt.close(fig); print('saved', name)

def new_fig(w, nrows, ncols, strips, p_in=2.25, gap=0.42, wspace=0.42, left=0.085, right=0.985, top=0.955, bottom=0.02, width_ratios=None):
    """fixed (not constrained) layout: each row = panels of height p_in (inches) + a legend strip of height strips[i] (inches) directly below them.
    The strip holds the tick labels, the axis label and the legend (drawn at the strip's bottom), so legends never touch the data."""
    H = (nrows * p_in + sum(strips) + (nrows - 1) * gap) / (top - bottom)
    fig = plt.figure(figsize=(w, H)); fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    mean_row = (H * (top - bottom) - (nrows - 1) * gap) / nrows
    fig._wspace = wspace; outer = fig.add_gridspec(nrows, 1, height_ratios=[p_in + s for s in strips], hspace=gap / mean_row); axes, legs = [], []
    for i in range(nrows):
        sub = outer[i].subgridspec(2, ncols, height_ratios=[p_in, strips[i]], hspace=0.0, wspace=wspace, width_ratios=width_ratios); ra, rl = [], []
        for j in range(ncols):
            ax = fig.add_subplot(sub[0, j]); lg = fig.add_subplot(sub[1, j]); lg.axis('off'); ra.append(ax); rl.append(lg)
        axes.append(ra); legs.append(rl)
    return fig, axes, legs

def legend_in(lg, handles=None, labels=None, ncol=2, ax=None, fontsize=None, **kw):
    """legend at the bottom of the strip lg; ncol is reduced until the legend fits the strip's width (+ 70 % of the inter-column gap);
    prints a warning when the legend is taller than the strip (then enlarge the strip)"""
    if handles is None: handles, labels = ax.get_legend_handles_labels()
    fig = lg.figure; fig.canvas.draw(); bb = lg.get_window_extent(); wmax = bb.width * (1 + 0.7 * getattr(fig, '_wspace', 0.4))
    for nc in range(ncol, 0, -1):
        leg = lg.legend(handles, labels, loc='lower center', ncol=nc, columnspacing=1.0, handlelength=1.8, handletextpad=0.5, borderaxespad=0.0, fontsize=fontsize, **kw)
        fig.canvas.draw(); lb = leg.get_window_extent()
        if lb.width <= wmax or nc == 1:
            if lb.height > bb.height - 0.3 * fig.dpi: print(f'  LEGEND OVERFLOW: strip {bb.height / fig.dpi:.2f} in, legend {lb.height / fig.dpi:.2f} in (+ labels) -- {labels[0][:30]}')
            return leg
        leg.remove()

# ----------------------------------------------------------------------------- data
def load():
    D = {}
    D['s2'] = pd.read_csv(LAYOUT['s2'] / 'forward_results.csv'); D['s2_series'] = pd.read_csv(LAYOUT['s2'] / 'forward_summary_by_series.csv')
    D['s2_resid_wl'] = pd.read_csv(LAYOUT['s2'] / 'forward_residual_vs_wavelength.csv')
    p_phot = LAYOUT['s2'] / 'photometric_invariance.csv'; D['phot'] = pd.read_csv(p_phot) if p_phot.exists() else None   # absent when the phase-angle spectra were not fetched
    D['s2_bands'] = pd.read_csv(LAYOUT['s2'] / 'band_parameters.csv')
    z = np.load(LAYOUT['s3a_matrix'] / 'benchmark_matrix.npz', allow_pickle=True); D['X'] = z['X']; D['ids'] = [str(s) for s in z['spectrum_id']]; D['qc'] = pd.read_csv(LAYOUT['s3a_matrix'] / 'spectra_qc.csv')
    D['mix'] = pd.read_csv(LAYOUT['bench'] / 'benchmark_mixtures_v0.csv'); D['em'] = pd.read_csv(LAYOUT['bench'] / 'benchmark_endmembers_v0.csv')
    for f in ('olivine_library', 'olivine_loo', 'pyroxene_library', 'pyroxene_loo', 'grainsize_validation', 'grainsize_calibration', 'parametric_fit_summary', 'parametric_fit_results', 'natural_endmembers_fit', 'weathering_transfer', 'library_level_offsets'):
        D[f] = pd.read_csv(LAYOUT['s3a_tables'] / f'{f}.csv')
    D['rule'] = json.load(open(LAYOUT['rule'])); D['model'] = FP.ParamModel(str(LAYOUT['lib']))
    for f in ('map_olFo_ol-opx', 'map_olFo_ol-opx-plag', 'map_old_ol-opx', 'map_FoFs_ol-opx', 'profile_intervals', 'twins_summary', 'validation_measured', 'shrinkage_calibration'):
        D[f] = pd.read_csv(LAYOUT['s3b'] / f'{f}.csv')
    D['profiles'] = {Path(p).stem: pd.read_csv(p) for p in glob.glob(str(LAYOUT['s3b'] / 'profile_*_*.csv')) if 'intervals' not in p}
    D['val_syn'] = {Path(p).stem: pd.read_csv(p) for p in glob.glob(str(LAYOUT['s3b'] / 'validation_synthetic_*.csv'))}
    for f in ('conformal_evaluation', 'width_vs_crb', 'measured_predictions', 'model_comparison', 'library_mismatch_endmembers'):
        D[f's4_{f}'] = pd.read_csv(LAYOUT['s4'] / f'{f}.csv')
    D['s4_pvl'] = {m: pd.read_csv(LAYOUT['s4'] / f'physics_vs_learned_{m}.csv') for m in ('lab', 'tel50', 'dawnfc', 'lab_noaug')}
    for f in ('band_sets_crb', 'learned_sets_accuracy', 'learned_sets_conformal', 'learned_sets_measured', 'mission_best_extra_filter', 'single_band_information'):
        D[f's5_{f}'] = pd.read_csv(LAYOUT['s5'] / f'{f}.csv')
    for f in ('nea_retrievals', 'nea_summary', 'nea_composites', 'nea_bandparam_retrievals', 'nea_bandparam_measured', 'nea_bandparam_reference_model', 'nea_mode_checks'):
        D[f's5b_{f}'] = pd.read_csv(LAYOUT['s5b'] / f'{f}.csv')
    return D

# ----------------------------------------------------------------------------- Figure 1: benchmark and forward-model validation
def fig1(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[1.15, 0.95], wspace=0.34)
    pos = {s: i for i, s in enumerate(D['ids'])}
    # (a) measured end-members and the olivine-bronzite series with the Hapke intimate-mixing prediction
    ax = A[0][0]; panel(ax, 'a'); em = D['em'].set_index('code'); mixes = D['mix']
    ol_sid, px_sid = em.loc['PO-81-C', 'primary_spectrum'], em.loc['PE-41-C', 'primary_spectrum']
    Rol, Rpx = D['X'][pos[ol_sid]], D['X'][pos[px_sid]]; mu0, mu = S2.mus(30, 0); wol, wpx = FP.w_from_reff(Rol, mu0, mu), FP.w_from_reff(Rpx, mu0, mu)
    ax.plot(GRID, Rol, color=W[2], lw=1.3, label='olivine (PO-81-C)'); ax.plot(GRID, Rpx, color=W[3], lw=1.3, label='orthopyroxene (PE-41-C)')
    for frac, mid, col in ((0.25, 'XT-CMP-030', W[0]), (0.50, 'XT-CMP-031', W[1]), (0.75, 'XT-CMP-032', W[4])):
        r = mixes[mixes['mix_id'] == mid].iloc[0]; Rm = D['X'][pos[r['primary_spectrum']]]
        F = np.array([frac / 3.32, (1 - frac) / 3.35]); F = F / F.sum(); Rh = S2.reff_from_w(F[0] * wol + F[1] * wpx, mu0, mu)
        ax.plot(GRID, Rm, color=col, lw=1.0, label=f'{int(frac * 100)} wt % olivine'); ax.plot(GRID, Rh, color=col, lw=1.0, ls='--')
    h, l = ax.get_legend_handles_labels(); h += [Line2D([], [], color=GREY, lw=1.0), Line2D([], [], color=GREY, lw=1.0, ls='--')]; l += ['measured mixture', 'Hapke mixing of the end-members']
    ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('reflectance (i = 30°, e = 0°)'); ax.set_xlim(0.4, 2.5); ax.set_ylim(0.08, 0.9); legend_in(L[0][0], h, l, ncol=2)
    # (b) parity: Hapke NNLS retrieved vs documented mass fractions (tier A); shaded band = median absolute error
    ax = A[0][1]; panel(ax, 'b'); s2 = D['s2']; ok = s2[(s2['status'] == 'ok') & (s2['tier'] == 'A')]
    fam_col = {'olivine': W[2], 'opx': W[3], 'cpx': W[0], 'plagioclase': W[1]}; fam_lab = {'olivine': 'olivine', 'opx': 'orthopyroxene', 'cpx': 'clinopyroxene', 'plagioclase': 'plagioclase'}; pts = {f: ([], []) for f in fam_col}
    for _, r in ok.iterrows():
        for i in range(1, 6):
            f = r.get(f'fam{i}')
            if isinstance(f, str) and f in fam_col and np.isfinite(r.get(f'true{i}', np.nan)) and np.isfinite(r.get(f'hap{i}', np.nan)): pts[f][0].append(r[f'true{i}']); pts[f][1].append(r[f'hap{i}'])
    allx = np.concatenate([np.array(v[0]) for v in pts.values()]); ally = np.concatenate([np.array(v[1]) for v in pts.values()]); mae = float(np.median(np.abs(ally - allx)))
    ax.fill_between([0, 1], [-mae, 1 - mae], [mae, 1 + mae], color=GREY, alpha=0.18, lw=0, label=f'± median absolute error ({mae:.3f})'); ax.plot([0, 1], [0, 1], color=GREY, lw=0.8)
    for f, (x, y) in pts.items(): ax.scatter(x, y, s=9, color=fam_col[f], alpha=0.75, linewidths=0, label=f'{fam_lab[f]} (n = {len(x)})')
    ax.set_xlabel('documented mass fraction'); ax.set_ylabel('retrieved mass fraction (Hapke, NNLS)'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); legend_in(L[0][1], ax=ax, ncol=3)
    # (c) abundance error per series: linear vs Hapke
    ax = A[1][0]; panel(ax, 'c'); ss = D['s2_series'].sort_values('mae_hap_pct'); xx = np.arange(len(ss))
    ax.bar(xx - 0.2, ss['mae_lin_pct'], width=0.4, color=W[5], label='linear (areal) mixing'); ax.bar(xx + 0.2, ss['mae_hap_pct'], width=0.4, color=W[0], label='Hapke intimate mixing')
    ax.set_xticks(xx); ax.set_xticklabels([f"{s} ({n})" for s, n in zip(ss['series'], ss['n'])], rotation=45, ha='right', fontsize=6); ax.set_ylabel('median abundance error (wt %)'); legend_in(L[1][0], ax=ax, ncol=2)
    # (d) residual structure vs wavelength
    ax = A[1][1]; panel(ax, 'd'); rw = D['s2_resid_wl']
    ax.plot(rw['wavelength_um'], 100 * rw['rms_resid_linear_tierA'], color=W[5], lw=1.1, label='linear mixing: RMS residual (× 100)')
    ax.plot(rw['wavelength_um'], 100 * rw['rms_resid_hapke_tierA'], color=W[0], lw=1.1, label='Hapke mixing: RMS residual (× 100)')
    ax.plot(rw['wavelength_um'], 100 * rw['median_rel_resid_hapke_tierA'], color=W[3], lw=1.1, ls='--', label='Hapke mixing: median relative residual (%)')
    ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('residual (units in the legend)'); ax.set_xlim(0.4, 2.5); ax.set_ylim(0, 17); legend_in(L[1][1], ax=ax, ncol=1)
    save(fig, 'Figure1')

# ----------------------------------------------------------------------------- Figure 2: the continuous forward model
def fig2(D):
    style(); fig, A, L = new_fig(7.2, 2, 2, strips=[0.95, 1.0], wspace=0.34); model = D['model']
    # (a) grain-size rule
    ax = A[0][0]; panel(ax, 'a'); gc = D['grainsize_calibration']; fam_col = {'ol': W[2], 'opx': W[3], 'cpx': W[0], 'plag': W[1]}; fam_lab = {'ol': 'olivine', 'opx': 'orthopyroxene', 'cpx': 'clinopyroxene', 'plag': 'plagioclase'}
    for k, (f, col) in enumerate(fam_col.items()):
        g = gc[gc['family'] == f]; ax.scatter(g['nominal_um'], g['d_eff_um'], s=16, color=col, label=fam_lab[f], marker='os^D'[k], linewidths=0)
    xx = np.array([5, 500]); r = D['rule']; ax.plot(xx, r['bounded'] * xx, color=GREY, lw=0.9, ls='--', label=f"rule: d = {r['bounded']:.2f} × class mean"); ax.plot(xx, xx, color=GREY, lw=0.6, ls=':', label='1:1')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('nominal sieve-class mean (µm)'); ax.set_ylabel('fitted effective diameter (µm)'); legend_in(L[0][0], ax=ax, ncol=3)
    # (b) olivine library: band-I centre vs Fo
    ax = A[0][1]; panel(ax, 'b'); ol = D['olivine_library']; kept = ol[ol['kept']]; rej = ol[~ol['kept']]
    ax.scatter(kept['Fo'], 1000 * kept['b1_center_d60'], s=18, color=W[0], label='synthetic olivine members kept (Dyar suite)', linewidths=0)
    if len(rej): ax.scatter(rej['Fo'], 1000 * rej['b1_center_d60'], s=18, color=W[0], marker='x', label='members rejected (> 12 nm off the trend)')
    Fo_grid = np.arange(0, 95.1, 5); bc = [S2.band_params(GRID, model.spectrum({'M': {'ol': 1.0}, 'Fo': float(Fo), 'd': 60.0, 'cpx_natural': True}))['b1_center'] * 1000 for Fo in Fo_grid]
    ax.plot(Fo_grid, bc, color=W[3], lw=1.3, label='library model (piecewise-linear log k in Fo), d = 60 µm')
    ax.set_ylim(1025, 1110); off = rej[1000 * rej['b1_center_d60'] < 1025]
    if len(off): ax.annotate(f"rejected member off-scale: {1000 * off['b1_center_d60'].min():.0f} nm at Fo {off['Fo'].iloc[0]:.0f}", xy=(off['Fo'].iloc[0], 1026), xytext=(1, 1030), fontsize=5.6, arrowprops=dict(arrowstyle='->', color=GREY, lw=0.6))
    ax.set_xlabel('Fo (mol %)'); ax.set_ylabel('band I centre at d = 60 µm (nm)'); legend_in(L[0][1], ax=ax, ncol=1)
    # (c) pyroxene library: band-II centre vs Fs
    ax = A[1][0]; panel(ax, 'c'); px = D['pyroxene_library']; opx = px[px['Wo'] <= 5]; cpx = px[px['Wo'] > 5]
    ax.scatter(opx['Fs'], 1000 * opx['b2_center_d60'], s=16, color=W[3], label='synthetic orthopyroxene members (Wo ≤ 5)', linewidths=0)
    ax.scatter(cpx['Fs'], 1000 * cpx['b2_center_d60'], s=16, color=W[4], marker='^', label='synthetic clinopyroxene members (Wo > 5)', linewidths=0)
    Fs_grid = np.arange(5, 95.1, 5); b2 = [S2.band_params(GRID, model.spectrum({'M': {'opx': 1.0}, 'Fs_opx': float(Fs), 'd': 60.0, 'cpx_natural': True}))['b2_center'] * 1000 for Fs in Fs_grid]
    ax.plot(Fs_grid, b2, color=W[0], lw=1.3, label='library model, orthopyroxene branch (Wo = 2), d = 60 µm'); ax.set_xlabel('Fs (mol %)'); ax.set_ylabel('band II centre at d = 60 µm (nm)'); legend_in(L[1][0], ax=ax, ncol=1)
    # (d) the abundance-error ladder
    ax = A[1][1]; panel(ax, 'd'); pf = D['parametric_fit_summary'].sort_values('mae_chemfixed_pct'); xx = np.arange(len(pf)); w = 0.27
    ax.bar(xx - w, pf['mae_chemfixed_pct'], width=w, color=W[0], label='chemistry and levels taken from the end-members')
    ax.bar(xx, pf['mae_chemfree_pct'], width=w, color=W[1], label='chemistry and levels free')
    ax.bar(xx + w, pf['mae_chem_size_free_pct'], width=w, color=W[3], label='chemistry, levels and grain size free')
    ax.set_xticks(xx); ax.set_xticklabels([f"{s} ({n})" for s, n in zip(pf['series'], pf['n'])], rotation=45, ha='right', fontsize=6); ax.set_ylabel('median abundance error (wt %)'); legend_in(L[1][1], ax=ax, ncol=1)
    save(fig, 'Figure2')

# ----------------------------------------------------------------------------- Figure 3: Cramér–Rao maps (no legends; colour bars only)
def fig3(D):
    style(); from matplotlib.colors import LogNorm
    m = D['map_olFo_ol-opx']; modes = [('lab', 'laboratory (absolute, SNR 1000)'), ('tel50', 'telescope (normalised, SNR 50)'), ('dawnfc', 'Dawn FC (7 bands, SNR 100)')]
    quants = [('sigma_M_ol', 'σ(olivine mass fraction)', (0.01, 0.5), [0.02, 0.05, 0.1, 0.2], '%.2f'), ('sigma_Fo', 'σ(Fo) (mol %)', (1, 40), [2, 5, 10, 20], '%.0f'), ('sigma_Fs_opx', 'σ(Fs) (mol %)', (1, 40), [2, 5, 10, 20], '%.0f')]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.6), sharex=True, sharey=True, layout='constrained')
    for i, (q, ql, lim, lev, fmt) in enumerate(quants):
        for j, (mode, ml) in enumerate(modes):
            ax = axes[i][j]; g = m[(m['mode'] == mode) & (m['variant'] == 'default')]; piv = g.pivot_table(index='Fo', columns='ol_frac', values=q); Z = np.clip(piv.values, lim[0], lim[1])
            im = ax.pcolormesh(piv.columns.values, piv.index.values, Z, cmap='cividis_r', norm=LogNorm(vmin=lim[0], vmax=lim[1]), shading='nearest')
            cs = ax.contour(piv.columns.values, piv.index.values, piv.values, levels=lev, colors=['#FFFFFF'], linewidths=0.6); ax.clabel(cs, fontsize=5, fmt=fmt, colors='#FFFFFF')
            if i == 0: ax.set_title(ml, fontsize=8)
            if j == 0: ax.set_ylabel('Fo (mol %)')
            if i == 2: ax.set_xlabel('olivine fraction of the mafic part')
            if j == 2: cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.06); cb.set_label(ql, fontsize=7); cb.ax.tick_params(labelsize=6)
            panel(ax, 'abcdefghi'[i * 3 + j], x=-0.08 if j else -0.3, y=1.02)
    save(fig, 'Figure3')

# ----------------------------------------------------------------------------- Figure 4: profile likelihoods (one legend strip below the grid)
def fig4(D):
    style(); states = [('OC-like', 'OC-like state'), ('px-rich', 'pyroxene-rich state'), ('ol-rich', 'olivine-rich state'), ('ternary', 'ternary state (30 wt % plagioclase)')]
    modes = [('lab', 'laboratory (absolute, SNR 1000)', W[0]), ('tel50', 'telescope (normalised, SNR 50)', W[1]), ('dawnfc', 'Dawn FC (7 bands, SNR 100)', W[3])]; pars = [('ilr1', 'olivine fraction of the mafic part'), ('Fo', 'Fo (mol %)'), ('logd', 'grain size d (µm)')]
    fig = plt.figure(figsize=(7.2, 7.8), layout='constrained'); gs = fig.add_gridspec(len(states) + 1, 3, height_ratios=[1, 1, 1, 1, 0.12]); axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(len(states))]
    for i, (tag, lab) in enumerate(states):
        for j, (par, pl) in enumerate(pars):
            ax = axes[i][j]
            for mode, ml, col in modes:
                key = f'profile_{tag}_{mode}_{par}'
                if key not in D['profiles']: continue
                p = D['profiles'][key].sort_values('x'); x = p['x'].values; y = p['dchi2'].values
                ax.plot(x, y, color=col, lw=1.2, marker='o', ms=2.5)
            ax.axhline(3.84, color=GREY, lw=0.8, ls='--'); ax.set_ylim(0, 25)
            if par == 'logd': ax.set_xscale('log')
            if j == 0: ax.set_ylabel('Δχ²')
            if i == len(states) - 1: ax.set_xlabel(pl)
            if j == 1: ax.set_title(lab, fontsize=8)
            panel(ax, 'abcdefghijkl'[i * 3 + j], x=-0.26 if j == 0 else -0.14, y=1.02)
    lg = fig.add_subplot(gs[len(states), :]); lg.axis('off')
    h = [Line2D([], [], color=col, lw=1.2, marker='o', ms=2.5, label=ml) for _, ml, col in modes] + [Line2D([], [], color=GREY, lw=0.8, ls='--', label='Δχ² = 3.84 (95 % profile interval)')]
    lg.legend(handles=h, loc='center', ncol=4, columnspacing=1.2, handlelength=2.0, borderaxespad=0.0)
    save(fig, 'Figure4')

# ----------------------------------------------------------------------------- Figure 5: distribution-free uncertainty, sim-to-real
def fig5(D):
    style(); ev = D['s4_conformal_evaluation']; cmp = D['s4_width_vs_crb']; meas = D['s4_measured_predictions']; prim = 'mlp'
    fig, A, L = new_fig(7.2, 2, 2, strips=[1.2, 1.15], wspace=0.34)
    # (a) reliability of the olivine-fraction intervals
    ax = A[0][0]; panel(ax, 'a'); modes = [('lab', 'laboratory', W[0]), ('tel50', 'telescope SNR 50', W[1]), ('dawnfc', 'Dawn FC', W[3]), ('lab_noaug', 'laboratory, no augmentation', W[4])]
    for mode, ml, col in modes:
        for sett, ls, mk in (('synthetic_test', '-', 'o'), ('measured_syncal', '--', 's'), ('measured_LOSO_recal', ':', '^')):
            g = ev[(ev['mode'] == mode) & (ev['model'] == prim) & (ev['set'] == sett) & (ev['target'] == 'M_ol')].sort_values('alpha', ascending=False)
            if len(g): ax.plot(1 - g['alpha'], g['coverage'], color=col, ls=ls, marker=mk, ms=3.5, lw=1.1)
    ax.plot([0.6, 1], [0.6, 1], color=GREY, lw=0.8); ax.set_xlim(0.62, 0.98); ax.set_ylim(0.25, 1.0); ax.set_xlabel('nominal coverage'); ax.set_ylabel('empirical coverage, olivine mass fraction')
    h = [Line2D([], [], color=col, lw=1.2, label=ml) for _, ml, col in modes] + [Line2D([], [], color=GREY, ls=ls, marker=mk, ms=3.5, lw=1.1, label=lab) for ls, mk, lab in (('-', 'o', 'synthetic test set'), ('--', 's', 'measured, synthetic calibration'), (':', '^', 'measured, LOSO recalibration'))]
    legend_in(L[0][0], h, [x.get_label() for x in h], ncol=2)
    # (b) conformal 68 % half-width vs the two Cramér–Rao bounds (laboratory mode)
    ax = A[0][1]; panel(ax, 'b'); g = cmp[cmp['mode'] == 'lab']
    ax.scatter(g['crb_M_ol'], g['hw68_M_ol'], s=7, color=W[5], alpha=0.6, linewidths=0, label='vs CRB, mixing-error covariance only')
    ax.scatter(g['crbaug_M_ol'], g['hw68_M_ol'], s=7, color=W[3], alpha=0.6, linewidths=0, label='vs CRB, mixing error + library mismatch')
    ax.plot([0, 0.25], [0, 0.25], color=GREY, lw=0.8, label='1:1'); ax.set_xlim(0, 0.25); ax.set_ylim(0, 0.25); ax.set_xlabel('Cramér–Rao σ (olivine mass fraction)'); ax.set_ylabel('conformal 68 % half-width, laboratory mode'); legend_in(L[0][1], ax=ax, ncol=1)
    # (c) measured mixtures: predicted vs documented olivine fraction (laboratory mode); band = median 90 % half-width
    ax = A[1][0]; panel(ax, 'c'); g = meas[meas['mode'] == 'lab']; hw = float(g['hw90_M_ol'].median())
    ax.fill_between([0, 1], [-hw, 1 - hw], [hw, 1 + hw], color=GREY, alpha=0.18, lw=0, label=f'± median 90 % half-width ({hw:.2f})'); ax.plot([0, 1], [0, 1], color=GREY, lw=0.8)
    for k, (ser, gg) in enumerate(g.groupby('series')): ax.scatter(gg['M_ol'], gg['pred_M_ol'], s=14, color=W[k % 7], marker='os^Dv<>'[k % 7], label=f'{ser} (n = {len(gg)})', linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('documented olivine mass fraction'); ax.set_ylabel('learned retrieval, laboratory mode'); legend_in(L[1][0], ax=ax, ncol=3)
    # (d) per-series coverage at 90 %
    ax = A[1][1]; panel(ax, 'd'); sers = ['XT-OBA', 'MP89', 'XT-OBA-fine', 'AEF-fine', 'McCoy-4']; rows = []
    for ser in sers:
        g = meas[(meas['mode'] == 'lab') & (meas['series'] == ser)]; g0 = meas[(meas['mode'] == 'lab_noaug') & (meas['series'] == ser)]
        rows.append((g0['covered90_M_ol'].mean(), g['covered90_M_ol'].mean(), np.nanmean(g['covered90_loso_M_ol'].astype(float))))
    rows = np.array(rows); xx = np.arange(len(sers)); w = 0.27
    ax.bar(xx - w, rows[:, 0], width=w, color=W[4], label='synthetic calibration, no augmentation'); ax.bar(xx, rows[:, 1], width=w, color=W[0], label='synthetic calibration, with augmentation'); ax.bar(xx + w, rows[:, 2], width=w, color=W[2], label='leave-one-series-out recalibration')
    ax.axhline(0.9, color=GREY, lw=0.8, ls='--', label='nominal 0.90'); ax.set_xticks(xx); ax.set_xticklabels(sers, rotation=30, ha='right', fontsize=6.5); ax.set_ylim(0, 1.05); ax.set_ylabel('coverage at nominal 90 %, olivine fraction'); legend_in(L[1][1], ax=ax, ncol=1)
    save(fig, 'Figure5')

# ----------------------------------------------------------------------------- Figure 6: optimal bands and mission filters
def fig6(D):
    style(); tab = D['s5_band_sets_crb']; model = D['model']
    fig = plt.figure(figsize=(7.2, 7.2), layout='constrained'); gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.05, 0.26])
    # (a) band positions over a model spectrum (set names written at the right of each row; legend strip not needed)
    ax = fig.add_subplot(gs[0, :]); panel(ax, 'a', x=-0.07)
    th = S3B.base_theta('ol-opx-plag', ol=0.6, Fo=80, Fs=20, d=60, s=0.5, plag=0.1); R = model.spectrum(th, mode='reff'); ax.plot(GRID, R / R.max(), color=GREY, lw=1.0)
    ax.text(1.02, 0.93, 'model spectrum: ol 0.6, Fo 80, Fs 20, 60 µm,\n10 wt % plagioclase, s = 0.5 (normalised to its maximum)', fontsize=5.8, ha='left', va='top', color='#555555')
    sets = [('opt4', 'optimal 4', W[0]), ('opt7', 'optimal 7', W[1]), ('opt12', 'optimal 12', W[2]), ('dawnfc', 'Dawn FC (7)', W[3]), ('onct', 'Hayabusa2 ONC-T (7)', W[4]), ('mapcam', 'OSIRIS-REx MapCam (4)', W[5])]
    for i, (tag, lab, col) in enumerate(sets):
        bands = json.loads(tab[tab['tag'] == tag]['bands'].iloc[0]); y0 = 1.62 - 0.1 * i
        for c, fw in bands: ax.plot([c - fw / 2, c + fw / 2], [y0, y0], color=col, lw=3.2, solid_capstyle='butt'); ax.plot([c], [y0], marker='|', color=col, ms=7)
        ax.text(2.53, y0, lab, fontsize=6.5, color=col, va='center')
    ax.set_xlim(0.4, 2.95); ax.set_ylim(0.2, 1.7); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('normalised reflectance')
    # (b) sigma(M_ol) vs k; (c) sigma(Fs) vs k
    fams = [('opt', 'optimal (D-criterion)', W[0], 'o'), ('uniform', 'uniform', W[1], 's'), ('random', 'random (median of 30 draws)', W[2], '^'), ('sens', 'sensitivity peaks', W[4], 'D'), ('pca', 'PCA peaks', W[5], 'v')]
    from matplotlib.ticker import ScalarFormatter, NullFormatter, FixedLocator
    for ax, col, lab, letter in ((fig.add_subplot(gs[1, 0]), 'sigma_M_ol', 'median σ(olivine mass fraction)', 'b'), (fig.add_subplot(gs[1, 1]), 'sigma_Fs_opx', 'median σ(Fs) (mol %)', 'c')):
        panel(ax, letter)
        for fam, fl, c_, mk in fams:
            g = tab[tab['family'] == fam].sort_values('k'); ax.plot(g['k'], g[col], color=c_, marker=mk, ms=3.5, lw=1.1, label=fl)
        for key, mk, c_, ml in (('dawnfc', '*', W[3], 'Dawn FC (7)'), ('onct', 'P', W[6], 'Hayabusa2 ONC-T (7)'), ('mapcam', 'X', W[5], 'OSIRIS-REx MapCam (4)')):
            g = tab[tab['family'] == key]; ax.scatter(g['k'], g[col], color=c_, marker=mk, s=70, zorder=5, edgecolors='#555555', linewidths=0.5, label=ml)
        hy = tab[tab['family'] == 'hyper']; ax.axhline(float(hy[col].iloc[0]), color=GREY, lw=0.8, ls='--', label='hyperspectral 0.45–2.45 µm, SNR 50')
        ax.set_yscale('log'); ax.set_xlabel('number of bands k'); ax.set_ylabel(lab + '\n(CRB with library mismatch)')
        ticks = [0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5] if col == 'sigma_M_ol' else [5, 7, 10, 15, 20, 30, 40, 60]
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_xticks([4, 5, 6, 7, 8, 10, 12])
        if letter == 'b': hh, ll = ax.get_legend_handles_labels()
    lg = fig.add_subplot(gs[2, :]); lg.axis('off'); lg.legend(hh, ll, loc='center', ncol=3, columnspacing=1.2, handlelength=2.0, borderaxespad=0.0)
    save(fig, 'Figure6')

# ----------------------------------------------------------------------------- Figure 7: the near-Earth asteroids
def fig7(D):
    style(); comp = D['s5b_nea_composites']; bp = D['s5b_nea_bandparam_retrievals']; bm = D['s5b_nea_bandparam_measured']; refs = D['s5b_nea_bandparam_reference_model']; l1 = D['s5b_nea_retrievals']; model = D['model']
    fig, A, L = new_fig(7.2, 2, 2, strips=[1.1, 1.8], wspace=0.34)
    # (a) spectra and the level-1 inadequacy
    ax = A[0][0]; panel(ax, 'a')
    for key, col in (('eros', W[0]), ('itokawa', W[2])):
        g = comp[comp['key'] == key]; labels = list(dict.fromkeys(g['label']))
        for i, lab in enumerate(labels):
            gg = g[g['label'] == lab].sort_values('wave'); nrm = np.interp(0.55 if gg['wave'].min() <= 0.55 else 1.25, gg['wave'], gg['refl'])
            ax.plot(gg['wave'], gg['refl'] / nrm, color=col, lw=0.6, alpha=0.5 if key == 'eros' else 1.0, label=('(433) Eros, 12 epochs (MITHNEOS + SMASS)' if key == 'eros' else '(25143) Itokawa (SMASS)') if i == 0 else None)
    le = l1[l1['key'] == 'eros']; r = le.iloc[int(np.argsort(le['phys_chi2_per_channel'].values)[len(le) // 2])]
    th = {'M': {'ol': max(r['phys_olpx'] * (1 - r['phys_M_plag']), 1e-3), 'opx': max((1 - r['phys_olpx']) * (1 - r['phys_M_plag']), 1e-3), 'plag': max(r['phys_M_plag'], 1e-3)}, 'Fo': r['phys_Fo'], 'Fs_opx': r['phys_Fs'], 'd': r['phys_d_um'], 's': r['phys_s'], 'l_ol': r['phys_l_ol'], 'l_opx': r['phys_l_opx'], 'l_plag': r['phys_l_plag'], 'cpx_natural': True}
    Rf = model.spectrum(th, mode='reff'); ax.plot(GRID, Rf / np.interp(0.55, GRID, Rf), color=W[3], lw=1.1, ls='--', label=f"best full-spectrum fit of one Eros epoch (χ²/n = {r['phys_chi2_per_channel']:.1f}; parameters at the bounds)")
    th2 = S3B.base_theta('ol-opx-plag', ol=0.58, Fo=77, Fs=22, d=60, s=0.0, plag=0.10); R2 = model.spectrum(th2, mode='reff'); ax.plot(GRID, R2 / np.interp(0.55, GRID, R2), color=W[1], lw=1.1, ls=':', label='model at the Eros reference composition, unweathered, d = 60 µm')
    ax.set_xlabel('wavelength (µm)'); ax.set_ylabel('reflectance / R(0.55 µm)'); ax.set_xlim(0.43, 2.5); ax.set_ylim(0.2, 2.1); legend_in(L[0][0], ax=ax, ncol=1)
    # (b) band-parameter plane
    ax = A[0][1]; panel(ax, 'b')
    sc = ax.scatter(bm['BI'], bm['BAR'], c=bm['olpx_true'], cmap='cividis', s=18, marker='s', vmin=0, vmax=1, edgecolors='#777777', linewidths=0.3, label='measured mixtures (colour: olivine / (olivine + pyroxene))')
    for key, col, mk in (('eros', W[0], '*'), ('itokawa', W[2], 'P')):
        g = bp[bp['key'] == key]; ax.scatter(g['BI'], g['BAR'], color=col, marker=mk, s=80, edgecolors='#333333', linewidths=0.5, zorder=5, label=('(433) Eros, 12 epochs' if key == 'eros' else '(25143) Itokawa'))
        gr = refs[refs['key'] == key].sort_values('d_um'); ax.plot(gr['BI'], gr['BAR'], color=col, lw=1.0, ls='--', marker='o', ms=3, label=f"model at the {'Eros' if key == 'eros' else 'Itokawa'} reference composition, d = 30, 60, 150 µm")
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06); cb.set_label('olivine / (olivine + pyroxene)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel('band I centre (µm)'); ax.set_ylabel('band area ratio'); ax.set_xlim(0.88, 1.08); ax.set_ylim(0, 1.8); legend_in(L[0][1], ax=ax, ncol=1)
    # (c) ol/(ol+px) per spectrum with 90 % band-parameter regions
    ax = A[1][0]; panel(ax, 'c'); x0 = 0; ticks, tlabels = [], []
    for key, col in (('eros', W[0]), ('itokawa', W[2])):
        g = bp[bp['key'] == key].sort_values('date_obs').reset_index(drop=True); xx = x0 + np.arange(len(g)); nm = 'Eros' if key == 'eros' else 'Itokawa'
        ax.bar(xx, g['olpx_hi90'] - g['olpx_lo90'], bottom=g['olpx_lo90'], width=0.55, color=col, alpha=0.28, linewidth=0, label=f'{nm}: 90 % region from the band parameters')
        ax.scatter(xx, g['olpx'], s=16, color=col, zorder=4, linewidths=0, label=f'{nm}: point retrieval')
        ax.plot([x0 - 0.5, x0 + len(g) - 0.5], [g['ref_olpx'].iloc[0]] * 2, color=col, lw=1.0, ls='--', label=f"{nm}: reference ({g['ref_olpx'].iloc[0]:.2f})")
        l1g = l1[l1['key'] == key].sort_values('date_obs').reset_index(drop=True); ax.scatter(xx, l1g['olpx'].values[:len(g)], marker='x', s=18, color=W[3], zorder=5, label='full-spectrum retrieval (model inadequate)' if key == 'eros' else None)
        ticks += list(xx); tlabels += [('Eros ' + str(d)[:7]) if key == 'eros' else 'Itokawa' for d in g['date_obs']]; x0 += len(g) + 1
    ax.set_ylim(0, 1.02); ax.set_xticks(ticks); ax.set_xticklabels(tlabels, rotation=90, fontsize=5.5); ax.set_xlabel('spectrum (date of observation)'); ax.set_ylabel('olivine / (olivine + pyroxene)'); legend_in(L[1][0], ax=ax, ncol=2)
    # (d) Fo and Fs per spectrum
    ax = A[1][1]; panel(ax, 'd'); x0 = 0; ticks, tlabels = [], []
    for key, col in (('eros', W[0]), ('itokawa', W[2])):
        g = bp[bp['key'] == key].sort_values('date_obs').reset_index(drop=True); xx = x0 + np.arange(len(g)); nm = 'Eros' if key == 'eros' else 'Itokawa'; c2 = W[1] if key == 'eros' else W[6]
        ax.bar(xx - 0.17, 2 * g['hw90_Fo'], bottom=g['Fo'] - g['hw90_Fo'], width=0.32, color=col, alpha=0.28, linewidth=0, label=f'{nm}: Fo, 90 % region'); ax.scatter(xx - 0.17, g['Fo'], s=14, color=col, zorder=4, linewidths=0)
        ax.bar(xx + 0.17, 2 * g['hw90_Fs'], bottom=g['Fs'] - g['hw90_Fs'], width=0.32, color=c2, alpha=0.35, linewidth=0, label=f'{nm}: Fs, 90 % region'); ax.scatter(xx + 0.17, g['Fs'], s=14, marker='s', color=c2, zorder=4, linewidths=0)
        ax.plot([x0 - 0.5, x0 + len(g) - 0.5], [g['ref_Fo'].iloc[0]] * 2, color=col, lw=1.0, ls='--', label=f"{nm}: reference Fo ({g['ref_Fo'].iloc[0]:.0f})"); ax.plot([x0 - 0.5, x0 + len(g) - 0.5], [g['ref_Fs'].iloc[0]] * 2, color=c2, lw=1.0, ls='--', label=f"{nm}: reference Fs ({g['ref_Fs'].iloc[0]:.0f})")
        ticks += list(xx); tlabels += [('Eros ' + str(d)[:7]) if key == 'eros' else 'Itokawa' for d in g['date_obs']]; x0 += len(g) + 1
    ax.set_ylim(0, 100); ax.set_xticks(ticks); ax.set_xticklabels(tlabels, rotation=90, fontsize=5.5); ax.set_xlabel('spectrum (date of observation)'); ax.set_ylabel('Fo, Fs (mol %)'); legend_in(L[1][1], ax=ax, ncol=2)
    save(fig, 'Figure7')

if __name__ == '__main__':
    D = load(); which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7']
    for w in which: globals()[f'fig{w}'](D)
