import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
GRID_N = 31
GRID_RANGE = 30.0


def load_grid(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    x = df['x_nm'].values.reshape(GRID_N, GRID_N)
    z = df['z_nm'].values.reshape(GRID_N, GRID_N)
    enorm = df['enorm'].values.reshape(GRID_N, GRID_N)
    e2 = df['e2'].values.reshape(GRID_N, GRID_N)
    return x, z, enorm, e2


def load_linecut(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df['z_nm'].values, df['enorm'].values


def plot_colormap(e2: np.ndarray, title_str: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize = (7, 6))
    # log10(|E|^2), clamp zeros
    e2_safe = np.where(e2 > 0, e2, np.nanmin(e2[e2 > 0]))
    im = ax.imshow(
        np.log10(e2_safe),
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
        origin = 'lower',
        cmap = 'hot',
        aspect = 'equal'
    )
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('log10(|E|^2)')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('z (nm)')
    ax.set_title(title_str)
    fig.tight_layout()
    fig.savefig(save_path, dpi = 150)
    plt.close(fig)
    print('[info] Saved {}'.format(save_path))


def plot_comparison(e2_matlab: np.ndarray, e2_python: np.ndarray,
        mode: str, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize = (18, 5))

    e2_safe_m = np.where(e2_matlab > 0, e2_matlab, np.nanmin(e2_matlab[e2_matlab > 0]))
    e2_safe_p = np.where(e2_python > 0, e2_python, np.nanmin(e2_python[e2_python > 0]))
    extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]

    vmin = min(np.log10(e2_safe_m).min(), np.log10(e2_safe_p).min())
    vmax = max(np.log10(e2_safe_m).max(), np.log10(e2_safe_p).max())

    # MATLAB
    im0 = axes[0].imshow(np.log10(e2_safe_m), extent = extent, origin = 'lower',
        cmap = 'hot', aspect = 'equal', vmin = vmin, vmax = vmax)
    axes[0].set_title('MATLAB {}'.format(mode))
    axes[0].set_xlabel('x (nm)')
    axes[0].set_ylabel('z (nm)')
    fig.colorbar(im0, ax = axes[0], label = 'log10(|E|^2)')

    # Python
    im1 = axes[1].imshow(np.log10(e2_safe_p), extent = extent, origin = 'lower',
        cmap = 'hot', aspect = 'equal', vmin = vmin, vmax = vmax)
    axes[1].set_title('Python {}'.format(mode))
    axes[1].set_xlabel('x (nm)')
    axes[1].set_ylabel('z (nm)')
    fig.colorbar(im1, ax = axes[1], label = 'log10(|E|^2)')

    # Relative error
    rel_err = np.abs(e2_python - e2_matlab) / np.maximum(np.abs(e2_matlab), 1e-30) * 100.0
    im2 = axes[2].imshow(rel_err, extent = extent, origin = 'lower',
        cmap = 'RdYlGn_r', aspect = 'equal')
    axes[2].set_title('{} relative error (%)'.format(mode))
    axes[2].set_xlabel('x (nm)')
    axes[2].set_ylabel('z (nm)')
    fig.colorbar(im2, ax = axes[2], label = 'Relative error (%)')

    fig.suptitle('{} NearField: MATLAB vs Python (max err = {:.2f}%)'.format(
        mode, np.nanmax(rel_err[np.isfinite(rel_err)])))
    fig.tight_layout()
    fig.savefig(save_path, dpi = 150)
    plt.close(fig)
    print('[info] Saved {}'.format(save_path))


def plot_linecut(mode: str, save_path: str) -> None:
    z_m, enorm_m = load_linecut(os.path.join(DATA_DIR, 'matlab_{}_linecut.csv'.format(mode)))
    z_p, enorm_p = load_linecut(os.path.join(DATA_DIR, 'python_{}_linecut.csv'.format(mode)))

    fig, axes = plt.subplots(2, 1, figsize = (8, 8), gridspec_kw = {'height_ratios': [3, 1]})

    # Top: |E| overlay
    axes[0].plot(z_m, enorm_m, 'b-', linewidth = 1.5, label = 'MATLAB')
    axes[0].plot(z_p, enorm_p, 'r--', linewidth = 1.5, label = 'Python')
    axes[0].set_xlabel('z (nm)')
    axes[0].set_ylabel('|E|')
    axes[0].set_title('{}: x=0 linecut, lambda=520nm'.format(mode))
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)

    # Bottom: relative error
    rel_err = np.abs(enorm_p - enorm_m) / np.maximum(np.abs(enorm_m), 1e-30) * 100.0
    axes[1].plot(z_m, rel_err, 'k-', linewidth = 1.0)
    axes[1].set_xlabel('z (nm)')
    axes[1].set_ylabel('Relative error (%)')
    axes[1].set_title('Max relative error: {:.2f}%'.format(np.nanmax(rel_err[np.isfinite(rel_err)])))
    axes[1].grid(True, alpha = 0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi = 150)
    plt.close(fig)
    print('[info] Saved {}'.format(save_path))


def print_summary(mode: str) -> None:
    _, _, enorm_m, e2_m = load_grid(os.path.join(DATA_DIR, 'matlab_{}.csv'.format(mode)))
    _, _, enorm_p, e2_p = load_grid(os.path.join(DATA_DIR, 'python_{}.csv'.format(mode)))

    rel_err_enorm = np.abs(enorm_p - enorm_m) / np.maximum(np.abs(enorm_m), 1e-30) * 100.0
    valid = np.isfinite(rel_err_enorm)

    print('[info] {} |E| relative error:'.format(mode))
    print('       mean = {:.4f}%'.format(np.mean(rel_err_enorm[valid])))
    print('       max  = {:.4f}%'.format(np.max(rel_err_enorm[valid])))
    print('       median = {:.4f}%'.format(np.median(rel_err_enorm[valid])))


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok = True)

    # --- stat ---
    print('[info] === stat comparison ===')
    has_matlab_stat = os.path.exists(os.path.join(DATA_DIR, 'matlab_stat.csv'))
    has_python_stat = os.path.exists(os.path.join(DATA_DIR, 'python_stat.csv'))

    if has_python_stat:
        _, _, enorm_p_s, e2_p_s = load_grid(os.path.join(DATA_DIR, 'python_stat.csv'))
        plot_colormap(e2_p_s, 'Python BEMStat |E|^2 (log10), lambda=520nm',
            os.path.join(FIG_DIR, 'stat_python.png'))

    if has_matlab_stat and has_python_stat:
        _, _, enorm_m_s, e2_m_s = load_grid(os.path.join(DATA_DIR, 'matlab_stat.csv'))
        plot_comparison(e2_m_s, e2_p_s, 'stat',
            os.path.join(FIG_DIR, 'stat_comparison.png'))
        print_summary('stat')

    if has_matlab_stat and has_python_stat:
        plot_linecut('stat', os.path.join(FIG_DIR, 'stat_linecut.png'))

    # --- ret ---
    print('[info] === ret comparison ===')
    has_matlab_ret = os.path.exists(os.path.join(DATA_DIR, 'matlab_ret.csv'))
    has_python_ret = os.path.exists(os.path.join(DATA_DIR, 'python_ret.csv'))

    if has_python_ret:
        _, _, enorm_p_r, e2_p_r = load_grid(os.path.join(DATA_DIR, 'python_ret.csv'))
        plot_colormap(e2_p_r, 'Python BEMRet |E|^2 (log10), lambda=520nm',
            os.path.join(FIG_DIR, 'ret_python.png'))

    if has_matlab_ret and has_python_ret:
        _, _, enorm_m_r, e2_m_r = load_grid(os.path.join(DATA_DIR, 'matlab_ret.csv'))
        plot_comparison(e2_m_r, e2_p_r, 'ret',
            os.path.join(FIG_DIR, 'ret_comparison.png'))
        print_summary('ret')

    if has_matlab_ret and has_python_ret:
        plot_linecut('ret', os.path.join(FIG_DIR, 'ret_linecut.png'))

    # --- Timing ---
    if os.path.exists(os.path.join(DATA_DIR, 'matlab_timing.csv')):
        df_mt = pd.read_csv(os.path.join(DATA_DIR, 'matlab_timing.csv'))
        print('[info] MATLAB timing:')
        for _, row in df_mt.iterrows():
            print('       {} = {:.4f} sec'.format(row['solver'], row['time_sec']))

    if os.path.exists(os.path.join(DATA_DIR, 'python_timing.csv')):
        df_pt = pd.read_csv(os.path.join(DATA_DIR, 'python_timing.csv'))
        print('[info] Python timing:')
        for _, row in df_pt.iterrows():
            print('       {} = {:.4f} sec'.format(row['solver'], row['time_sec']))

    print('[info] Comparison complete.')
