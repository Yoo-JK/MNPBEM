"""
13_shapes comparison: overlay MATLAB vs Python extinction spectra
and produce shapes_overview.png with all 7 shapes on one plot.

Generates:
  - {shape}_comparison.png  (7 files)
  - shapes_overview.png     (1 file)
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/data'
FIG_DIR = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/figures'

SHAPES = [
    'trisphere',
    'trirod',
    'tricube',
    'tritorus',
    'trispheresegment',
    'trispherescale',
    'tripolygon',
]

TITLES = {
    'trisphere':        'trisphere(144, 20)',
    'trirod':           'trirod(10, 40, [15,15,15])',
    'tricube':          'tricube(10, 20)',
    'tritorus':         'tritorus(15, 5, [20,20])',
    'trispheresegment': 'trispheresegment (hemisphere, d=20)',
    'trispherescale':   'scale(trisphere(144,20), [1,1,2])',
    'tripolygon':       'tripolygon (hexagon + EdgeProfile)',
}


def load_csv(path):
    """Load CSV with header row."""
    with open(path) as f:
        header = f.readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return header, data


def rms_relative_error(a, b):
    """RMS of relative error |a-b|/max(|b|, eps)."""
    denom = np.maximum(np.abs(b), 1e-30)
    return np.sqrt(np.mean(((a - b) / denom) ** 2))


def make_comparison(shape):
    """Generate comparison figure for a single shape."""
    matlab_path = os.path.join(DATA_DIR, '{}_matlab.csv'.format(shape))
    python_path = os.path.join(DATA_DIR, '{}_python.csv'.format(shape))

    header_m, data_m = load_csv(matlab_path)
    header_p, data_p = load_csv(python_path)

    wl_m = data_m[:, 0]
    wl_p = data_p[:, 0]

    ncols = data_m.shape[1] - 1  # number of extinction columns

    fig, axes = plt.subplots(ncols, 2, figsize=(14, 5 * ncols), squeeze=False)

    for ic in range(ncols):
        ext_m = data_m[:, 1 + ic]
        ext_p = data_p[:, 1 + ic]

        # Interpolate python to matlab grid if needed
        if not np.allclose(wl_m, wl_p, atol=0.01):
            ext_p = np.interp(wl_m, wl_p, ext_p)

        rms_err = rms_relative_error(ext_p, ext_m)

        col_label = header_m[1 + ic] if ncols > 1 else 'ext'

        # Left: overlay
        ax = axes[ic, 0]
        ax.plot(wl_m, ext_m, 'b-', lw=1.5, label='MATLAB')
        ax.plot(wl_m, ext_p, 'r--', lw=1.5, label='Python')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm$^2$)')
        ax.set_title('{} -- {} (RMS err: {:.2e})'.format(TITLES[shape], col_label, rms_err))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Right: relative error
        ax2 = axes[ic, 1]
        rel_err = np.abs(ext_p - ext_m) / np.maximum(np.abs(ext_m), 1e-30)
        ax2.semilogy(wl_m, rel_err, 'k-', lw=1)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Relative error')
        ax2.set_title('Relative error -- {}'.format(col_label))
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, '{}_comparison.png'.format(shape))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('[saved] {}'.format(out_path))

    return rms_err


def make_overview():
    """Generate shapes_overview.png with all 7 shapes on one plot."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    colors_m = 'b'
    colors_p = 'r'

    for idx, shape in enumerate(SHAPES):
        ax = axes[idx]
        matlab_path = os.path.join(DATA_DIR, '{}_matlab.csv'.format(shape))
        python_path = os.path.join(DATA_DIR, '{}_python.csv'.format(shape))

        _, data_m = load_csv(matlab_path)
        _, data_p = load_csv(python_path)

        wl = data_m[:, 0]

        if shape == 'trirod':
            # x-pol and z-pol
            ax.plot(wl, data_m[:, 1], 'b-', lw=1.5, label='MATLAB x-pol')
            ax.plot(wl, data_m[:, 2], 'b--', lw=1.5, label='MATLAB z-pol')
            ax.plot(wl, data_p[:, 1], 'r-', lw=1, alpha=0.8, label='Python x-pol')
            ax.plot(wl, data_p[:, 2], 'r--', lw=1, alpha=0.8, label='Python z-pol')
        else:
            ax.plot(wl, data_m[:, 1], 'b-', lw=1.5, label='MATLAB')
            ax.plot(wl, data_p[:, 1], 'r--', lw=1.5, label='Python')

        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Extinction (nm$^2$)', fontsize=9)
        ax.set_title(TITLES[shape], fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(SHAPES), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('13_shapes: BEMStat Extinction -- MATLAB vs Python (7 shapes)',
                 fontsize=14, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, 'shapes_overview.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('[saved] {}'.format(out_path))


def main():
    print('=== 13_shapes Comparison ===')
    for shape in SHAPES:
        make_comparison(shape)

    make_overview()
    print('\n[info] All comparison figures generated.')


if __name__ == '__main__':
    main()
