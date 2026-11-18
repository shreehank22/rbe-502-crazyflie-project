"""Overlays the position-error components of several runs (Task 2: gain sensitivity).

Each run is a folder written by trajectory_tracking.py that contains log.npz.

Examples (from the examples folder):

    python compare_errors.py --traj circle
    python compare_errors.py --traj diamond
    python compare_errors.py --traj circle --runs "baseline=results/circle_A" "K_v x 0.5=results/circle_B"
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

#### Default Task 2 runs (baseline: wn = 10 rad/s, zeta = 0.9) ##################
DEFAULT_RUNS = {
    'circle': {
        'baseline':                  'results/circle_15s_20261117-120213',
        r'$K_v \times 0.5$':         'results/circle_15s_20261117-143600',
        r'$K_x \times 0.5$':         'results/circle_15s_20261117-143904',
    },
    'diamond': {
        'baseline':                  'results/diamond_8s_20261117-120237',
        r'$K_v \times 0.5$':         'results/diamond_8s_20261117-143530',
        r'$K_x \times 0.5$':         'results/diamond_8s_20261117-143937',
    },
}
STYLES = ['-', '--', ':', '-.']


def load_run(folder):
    """Returns t, pos_err (N, 3) over the trajectory duration, and the per-axis and norm RMSE."""
    log = np.load(os.path.join(folder, 'log.npz'))
    t = log['t']
    mask = t <= float(log['traj_duration']) + 1e-9          # same window as the provided RMSE
    err = log['pos_err'][mask]
    rmse_xyz = np.sqrt(np.mean(err**2, axis=0))
    rmse_norm = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return t[mask], err, rmse_xyz, rmse_norm


def compare(traj, runs, output, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    print(f"\n{traj}: RMSE [mm]")
    print(f"{'run':>18s} {'x':>7s} {'y':>7s} {'z':>7s} {'norm':>7s}")
    for k, (label, folder) in enumerate(runs.items()):
        t, err, rmse_xyz, rmse_norm = load_run(folder)
        plain = label.replace('$', '').replace('\\times', 'x').replace('K_', 'K')
        print(f"{plain:>18s} " + ' '.join(f"{1e3 * v:7.2f}" for v in (*rmse_xyz, rmse_norm)))
        for i in range(3):
            axs[i].plot(t, 1e3 * err[:, i], STYLES[k % len(STYLES)], lw=1.2,
                        label=f"{label} (RMSE {1e3 * rmse_xyz[i]:.2f} mm)")
    for i, name in enumerate('xyz'):
        axs[i].set_ylabel(f'$e_{name}$ [mm]')
        axs[i].grid(True, alpha=0.4)
        axs[i].legend(fontsize=8, loc='upper right')
    axs[-1].set_xlabel('t [s]')
    axs[0].set_title(title or f'Position error, {traj}: gain sensitivity')
    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output)
    print(f"Saved {os.path.abspath(output)}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay the position errors of several runs')
    parser.add_argument('--traj', default='circle', choices=list(DEFAULT_RUNS))
    parser.add_argument('--runs', nargs='+', metavar='LABEL=FOLDER', default=None,
                        help='Runs to compare; defaults to the Task 2 runs of --traj')
    parser.add_argument('--output', default=None, help='Output file (default: figures/task2_<traj>.pdf)')
    parser.add_argument('--title', default=None, help='Figure title')
    parser.add_argument('--show', action='store_true', help='Show the figure')
    args = parser.parse_args()

    # split on the LAST '=' so labels may contain '=' (e.g. "T = 8 s=results/...")
    runs = dict(r.rsplit('=', 1) for r in args.runs) if args.runs else DEFAULT_RUNS[args.traj]
    output = args.output or os.path.join('figures', f'task2_{args.traj}.pdf')
    compare(args.traj, runs, output, args.title)
    if args.show:
        plt.show()
