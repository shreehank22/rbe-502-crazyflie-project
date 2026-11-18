"""Overlays the 3D paths of several runs on the desired path (Task 3: trajectory duration).

Left: 3D view with equal axis scaling. Right: the 2D projection in the plane of the motion
(x-y top view for the circle, y-z side view for the diamond), where deviations are easier to see.

Usage (from the examples folder):
    python plot_paths.py --traj circle
    python plot_paths.py --traj diamond
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RUNS = {
    'circle': {
        'T = 15 s': 'results/circle_15s_20261117-120213',
        'T = 12 s': 'results/circle_scaled_12s_20261117-183215',
        'T = 9 s':  'results/circle_scaled_9s_20261117-183217',
    },
    'diamond': {
        'T = 8 s':   'results/diamond_8s_20261117-120237',
        'T = 5 s':   'results/diamond_5s_20261117-181924',
        'T = 3.5 s': 'results/diamond_3.5s_20261117-182422',
    },
}
PLANE = {'circle': (0, 1, 'x', 'y', 'top view (x-y)'),
         'diamond': (1, 2, 'y', 'z', 'side view (y-z)')}
STYLES = ['-', '--', ':']


def load(folder):
    """Returns the actual and desired positions over the trajectory window."""
    log = np.load(os.path.join(folder, 'log.npz'))
    mask = log['t'] <= float(log['traj_duration']) + 1e-9
    return log['pos'][mask], log['pos_d'][mask]


def set_equal_3d(ax, pts):
    """Equal scaling on the three axes, so the path shape is not distorted."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))


def plot_paths(traj, runs, output):
    i, j, ni, nj, view = PLANE[traj]
    fig = plt.figure(figsize=(11, 5))
    ax3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    all_pts = []
    for k, (label, folder) in enumerate(runs.items()):
        pos, pos_d = load(folder)
        if k == 0:                                    # same geometric path for every duration
            ax3.plot(*pos_d.T, 'k', lw=2.5, alpha=0.35, label='desired')
            ax2.plot(pos_d[:, i], pos_d[:, j], 'k', lw=2.5, alpha=0.35, label='desired')
            all_pts.append(pos_d)
        max_dev = 1e3 * np.linalg.norm(pos - pos_d, axis=1).max()
        ax3.plot(*pos.T, STYLES[k % 3], lw=1.3, label=label)
        ax2.plot(pos[:, i], pos[:, j], STYLES[k % 3], lw=1.3,
                 label=f"{label} (max error {max_dev:.0f} mm)")
        all_pts.append(pos)

    set_equal_3d(ax3, np.vstack(all_pts))
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.set_title(f'3D path, {traj}')
    ax3.legend(fontsize=8, loc='upper left')

    ax2.set_aspect('equal', adjustable='datalim')
    ax2.set_xlabel(f'{ni} [m]'); ax2.set_ylabel(f'{nj} [m]')
    ax2.set_title(f'{traj}, {view}')
    ax2.grid(True, alpha=0.4)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output)
    print(f"Saved {os.path.abspath(output)}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Overlay the 3D paths of several runs')
    parser.add_argument('--traj', default='circle', choices=list(DEFAULT_RUNS))
    parser.add_argument('--runs', nargs='+', metavar='LABEL=FOLDER', default=None)
    parser.add_argument('--output', default=None, help='default: figures/task3_<traj>_paths.pdf')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    runs = dict(r.rsplit('=', 1) for r in args.runs) if args.runs else DEFAULT_RUNS[args.traj]
    plot_paths(args.traj, runs, args.output or os.path.join('figures', f'task3_{args.traj}_paths.pdf'))
    if args.show:
        plt.show()
