import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.trajectory import circle as circle_module
import gym_pybullet_drones.examples.trajectory_tracking as tt
from gym_pybullet_drones.utils.utils import str2bool

M = 0.027
G = 9.8
F_MAX = 2.25*M*G
E3 = np.array([0.0, 0.0, 1.0])
T0 = 15.0                                       # default circle duration [s]

DURATIONS = [15, 12, 11, 10, 9.5, 9, 8.5, 8, 7.5, 7]   # candidates for the table
CHOSEN = [15, 12, 9]                                   # durations used in Task 3


def circle_scaled(t, T=T0):
    """Desired state of the circle with total duration T, by time scaling the provided circle."""
    s = T0 / T
    d = circle_module.circle(s * t)
    return {'pos': np.asarray(d['pos'], dtype=float),
            'vel': s * np.asarray(d['vel'], dtype=float),
            'acc': s**2 * np.asarray(d['acc'], dtype=float),
            'jerk': s**3 * np.asarray(d['jerk'], dtype=float),
            'yaw': d['yaw'],
            'yawdot': s * d['yawdot']}


def f_d_of(T, n=3000):
    """Returns t and the thrust f_d(t) = m ||a_d(t) + g e3|| for a circle of duration T."""
    t = np.linspace(0, T, n)
    acc = np.array([circle_scaled(ti, T)['acc'] for ti in t])
    return t, M*np.linalg.norm(acc + G*E3, axis=1), acc


def rmse_of(log):
    mask = log['t'] <= float(log['traj_duration']) + 1e-9
    err = log['pos_err'][mask]
    return np.sqrt(np.mean(err**2, axis=0)), np.sqrt(np.mean(np.sum(err**2, axis=1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 3 for the time-scaled circle')
    parser.add_argument('--durations', nargs='+', type=float, default=CHOSEN)
    parser.add_argument('--gui', default=False, type=str2bool)
    parser.add_argument('--output_folder', default='results')
    args = parser.parse_args()
    chosen = args.durations

    #### Table: peak f_d and feedforward tilt for each candidate duration ######
    print(f"{'T [s]':>6s} {'peak f_d / mg':>14s} {'peak tilt [deg]':>16s} {'in [1.5, 2.25]?':>16s}")
    for T in DURATIONS:
        _, f_d, acc = f_d_of(T)
        v = acc + G*E3
        tilt = np.degrees(np.arccos(v[:, 2] / np.linalg.norm(v, axis=1))).max()
        peak = f_d.max()/(M*G)
        print(f"{T:6.2f} {peak:14.3f} {tilt:16.1f} {'yes' if 1.5 <= peak <= 2.25 else '':>16s}")

    #### Run the simulations with the provided tracking script #################
    tt.TRAJECTORIES['circle_scaled'] = (circle_scaled, T0)
    folders = {}
    for T in chosen:
        before = set(os.listdir(args.output_folder)) if os.path.isdir(args.output_folder) else set()
        tt.run(traj='circle_scaled', traj_duration=T, gui=args.gui, plot=False,
               output_folder=args.output_folder)
        new = sorted(set(os.listdir(args.output_folder)) - before)
        folders[T] = os.path.join(args.output_folder, new[-1])

    #### Results table #########################################################
    print(f"\n{'T [s]':>6s} {'x':>7s} {'y':>7s} {'z':>7s} {'norm':>7s}   [mm]   "
          f"{'peak f_d':>9s} {'peak f':>7s} [mg] {'max tilt':>9s}  folder")
    logs = {}
    for T, folder in folders.items():
        log = np.load(os.path.join(folder, 'log.npz'))
        logs[T] = log
        r, n = rmse_of(log)
        _, f_d, _ = f_d_of(T)
        mask = log['t'] <= T + 1e-9
        tilt = np.degrees(np.arccos(np.clip(log['rot'][:, 2, 2], -1, 1))).max()
        print(f"{T:6.2f} " + ' '.join(f"{1e3*v:7.2f}" for v in (*r, n)) +
              f"          {f_d.max()/(M*G):9.2f} {log['thrust'][mask].max()/(M*G):7.2f}      "
              f"{tilt:9.1f}  {folder}")

    os.makedirs('figures', exist_ok=True)

    #### Figure 1: f_d(t) for the chosen durations, with f_max #################
    fig, ax = plt.subplots(figsize=(7, 4))
    for T in chosen:
        t, f_d, _ = f_d_of(T)
        ax.plot(t, f_d, label=f"T = {T:g} s (peak {f_d.max()/(M*G):.2f} mg)")
    ax.axhline(F_MAX, linestyle='--', color='k', label=r'$f_{max} = 2.25\,mg$')
    ax.axhline(M*G, linestyle=':', color='gray', label=r'$mg$ (hover)')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(r'$f_d$ [N]')
    ax.set_title(r'Required thrust $f_d(t) = m\,\|a_d + g e_3\|$, circle')
    ax.set_ylim(top=0.72)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    fig.tight_layout()
    fig.savefig('figures/task3_circle_fd.pdf')

    #### Figure 2: position error components ####################################
    styles = ['-', '--', ':']
    fig, axs = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    for k, (T, log) in enumerate(logs.items()):
        mask = log['t'] <= T + 1e-9
        r, _ = rmse_of(log)
        for i in range(3):
            axs[i].plot(log['t'][mask], 1e3*log['pos_err'][mask, i], styles[k % 3], lw=1.2,
                        label=f"T = {T:g} s (RMSE {1e3*r[i]:.2f} mm)")
    for i, name in enumerate('xyz'):
        axs[i].set_ylabel(f'$e_{name}$ [mm]')
        axs[i].grid(True, alpha=0.4)
        axs[i].legend(fontsize=8, loc='upper right')
    axs[-1].set_xlabel('t [s]')
    axs[0].set_title('Position error, circle: trajectory duration')
    fig.tight_layout()
    fig.savefig('figures/task3_circle_errors.pdf')

    #### Figure 3: actual thrust vs ideal f_d ###################################
    fig, axs = plt.subplots(len(logs), 1, figsize=(7, 2.4*len(logs)), sharex=True)
    for ax, (T, log) in zip(np.atleast_1d(axs), logs.items()):
        mask = log['t'] <= T + 1e-9
        t, f_d, _ = f_d_of(T)
        ax.plot(log['t'][mask], log['thrust'][mask], lw=1.3,
                label=f"actual $f$ (peak {log['thrust'][mask].max()/(M*G):.2f} mg)")
        ax.plot(t, f_d, '--', lw=1.3, label=f"ideal $f_d$ (peak {f_d.max()/(M*G):.2f} mg)")
        ax.axhline(F_MAX, linestyle='--', color='k', lw=1, label=r'$f_{max}$')
        ax.axhline(M*G, linestyle=':', color='gray', lw=1, label=r'$mg$')
        ax.set_ylabel('thrust [N]')
        ax.set_title(f"T = {T:g} s", fontsize=10)
        ax.set_ylim(0.1, 0.72)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
    np.atleast_1d(axs)[-1].set_xlabel('t [s]')
    fig.suptitle('Total thrust: actual $f(t)$ vs ideal $f_d(t)$, circle')
    fig.tight_layout()
    fig.savefig('figures/task3_circle_thrust.pdf')
    print("\nSaved figures/task3_circle_fd.pdf, figures/task3_circle_errors.pdf, figures/task3_circle_thrust.pdf")
