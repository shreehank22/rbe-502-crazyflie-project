import os

import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.trajectory.diamond import diamond

M = 0.027
G = 9.8
F_MAX = 2.25*M*G
E3 = np.array([0.0, 0.0, 1.0])

DURATIONS = [8, 6, 5, 4,3.75, 3.5,3.25, 3, 2.5, 2]    # candidates for the table
CHOSEN = [8, 5, 3.5]                          # durations used in Task 3


def f_d_of(T, n=1000):

    t = np.linspace(0, T, n)
    acc = np.zeros((len(t), 3))
    for i, ti in enumerate(t):
        acc[i, :] = diamond(ti, tfinal=T)['acc']
    f_d = M*np.linalg.norm(acc + G*E3, axis=1)
    return t, f_d


#### Table: peak f_d for each candidate duration ##############################
print(f"{'T [s]':>6s} {'peak f_d / mg':>14s} {'in [1.5, 2.25]?':>16s}")
for T in DURATIONS:
    _, f_d = f_d_of(T)
    peak = f_d.max()/(M*G)
    print(f"{T:6.2f} {peak:14.3f} {'yes' if 1.5 <= peak <= 2.25 else '':>16s}")

#### Plot: f_d(t) for the chosen durations, with f_max ########################
fig, ax = plt.subplots(figsize=(7, 4))
for T in CHOSEN:
    t, f_d = f_d_of(T)
    ax.plot(t, f_d, label=f"T = {T} s (peak {f_d.max()/(M*G):.2f} mg)")
ax.axhline(F_MAX, linestyle='--', color='k', label=r'$f_{max} = 2.25\,mg$')
ax.axhline(M*G, linestyle=':', color='gray', label=r'$mg$ (hover)')
ax.set_xlabel('t [s]')
ax.set_ylabel(r'$f_d$ [N]')
ax.set_title(r'Required thrust $f_d(t) = m\,\|a_d + g e_3\|$, diamond')
ax.grid(True, alpha=0.4)
ax.set_ylim(top=0.72)
ax.legend(fontsize=8, loc='upper right', ncol=2)
fig.tight_layout()

os.makedirs('figures', exist_ok=True)
fig.savefig('figures/task3_fd.pdf')
fig.savefig('figures/task3_fd.png', dpi=150)
print(f"Saved {os.path.abspath('figures/task3_fd.pdf')}")

#### Plot: actual thrust f(t) from the runs vs the ideal f_d(t) ###############
RUNS = {                                    # duration -> run folder from trajectory_tracking.py
    8:   'results/diamond_8s_20261117-120237',
    5:   'results/diamond_5s_20261117-181924',
    3.5: 'results/diamond_3.5s_20261117-182422',
}

fig, axs = plt.subplots(len(RUNS), 1, figsize=(7, 7), sharex=True)
print(f"\n{'T [s]':>6s} {'peak f_d / mg':>14s} {'peak f / mg':>12s}")
for ax, (T, folder) in zip(axs, RUNS.items()):
    log = np.load(os.path.join(folder, 'log.npz'))
    mask = log['t'] <= T + 1e-9                     # trajectory window only
    t_run, f_run = log['t'][mask], log['thrust'][mask]
    t, f_d = f_d_of(T)
    ax.plot(t_run, f_run, lw=1.3, label=f"actual $f$ (peak {f_run.max()/(M*G):.2f} mg)")
    ax.plot(t, f_d, '--', lw=1.3, label=f"ideal $f_d$ (peak {f_d.max()/(M*G):.2f} mg)")
    ax.axhline(F_MAX, linestyle='--', color='k', lw=1, label=r'$f_{max}$')
    ax.axhline(M*G, linestyle=':', color='gray', lw=1, label=r'$mg$')
    ax.set_ylabel('thrust [N]')
    ax.set_title(f"T = {T} s", fontsize=10)
    ax.set_ylim(0.1, 0.65)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    print(f"{T:6.2f} {f_d.max()/(M*G):14.3f} {f_run.max()/(M*G):12.3f}")
axs[-1].set_xlabel('t [s]')
fig.suptitle('Total thrust: actual $f(t)$ vs ideal $f_d(t)$, diamond')
fig.tight_layout()
fig.savefig('figures/task3_thrust.pdf')
fig.savefig('figures/task3_thrust.png', dpi=150)
print(f"Saved {os.path.abspath('figures/task3_thrust.pdf')}")
