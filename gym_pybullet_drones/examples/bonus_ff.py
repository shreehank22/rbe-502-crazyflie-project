"""Bonus: tracking with and without the angular velocity feedforward Omega_d.

Runs the shortest Task 3 durations (diamond 3.5 s, circle 9 s) with GeometricControl
(Omega_d = 0) and GeometricControlFF (Omega_d from differential flatness), and compares
the per-axis RMSE and the attitude error.

The provided trajectories return jerk = 0, and trajectory_tracking.run() does not pass the
jerk to the controller, so this script uses the same simulation loop as run() and computes
the desired jerk numerically, by a central difference of the desired acceleration:

    j_d(t) = (a_d(t + h) - a_d(t - h)) / (2 h),   h = 1e-4 s   (one-sided at t = 0).

Usage (from the examples folder):
    python bonus_ff.py                      # both trajectories, headless
    python bonus_ff.py --gui true           # with the PyBullet GUI
"""
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.GeometricControl import GeometricControl
from gym_pybullet_drones.control.GeometricControlFF import GeometricControlFF
from gym_pybullet_drones.utils.TrackingLogger import TrackingLogger, attitude_error
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.trajectory.diamond import diamond
from gym_pybullet_drones.examples.task3_circle import circle_scaled
from gym_pybullet_drones.examples.trajectory_tracking import (SIMULATION_FREQ_HZ, CONTROL_FREQ_HZ,
                                                              HOVER_SEC)

JERK_H = 1e-4      # step for the central difference of the desired acceleration [s]


CASES = {          # name: (trajectory function of (t, T), duration T [s])
    'diamond': (lambda t, T: diamond(t, tfinal=T), 3.5),
    'circle':  (circle_scaled, 9.0),
}


def numerical_jerk(traj_fn, t, T):
    """Central difference of the desired acceleration (one-sided at t = 0)."""
    a_plus = np.asarray(traj_fn(t + JERK_H, T)['acc'], dtype=float)
    a_minus = np.asarray(traj_fn(max(t - JERK_H, 0.0), T)['acc'], dtype=float)
    return (a_plus - a_minus) / (JERK_H + min(t, JERK_H))


def desired_with_jerk(name, t):
    """Desired state with the numerically differentiated jerk."""
    traj_fn, T = CASES[name]
    des = dict(traj_fn(t, T))
    des['jerk'] = numerical_jerk(traj_fn, t, T)
    return des


def run(name, use_ff, gui=False, output_folder='results'):
    """Same simulation loop as trajectory_tracking.run(), with the jerk passed to the controller."""
    traj_fn, T = CASES[name]
    ctrl = GeometricControlFF(drone_model=DroneModel.CF2X) if use_ff else GeometricControl(drone_model=DroneModel.CF2X)
    logger = TrackingLogger(traj_name=f"{name}_{'ff' if use_ff else 'noff'}", traj_duration=T)

    init_pos = np.array(traj_fn(0.0, T)['pos'], dtype=float)
    rest_height = ctrl._getURDFParameter('length') / 2 - ctrl._getURDFParameter('collision_z_offset')
    init_pos[2] = max(init_pos[2], rest_height)
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=init_pos.reshape(1, 3),
                     initial_rpys=np.zeros((1, 3)), physics=Physics.PYB, pyb_freq=SIMULATION_FREQ_HZ,
                     ctrl_freq=CONTROL_FREQ_HZ, gui=gui, record=False)
    obs, _ = env.reset()

    import time
    start = time.time()
    for i in range(int(round((T + HOVER_SEC) * env.CTRL_FREQ))):
        t = i * env.CTRL_TIMESTEP
        des = desired_with_jerk(name, t)
        kwargs = dict(state=obs[0], target_pos=des['pos'], target_vel=des['vel'],
                      target_acc=des['acc'], target_yaw=des['yaw'])
        if use_ff:
            kwargs.update(target_jerk=des['jerk'], target_yawdot=des['yawdot'])
        rpm, info = ctrl.computeControlFromState(**kwargs)
        logger.log(t, obs[0], des, info, rpm)
        obs, _, _, _, _ = env.step(rpm.reshape(1, 4))
        if gui:
            sync(i, start, env.CTRL_TIMESTEP)
    env.close()

    folder = os.path.join(output_folder, f"bonus_{name}_{'ff' if use_ff else 'noff'}_{T:g}s_"
                                         f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    logger.save(folder)
    logger.plot(folder=folder, show=False)
    return folder


def metrics(folder):
    log = np.load(os.path.join(folder, 'log.npz'))
    t = log['t']; m = t <= float(log['traj_duration']) + 1e-9
    e = log['pos_err'][m]
    rmse = np.sqrt(np.mean(e**2, axis=0)); norm = np.sqrt(np.mean(np.sum(e**2, axis=1)))
    eR = np.linalg.norm(log['att_err'][m], axis=1)
    ang = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', log['rot_d'][m], log['rot'][m]) - 1) / 2, -1, 1)))
    return dict(t=t[m], eR=eR, rmse=rmse, norm=norm, eR_rms=np.sqrt(np.mean(eR**2)), ang_max=ang.max(),
                tilt=np.degrees(np.arccos(np.clip(log['rot'][m, 2, 2], -1, 1))).max(),
                fpk=log['thrust'][m].max() / (0.027 * 9.8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bonus: Omega_d feedforward comparison')
    parser.add_argument('--gui', default=False, type=str2bool)
    parser.add_argument('--output_folder', default='results')
    args = parser.parse_args()

    res = {}
    for name in CASES:
        for use_ff in (False, True):
            res[(name, use_ff)] = metrics(run(name, use_ff, args.gui, args.output_folder))

    print(f"\n{'case':>16s} {'x':>7s} {'y':>7s} {'z':>7s} {'norm':>7s} [mm]  "
          f"{'rms|e_R|':>9s} {'max angle':>10s} {'max tilt':>9s} {'peak f':>7s}")
    for (name, ff), r in res.items():
        print(f"{name + (' FF' if ff else ' no FF'):>16s} " + ' '.join(f"{1e3 * v:7.2f}" for v in (*r['rmse'], r['norm'])) +
              f"       {r['eR_rms']:9.4f} {r['ang_max']:9.1f}° {r['tilt']:8.1f}° {r['fpk']:6.2f}mg")

    #### Figure: attitude error norm with and without feedforward ##############
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, name in zip(axs, CASES):
        for ff, style in ((False, '-'), (True, '--')):
            r = res[(name, ff)]
            ax.plot(r['t'], r['eR'], style, lw=1.3,
                    label=f"{'with' if ff else 'without'} $\\Omega_d$ (RMSE$_e$ {1e3 * r['norm']:.1f} mm)")
        ax.set_title(f"{name}, T = {CASES[name][1]:g} s")
        ax.set_xlabel('t [s]'); ax.set_ylabel(r'$\|e_R\|$')
        ax.grid(True, alpha=0.4); ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/bonus_eR.pdf')
    fig.savefig('figures/bonus_eR.png', dpi=150)
    print("Saved figures/bonus_eR.pdf")
