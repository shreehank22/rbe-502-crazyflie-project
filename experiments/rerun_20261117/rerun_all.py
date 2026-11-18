"""Re-runs every experiment from the tutoring session with gains injected in memory.

The user's GeometricControl.py is NOT modified: a subclass calls the user's __init__ and
then overrides only the gain matrices. All control laws are the user's own code.
"""
import os, sys, io, json, contextlib
import numpy as np

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

import gym_pybullet_drones.examples.trajectory_tracking as tt
import gym_pybullet_drones.examples.hover_test as ht
from gym_pybullet_drones.control.GeometricControl import GeometricControl

M = 0.027
J = np.array([1.4e-5, 1.4e-5, 2.17e-5])
KF = 3.16e-10


def pos_gains(wn, zeta=0.9):
    return M * wn**2, 2 * zeta * M * wn


def att_gains(wn_rp, wn_yaw=15.0, zeta=0.9):
    wn = np.array([wn_rp, wn_rp, wn_yaw])
    return np.diag(J * wn**2), np.diag(2 * zeta * J * wn)


BASE_KR, BASE_KOM = np.diag([8.75e-3, 8.75e-3, 4.88e-3]), np.diag([6.3e-4, 6.3e-4, 5.86e-4])


def make_ctrl(kx, kv, KR=BASE_KR, KOm=BASE_KOM):
    class Patched(GeometricControl):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.K_x = np.diag([kx] * 3)
            self.K_v = np.diag([kv] * 3)
            self.K_R = KR.copy()
            self.K_Omega = KOm.copy()
    return Patched


def newest(folder, prefix):
    ds = [d for d in os.listdir(folder) if d.startswith(prefix)]
    return os.path.join(folder, max(ds, key=lambda d: os.path.getmtime(os.path.join(folder, d))))


def traj_metrics(run_dir, max_rotor):
    L = np.load(os.path.join(run_dir, 'log.npz'))
    t = L['t']; T = float(L['traj_duration']); m = t <= T + 1e-9
    e = L['pos_err'][m]; tm = t[m]
    r = np.sqrt(np.mean(e**2, 0)); n = np.sqrt(np.mean(np.sum(e**2, 1)))
    en = np.linalg.norm(e, axis=1)
    tilt = np.degrees(np.arccos(np.clip(L['rot'][:, 2, 2], -1, 1)))
    # true attitude-error angle: arccos((tr(R_d^T R) - 1) / 2). (degrees(|e_R|) = degrees(sin theta) saturates at 57.3)
    att = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', L['rot_d'], L['rot']) - 1) / 2, -1, 1)))
    F = KF * L['rpm']**2
    lat = 1 if 'diamond' in str(L['traj_name']) else 0     # main lateral axis
    el = e[:, lat]
    zc = int(np.sum(np.diff(np.sign(el)) != 0))
    # dominant frequency of the lateral error (detrended)
    x = el - np.polyval(np.polyfit(tm, el, 3), tm)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), d=tm[1] - tm[0])
    k = 1 + np.argmax(spec[1:])
    inner = (tm >= 0.5) & (tm <= T - 0.5)
    n_inner = np.sqrt(np.mean(en[inner]**2))
    return dict(rmse_x=r[0]*1e3, rmse_y=r[1]*1e3, rmse_z=r[2]*1e3, rmse_norm=n*1e3,
                rmse_norm_excl_ends=n_inner*1e3, max_err=en.max()*1e3, t_max_err=float(tm[en.argmax()]),
                max_tilt_deg=tilt.max(), max_att_err_deg=att.max(), t_max_att=float(t[att.argmax()]),
                lateral_sign_changes=zc, lateral_dom_freq_hz=float(freqs[k]),
                min_rotor_N=F.min(), max_rotor_N=F.max(),
                frac_rotor_at_zero=float(np.mean(F.min(1) < 1e-5)),
                frac_rotor_at_max=float(np.mean(F.max(1) > max_rotor - 1e-5)),
                max_thrust_N=float(L['thrust'].max()), z0=float(L['pos'][0, 2]),
                run_dir=os.path.basename(run_dir))


def run_traj(name, traj, kx, kv, KR=BASE_KR, KOm=BASE_KOM):
    tt.GeometricControl = make_ctrl(kx, kv, KR, KOm)
    folder = os.path.join(OUT, 'runs')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tt.run(traj=traj, gui=False, plot=False, output_folder=folder)
    out = buf.getvalue()
    with open(os.path.join(OUT, 'console.log'), 'a') as f:
        f.write(f"\n===== {name} | {traj} | Kx={kx:.4f} Kv={kv:.4f} | K_R={np.diag(KR)} K_Om={np.diag(KOm)} =====\n")
        f.write('\n'.join(l for l in out.splitlines() if l.startswith(('Trajectory', 'RMSE', 'Controller', 'Log'))) + '\n')
    rd = newest(folder, traj)
    ctrl = tt.GeometricControl()
    met = traj_metrics(rd, ctrl.MAX_ROTOR_THRUST)
    score = [l for l in out.splitlines() if l.startswith('Controller')]
    met.update(name=name, traj=traj, kx=kx, kv=kv, score=score[0].split(':')[-1].strip() if score else None)
    print(f"{name:28s} {traj:8s} norm={met['rmse_norm']:.2f}mm x/y/z={met['rmse_x']:.2f}/{met['rmse_y']:.2f}/{met['rmse_z']:.2f} "
          f"att={met['max_att_err_deg']:.1f} zc={met['lateral_sign_changes']} score={met['score']}", file=sys.stderr)
    return met


def run_hover(name, rpy, KR=BASE_KR, KOm=BASE_KOM):
    ht.GeometricControl = make_ctrl(0, 0, KR, KOm)
    folder = os.path.join(OUT, 'runs')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ht.run(init_rpy=rpy, gui=False, plot=False, output_folder=folder)
    out = buf.getvalue()
    with open(os.path.join(OUT, 'console.log'), 'a') as f:
        f.write(f"\n===== {name} | hover rpy={rpy} | K_R={np.diag(KR)} K_Om={np.diag(KOm)} =====\n")
        f.write('\n'.join(l for l in out.splitlines() if l.startswith(('Hover', 'Commands', 'PASS', 'FAIL', 'Note', 'Log'))) + '\n')
    rd = newest(folder, 'hover')
    L = np.load(os.path.join(rd, 'log.npz'))
    F = L['rotor_thrusts']
    res.update(name=name, rpy=list(rpy), min_rotor_N=float(F.min()), max_rotor_N=float(F.max()),
               t_min_rotor=float(L['t'][np.unravel_index(F.argmin(), F.shape)[0]]),
               frac_rotor_at_zero=float(np.mean(F.min(1) < 1e-5)),
               max_moment=np.abs(L['moment']).max(0).tolist(), run_dir=os.path.basename(rd))
    print(f"{name:28s} hover {rpy} settle={res['settling_time']} minF={res['min_rotor_N']:.4f}", file=sys.stderr)
    return res


results = {'hover': [], 'traj': []}

# 1. Hover tests (baseline attitude gains: roll/pitch wn=25, yaw wn=15, zeta=0.9)
results['hover'].append(run_hover('hover_default', [20, -15, 30]))
results['hover'].append(run_hover('hover_roll45', [45, 0, 0]))

# 2. Position-loop wn sweep (zeta = 0.9), extended past 12 to find the breaking point
for wn in [4, 6, 8, 10, 12, 14, 16, 20]:
    kx, kv = pos_gains(wn)
    for traj in ['circle', 'diamond']:
        m = run_traj(f'sweep_wn{wn}', traj, kx, kv); m['wn'] = wn; results['traj'].append(m)

# 3. Task 2 sensitivity at baseline wn=10
kx0, kv0 = pos_gains(10)
for name, kx, kv in [('task2_Kv_x0.5', kx0, 0.5 * kv0), ('task2_Kx_x0.5', 0.5 * kx0, kv0)]:
    for traj in ['circle', 'diamond']:
        m = run_traj(name, traj, kx, kv); m['wn'] = None; results['traj'].append(m)

# 4. Timescale-separation test: faster attitude loop (roll/pitch wn=35)
KR35, KOm35 = att_gains(35)
results['hover'].append(run_hover('hover_default_att35', [20, -15, 30], KR35, KOm35))
results['hover'].append(run_hover('hover_roll45_att35', [45, 0, 0], KR35, KOm35))
for wn in [10, 12, 14, 16]:
    kx, kv = pos_gains(wn)
    for traj in ['circle', 'diamond']:
        m = run_traj(f'att35_wn{wn}', traj, kx, kv, KR35, KOm35); m['wn'] = wn; results['traj'].append(m)


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


with open(os.path.join(OUT, 'results.json'), 'w') as f:
    json.dump(clean(results), f, indent=1)
print('done', file=sys.stderr)
