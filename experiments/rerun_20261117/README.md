# Re-run of all tuning experiments (2026-11-17)

Every experiment from the tutoring session was re-run headless with one script, `rerun_all.py`.
The script subclasses `GeometricControl` and overrides **only the gain matrices**, so all the
control laws are the code in `gym_pybullet_drones/control/GeometricControl.py`, unmodified.

- `console.log` holds the simulator output (RMSE and score lines) for every run.
- `results.json` holds all computed metrics.
- `runs/<traj>_<time>/` holds `log.npz` and the standard figures for each run.

Reproduce with (from the repo root, `drones` env active):

```sh
python experiments/rerun_20261117/rerun_all.py experiments/rerun_20261117
```

All gain designs use ζ = 0.9, k_x = m ωₙ², k_v = 2ζ m ωₙ (m = 0.027 kg).
Unless stated otherwise, the attitude gains are the baseline: roll/pitch ωₙ = 25, yaw ωₙ = 15
(K_R = diag(8.75e-3, 8.75e-3, 4.88e-3), K_Ω = diag(6.3e-4, 6.3e-4, 5.86e-4)).

The re-run reproduced the original interactive runs exactly (to 0.01 mm). The simulation is deterministic.

---

## 1. Hover tests (attitude loop)

| Test | Attitude gains | Settling (< 1°) | Prediction | Min rotor thrust | Rotor at 0 N? |
|---|---|---|---|---|---|
| default (20, −15, 30)° | baseline | **0.295 s** | 0.30 s (yaw-limited) | 0.0036 N | no |
| roll 45° | baseline | **0.195 s** | 0.18 s (roll ωₙ = 25) | 0.0110 N | no |
| default (20, −15, 30)° | roll/pitch ωₙ = 35 | 0.295 s | unchanged (yaw-limited) | 0.0000 N | yes, 0.3% of samples |
| roll 45° | roll/pitch ωₙ = 35 | 0.145 s | faster | 0.0000 N | yes, 0.3% of samples |

All four PASS.

## 2. Position-loop ωₙ sweep

| ωₙ | k_x | k_v | Circle ‖e‖ (x / y / z) [mm] | Diamond ‖e‖ (x / y / z) [mm] | Diamond ‖e‖ excl. first/last 0.5 s | Circle sign changes | Diamond sign changes | Max tilt (circle / diamond) | Stable? | Score (c / d) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.432 | 0.194 | **12.81** (8.70 / 9.22 / 1.84) | **7.39** (0.50 / 5.56 / 4.85) | 6.07 | 6 | 12 | 31.0° / 8.3° | yes | 100 / 100 |
| 6 | 0.972 | 0.292 | **6.62** (4.53 / 4.76 / 0.83) | **4.73** (0.23 / 3.12 / 3.55) | 3.28 | 6 | 11 | 30.3° / 8.9° | yes | 100 / 100 |
| 8 | 1.728 | 0.389 | **3.94** (2.70 / 2.84 / 0.46) | **4.20** (0.14 / 2.67 / 3.24) | 2.09 | 6 | 13 | 29.9° / 10.5° | yes | 100 / 100 |
| **10** | **2.700** | **0.486** | **2.59** (1.78 / 1.87 / 0.30) | **4.17** (0.10 / 2.81 / 3.07) | 1.76 | 6 | 21 | 29.8° / 14.4° | yes | 100 / 100 |
| 12 | 3.888 | 0.583 | **1.85** (1.26 / 1.34 / 0.21) | **4.99** (0.08 / 3.94 / 3.07) | 3.42 | **50** | **41** | 29.9° / 18.1° | marginal | 100 / 100 |
| 14 | 5.292 | 0.680 | 30.10 | 32.95 | 31.07 | 72 | 34 | **120° / 180°** | **no (flips)** | 100* / 100* |
| 16 | 6.912 | 0.778 | 1929 | 851 | — | — | — | 180° / 180° | **no (crash)** | 0 / 0 |
| 20 | 10.80 | 0.972 | 1028 | 1236 | — | — | — | 180° / 180° | **no (crash)** | 0 / 0 |

\* At ωₙ = 14 the RMSE stays under 50 mm, so the script prints 100. But the quadrotor tilts to 120° on the circle and **turns upside down (180°)** on the diamond. Its yaw also spins to about ±179°, and the true attitude-error angle reaches 178.6° (circle) and 167.8° (diamond). Under the grading rule "or the quadrotor becomes unstable", that counts as a fail.

Peak true attitude-error angle for the stable runs (circle / diamond): ωₙ = 4: 8.0° / 4.1°, 6: 7.6° / 3.9°, 8: 7.5° / 7.6°, 10: 7.4° / 12.5°, 12: 7.6° / 17.7°.

"Sign changes" is the number of zero crossings of the main lateral error (e_x for the circle, e_y for the diamond) during the trajectory. It's a rough measure of oscillation.

## 3. Task 2: gain sensitivity (baseline ωₙ = 10)

| Case | k_x | k_v | ωₙ | ζ | Circle ‖e‖ (x / y / z) | Diamond ‖e‖ (x / y / z) | Circle sign changes | Diamond sign changes | Diamond error dominant freq | Diamond peak attitude-error angle |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 2.700 | 0.486 | 10.0 | 0.90 | 2.59 (1.78 / 1.87 / 0.30) | 4.17 (0.10 / 2.81 / 3.07) | 6 | 21 | — | 12.5° |
| K_v × 0.5 | 2.700 | 0.243 | 10.0 | 0.45 | 3.05 (2.05 / 2.23 / 0.33), **+18%** | 14.12 (0.31 / 13.67 / 3.50), **×3.4** | **41** | 27 | **1.75 Hz = 11.0 rad/s** | 15.2° |
| K_x × 0.5 | 1.350 | 0.486 | 7.07 | 1.27 | 4.11 (2.79 / 2.96 / 0.55), **+59%** | 4.19 (0.17 / 2.35 / 3.47), **same** | 6 | 11 | — | **6.1°** |

## 4. Timescale-separation test: faster attitude loop (roll/pitch ωₙ = 35, yaw unchanged)

| Position ωₙ | Ratio (att / pos) | Circle ‖e‖ | Diamond ‖e‖ | Diamond excl. ends | Diamond peak attitude-error angle | Same position ωₙ with baseline attitude (circle / diamond) |
|---|---|---|---|---|---|---|
| 10 | 3.5 | **1.65** | **3.90** | 0.97 | 10.2° | 2.59 / 4.17 |
| 12 | 2.9 | **1.17** | **3.99** | 0.80 | 14.3° | 1.85 / 4.99 |
| 14 | 2.5 | 0.87 | 4.17 | 0.95 | 19.5° | **flips** |
| 16 | 2.2 | 0.67 | 5.34 | 1.77 | **161.8°** at t = 7.94 s (R_d inverted, see below), rotors hit 0 N | crash |

---

## 5. Verification of the inferences made during the session

| # | Inference | Verdict | Evidence |
|---|---|---|---|
| 1 | Hover settling ≈ 4/(ζωₙ): about 0.30 s (yaw-limited) for the default test, about 0.18 s for roll-only | ✅ Confirmed | 0.295 s and 0.195 s |
| 2 | At 45° roll, rotors 1 and 2 drop to about 0.011 N and do **not** clip | ✅ Confirmed | Min rotor thrust 0.0110 N |
| 3 | Default hover: F₄ starts near 0.003 N, with little margin | ✅ Confirmed | Min rotor thrust 0.0036 N |
| 4 | The circle requires about 29° of tilt (v_peak ≈ 2.36 m/s, a = v²/r ≈ 5.5 m/s²) | ✅ Confirmed | Measured 29.7–31.0° in every stable run |
| 5 | Tilt comes from the feedforward m·a_d, so it barely depends on the gains | ✅ Confirmed | Circle max tilt 31.0° → 29.8° over ωₙ = 4 → 12 |
| 6 | RMSE ∝ 1/ωₙ² | ⚠️ Approximately | Log-log slope for the circle is **−1.77** (not −2). For the diamond it's −0.4 overall, because of the ground offset (#7). Excluding the ends, the diamond follows the trend up to ωₙ = 10 |
| 7 | The diamond has an error floor because it starts and ends on the ground: the CoM rests at z = 12.5 mm, while the target is z = 0 | ✅ Confirmed | z₀ = 12.5 mm. At ωₙ = 10, RMSE is 4.17 mm overall vs **1.76 mm** without the first and last 0.5 s. Max error occurs at t = 8.00 s |
| 8 | x/y error is larger than z on the circle because lateral motion needs tilting (attitude lag) | ✅ Consistent | Circle z RMSE is about 6× smaller than x and y at every ωₙ |
| 9 | Diamond degradation at ωₙ ≥ 10 comes from loss of timescale separation, **not** rotor saturation | ✅ Confirmed | No rotor hits 0 N or max at ωₙ = 10 or 12. With a faster attitude loop (§4), the diamond at ωₙ = 12 improves from 4.99 to 3.99 mm and the trend reverses |
| 10 | "Circle: stiffer is better throughout" | ❌ **Wrong past 12** | At ωₙ = 12 the circle's sign changes jump 6 → 50 (onset of oscillation). At ωₙ = 14 it goes unstable |
| 11 | ωₙ = 10 is a better choice than 12 because of margin | ✅ **Strongly confirmed** | The stability edge is between 12 and 14. The 12 → 14 step goes from 1.85 mm to flipping. ωₙ = 12 was closer to the edge than the session suggested |
| 12 | K_v × 0.5 → sustained oscillation at about ωₙ, not lag | ✅ Confirmed | Diamond dominant error frequency is 1.75 Hz = 11.0 rad/s (ωₙ = 10). Circle sign changes go 6 → 41 |
| 13 | K_v × 0.5 → damage "concentrated in transients" (original prediction) | ❌ Wrong (corrected during the session) | The diamond's y error stays at 10–25 mm for the whole run |
| 14 | K_x × 0.5 → overdamped lag with no oscillation | ✅ Confirmed | Circle sign changes stay at 6 (same as baseline), and RMSE goes up 59% |
| 15 | K_x × 0.5 → circle ≈ 5 mm | ⚠️ Close | Measured 4.11 mm. ζ = 1.27 smooths more than the 1/ωₙ² estimate predicts |
| 16 | K_x × 0.5 → diamond unchanged and attitude error lower (more separation) | ✅ Confirmed | 4.19 vs 4.17 mm. Peak attitude-error angle 12.5° → 6.1°, sign changes 21 → 11 |
| 17 | With a faster attitude loop, the hover test rotors hit 0 N but still PASS | ✅ Confirmed | 0.3% of samples at 0 N, both tests PASS |

### New findings from the extended runs

- **The instability edge is about 2× separation.** With baseline attitude (ωₙ = 25), position ωₙ = 12 (ratio 2.1) is marginal and 14 (1.8) flips. With attitude ωₙ = 35, position ωₙ = 16 (2.2) is unsafe (next bullet). A ratio of **2.5× or more** is safe in both cases.
- **Inverted R_d near the ground.** In every unstable run, the desired thrust axis b₃,d points **downward** (R_d[2,2] < 0) for a while. For att35 / ωₙ = 16 on the diamond, this happens for 14 samples (70 ms) from t = 7.875 s. The drone is about 2 cm above its final ground target, and the stiff position loop asks for a net downward force, so F_d,z < 0 and eq. (9) sets b₃,d = F_d/‖F_d‖ upside down. The real drone never flips there (max tilt 27°). The 161.8° attitude error is between R and the inverted R_d, not a flip of R. The rotors can't push down, so the loop is asking for something physically impossible. Eq. (9) assumes F_d stays well above horizontal, which the PDF notes holds "as long as F_d is not horizontal".
- **At ωₙ = 12 a faster mode appears.** The diamond's dominant error frequency is 2.75 Hz (≈ 17 rad/s), not the position ωₙ. That's the outer and inner loops interacting, which is a sign of approaching instability.
- **Possible improvement:** attitude ωₙ = 35 with position ωₙ = 10 gives circle 1.65 mm and diamond 3.90 mm, both better than the current baseline (2.59 / 4.17). The cost is that the rotors briefly hit 0 N in the hover test.

### Attitude-error metric

All attitude-error angles in this document are the **true rotation angle** between R and R_d, θ = arccos((tr(R_dᵀR) − 1)/2), computed from the logged `rot` and `rot_d`.

The first version of this document reported degrees(‖e_R‖) instead. Since ‖e_R‖ = sin θ, that matches θ to within 0.4° for every stable run (all below 20°), but it saturates at 57.3° when θ ≥ 90°. It hid the 162–180° errors in the unstable runs and has been replaced.
