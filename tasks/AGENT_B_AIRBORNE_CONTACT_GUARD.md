# AGENT B — Airborne / Contact-Loss Guard

**Status (2026-07-13): SOFTWARE COMPLETE; HARDWARE VALIDATION UNFINISHED.**

## Purpose

Prevent the low-level dance policy from continuing to thrash when the G1 is suspended,
barely grounded, or loses credible foot contact. This lane applies to
`ground-run-legodom`, the current show runtime.

The `unitree_hg` LowState used by this G1 does **not** expose foot-force samples. The guard
therefore uses conservative leg-kinematic evidence already produced by `LegOdometry`:
each foot independently implies a base velocity under the assumption that it is planted.
Sustained gross disagreement, while both feet imply substantial motion, is treated as a
contact-confidence collapse.

## Completed in `feat/airborne-contact-guard`

1. **Pre-release contact advisory/enforcement** in `pipeline/deploy_runtime.py`:
   - Samples both per-foot kinematic estimates while onboard balance still owns the robot.
   - Runs before `_release_motion_service`, so `AIRBORNE_START_GUARD=enforce` refuses without
     releasing the safe onboard controller.
   - Default is `AIRBORNE_START_GUARD=advisory`; `off` and supervised `enforce` modes exist.
   - Resets the odometry filter after sampling so precheck state cannot leak into the policy.
2. **Debounced in-loop trip**:
   - A normal one-foot step cannot trip on disagreement alone; the slower/stance foot must
     also exceed the configured speed threshold.
   - Defaults: disagreement `>=2.25 m/s`, both-foot speed `>=0.60 m/s`, for 12 consecutive
     ticks (240 ms at 50 Hz).
   - Enforcement is explicitly opt-in with `AIRBORNE_TRIP=1` until robot validation passes.
   - A confirmed fault raises into the existing proven exception path: damping, then soft
     handoff to onboard control. It does not introduce a new shutdown path.
3. **Telemetry and calibration**:
   - Every leg-odometry run records the start assessment, guard configuration, candidate
     count, longest candidate run, and observed maxima—even while enforcement is disabled.
   - `tools/airborne_guard_replay.py` reproduces the false-positive analysis without robot I/O.
   - Replay evidence: `data/telemetry/airborne_guard_20260713/replay_summary.json`.
   - Corpus result: 10 real ground runs / 15,588 ticks; 94 isolated candidate ticks;
     longest run 7 ticks; **0/10 files would trip** the 12-tick guard.
4. **Tests**:
   - `tests/test_airborne_contact_guard.py` covers step suppression, gross disagreement,
     non-finite fail-closed behavior, debounce/reset, enforcement-off behavior, passive
     precheck advisory/refusal, and precheck-before-release integration ordering.

## Unfinished — requires the G1 and a human

These items are intentionally **not** marked complete and must not be inferred from the
offline replay:

1. **Suspended gantry validation:** with the damping remote in hand and feet clearly off the
   floor, run first in advisory mode and capture telemetry. Then, only under the supervised
   robot procedure, try `AIRBORNE_START_GUARD=enforce AIRBORNE_TRIP=1` and verify an obvious
   moving/contact-loss condition refuses or damps before sustained thrashing.
2. **Grounded false-positive validation:** run a complete normal dance with both feet loaded,
   first advisory, then opt-in enforcement. It must complete without a guard trip. Repeat
   across normal steps and the fastest choreography section.
3. **Threshold sign-off:** compare suspended telemetry against the committed grounded corpus.
   Adjust environment thresholds only if the two populations separate with margin; commit the
   new raw evidence. Keep `AIRBORNE_TRIP` default-off until this passes.
4. **Default promotion:** only after both gantry and grounded tests pass may enforcement become
   the default. Record the result in `PROJECT_STATE.md`.

## Structural limitation — still unfinished by available sensors

A **perfectly still suspended robot can look identical to a perfectly still grounded robot**
in q/dq/IMU kinematics. The passive precheck can catch swinging, moving, or internally
inconsistent suspension, but it cannot prove that the feet carry weight. Fully covering the
motionless case requires one of:

- a vendor-supported contact/ground-reaction signal;
- validated motor-torque/load inference with a hardware-measured separation margin; or
- an external load/contact sensor.

Until then, the operator must visually confirm both feet are flat and fully loaded before
arming, with the physical damping remote in hand. This guard supplements that procedure; it
does not replace it.

## Reproduce offline

```bash
PYTHONPATH=. python tools/airborne_guard_replay.py
pytest -q tests/test_airborne_contact_guard.py
```

## Hardware validation settings (supervised only)

```bash
AIRBORNE_START_GUARD=enforce \
AIRBORNE_TRIP=1 \
python -m pipeline.deploy_runtime --mode ground-run-legodom ...
```

All normal robot-motion gates still apply: explicit human confirmation, gantry/clear space,
damping remote in hand, and a simulation-verified policy.
