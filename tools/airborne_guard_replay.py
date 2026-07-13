#!/usr/bin/env python3
"""Replay the airborne/contact-loss heuristic over recorded leg-odometry runs.

This sends no commands and never contacts the robot. It recomputes each foot's implied
base velocity from recorded q/dq/IMU samples, applies deploy_runtime's candidate predicate,
and reports whether the configured debounce would have false-tripped any saved run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.deploy_runtime import (
    AIRBORNE_BOTH_SPEED_MPS,
    AIRBORNE_CONFIRM_TICKS,
    AIRBORNE_DIVERGENCE_MPS,
    _airborne_contact_signal,
    quat_wxyz_to_mat,
)
from pipeline.leg_odometry import LegOdometry


def _max_true_run(values):
    run = longest = 0
    for value in values:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return longest


def analyze_file(path: Path, divergence_mps: float, both_speed_mps: float):
    data = np.load(path, allow_pickle=False)
    required = ("q", "dq", "imu_quat", "gyro", "joint_order")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path}: missing telemetry keys {missing}")
    odom = LegOdometry([str(name) for name in data["joint_order"]])
    candidate, divergence, both_speed = [], [], []
    for q, dq, quat, gyro in zip(
            data["q"], data["dq"], data["imu_quat"], data["gyro"]):
        _, _, info = odom.estimate(q, dq, quat_wxyz_to_mat(quat), gyro)
        bad, _, metrics = _airborne_contact_signal(
            info, divergence_mps=divergence_mps, both_speed_mps=both_speed_mps)
        candidate.append(bool(bad))
        divergence.append(metrics["divergence_mps"])
        both_speed.append(metrics["min_speed_mps"])
    div = np.asarray(divergence, float)
    speed = np.asarray(both_speed, float)
    return {
        "file": path.as_posix(),
        "ticks": len(candidate),
        "candidate_ticks": int(sum(candidate)),
        "max_candidate_run_ticks": _max_true_run(candidate),
        "divergence_mps": {
            "p95": round(float(np.percentile(div, 95)), 4),
            "p99": round(float(np.percentile(div, 99)), 4),
            "max": round(float(div.max()), 4),
        },
        "both_speed_mps": {
            "p95": round(float(np.percentile(speed, 95)), 4),
            "p99": round(float(np.percentile(speed, 99)), 4),
            "max": round(float(speed.max()), 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path,
                        help="telemetry npz files (default: all ground-run-legodom captures)")
    parser.add_argument("--divergence", type=float, default=AIRBORNE_DIVERGENCE_MPS)
    parser.add_argument("--both-speed", type=float, default=AIRBORNE_BOTH_SPEED_MPS)
    parser.add_argument("--confirm-ticks", type=int, default=AIRBORNE_CONFIRM_TICKS)
    args = parser.parse_args()
    paths = args.paths or sorted(Path("data/telemetry").glob("*ground-run-legodom.npz"))
    if not paths:
        raise SystemExit("no ground-run-legodom telemetry files found")
    runs = [analyze_file(path, args.divergence, args.both_speed) for path in paths]
    result = {
        "schema": "airborne-contact-guard-replay/v1",
        "thresholds": {
            "divergence_mps": args.divergence,
            "both_speed_mps": args.both_speed,
            "confirm_ticks": args.confirm_ticks,
        },
        "corpus": {
            "files": len(runs),
            "ticks": sum(run["ticks"] for run in runs),
            "candidate_ticks": sum(run["candidate_ticks"] for run in runs),
            "max_candidate_run_ticks": max(run["max_candidate_run_ticks"] for run in runs),
            "would_trip_files": [run["file"] for run in runs
                                 if run["max_candidate_run_ticks"] >= args.confirm_ticks],
        },
        "runs": runs,
        "scope": ("Offline false-positive replay only. It does not prove suspended-robot "
                  "detection; that remains a supervised gantry test."),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
