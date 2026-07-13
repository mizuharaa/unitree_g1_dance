"""Airborne/contact-loss guard tests — pure numpy, no SDK, robot, or policy artifacts."""
import os

import numpy as np
import pytest

import pipeline.deploy_runtime as dr


def _info(left, right):
    return {"per_foot_v": [np.asarray(left, float), np.asarray(right, float)]}


def test_contact_signal_ignores_an_ordinary_one_foot_step(monkeypatch):
    """Large disagreement alone is expected in swing; the planted/slower foot suppresses it."""
    monkeypatch.setattr(dr, "AIRBORNE_DIVERGENCE_MPS", 2.0)
    monkeypatch.setattr(dr, "AIRBORNE_BOTH_SPEED_MPS", 0.6)
    candidate, _, metrics = dr._airborne_contact_signal(
        _info([0.15, 0.0, 0.0], [3.0, 0.0, 0.0]))
    assert metrics["divergence_mps"] > 2.0
    assert metrics["min_speed_mps"] < 0.6
    assert candidate is False


def test_contact_signal_flags_gross_two_foot_disagreement(monkeypatch):
    monkeypatch.setattr(dr, "AIRBORNE_DIVERGENCE_MPS", 2.0)
    monkeypatch.setattr(dr, "AIRBORNE_BOTH_SPEED_MPS", 0.6)
    candidate, reason, metrics = dr._airborne_contact_signal(
        _info([1.4, 0.0, 0.0], [-1.3, 0.0, 0.0]))
    assert candidate is True
    assert metrics["divergence_mps"] == pytest.approx(2.7)
    assert "disagreement" in reason


def test_contact_signal_fails_closed_on_missing_or_nonfinite_evidence():
    for bad in (None, {}, _info([np.nan, 0, 0], [0, 0, 0])):
        candidate, reason, metrics = dr._airborne_contact_signal(bad)
        assert candidate is True
        assert metrics["valid"] is False
        assert "unavailable" in reason


def test_inloop_trip_is_debounced_and_a_good_tick_resets(monkeypatch):
    monkeypatch.setattr(dr, "AIRBORNE_DIVERGENCE_MPS", 2.0)
    monkeypatch.setattr(dr, "AIRBORNE_BOTH_SPEED_MPS", 0.6)
    monkeypatch.setattr(dr, "AIRBORNE_CONFIRM_TICKS", 3)
    bad = _info([1.4, 0, 0], [-1.3, 0, 0])
    good = _info([0.1, 0, 0], [2.5, 0, 0])  # one-foot step, not contact loss

    ticks, _ = dr._check_airborne_contact(0, bad, 10, enforce=True)
    assert ticks == 1
    ticks, _ = dr._check_airborne_contact(ticks, bad, 11, enforce=True)
    assert ticks == 2
    ticks, _ = dr._check_airborne_contact(ticks, good, 12, enforce=True)
    assert ticks == 0

    ticks, _ = dr._check_airborne_contact(0, bad, 20, enforce=True)
    ticks, _ = dr._check_airborne_contact(ticks, bad, 21, enforce=True)
    with pytest.raises(RuntimeError, match="AIRBORNE/CONTACT LOSS"):
        dr._check_airborne_contact(ticks, bad, 22, enforce=True)


def test_inloop_trip_disabled_still_measures_but_never_raises(monkeypatch):
    monkeypatch.setattr(dr, "AIRBORNE_CONFIRM_TICKS", 2)
    bad = _info([2.0, 0, 0], [-2.0, 0, 0])
    ticks = 0
    for tick in range(10):
        ticks, metrics = dr._check_airborne_contact(ticks, bad, tick, enforce=False)
    assert ticks == 10
    assert metrics["valid"] is True


def test_prestart_assessment_requires_a_majority_of_suspect_samples(monkeypatch):
    monkeypatch.setattr(dr, "AIRBORNE_START_DIVERGENCE_MPS", 0.75)
    monkeypatch.setattr(dr, "AIRBORNE_START_BOTH_SPEED_MPS", 0.20)
    clear = _info([0.02, 0, 0], [0.03, 0, 0])
    suspect = _info([0.8, 0, 0], [-0.8, 0, 0])
    assert dr._assess_feet_planted([clear, clear, suspect])["clear"] is True
    result = dr._assess_feet_planted([clear, suspect, suspect])
    assert result["clear"] is False
    assert result["candidate_samples"] == 2


class _FakeLegOdom:
    def __init__(self, infos):
        self.infos = iter(infos)
        self.reset_calls = 0

    def estimate(self, *args, **kwargs):
        return np.zeros(3), 0.8, next(self.infos)

    def reset_filter(self):
        self.reset_calls += 1


def _state():
    return (np.zeros(29), np.zeros(29), np.array([1.0, 0, 0, 0]),
            np.zeros(3), object())


def test_prestart_enforce_refuses_before_control_release(monkeypatch):
    suspect = _info([0.8, 0, 0], [-0.8, 0, 0])
    odo = _FakeLegOdom([suspect, suspect, suspect])
    monkeypatch.setattr(dr, "AIRBORNE_START_GUARD", "enforce")
    monkeypatch.setattr(dr, "AIRBORNE_START_SAMPLES", 3)
    monkeypatch.setattr(dr, "read_state", lambda *a, **k: _state())
    monkeypatch.setattr(dr.time, "sleep", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="nothing was released"):
        dr._check_feet_planted(odo, object(), _state())
    assert odo.reset_calls == 1


def test_prestart_advisory_records_same_failure_without_refusing(monkeypatch, capsys):
    suspect = _info([0.8, 0, 0], [-0.8, 0, 0])
    odo = _FakeLegOdom([suspect, suspect, suspect])
    monkeypatch.setattr(dr, "AIRBORNE_START_GUARD", "advisory")
    monkeypatch.setattr(dr, "AIRBORNE_START_SAMPLES", 3)
    monkeypatch.setattr(dr, "read_state", lambda *a, **k: _state())
    monkeypatch.setattr(dr.time, "sleep", lambda *a, **k: None)
    result = dr._check_feet_planted(odo, object(), _state())
    assert result["clear"] is False
    assert "AIRBORNE ADVISORY" in capsys.readouterr().out


def test_defaults_are_advisory_and_trip_opt_in():
    if "AIRBORNE_START_GUARD" not in os.environ:
        assert dr.AIRBORNE_START_GUARD == "advisory"
    if "AIRBORNE_TRIP" not in os.environ:
        assert dr.AIRBORNE_TRIP is False
    assert dr.AIRBORNE_CONFIRM_TICKS >= 2


class _ModeExit(BaseException):
    pass


def test_legodom_mode_runs_contact_precheck_before_releasing_onboard(monkeypatch):
    """Integration seam: even enforcement failure occurs while onboard still owns control."""
    import pipeline.leg_odometry as lo

    class Meta:
        n = 29
        joint_order = [f"joint_{i}" for i in range(29)]
        default = np.zeros(29)
        kp = np.ones(29)
        kd = np.ones(29)
        action_scale = np.ones(29)

    class Ref:
        T = 1
        apos = np.zeros((1, 3))

        @staticmethod
        def at(_tick):
            return None, None, np.zeros(3)

    class Msg:
        mode_machine = 5

    class LegOdom:
        def __init__(self, *_a, **_k):
            pass

        def reset_filter(self):
            pass

        def estimate(self, *_a, **_k):
            return np.zeros(3), 0.8, _info([0, 0, 0], [0, 0, 0])

    class Telem:
        def __init__(self, _mode, _meta, extra=None):
            self.extra = extra or {}

        def add(self, *_a, **_k):
            pass

    class Clock:
        def tick(self, *_a, **_k):
            pass

        def report(self):
            return {}

    events = []
    state = (np.zeros(29), np.zeros(29), np.array([1.0, 0, 0, 0]),
             np.zeros(3), Msg())
    monkeypatch.setenv("CONFIRMED_BY_HUMAN", "alois")
    monkeypatch.setattr(lo, "LegOdometry", LegOdom)
    monkeypatch.setattr(dr, "AIRBORNE_START_SAMPLES", 1)
    monkeypatch.setattr(dr, "AIRBORNE_START_GUARD", "advisory")
    monkeypatch.setattr(dr, "ARM_GROUND_KP_SCALE", 1.0)
    monkeypatch.setattr(dr, "make_dds", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "lowstate_subscriber", lambda: object())
    monkeypatch.setattr(dr, "read_state", lambda *_a, **_k: state)
    monkeypatch.setattr(dr, "_check_start_upright", lambda *_a: None)
    monkeypatch.setattr(dr, "_check_start_near_default", lambda *_a: None)
    original_check = dr._check_feet_planted

    def check(*args, **kwargs):
        events.append("contact_check")
        return original_check(*args, **kwargs)

    monkeypatch.setattr(dr, "_check_feet_planted", check)
    monkeypatch.setattr(dr, "_lowcmd_setup", lambda: (None, None, None))
    monkeypatch.setattr(dr, "_install_damp_on_signals", lambda: None)
    monkeypatch.setattr(dr, "_release_motion_service", lambda: events.append("release"))
    monkeypatch.setattr(dr, "_ramp_to_pose", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "_hold", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "_align_reference", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "_send_cmd", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "_damp", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "run_policy", lambda *_a, **_k: np.zeros(29))
    monkeypatch.setattr(dr, "action_to_target", lambda *_a, **_k: np.zeros(29))
    monkeypatch.setattr(dr, "Telemetry", Telem)
    monkeypatch.setattr(dr, "TickClock", Clock)
    monkeypatch.setattr(dr.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(dr, "_finalize_and_exit",
                        lambda *_a, **_k: (_ for _ in ()).throw(_ModeExit()))

    with pytest.raises(_ModeExit):
        dr.mode_ground_run_legodom(Meta(), object(), Ref(), "iface", True, 0.02, "damp")
    assert events.index("contact_check") < events.index("release")
