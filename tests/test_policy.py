from __future__ import annotations

from vision.policy import AnalysisPolicy


def test_window_change_triggers_analysis():
    p = AnalysisPolicy(max_calls_per_hour=100, min_seconds_between_analysis=0.0)
    d = p.decide(window_changed=True, pixel_changed=False, is_volatile=False)
    assert d.should_analyze and d.reason == "window_changed"


def test_pixel_change_triggers_when_not_volatile():
    p = AnalysisPolicy(max_calls_per_hour=100, min_seconds_between_analysis=0.0)
    d = p.decide(window_changed=False, pixel_changed=True, is_volatile=False)
    assert d.should_analyze and d.reason == "pixel_change"


def test_volatile_pixel_change_is_suppressed():
    p = AnalysisPolicy(max_calls_per_hour=100, min_seconds_between_analysis=0.0)
    d = p.decide(window_changed=False, pixel_changed=True, is_volatile=True)
    assert not d.should_analyze and d.reason == "volatile_backoff"


def test_no_change_no_analysis():
    p = AnalysisPolicy(max_calls_per_hour=100, min_seconds_between_analysis=0.0)
    d = p.decide(window_changed=False, pixel_changed=False, is_volatile=False)
    assert not d.should_analyze and d.reason == "no_change"


def test_hourly_budget_enforced():
    p = AnalysisPolicy(max_calls_per_hour=2, min_seconds_between_analysis=0.0)
    now = 1000.0
    p.record_call(now)
    p.record_call(now + 1)
    d = p.decide(window_changed=True, pixel_changed=False, is_volatile=False, now=now + 2)
    assert not d.should_analyze and d.reason == "hourly_budget_exhausted"


def test_cooldown_enforced():
    p = AnalysisPolicy(max_calls_per_hour=100, min_seconds_between_analysis=5.0)
    now = 1000.0
    p.record_call(now)
    d = p.decide(window_changed=True, pixel_changed=False, is_volatile=False, now=now + 1)
    assert not d.should_analyze and d.reason == "cooldown_active"
    d2 = p.decide(window_changed=True, pixel_changed=False, is_volatile=False, now=now + 6)
    assert d2.should_analyze
