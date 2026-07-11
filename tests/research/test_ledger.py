"""실행 원장 회귀 테스트 — 계수·예산·persist."""

from __future__ import annotations

from src.research.validation.ledger import RunLedger


def test_record_and_count(tmp_path):
    led = RunLedger(str(tmp_path / "ledger.jsonl"))
    assert led.comparison_count() == 0
    led.record({"model": "prior"}, {"ba_mean": 0.5}, campaign="R1")
    led.record({"model": "uniform"}, {"ba_mean": 0.33}, campaign="R1")
    assert led.comparison_count("R1") == 2
    assert led.comparison_count() == 2


def test_budget_and_remaining(tmp_path):
    led = RunLedger(str(tmp_path / "l.jsonl"))
    led.register_budget("R1", 3)
    assert led.remaining("R1") == 3
    led.record({}, {}, campaign="R1")
    assert led.remaining("R1") == 2
    assert not led.over_budget("R1")
    led.record({}, {}, campaign="R1")
    led.record({}, {}, campaign="R1")
    led.record({}, {}, campaign="R1")
    assert led.over_budget("R1")


def test_non_comparison_not_counted(tmp_path):
    led = RunLedger(str(tmp_path / "l.jsonl"))
    led.record({}, {}, campaign="R1", is_comparison=False)
    assert led.comparison_count("R1") == 0


def test_persistence_reload(tmp_path):
    p = str(tmp_path / "l.jsonl")
    RunLedger(p).record({"m": "a"}, {}, campaign="R1")
    assert RunLedger(p).comparison_count("R1") == 1  # 새 인스턴스가 파일에서 복원
