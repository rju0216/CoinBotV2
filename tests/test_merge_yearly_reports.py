"""scripts/merge_yearly_reports.py 갱신 검증 (단계 15).

신규 BacktestEngine.write_reports 가 생성한 trades.csv 의
`strategy_name` 컬럼이 정상 집계되는지, 그리고 옛 `owner` 컬럼이 있는
구식 리포트도 fallback으로 처리되는지 확인.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parent.parent


def _load_merge_module():
    """scripts/merge_yearly_reports.py 를 동적으로 import."""
    spec = importlib.util.spec_from_file_location(
        "merge_yearly_reports",
        str(ROOT / "scripts" / "merge_yearly_reports.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mr():
    return _load_merge_module()


def _make_trades_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    return df


def _make_equity_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "balance": [10000, 10100, 10050, 10200, 10300, 10250, 10400, 10350, 10500, 10450],
            "equity": [10000, 10100, 10050, 10200, 10300, 10250, 10400, 10350, 10500, 10450],
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def test_strategy_name_split(mr):
    """신규 strategy_name 컬럼으로 by_strategy_name 집계."""
    trades = _make_trades_df([
        {
            "strategy_name": "macross", "side": "long", "size": 0.1,
            "entry_time": "2024-01-01", "exit_time": "2024-01-02",
            "pnl": 100.0, "exit_reason": "tp_hit",
        },
        {
            "strategy_name": "macross", "side": "short", "size": 0.1,
            "entry_time": "2024-01-03", "exit_time": "2024-01-04",
            "pnl": -50.0, "exit_reason": "sl_hit",
        },
        {
            "strategy_name": "breakout", "side": "long", "size": 0.05,
            "entry_time": "2024-01-05", "exit_time": "2024-01-06",
            "pnl": 75.0, "exit_reason": "tp_hit",
        },
    ])
    equity = _make_equity_df()
    result = mr.compute_merged_metrics(trades, equity, 10000.0, 10125.0)

    assert "by_strategy_name" in result
    assert "by_owner" not in result, "옛 키 by_owner 는 더 이상 사용하지 않아야 함"

    by_strategy = result["by_strategy_name"]
    assert set(by_strategy.keys()) == {"macross", "breakout"}
    assert by_strategy["macross"]["trades"] == 2
    assert by_strategy["macross"]["pnl"] == 50.0  # 100 - 50
    assert by_strategy["macross"]["win_rate_pct"] == 50.0
    assert by_strategy["breakout"]["trades"] == 1
    assert by_strategy["breakout"]["pnl"] == 75.0
    assert by_strategy["breakout"]["win_rate_pct"] == 100.0


def test_owner_fallback(mr):
    """옛 trades.csv (owner 컬럼만 있고 strategy_name 없음) 도 처리."""
    trades = _make_trades_df([
        {
            "owner": "trend", "side": "long", "size": 0.1,
            "entry_time": "2024-01-01", "exit_time": "2024-01-02",
            "pnl": 100.0, "exit_reason": "tp_hit",
        },
        {
            "owner": "scalping", "side": "short", "size": 0.1,
            "entry_time": "2024-01-03", "exit_time": "2024-01-04",
            "pnl": -30.0, "exit_reason": "sl_hit",
        },
    ])
    equity = _make_equity_df()
    result = mr.compute_merged_metrics(trades, equity, 10000.0, 10070.0)

    # strategy_name 없으면 owner 로 fallback
    assert "by_strategy_name" in result
    by_strategy = result["by_strategy_name"]
    assert set(by_strategy.keys()) == {"trend", "scalping"}
    assert by_strategy["trend"]["pnl"] == 100.0
    assert by_strategy["scalping"]["pnl"] == -30.0


def test_no_strategy_column_skips_split(mr):
    """strategy_name 도 owner 도 없는 trades 는 split을 비움."""
    trades = _make_trades_df([
        {
            "side": "long", "size": 0.1,
            "entry_time": "2024-01-01", "exit_time": "2024-01-02",
            "pnl": 50.0, "exit_reason": "tp_hit",
        },
    ])
    equity = _make_equity_df()
    result = mr.compute_merged_metrics(trades, equity, 10000.0, 10050.0)
    assert result["by_strategy_name"] == {}


def test_exit_reason_and_direction_splits_unchanged(mr):
    """exit_reason / side 분할 로직은 그대로 작동."""
    trades = _make_trades_df([
        {
            "strategy_name": "x", "side": "long", "size": 0.1,
            "entry_time": "2024-01-01", "exit_time": "2024-01-02",
            "pnl": 100.0, "exit_reason": "tp_hit",
        },
        {
            "strategy_name": "x", "side": "short", "size": 0.1,
            "entry_time": "2024-01-03", "exit_time": "2024-01-04",
            "pnl": -50.0, "exit_reason": "sl_hit",
        },
    ])
    equity = _make_equity_df()
    result = mr.compute_merged_metrics(trades, equity, 10000.0, 10050.0)

    assert "by_exit_reason" in result
    assert set(result["by_exit_reason"].keys()) == {"tp_hit", "sl_hit"}
    assert "by_direction" in result
    assert set(result["by_direction"].keys()) == {"long", "short"}


def test_integrated_metrics_unchanged(mr):
    """integrated 섹션의 핵심 필드들은 신규 포맷과도 호환."""
    trades = _make_trades_df([
        {
            "strategy_name": "x", "side": "long", "size": 0.1,
            "entry_time": "2024-01-01", "exit_time": "2024-01-02",
            "pnl": 100.0, "exit_reason": "tp_hit",
        },
    ])
    equity = _make_equity_df()
    result = mr.compute_merged_metrics(trades, equity, 10000.0, 10100.0)

    integrated = result["integrated"]
    assert integrated["initial_balance"] == 10000.0
    assert integrated["final_balance"] == 10100.0
    assert integrated["total_trades"] == 1
    assert integrated["winning_trades"] == 1
    assert integrated["win_rate_pct"] == 100.0
    assert integrated["total_return_pct"] == 1.0
