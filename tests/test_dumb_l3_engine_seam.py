"""멍청3층 ↔ 엔진 seam 통합테스트 (Phase 4 Step 4.2, F-11 / CLAUDE 16).

관문2 채점(Step 4.3)을 믿으려면, 배선이 **검증된 예측을 미래참조 없이·정확한 계산으로**
거래하는지 먼저 못박아야 한다. 합성 캔들 + 합성 예측 아티팩트로 엔진을 관통시켜 seam
불변식을 회귀로 박제한다(일회성 시연 아님):

- **진입 타이밍(lookahead 0)**: 예측은 봉 t(직전 완성봉)를 보고, 엔진은 **봉 t+1 open** 진입.
  entry_price = open[t+1] (예측봉 close 아님) — 미래참조 차단(INFRA §4-3).
- **배리어 실현(D-b①)**: TP/SL = 실진입가 기준 entry·(1±barrier_frac), 방향별 대칭.
- **만기(expire)**: 진입 후 N봉 경과 → force_exit.
- **정합성(CLAUDE 10)**: Σtrades.pnl == 최종잔고 변화.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.core.enums import SignalSide
from src.strategy.plugins.dumb_l3 import DumbL3
from src.strategy.registry import register_strategy, reset_registry_for_testing


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    register_strategy(DumbL3)          # discovery 우회, 실제 플러그인 등록
    yield
    reset_registry_for_testing()


def _candles(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(open,high,low,close), ...] 15m 간격 UTC."""
    idx = pd.date_range("2021-01-01 00:00", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "timestamp"
    return df


def _artifact(tmp_path, idx, long_at=None, short_at=None, bf=0.01):
    """모든 봉 HOLD(0.30/0.30/0.40) + long_at/short_at 위치만 강신호. barrier_frac=bf."""
    df = pd.DataFrame({
        "ens_up": [0.30] * len(idx), "ens_down": [0.30] * len(idx),
        "ens_expire": [0.40] * len(idx), "barrier_frac": [bf] * len(idx),
    }, index=idx)
    if long_at is not None:
        df.loc[idx[long_at], ["ens_up", "ens_down", "ens_expire"]] = [0.90, 0.05, 0.05]
    if short_at is not None:
        df.loc[idx[short_at], ["ens_up", "ens_down", "ens_expire"]] = [0.05, 0.90, 0.05]
    df.index.name = "timestamp"
    p = tmp_path / "art.parquet"
    df.to_parquet(p)
    return str(p)


def _run(candles, artifact, horizon_bars=24, theta=0.40, notional=10_000.0):
    cfg = {
        "strategies": {"active": ["dumb_l3"]},
        "dumb_l3": {"artifact_path": artifact, "theta": theta, "notional": notional,
                    "horizon_bars": horizon_bars},
        "backtest": {"initial_balance": 10_000.0},
    }
    eng = BacktestEngine(cfg, candles.index[0].to_pydatetime(), candles.index[-1].to_pydatetime())
    eng.inject_candles({"15m": candles})
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()
    return eng


def test_entry_timing_no_lookahead_and_tp(tmp_path):
    # bar1 close=100000 이지만 bar2 open=100100 → 진입가는 bar2 open (예측봉 close 아님)
    c = _candles([
        (100000, 100000, 100000, 100000),   # 0
        (100000, 100000, 100000, 100000),   # 1  ← 예측 LONG (pred_key)
        (100100, 100150, 100050, 100100),   # 2  ← 진입 @ open 100100
        (100100, 101300, 100100, 101200),   # 3  ← high 101300 ≥ TP → TP hit
        (100000, 100050, 99950, 100000),    # 4
        (100000, 100050, 99950, 100000),    # 5
    ])
    art = _artifact(tmp_path, c.index, long_at=1, bf=0.01)
    eng = _run(c, art)

    assert len(eng.trades) == 1
    t = eng.trades[0]
    # 진입 타이밍: pred_key=idx[1] → 진입은 idx[2](t+1봉), 가격은 그 봉 open(100100), NOT 예측봉 close(100000)
    assert t["entry_time"] == c.index[2]
    assert t["entry_time"] == c.index[1] + pd.Timedelta(minutes=15)
    assert t["entry_price"] == pytest.approx(100100.0)
    assert t["side"] == "long"
    # 배리어 실현(D-b①): TP = entry·(1+0.01)
    assert t["exit_reason"] == "tp_hit"
    assert t["exit_price"] == pytest.approx(100100.0 * 1.01)
    # 정합성(CLAUDE 10)
    assert sum(x["pnl"] for x in eng.trades) == pytest.approx(eng.balance - 10_000.0)


def test_short_flips_barriers_and_tp(tmp_path):
    # SHORT: 가격 하락이 이익. TP = entry·(1-bf) 를 low 가 침
    c = _candles([
        (100000, 100000, 100000, 100000),   # 0
        (100000, 100000, 100000, 100000),   # 1  ← 예측 SHORT
        (100100, 100150, 100050, 100100),   # 2  ← 진입 @ 100100
        (100100, 100100, 98800, 99000),     # 3  ← low 98800 ≤ TP(=100100·0.99=99099) → TP hit
        (100000, 100050, 99950, 100000),    # 4
    ])
    art = _artifact(tmp_path, c.index, short_at=1, bf=0.01)
    eng = _run(c, art)

    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert t["side"] == "short"
    assert t["exit_reason"] == "tp_hit"
    assert t["exit_price"] == pytest.approx(100100.0 * 0.99)   # 대칭 반전 확인
    assert t["pnl"] > 0                                          # 하락 → short 이익
    assert sum(x["pnl"] for x in eng.trades) == pytest.approx(eng.balance - 10_000.0)


def test_expire_timeout(tmp_path):
    # 배리어 미터치 → N봉 만기 force_exit. horizon_bars=3 → 진입 idx[2] 후 idx[5] 에 만기
    flat = (100100, 100150, 100050, 100100)
    c = _candles([
        (100000, 100000, 100000, 100000),   # 0
        (100000, 100000, 100000, 100000),   # 1  ← 예측 LONG
        flat,                                # 2  ← 진입 @ 100100 (TP101101/SL99099 미터치)
        flat, flat, flat, flat, flat,        # 3..7
    ])
    art = _artifact(tmp_path, c.index, long_at=1, bf=0.01)
    eng = _run(c, art, horizon_bars=3)

    assert len(eng.trades) == 1
    t = eng.trades[0]
    assert t["exit_reason"] == "force_exit"
    # 진입 idx[2] + 3봉 = idx[5] 에 만기 청산
    assert t["exit_time"] == c.index[5]
    assert sum(x["pnl"] for x in eng.trades) == pytest.approx(eng.balance - 10_000.0)


def test_no_trade_when_all_below_theta(tmp_path):
    c = _candles([(100000, 100050, 99950, 100000)] * 6)
    art = _artifact(tmp_path, c.index)      # 강신호 없음 → 전부 HOLD
    eng = _run(c, art)
    assert len(eng.trades) == 0
    assert eng.balance == pytest.approx(10_000.0)
