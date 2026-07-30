"""Phase 6 Step 6.0 — 정책 하네스 회귀 (로직 fast 유닛 + 등가성 통합).

- fast 유닛(합성 trades, 엔진無): _winrate_breakeven·_apply_funding·_analyze·증강 로직.
- 등가성 통합: 하네스가 scratchpad frontier_gate2 수치(4h_4d net +$9,890) 재현.
  무거운 1m 백테라 **PHASE6_HEAVY_EQUIV=1 + 데이터 존재** 시에만 실행(기본 skip — suite 속도·
  백테 승인정책 존중). 규칙17 박제는 커밋된 이 테스트로 유지.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.experiments import policy_eval as pe

CANDLE_DIR = "data/candles"
ART_4H_4D = "data/research/phase5_frontier/oos_4h_4d_x3.parquet"


# ---- fast 유닛 (엔진無) --------------------------------------------------

def test_winrate_breakeven_symmetric():
    # 3승(+10) 2패(-10): 승률 60%, breakeven = 10/(10+10)=50%
    pnl = pd.Series([10, 10, 10, -10, -10], dtype=float)
    wr, be = pe._winrate_breakeven(pnl)
    assert wr == 60.0
    assert be == 50.0


def test_winrate_breakeven_skewed_payoff():
    # 이익 크고(+30) 손실 작으면(-10) breakeven 승률이 낮아짐(25%)
    pnl = pd.Series([30, -10, -10, -10], dtype=float)
    wr, be = pe._winrate_breakeven(pnl)
    assert wr == 25.0
    assert be == 25.0  # 10/(30+10)


def test_apply_funding_long_pays_short_receives(tmp_path):
    fcsv = tmp_path / "f.csv"
    # 2개 펀딩 구간, rate 0.001 (0.1%)
    ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True)
    ms = [int(t.timestamp() * 1000) for t in ts]   # 해상도 무관 ms (실 csv 포맷)
    pd.DataFrame({"timestamp": ms, "funding_rate": [0.001, 0.001]}).to_csv(fcsv, index=False)
    df = pd.DataFrame({
        "entry_time": pd.to_datetime(["2023-12-31 20:00", "2023-12-31 20:00"], utc=True),
        "exit_time": pd.to_datetime(["2024-01-01 12:00", "2024-01-01 12:00"], utc=True),
        "side": ["LONG", "SHORT"],
        "size": [0.1, 0.1], "entry_price": [100_000.0, 100_000.0],   # 노셔널 10,000
    })
    fund = pe._apply_funding(df, str(fcsv))
    # 두 펀딩 구간 모두 보유(0.002 합). 롱=-0.002*10000=-20, 숏=+20
    assert fund.iloc[0] == pytest.approx(-20.0)
    assert fund.iloc[1] == pytest.approx(+20.0)


def test_apply_funding_scales_with_trade_notional():
    """★I-008 회귀★ 펀딩은 초기자본이 아니라 **그 거래의 노셔널**에 부과된다.

    N-트랜치(notional=init/N)에서 init 고정이면 N배 과대계상 → 이 테스트가 재발을 막는다."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        fcsv = Path(td) / "f.csv"
        ts = pd.to_datetime(["2024-01-01 00:00"], utc=True)
        pd.DataFrame({"timestamp": [int(ts[0].timestamp() * 1000)],
                      "funding_rate": [0.001]}).to_csv(fcsv, index=False)
        df = pd.DataFrame({
            "entry_time": pd.to_datetime(["2023-12-31 20:00"] * 3, utc=True),
            "exit_time": pd.to_datetime(["2024-01-01 04:00"] * 3, utc=True),
            "side": ["LONG"] * 3,
            # 노셔널 10,000 / 2,500(N=4) / 833.33(N=12)
            "size": [0.1, 0.025, 0.0083333], "entry_price": [100_000.0] * 3,
        })
        fund = pe._apply_funding(df, str(fcsv))
        assert fund.iloc[0] == pytest.approx(-10.0)          # 0.001 × 10,000
        assert fund.iloc[1] == pytest.approx(-2.5)           # 0.001 × 2,500
        assert fund.iloc[2] == pytest.approx(-0.83333, rel=1e-4)


def test_occupancy_sequential_trades_never_overlap():
    """★I-009 회귀★ exit_time == 다음 entry_time 인 순차 거래는 동시성 1 이어야 한다.

    (구버그: 동시각에 진입을 먼저 세어 max_concurrent 가 용량+1 로 부풀었음)"""
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"],
                       utc=True)
    df = pd.DataFrame({"entry_time": [t[0], t[1]], "exit_time": [t[1], t[2]]})
    occ = pe._occupancy(df)
    assert occ["max_concurrent"] == 1
    assert occ["time_in_market_pct"] == pytest.approx(100.0)


def test_occupancy_true_overlap_counted():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00", "2024-01-01 04:00",
                        "2024-01-01 06:00"], utc=True)
    df = pd.DataFrame({"entry_time": [t[0], t[1]], "exit_time": [t[2], t[3]]})
    assert pe._occupancy(df)["max_concurrent"] == 2


def test_bp_metrics_use_realized_notional(tmp_path):
    """★I-010 회귀★ conviction 사이징(노셔널 가변)에서 bp 는 **실현 노셔널**로 정규화된다.

    선언 notional(10,000) 로 나누면 실노셔널 20,000 거래의 bp 가 2배 부풀려진다."""
    trades = [
        dict(side="LONG", size=0.1, entry_price=100_000.0,          # 노셔널 10,000
             entry_time=pd.Timestamp("2024-01-01", tz="UTC"),
             exit_time=pd.Timestamp("2024-01-02", tz="UTC"),
             pnl=100.0, trading_fee=0.0, funding_fee=0.0, exit_reason="tp_hit"),
        dict(side="LONG", size=0.2, entry_price=100_000.0,          # 노셔널 20,000
             entry_time=pd.Timestamp("2024-01-03", tz="UTC"),
             exit_time=pd.Timestamp("2024-01-04", tz="UTC"),
             pnl=200.0, trading_fee=0.0, funding_fee=0.0, exit_reason="tp_hit"),
    ]
    out = pe._analyze({"notional": 10_000.0}, trades, "data/funding/__nope__.csv",
                      init=10_000.0)
    # 총 pnl 300 / 총 실현노셔널 30,000 = 100bp (선언값 기준이면 150bp 로 과대)
    assert out["net_bp_per_trade"] == pytest.approx(100.0)
    assert out["total_notional"] == pytest.approx(30_000.0)
    assert out["notional_per_trade"] == pytest.approx(15_000.0)


def test_ensemble_requires_explicit_notional():
    """★I-013 회귀★ 앙상블 슬리브가 notional 을 빠뜨리면 조용히 전액 배정되지 않고 하드페일."""
    ens = [("econ_l3_h3", {"artifact_path": "x", "horizon_bars": 18}),      # notional 누락
           ("econ_l3_h4", {"artifact_path": "y", "horizon_bars": 24, "notional": 100.0})]
    with pytest.raises(ValueError, match="notional 미지정"):
        pe._run_engine({}, "2021-01-01", "2021-02-01", 10_000.0,
                       max_slots=42, ensemble=ens)


def test_apply_funding_missing_file_zero():
    df = pd.DataFrame({"entry_time": pd.to_datetime(["2024-01-01"], utc=True),
                       "exit_time": pd.to_datetime(["2024-01-02"], utc=True),
                       "side": ["LONG"]})
    fund = pe._apply_funding(df, "data/funding/__nope__.csv")
    assert (fund == 0.0).all()


def _synth_trades():
    """2년(2023·2024) 걸친 합성 trades — 연도별 파티션·net 검증용."""
    rows = []
    # 2023: 2승(+100) 1패(-50) → net +150
    for t, pnl in [("2023-03-01", 100), ("2023-06-01", 100), ("2023-09-01", -50)]:
        rows.append(dict(entry_time=f"{t} 00:00", exit_time=f"{t} 08:00",
                         side="LONG", size=0.1, entry_price=100_000.0,   # 노셔널 10,000
                         pnl=float(pnl), trading_fee=5.0, funding_fee=0.0,
                         exit_reason="tp_hit" if pnl > 0 else "sl_hit"))
    # 2024: 1승(+80) 2패(-60) → net -40
    for t, pnl in [("2024-02-01", 80), ("2024-05-01", -60), ("2024-08-01", -60)]:
        rows.append(dict(entry_time=f"{t} 00:00", exit_time=f"{t} 08:00",
                         side="SHORT", size=0.1, entry_price=100_000.0,  # 노셔널 10,000
                         pnl=float(pnl), trading_fee=5.0, funding_fee=0.0,
                         exit_reason="tp_hit" if pnl > 0 else "sl_hit"))
    return rows


def test_analyze_by_year_and_totals():
    spec = {"decision_tf": "4h", "horizon_bars": 24, "artifact_path": "?", "theta": 0.40}
    out = pe._analyze(spec, _synth_trades(), funding_csv="data/funding/__nope__.csv", init=10_000.0)
    assert out["n_trades"] == 6
    assert out["net"] == pytest.approx(110.0)  # 2023 net150 + 2024 net(-40) = 110
    assert out["by_year"]["2023"]["net"] == pytest.approx(150.0)
    assert out["by_year"]["2023"]["net_sign"] == "+"
    assert out["by_year"]["2024"]["net"] == pytest.approx(-40.0)
    assert out["by_year"]["2024"]["net_sign"] == "-"
    assert out["win_rate"] == pytest.approx(50.0)  # 3승/6


def test_augment_adds_causal_cols_preserves_preds(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    art = pd.DataFrame({"ens_up": [0.5] * 5, "ens_down": [0.3] * 5,
                        "regime": ["up|high"] * 5}, index=idx)  # 기존 analysis regime
    p = tmp_path / "a.parquet"
    art.to_parquet(p)

    def _fake_frame(tf, index, candle_dir):
        return pd.DataFrame({"regime_trend": ["up", "down", "up", "flat", "up"],
                             "regime_vol": ["high", "low", "high", "low", "high"],
                             "regime": ["x"] * 5}, index=index)

    monkeypatch.setattr(pe, "causal_regime_frame_for", _fake_frame)
    aug, cov = pe.augment_artifact_causal_regime(str(p), "4h", str(tmp_path))
    # 신규 causal 열 추가 + 기존 예측/analysis regime 불변
    assert list(aug["causal_regime_trend"]) == ["up", "down", "up", "flat", "up"]
    assert list(aug["causal_regime_vol"]) == ["high", "low", "high", "low", "high"]
    assert list(aug["ens_up"]) == [0.5] * 5
    assert list(aug["regime"]) == ["up|high"] * 5  # analysis regime 덮어쓰지 않음
    assert cov["n_rows"] == 5 and cov["trend_nan"] == 0


# ---- 등가성 통합 (무거운 1m 백테, opt-in) --------------------------------

_HEAVY = os.environ.get("PHASE6_HEAVY_EQUIV") == "1"
_has_data = os.path.exists(ART_4H_4D) and os.path.exists(
    os.path.join(CANDLE_DIR, "BTC_USDT_USDT_1m.csv"))


@pytest.mark.skipif(not (_HEAVY and _has_data),
                    reason="PHASE6_HEAVY_EQUIV=1 + 1m/아티팩트 데이터 필요 (무거운 백테)")
def test_equivalence_frontier_gate2_4h_4d():
    """하네스 baseline(4h_4d) = frontier_gate2 net +$9,890 재현 (규칙17 박제)."""
    spec = {"decision_tf": "4h", "horizon_bars": 24, "artifact_path": ART_4H_4D,
            "fill_tf": "1m", "theta": 0.40, "no_tp": False, "no_timeout": False,
            "allow_reverse": False, "sizing": "fixed", "mtf_gate": False}
    out = pe.evaluate_policy(spec, start="2020-12-28", end="2026-04-23")
    assert out["n_trades"] == 338
    assert out["net"] == pytest.approx(9890, abs=60)  # frontier_gate2 기록값
