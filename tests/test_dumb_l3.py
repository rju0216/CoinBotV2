"""멍청3층 플러그인(DumbL3) 단위 테스트 (Phase 4 Step 4.1).

박제 OOS 예측 아티팩트를 합성(실데이터 비의존)해 배선 정책을 검증한다:
- θ 진입 게이트 (HOLD/LONG/SHORT)
- D-b① 배리어 앵커링 (TP/SL = 실진입가 기준, 방향별 대칭 반전)
- 고정 notional 사이징
- 만기(N봉) should_force_exit
- OOS 밖(예측 부재) → 무거래, require_dir_gt_expire 게이트, pred_source 선택
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.core.enums import PositionSide, PositionStatus, SignalSide
from src.core.types import Position, StrategyContext
from src.strategy.plugins.dumb_l3 import DumbL3


@pytest.fixture
def artifact_path(tmp_path):
    """합성 예측 아티팩트: 3봉(강LONG·강SHORT·저확신) + barrier_frac. 15m 간격 UTC."""
    idx = pd.date_range("2021-01-01 00:00", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({
        # ens_* (앙상블) + s0_* (개별) 둘 다 넣어 pred_source 분기 검증
        "ens_up":     [0.60, 0.20, 0.34],
        "ens_down":   [0.20, 0.55, 0.33],
        "ens_expire": [0.20, 0.25, 0.33],
        "s0_up":      [0.10, 0.10, 0.10],
        "s0_down":    [0.80, 0.80, 0.80],
        "s0_expire":  [0.10, 0.10, 0.10],
        "barrier_frac": [0.02, 0.03, 0.01],
        "mtf1h_kf_slope": [0.5, 0.5, 0.5],   # 1h 추세 상승(>0) — LONG 정렬, SHORT 역행
        "y_true":  ["up", "down", "expire"],
        "regime":  ["up|high", "down|low", "flat|low"],
    }, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / "art.parquet"
    df.to_parquet(p)
    return str(p), idx


def _ctx(key, price=100_000.0, now=None, position=None):
    prices = pd.DataFrame({"open": [price], "high": [price], "low": [price], "close": [price]},
                          index=[key])
    return StrategyContext(candles={"15m": prices}, current_price=price, balance=10_000.0,
                           position=position, is_slot_occupied=position is not None,
                           params={}, now=now or key)


def test_long_above_theta_barrier_anchor(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "theta": 0.40, "notional": 10_000.0})
    entry = 100_000.0
    ctx = _ctx(idx[0], price=entry)
    sig = p.generate_signal(ctx)
    assert sig.side == SignalSide.LONG and sig.is_actionable
    sl = p.compute_stop_loss(ctx, sig)
    tp = p.compute_take_profit(ctx, sig, sl)
    # D-b①: 실진입가 기준, LONG → SL 아래·TP 위, 폭 = barrier_frac(0.02)
    assert sl == pytest.approx(entry * (1 - 0.02))
    assert tp == pytest.approx(entry * (1 + 0.02))
    # 고정 notional
    assert p.compute_position_size(ctx, sig, sl) == pytest.approx(10_000.0 / entry)


def test_short_flips_barriers(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "theta": 0.40})
    entry = 100_000.0
    ctx = _ctx(idx[1], price=entry)
    sig = p.generate_signal(ctx)
    assert sig.side == SignalSide.SHORT
    sl = p.compute_stop_loss(ctx, sig)
    tp = p.compute_take_profit(ctx, sig, sl)
    # SHORT → SL 위·TP 아래 (대칭 반전), 폭 = 0.03
    assert sl == pytest.approx(entry * (1 + 0.03))
    assert tp == pytest.approx(entry * (1 - 0.03))


def test_hold_below_theta(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "theta": 0.40})
    # 3봉: max(up,down)=0.34 < 0.40 → HOLD
    sig = p.generate_signal(_ctx(idx[2]))
    assert sig.side == SignalSide.HOLD and not sig.is_actionable


def test_oos_miss_holds(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "theta": 0.40})
    # 아티팩트에 없는 봉 → 무거래
    key = idx[-1] + timedelta(days=999)
    sig = p.generate_signal(_ctx(key))
    assert sig.side == SignalSide.HOLD


def test_require_dir_gt_expire_gate(artifact_path):
    path, idx = artifact_path
    # theta 낮춰 방향은 통과시키되, dir>expire 조건으로 걸러지는지: 저확신봉(0.34 vs exp0.33)
    p_off = DumbL3({"artifact_path": path, "theta": 0.30, "require_dir_gt_expire": False})
    p_on = DumbL3({"artifact_path": path, "theta": 0.30, "require_dir_gt_expire": True})
    # idx[2]: up=0.34 > expire=0.33 → on 도 통과 (dir>expire)
    assert p_off.generate_signal(_ctx(idx[2])).is_actionable
    assert p_on.generate_signal(_ctx(idx[2])).is_actionable
    # up=down 근접이지만 up>=down 이라 LONG, 0.34>0.33 통과. 경계 확인용.


def test_pred_source_selects_seed(artifact_path):
    path, idx = artifact_path
    # s0: down=0.80 → 어느 봉이든 SHORT (ens 와 다른 방향)
    p = DumbL3({"artifact_path": path, "theta": 0.40, "pred_source": "s0"})
    sig = p.generate_signal(_ctx(idx[0]))     # ens 로는 LONG 이지만 s0 로는 SHORT
    assert sig.side == SignalSide.SHORT


def test_mtf_gate_blocks_counter_trend(artifact_path):
    path, idx = artifact_path
    # 아티팩트 1h추세 = +0.5(상승). idx[0]=강LONG(정렬→통과), idx[1]=강SHORT(역행→차단).
    p = DumbL3({"artifact_path": path, "theta": 0.40, "mtf_gate": True})
    assert p.generate_signal(_ctx(idx[0])).side == SignalSide.LONG    # 정렬 → 진입
    assert p.generate_signal(_ctx(idx[1])).side == SignalSide.HOLD    # 역행 SHORT → 차단
    # 게이트 off 면 idx[1] SHORT 진입
    p_off = DumbL3({"artifact_path": path, "theta": 0.40, "mtf_gate": False})
    assert p_off.generate_signal(_ctx(idx[1])).side == SignalSide.SHORT


def test_force_exit_timeout(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "horizon_bars": 24})   # 24봉 × 15m = 6h
    pos = Position(side=PositionSide.LONG, size=0.1, entry_price=100_000.0,
                   entry_time=idx[0], strategy_name="dumb_l3", status=PositionStatus.OPEN)
    # 6h 직전 → 유지
    ctx_before = _ctx(idx[0], now=idx[0] + timedelta(hours=6) - timedelta(minutes=1), position=pos)
    assert p.should_force_exit(ctx_before, pos) is None
    # 6h 도달 → 만기 청산
    ctx_at = _ctx(idx[0], now=idx[0] + timedelta(hours=6), position=pos)
    dec = p.should_force_exit(ctx_at, pos)
    assert dec is not None and dec.note == "expire_timeout"


def test_no_reverse_by_default(artifact_path):
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path})
    pos = Position(side=PositionSide.LONG, size=0.1, entry_price=100_000.0,
                   entry_time=idx[0], strategy_name="dumb_l3", status=PositionStatus.OPEN)
    from src.core.types import Signal
    assert p.should_reverse(_ctx(idx[1], position=pos), pos, Signal(SignalSide.SHORT)) is False


# ---- Phase 5.A 홀딩 지렛대 (no_tp / no_timeout) ----

def test_no_tp_removes_take_profit(artifact_path):
    """no_tp=True → TP None(SL 유지). 승자 태우기 지렛대(C·D)."""
    path, idx = artifact_path
    entry = 100_000.0
    p = DumbL3({"artifact_path": path, "theta": 0.40, "no_tp": True})
    ctx = _ctx(idx[0], price=entry)
    sig = p.generate_signal(ctx)
    assert sig.side == SignalSide.LONG and sig.is_actionable
    # TP 제거되나 SL 은 유지 (배리어 하벽)
    assert p.compute_take_profit(ctx, sig, None) is None
    assert p.compute_stop_loss(ctx, sig) == pytest.approx(entry * (1 - 0.02))
    # 기본(no_tp 미지정)은 TP 유지 — 회귀 방어
    p_def = DumbL3({"artifact_path": path, "theta": 0.40})
    assert p_def.compute_take_profit(ctx, p_def.generate_signal(ctx), None) is not None


def test_no_timeout_removes_force_exit(artifact_path):
    """no_timeout=True → 만기 도달해도 force_exit None(런 유지). 지렛대(B·D)."""
    path, idx = artifact_path
    p = DumbL3({"artifact_path": path, "horizon_bars": 24, "no_timeout": True})
    pos = Position(side=PositionSide.LONG, size=0.1, entry_price=100_000.0,
                   entry_time=idx[0], strategy_name="dumb_l3", status=PositionStatus.OPEN)
    # 만기(6h) 도달·초과해도 청산 안 함
    ctx_at = _ctx(idx[0], now=idx[0] + timedelta(hours=6), position=pos)
    ctx_far = _ctx(idx[0], now=idx[0] + timedelta(days=30), position=pos)
    assert p.should_force_exit(ctx_at, pos) is None
    assert p.should_force_exit(ctx_far, pos) is None
    # 기본(no_timeout 미지정)은 만기 청산 유지 — 회귀 방어
    p_def = DumbL3({"artifact_path": path, "horizon_bars": 24})
    assert p_def.should_force_exit(ctx_at, pos) is not None


def test_full_holding_config_D(artifact_path):
    """셀 D = no_tp + no_timeout + allow_reverse: 청산구 = SL / reverse 만."""
    path, idx = artifact_path
    entry = 100_000.0
    p = DumbL3({"artifact_path": path, "theta": 0.40,
                "no_tp": True, "no_timeout": True, "allow_reverse": True})
    ctx = _ctx(idx[0], price=entry)
    sig = p.generate_signal(ctx)
    # TP·만기 제거, SL 유지
    assert p.compute_take_profit(ctx, sig, None) is None
    assert p.compute_stop_loss(ctx, sig) == pytest.approx(entry * (1 - 0.02))
    pos = Position(side=PositionSide.LONG, size=0.1, entry_price=entry,
                   entry_time=idx[0], strategy_name="dumb_l3", status=PositionStatus.OPEN)
    assert p.should_force_exit(_ctx(idx[0], now=idx[0] + timedelta(days=30), position=pos), pos) is None
    # reverse 는 반대 신호에만 발화 (같은 방향/HOLD → 유지)
    from src.core.types import Signal
    ctx_pos = _ctx(idx[1], position=pos)
    assert p.should_reverse(ctx_pos, pos, Signal(SignalSide.SHORT)) is True   # 반대 → 청산
    assert p.should_reverse(ctx_pos, pos, Signal(SignalSide.LONG)) is False   # 같은 방향 → 유지


# ---- Phase 5 지평 프론티어: decision_tf (P2) / conviction 사이징 (P3) ----

def test_decision_tf_parameterizes_entry_timeframe(artifact_path):
    """decision_tf → entry_timeframe·required_timeframes·만기간격 재구성. 기본은 15m 보존."""
    path, _ = artifact_path
    # 기본: 15m
    p0 = DumbL3({"artifact_path": path})
    assert p0.entry_timeframe == "15m" and p0.required_timeframes == ["15m"]
    # 1h 결정 TF + 1m 체결
    p1 = DumbL3({"artifact_path": path, "decision_tf": "1h", "fill_tf": "1m", "horizon_bars": 24})
    assert p1.entry_timeframe == "1h"
    assert p1.required_timeframes == ["1h", "1m"]
    # 만기 = N봉 × 1h 간격 (24봉 = 1일)
    assert p1._timeout == timedelta(days=1)
    # 4h 결정
    p4 = DumbL3({"artifact_path": path, "decision_tf": "4h", "horizon_bars": 6})
    assert p4.entry_timeframe == "4h" and p4._timeout == timedelta(days=1)   # 6봉×4h=1일


def test_conviction_sizing_scales_with_confidence(artifact_path):
    """sizing='conviction': notional 이 신뢰도 [θ,1]→[1,max_mult] 선형. fixed 는 불변."""
    path, idx = artifact_path
    entry = 100_000.0
    theta, mm = 0.40, 3.0
    p = DumbL3({"artifact_path": path, "theta": theta, "notional": 10_000.0,
                "sizing": "conviction", "conviction_max_mult": mm})
    ctx = _ctx(idx[0], price=entry)
    from src.core.types import Signal
    # p_dir = θ → mult 1 (base)
    s_lo = Signal(SignalSide.LONG, confidence=theta)
    assert p.compute_position_size(ctx, s_lo, None) == pytest.approx(10_000.0 / entry)
    # p_dir = 1.0 → mult max_mult
    s_hi = Signal(SignalSide.LONG, confidence=1.0)
    assert p.compute_position_size(ctx, s_hi, None) == pytest.approx(mm * 10_000.0 / entry)
    # 중간(θ+(1-θ)/2) → mult 1+(mm-1)/2 = 2.0
    s_mid = Signal(SignalSide.LONG, confidence=theta + (1 - theta) / 2)
    assert p.compute_position_size(ctx, s_mid, None) == pytest.approx(2.0 * 10_000.0 / entry)
    # fixed 는 신뢰도 무관 불변
    pf = DumbL3({"artifact_path": path, "theta": theta, "notional": 10_000.0})
    assert pf.compute_position_size(ctx, s_hi, None) == pytest.approx(10_000.0 / entry)
