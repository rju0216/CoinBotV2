"""AdaptiveL3(적응 청산 3층) 단위 테스트 (Phase 6 Step 6.2).

DumbL3 진입 재사용 + 청산 재설계 검증:
- E2 sl_mult: 진입 SL 배수(0.5=tight)
- E4 breakeven: 이익 R배 도달 시 SL→진입가(한 번), 상태 초기화(on_position_opened)
- E1 signal_decay: 보유 방향 fresh p_dir<θ 조기청산 + 만기 backstop 유지
- 기본값(레버 off) = DumbL3 동작(SL=full barrier·신호감쇠 없음)
합성 아티팩트(4h) — 실데이터 비의존.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, StrategyContext
from src.strategy.plugins.adaptive_l3 import AdaptiveL3


@pytest.fixture
def art(tmp_path):
    """3봉 4h: bar0 강LONG(up0.60)·bar1 신호감쇠(up0.30<θ)·bar2 강LONG. barrier_frac 0.02."""
    idx = pd.date_range("2021-01-01 00:00", periods=3, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "ens_up":     [0.60, 0.30, 0.55],
        "ens_down":   [0.20, 0.30, 0.20],
        "ens_expire": [0.20, 0.40, 0.25],
        "barrier_frac": [0.02, 0.02, 0.02],
    }, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / "a.parquet"
    df.to_parquet(p)
    return str(p), idx


def _ctx(key, price=100_000.0, position=None, now=None):
    df = pd.DataFrame({"open": [price], "high": [price], "low": [price], "close": [price]},
                      index=[key])
    return StrategyContext(candles={"4h": df}, current_price=price, balance=10_000.0,
                           position=position, is_slot_occupied=position is not None,
                           params={}, now=now or key)


def _ctx_multi(idx, price, position, now):
    """다봉 4h df + 임의 now (완성봉 가드 검증용 — df 에 진행 중 봉 포함)."""
    df = pd.DataFrame({"open": [price] * len(idx), "high": [price] * len(idx),
                       "low": [price] * len(idx), "close": [price] * len(idx)}, index=idx)
    return StrategyContext(candles={"4h": df}, current_price=price, balance=10_000.0,
                           position=position, is_slot_occupied=True, params={}, now=now)


def _pos(entry=100_000.0, side=PositionSide.LONG, sl=98_000.0, tp=102_000.0, t=None):
    return Position(side=side, size=0.1, entry_price=entry, entry_time=t,
                    strategy_name="adaptive_l3", stop_loss=sl, take_profit=tp)


def test_default_matches_dumb_sl(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40})
    ctx = _ctx(idx[0])
    sig = p.generate_signal(ctx)
    assert sig.side == SignalSide.LONG
    # sl_mult 기본 1.0 → full barrier (dumb 동일)
    assert p.compute_stop_loss(ctx, sig) == pytest.approx(100_000.0 * (1 - 0.02))


def test_e2_tight_sl(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40, "sl_mult": 0.5})
    ctx = _ctx(idx[0])
    sig = p.generate_signal(ctx)
    # 0.5 배 → SL = entry*(1 - 0.5*0.02) = entry*0.99
    assert p.compute_stop_loss(ctx, sig) == pytest.approx(100_000.0 * (1 - 0.01))


def test_e2_tight_sl_short(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40, "sl_mult": 0.5})
    # SHORT 방향 신호 강제(ens_down 큰 봉 없어 직접 Signal 구성)
    from src.core.types import Signal
    sig = Signal(SignalSide.SHORT, confidence=0.6, meta={"barrier_frac": 0.02})
    assert p.compute_stop_loss(_ctx(idx[0]), sig) == pytest.approx(100_000.0 * (1 + 0.01))


def test_e4_breakeven_arms_at_1R(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40,
                    "breakeven": True, "breakeven_r": 1.0})
    pos = _pos(entry=100_000.0, sl=98_000.0, t=idx[0])   # R=2000, BE 트리거=102000
    p.on_position_opened(pos)
    assert p._be_trigger == pytest.approx(102_000.0)
    # 이익 미달 → SL 유지
    assert p.update_stop_loss(_ctx(idx[1], price=101_000.0, position=pos), pos) is None
    # 1R 도달 → SL 진입가로
    assert p.update_stop_loss(_ctx(idx[1], price=102_000.0, position=pos), pos) == pytest.approx(100_000.0)
    # armed → 재호출 None(한 번만)
    assert p.update_stop_loss(_ctx(idx[1], price=103_000.0, position=pos), pos) is None


def test_e4_breakeven_short(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40, "breakeven": True})
    pos = _pos(entry=100_000.0, side=PositionSide.SHORT, sl=102_000.0, tp=98_000.0, t=idx[0])
    p.on_position_opened(pos)          # R=2000, SHORT 트리거=98000
    assert p._be_trigger == pytest.approx(98_000.0)
    assert p.update_stop_loss(_ctx(idx[1], price=98_000.0, position=pos), pos) == pytest.approx(100_000.0)


def test_e1_signal_decay_completed_bar_no_lookahead(art):
    """★lookahead 가드★: 진행 중 봉(df 에 존재)이 아니라 **완성봉** 예측으로 판단.

    idx=00:00/04:00/08:00, ens_up=0.60/0.30/0.55. df 엔 세 봉 다 존재하나,
    now 시점에 **마감된** 봉만 써야 한다(구 버그: df.index[-1]=진행중봉 예측 = 미래누수)."""
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40,
                    "signal_decay": True, "horizon_bars": 1000})
    pos = _pos(t=idx[0])
    p.on_position_opened(pos)
    # now=04:37 (04:00봉 진행중). 완성봉 마지막=00:00(up0.60≥θ) → HOLD.
    # (구 버그면 진행중 04:00봉 up0.30 을 읽어 잘못 EXIT)
    now1 = idx[1] + pd.Timedelta(minutes=37)
    assert p.should_force_exit(_ctx_multi(idx, 100_000.0, pos, now1), pos) is None
    # now=08:37 (08:00봉 진행중). 완성봉 마지막=04:00(up0.30<θ) → 신호감쇠 EXIT
    now2 = idx[2] + pd.Timedelta(minutes=37)
    dec = p.should_force_exit(_ctx_multi(idx, 100_000.0, pos, now2), pos)
    assert dec is not None and dec.note == "signal_decay"


def test_e1_no_exit_before_any_completed_bar(art):
    """완성봉이 아직 없으면(초기) 신호감쇠 평가 스킵 → HOLD."""
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40,
                    "signal_decay": True, "horizon_bars": 1000})
    pos = _pos(t=idx[0])
    p.on_position_opened(pos)
    now0 = idx[0] + pd.Timedelta(minutes=10)   # 첫 봉도 미완성
    assert p.should_force_exit(_ctx_multi(idx, 100_000.0, pos, now0), pos) is None


def test_e1_timeout_backstop_kept(art):
    path, idx = art
    # horizon 1봉 → bar1(4h 경과)서 만기 backstop 발동(신호 살아있어도)
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40,
                    "signal_decay": True, "horizon_bars": 1})
    pos = _pos(t=idx[0])
    p.on_position_opened(pos)
    dec = p.should_force_exit(_ctx(idx[2], position=pos, now=idx[2]), pos)  # 8h 경과 > 1봉
    assert dec is not None and dec.note == "expire_timeout"


def test_default_no_forced_exit_before_timeout(art):
    path, idx = art
    p = AdaptiveL3({"artifact_path": path, "decision_tf": "4h", "theta": 0.40, "horizon_bars": 1000})
    pos = _pos(t=idx[0])
    p.on_position_opened(pos)
    # 레버 전부 off → 만기 전 강제청산 없음
    assert p.should_force_exit(_ctx(idx[1], position=pos, now=idx[1]), pos) is None
