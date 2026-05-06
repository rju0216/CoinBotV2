"""Ensemble plugin 단위 테스트 (BP-3-2).

mock sub-plugin을 ensemble._sub_instances에 직접 주입하여 voting 로직만 검증.
sub-plugin의 모델 로드/추론은 별도 테스트 (test_ml_*.py)에서 검증.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.plugins.ensemble import Ensemble
from src.strategy.registry import reset_registry_for_testing


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    # ensemble은 plugins discovery에서 자동 등록되지만 본 테스트는 직접 instantiate
    yield
    reset_registry_for_testing()


def _ctx(close: float = 67000.0) -> StrategyContext:
    timestamps = pd.date_range(
        "2024-01-01", periods=30, freq="15min", tz="UTC"
    )
    df = pd.DataFrame(
        {
            "open": [close] * 30,
            "high": [close + 100] * 30,
            "low": [close - 100] * 30,
            "close": [close] * 30,
            "volume": [1.0] * 30,
        },
        index=timestamps,
    )
    df.index.name = "timestamp"
    return StrategyContext(
        candles={"15m": df},
        current_price=close,
        balance=10000.0,
        position=None,
        is_slot_occupied=False,
        params={},
        now=datetime(2024, 1, 1, 7, 30, tzinfo=timezone.utc),
        precomputed_features=None,
    )


class _StubSubModel(StrategyModule):
    """원하는 probs를 그대로 반환하는 mock sub-plugin."""
    name = "_stub"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def __init__(self, probs: list[float] | None, raise_on_signal: bool = False):
        super().__init__({})
        self._probs = probs
        self._raise = raise_on_signal

    def generate_signal(self, ctx):
        if self._raise:
            raise RuntimeError("simulated sub-model failure")
        if self._probs is None:
            return Signal(side=SignalSide.HOLD)  # early HOLD, meta 없음
        meta = {"probs": self._probs}
        # ensemble은 side 무시하고 meta["probs"]만 사용 — 임의로 HOLD 반환
        return Signal(side=SignalSide.HOLD, confidence=max(self._probs), meta=meta)

    def compute_stop_loss(self, ctx, s): return ctx.current_price * 0.99
    def compute_take_profit(self, ctx, s, sl): return ctx.current_price * 1.02


def _make_ensemble(
    sub_probs: dict[str, list[float] | None],
    *,
    min_models: int = 2,
    confidence_threshold: float = 0.55,
) -> Ensemble:
    params = {
        "risk_per_trade_pct": 0.01,
        "max_leverage": 5,
        "sub_models": list(sub_probs.keys()),
        "min_models": min_models,
        "confidence_threshold": confidence_threshold,
        "entry_timeframe": "15m",
        "required_timeframes": ["15m"],
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "reward_risk_ratio": 2.0,
    }
    eng = Ensemble(params)
    # sub_instances 직접 주입 (ensure_sub_models 우회)
    for name, probs in sub_probs.items():
        eng._sub_instances[name] = _StubSubModel(probs)
    return eng


# ---- Soft voting ----


class TestSoftVoting:
    def test_average_of_probs(self):
        """3 모델 평균 [0.1+0.7+0.4, 0.2+0.2+0.3, 0.7+0.1+0.3] / 3 = [0.4, 0.233, 0.367]
        argmax = 0 (SHORT). confidence 0.4 < 0.55 → HOLD"""
        eng = _make_ensemble({
            "m1": [0.1, 0.2, 0.7],
            "m2": [0.7, 0.2, 0.1],
            "m3": [0.4, 0.3, 0.3],
        })
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.HOLD
        assert pytest.approx(sig.meta["probs"][0], rel=1e-3) == 0.4
        assert pytest.approx(sig.meta["probs"][2], rel=1e-3) == 0.367

    def test_long_signal_when_avg_long_above_threshold(self):
        eng = _make_ensemble({
            "m1": [0.1, 0.1, 0.8],
            "m2": [0.1, 0.2, 0.7],
            "m3": [0.2, 0.1, 0.7],
        })
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.LONG
        assert sig.confidence >= 0.55

    def test_short_signal_when_avg_short_above_threshold(self):
        eng = _make_ensemble({
            "m1": [0.8, 0.1, 0.1],
            "m2": [0.7, 0.2, 0.1],
            "m3": [0.7, 0.1, 0.2],
        })
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.SHORT

    def test_contributors_recorded(self):
        eng = _make_ensemble({
            "m1": [0.1, 0.1, 0.8],
            "m2": [0.1, 0.2, 0.7],
        })
        sig = eng.generate_signal(_ctx())
        assert sig.meta["contributors"] == ["m1", "m2"]


# ---- min_models 보호 ----


class TestMinModels:
    def test_holds_when_below_min_at_init(self):
        # sub_models 4개 명시했지만 instances 1개만 등록 → active 1 < min 2 → HOLD
        params = {
            "risk_per_trade_pct": 0.01,
            "max_leverage": 5,
            "sub_models": ["m1", "m2", "m3", "m4"],
            "min_models": 2,
            "confidence_threshold": 0.55,
            "entry_timeframe": "15m",
        }
        eng = Ensemble(params)
        eng._sub_instances["m1"] = _StubSubModel([0.1, 0.1, 0.8])
        eng._failed_models = {"m2", "m3", "m4"}
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.HOLD

    def test_runs_when_at_min(self):
        eng = _make_ensemble(
            {"m1": [0.1, 0.1, 0.8], "m2": [0.2, 0.1, 0.7]},
            min_models=2,
        )
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.LONG


# ---- 실패 처리 ----


class TestFailureHandling:
    def test_runtime_failure_marks_permanent_skip(self):
        eng = _make_ensemble({
            "m1": [0.1, 0.1, 0.8],
            "m2": [0.1, 0.1, 0.8],
            "m3": None,  # placeholder
        })
        # m3는 추론 시 raise하도록 교체
        eng._sub_instances["m3"] = _StubSubModel(None, raise_on_signal=True)

        # 첫 호출: m3 실패 → skip + warning. m1+m2로 voting (LONG)
        sig1 = eng.generate_signal(_ctx())
        assert sig1.side == SignalSide.LONG
        assert "m3" in eng.get_failed_models()
        assert eng.get_active_models_count() == 2

        # 두 번째 호출: m3 영구 skip. 호출 안 됨
        sig2 = eng.generate_signal(_ctx())
        assert sig2.side == SignalSide.LONG

    def test_early_hold_skipped_this_bar_only(self):
        """probs=None (early HOLD)은 이번 봉만 skip. _failed_models에 추가 안 됨."""
        eng = _make_ensemble({
            "m1": [0.1, 0.1, 0.8],
            "m2": [0.1, 0.1, 0.8],
            "m3": None,  # early HOLD (no probs)
        })
        sig = eng.generate_signal(_ctx())
        # m3 빠진 m1+m2로 voting (둘 다 LONG)
        assert sig.side == SignalSide.LONG
        assert "m3" not in eng.get_failed_models()  # 영구 disable 아님

    def test_all_fail_returns_hold(self):
        eng = _make_ensemble({
            "m1": None,
            "m2": None,
        })
        sig = eng.generate_signal(_ctx())
        assert sig.side == SignalSide.HOLD


# ---- SL/TP ----


class TestSLTP:
    def test_long_sl_below_entry(self):
        eng = _make_ensemble({"m1": [0.1, 0.1, 0.8], "m2": [0.1, 0.1, 0.8]})
        ctx = _ctx(close=67000.0)
        signal = Signal(side=SignalSide.LONG)
        sl = eng.compute_stop_loss(ctx, signal)
        tp = eng.compute_take_profit(ctx, signal, sl)
        assert sl < ctx.current_price
        assert tp > ctx.current_price
        # rr 2.0 → tp distance = 2 × sl distance
        assert pytest.approx(tp - ctx.current_price, rel=1e-3) == 2 * (
            ctx.current_price - sl
        )

    def test_short_sl_above_entry(self):
        eng = _make_ensemble({"m1": [0.8, 0.1, 0.1], "m2": [0.8, 0.1, 0.1]})
        ctx = _ctx(close=67000.0)
        signal = Signal(side=SignalSide.SHORT)
        sl = eng.compute_stop_loss(ctx, signal)
        tp = eng.compute_take_profit(ctx, signal, sl)
        assert sl > ctx.current_price
        assert tp < ctx.current_price


# ---- BL-2 OOS warm-up: extract_train_meta override (I-BL003 fix) ----


class _SubWithMeta(_StubSubModel):
    """extract_train_meta가 미리 주입된 (cutoff, acc) 반환."""
    def __init__(self, cutoff_dt, acc):
        super().__init__(probs=[0.1, 0.1, 0.8])
        self._cutoff = cutoff_dt
        self._acc = acc

    def extract_train_meta(self):
        return self._cutoff, self._acc


class TestEnsembleExtractTrainMeta:
    def test_aggregates_min_cutoff_mean_acc(self):
        """4 sub-plugin 집계: cutoff=min, acc=mean."""
        eng = _make_ensemble({"m1": [0.1, 0.1, 0.8], "m2": [0.1, 0.1, 0.8]})
        # _make_ensemble은 _StubSubModel만 주입. _SubWithMeta로 교체.
        eng._sub_instances["m1"] = _SubWithMeta(
            datetime(2024, 12, 31, tzinfo=timezone.utc), 0.74
        )
        eng._sub_instances["m2"] = _SubWithMeta(
            datetime(2026, 5, 4, tzinfo=timezone.utc), 0.76
        )
        cutoff, acc = eng.extract_train_meta()
        assert cutoff == datetime(2024, 12, 31, tzinfo=timezone.utc)
        assert acc == pytest.approx(0.75)

    def test_partial_missing_acc(self):
        """sub-plugin 중 일부 acc=None이어도 cutoff은 산출, acc는 가용분 평균."""
        eng = _make_ensemble({"m1": [0.1, 0.1, 0.8], "m2": [0.1, 0.1, 0.8]})
        eng._sub_instances["m1"] = _SubWithMeta(
            datetime(2026, 5, 4, tzinfo=timezone.utc), 0.78
        )
        eng._sub_instances["m2"] = _SubWithMeta(
            datetime(2026, 5, 4, tzinfo=timezone.utc), None
        )
        cutoff, acc = eng.extract_train_meta()
        assert cutoff == datetime(2026, 5, 4, tzinfo=timezone.utc)
        assert acc == pytest.approx(0.78)

    def test_all_missing_returns_none(self):
        """모든 sub-plugin train_meta 추출 실패 시 (None, None)."""
        eng = _make_ensemble({"m1": [0.1, 0.1, 0.8], "m2": [0.1, 0.1, 0.8]})
        eng._sub_instances["m1"] = _SubWithMeta(None, None)
        eng._sub_instances["m2"] = _SubWithMeta(None, None)
        assert eng.extract_train_meta() == (None, None)

    def test_get_sub_instances_returns_dict_copy(self):
        """get_sub_instances는 dict copy 반환 — 외부 변경이 내부 state에 영향 없음."""
        eng = _make_ensemble({"m1": [0.1, 0.1, 0.8], "m2": [0.1, 0.1, 0.8]})
        subs = eng.get_sub_instances()
        assert set(subs.keys()) == {"m1", "m2"}
        subs["m3"] = _StubSubModel(probs=[0.5, 0.3, 0.2])
        assert "m3" not in eng._sub_instances
