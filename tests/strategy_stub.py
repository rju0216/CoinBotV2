"""테스트 전용 최소 전략 base.

production 의 StrategyModule 은 거래 정책 6개를 abstract 로 강제한다(엔진에 정책
기본값 없음). 테스트는 매번 6개를 다 구현하기 번거로우므로, 여기 trivial 기본
구현을 제공하는 StubStrategy 를 두고 테스트는 필요한 메서드만 override 한다.

이 편의 base 는 tests 전용이며 src 에는 없다 — 엔진/모델 분리 원칙 유지.
"""

from __future__ import annotations

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule


class StubStrategy(StrategyModule):
    name = "stub"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]
    sl_tp_fill_priority = "sl_first"

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        return Signal(side=SignalSide.HOLD)

    def compute_stop_loss(self, ctx, signal):
        return None

    def compute_take_profit(self, ctx, signal, stop_loss):
        return None

    def compute_position_size(self, ctx, signal, stop_loss):
        return 0.01

    def should_reverse(self, ctx, position, new_signal):
        return False

    def allow_entry(self, ctx):
        return True
