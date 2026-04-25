"""StrategyModule 추상 클래스.

뼈대 프로토타입에서 모든 전략 플러그인이 구현해야 하는 인터페이스.
필수 메서드 3개(generate_signal, compute_stop_loss, compute_take_profit)와
선택 훅 6개를 정의. 엔진은 이 인터페이스만을 통해 전략과 상호작용한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ExitDecision, Position, Signal, StrategyContext


class StrategyModule(ABC):
    # ---- 클래스 속성 (서브클래스가 반드시 선언) ----
    name: str = ""
    entry_timeframe: str = ""
    required_timeframes: list[str] = []
    supports_pyramiding: bool = False

    def __init__(self, params: dict[str, Any]) -> None:
        """
        Args:
            params: config[self.name] 섹션의 dict.
                    엔진이 registry.load_active_strategies에서 주입.
        """
        self.params = params or {}

    # ---- 필수 구현 ----

    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> Signal:
        """진입 신호 생성. ctx.is_slot_occupied=False일 때만 엔진이 호출."""

    @abstractmethod
    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        """진입 직전 SL 가격 산정."""

    @abstractmethod
    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        """진입 직전 TP 가격 산정."""

    # ---- 선택 훅 (기본 no-op) ----

    def on_bar_close(self, ctx: StrategyContext, timeframe: str) -> None:
        """봉 마감 시 호출. 전략의 보조 TF 상태 갱신용."""
        return None

    def update_stop_loss(
        self, ctx: StrategyContext, position: Position
    ) -> float | None:
        """동적 SL 갱신 (trailing stop 등). None 반환 시 기존 값 유지."""
        return None

    def should_force_exit(
        self, ctx: StrategyContext, position: Position
    ) -> ExitDecision | None:
        """전략 특화 강제 청산 판단 (timeout, regime 변화 등). None=유지."""
        return None

    def on_position_opened(self, position: Position) -> None:
        return None

    def on_position_closed(self, position: Position, pnl: float) -> None:
        return None

    def generate_pyramid_signal(
        self, ctx: StrategyContext, position: Position
    ) -> Signal | None:
        """피라미딩 신호. supports_pyramiding=True일 때만 엔진이 호출."""
        return None
