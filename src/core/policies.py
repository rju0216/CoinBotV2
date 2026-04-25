"""엔진 전역 정책 — 역방향 신호 처리 등."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal


def signal_opposes_position(signal: Signal, position: Position | None) -> bool:
    if position is None or position.side == PositionSide.NONE:
        return False
    if signal.side == SignalSide.LONG and position.side == PositionSide.SHORT:
        return True
    if signal.side == SignalSide.SHORT and position.side == PositionSide.LONG:
        return True
    return False


class ReverseSignalPolicy(ABC):
    @abstractmethod
    def should_reverse(
        self,
        position: Position,
        new_signal: Signal,
        current_strategy_name: str,
        new_strategy_name: str,
    ) -> bool:
        ...


class IgnoreReversePolicy(ReverseSignalPolicy):
    """C-1 (default): 보유 전략의 청산 규칙이 끝날 때까지 대기."""

    def should_reverse(
        self,
        position: Position,
        new_signal: Signal,
        current_strategy_name: str,
        new_strategy_name: str,
    ) -> bool:
        return False


class ReverseOnOppositePolicy(ReverseSignalPolicy):
    """반대 신호 시 청산 후 즉시 신규 진입."""

    def should_reverse(
        self,
        position: Position,
        new_signal: Signal,
        current_strategy_name: str,
        new_strategy_name: str,
    ) -> bool:
        return signal_opposes_position(new_signal, position)


class SameStrategyReversePolicy(ReverseSignalPolicy):
    """같은 전략이 반대 신호를 낸 경우에만 reverse."""

    def should_reverse(
        self,
        position: Position,
        new_signal: Signal,
        current_strategy_name: str,
        new_strategy_name: str,
    ) -> bool:
        if current_strategy_name != new_strategy_name:
            return False
        return signal_opposes_position(new_signal, position)


_POLICY_REGISTRY: dict[str, type[ReverseSignalPolicy]] = {
    "ignore": IgnoreReversePolicy,
    "reverse": ReverseOnOppositePolicy,
    "same_strategy_only": SameStrategyReversePolicy,
}


def build_reverse_policy(name: str) -> ReverseSignalPolicy:
    if name not in _POLICY_REGISTRY:
        raise ValueError(
            f"Unknown reverse_signal_policy '{name}'. "
            f"Valid options: {list(_POLICY_REGISTRY.keys())}"
        )
    return _POLICY_REGISTRY[name]()
