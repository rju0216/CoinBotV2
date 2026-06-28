"""reverse 판정 공식 (opt-in). 모델의 should_reverse 에서 골라 쓴다."""

from __future__ import annotations

from src.core.enums import PositionSide, SignalSide
from src.core.types import Position, Signal


def signal_opposes_position(signal: Signal, position: Position | None) -> bool:
    """신호가 보유 포지션과 반대 방향이면 True.

    흔한 reverse 규칙: `return signal_opposes_position(new_signal, position)`.
    """
    if position is None or position.side == PositionSide.NONE:
        return False
    if signal.side == SignalSide.LONG and position.side == PositionSide.SHORT:
        return True
    if signal.side == SignalSide.SHORT and position.side == PositionSide.LONG:
        return True
    return False
