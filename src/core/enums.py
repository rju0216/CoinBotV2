"""뼈대에서 사용하는 공용 Enum 정의."""

from enum import Enum


class SignalSide(Enum):
    """전략 generate_signal 반환값."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class PositionSide(Enum):
    """포지션 보유 방향."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"


class ExitReason(Enum):
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    FORCE_EXIT = "force_exit"
    REVERSE_SIGNAL = "reverse_signal"
    ENGINE_SHUTDOWN = "engine_shutdown"
