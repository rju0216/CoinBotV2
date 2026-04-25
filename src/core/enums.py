"""뼈대 프로토타입에서 사용하는 공용 Enum 정의."""

from enum import Enum


class SignalSide(Enum):
    """전략 generate_signal 반환값."""

    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class OrderSide(Enum):
    """거래소 주문 방향."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionSide(Enum):
    """포지션 보유 방향."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    ORPHAN = "orphan"


class ExitReason(Enum):
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    FORCE_EXIT = "force_exit"
    DRAWDOWN_LOCK = "drawdown_lock"
    DAILY_LOSS_CAP = "daily_loss_cap"
    REVERSE_SIGNAL = "reverse_signal"
    MANUAL = "manual"
    ENGINE_SHUTDOWN = "engine_shutdown"


class EventType(str, Enum):
    """EventBus 이벤트 종류 표준화."""

    BAR_CLOSED = "bar_closed"
    SIGNAL_GENERATED = "signal_generated"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    EQUITY_UPDATED = "equity_updated"
    DRAWDOWN_LOCKED = "drawdown_locked"
    DAILY_LOSS_LOCKED = "daily_loss_locked"
    ERROR = "error"
