"""StrategyModule 추상 클래스.

엔진은 이 인터페이스만을 통해 전략(모델)과 상호작용한다. **모든 거래 정책**
— 진입 신호·포지션 사이징·SL/TP·reverse 여부·진입 게이트(리스크) — 은 전략이
소유하며, 엔진은 그 결정을 받아 *집행*만 한다. 엔진에는 정책 기본값이 없다.

필수 구현 6개(generate_signal / compute_stop_loss / compute_take_profit /
compute_position_size / should_reverse / allow_entry)와 선택 훅 5개를 정의.

사이징 공식·피처·리스크 게이트는 모델이 자체 구현한다 — 엔진은 이 인터페이스만
호출하며 정책 기본값이나 공용 헬퍼를 강요하지 않는다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.core.types import ExitDecision, Position, Signal, StrategyContext

logger = logging.getLogger(__name__)


class StrategyModule(ABC):
    # ---- 클래스 속성 (서브클래스가 반드시 선언) ----
    name: str = ""
    entry_timeframe: str = ""
    required_timeframes: list[str] = []
    # 같은 캔들에 SL·TP 가 동시에 닿았을 때 체결 우선순위 — "sl_first" | "tp_first".
    # 백테 체결 시뮬(check_candle_sl_tp)에서 동시 도달 판정에 사용.
    # 기본값 없음(정책이므로 모델이 선언) — registry 가 유효값 강제.
    sl_tp_fill_priority: str

    def __init__(self, params: dict[str, Any]) -> None:
        """
        Args:
            params: config[self.name] 섹션의 dict.
                    엔진이 registry.load_active_strategies에서 주입.
        """
        self.params = params or {}

    # ---- 필수 구현: 거래 결정(정책) ----

    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> Signal:
        """진입/reverse 후보 신호 생성. 슬롯이 비었을 때는 진입 평가로, 슬롯이
        찼을 때는 보유 전략의 should_reverse 판단용 후보 신호로 엔진이 호출한다."""

    @abstractmethod
    def compute_stop_loss(
        self, ctx: StrategyContext, signal: Signal
    ) -> float | None:
        """진입 직전 SL 가격 산정.

        None 반환 시 standing SL 미설정 — check_candle_sl_tp 의 SL 체크 skip.
        청산을 전적으로 should_force_exit(명령형)로 가져가는 모델용. SL 가격을
        설정하면 엔진이 그 값을 캔들 high/low 교차로 감지해 청산한다(선언형).
        """

    @abstractmethod
    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float | None
    ) -> float | None:
        """진입 직전 TP 가격 산정. None 반환 시 TP 미설정 (trailing SL 등)."""

    @abstractmethod
    def compute_position_size(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float | None
    ) -> float:
        """포지션 크기(계약/BTC) 산정 — 사이징 공식·레버리지·변동성 조정 전부 모델 소유.

        stop_loss 가 None 이면(standing SL 없음) SL 거리 기반 공식은 쓸 수 없으므로
        모델이 다른 사이징 방식을 택해야 한다. 0 이하 반환 시 엔진은 진입을 건너뛴다.
        """

    @abstractmethod
    def should_reverse(
        self, ctx: StrategyContext, position: Position, new_signal: Signal
    ) -> bool:
        """슬롯이 찬 상태에서 actionable 신호가 났을 때, 보유 포지션을 청산하고
        새 신호로 reverse 진입할지 여부. position 은 현재 보유 포지션,
        new_signal 은 이번 봉의 신호. True 면 엔진이 청산 후 재진입을 시도한다.
        """

    @abstractmethod
    def allow_entry(self, ctx: StrategyContext) -> bool:
        """신규 진입 허용 여부(진입 게이트). DD락·일일손실 한도·기타 리스크 정책을
        모델이 여기서 결정한다. ctx.account(잔액·equity·peak·daily_pnl·dd)를 참조.
        False 면 엔진은 진입을 건너뛴다.
        """

    # ---- 선택 훅 (기본 no-op — '행동 없음'이며 정책 아님) ----

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
        """명령형 청산 판단 (timeout, regime 변화 등). None=유지.
        SL/TP 를 None 으로 두고 청산을 전적으로 여기서 결정할 수도 있다.
        """
        return None

    def on_position_opened(self, position: Position) -> None:
        return None

    def on_position_closed(self, position: Position, pnl: float) -> None:
        return None
