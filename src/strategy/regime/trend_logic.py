"""추세 매매 로직 (type=trend) — 순수 매매 규칙 (S1).

스펙 §3. 디스패처 플러그인(RegimeQuantStrategy)이 RegimeService 가 낸 Contract 를
받아 이 클래스에 위임한다. 이 클래스는 엔진/StrategyModule 인터페이스를 모른다 —
Contract + candles + Position 만 보고 매매 결정을 낸다 (캡슐화).

평균회귀(RangeLogic)와 **코드 분리**(독립 진화) — 사이징 형태가 같아도 공유 안 함
(스펙 §5.1). 공유는 opt-in 헬퍼(risk_based_size)·레짐 계산(RegimeService)에 한함.

핵심 (스펙 §3):
  - 진입: 관대(DD-1) — flat ∧ trend ∧ direction ∧ conf≥θ_trend → 진입.
  - 초기 SL = 진입가 ∓ k_trend_sl·ATR (standing).
  - TP 없음 = 순수 트레일링.
  - 트레일링: 동방향 trend 일 때만 유리 방향으로 SL 추적(절대 안 내림),
    trend→range 면 동결(None), 적대(반대방향 trend)는 force_exit 로 청산.
"""

from __future__ import annotations

from typing import Any

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import ExitDecision, Position, Signal
from src.strategy.helpers.sizing import risk_based_size
from src.strategy.regime.contract import Contract, RegimeDirection


class TrendLogic:
    """추세 매매 정책. 상태 없음 (트레일링은 Position.stop_loss 에 누적)."""

    def __init__(self, params: dict[str, Any]) -> None:
        p = params or {}
        self.theta_trend = float(p.get("theta_trend", 0.6))
        self.k_trend_sl = float(p.get("k_trend_sl", 2.0))
        self.k_trend_trail = float(p.get("k_trend_trail", 2.5))
        self.risk_per_trade_pct = float(p.get("risk_per_trade_pct", 0.01))
        self.max_leverage = float(p.get("max_leverage", 5.0))

    # ---- 진입 ----

    def entry_signal(self, contract: Contract) -> Signal | None:
        """관대 진입(DD-1): trend ∧ direction(long/short) ∧ conf≥θ_trend → Signal.

        confidence 게이트가 진입 1차 스위치 (스펙 §3.1). meta 에 contract.volatility
        (진입 시 ATR)를 담아 SL/사이징이 동일 ATR 로 계산하게 한다.
        """
        if not contract.is_trend:
            return None
        if contract.direction == RegimeDirection.LONG:
            side = SignalSide.LONG
        elif contract.direction == RegimeDirection.SHORT:
            side = SignalSide.SHORT
        else:
            return None
        if contract.confidence < self.theta_trend:
            return None
        return Signal(
            side=side,
            confidence=contract.confidence,
            meta={"volatility": contract.volatility},
        )

    def initial_stop_loss(
        self, signal: Signal, entry_price: float, volatility: float
    ) -> float:
        """초기 SL = 진입가 ∓ k_trend_sl·ATR (진입 시 1회). standing SL."""
        dist = self.k_trend_sl * volatility
        if signal.side == SignalSide.LONG:
            return entry_price - dist
        return entry_price + dist

    def take_profit(self) -> None:
        """추세는 TP 없음 (순수 트레일링)."""
        return None

    def position_size(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss: float,
        balance: float,
    ) -> float:
        """O-2: size = risk_based_size(진입, SL 거리, …) × confidence.

        SL 거리 = |entry - stop| 가 리스크 예산을 고정하고, confidence 가 그 위에
        곱해져 확신만큼 키운다 (사이즈만 조절·SL 불변 → 거래당 손실 비율 일정).
        """
        base = risk_based_size(
            entry_price,
            stop_loss,
            balance,
            risk_per_trade_pct=self.risk_per_trade_pct,
            max_leverage=self.max_leverage,
        )
        return base * signal.confidence

    # ---- 보유 중 (매 봉) ----

    def trailing_stop(
        self, contract: Contract, position: Position, current_price: float
    ) -> float | None:
        """트레일링 SL. 동방향 trend 일 때만 유리 방향으로 추적, 그 외 None(동결).

        - 동방향 trend: LONG SL=max(기존, price−k·ATR) / SHORT SL=min(기존, price+k·ATR).
          유리 방향만 갱신(절대 안 내림) → 안 유리하면 None(유지).
        - trend→range: None(그 자리 동결, range 동안 그 SL 이 방어선) (스펙 §3.3).
        - 적대 trend: None(동결) — 청산은 force_exit 가 담당.
        """
        if not contract.is_trend:
            return None
        same_dir = (
            position.side == PositionSide.LONG
            and contract.direction == RegimeDirection.LONG
        ) or (
            position.side == PositionSide.SHORT
            and contract.direction == RegimeDirection.SHORT
        )
        if not same_dir:
            return None
        dist = self.k_trend_trail * contract.volatility
        cur = position.stop_loss
        if position.side == PositionSide.LONG:
            new_sl = current_price - dist
            return new_sl if (cur is None or new_sl > cur) else None
        new_sl = current_price + dist
        return new_sl if (cur is None or new_sl < cur) else None

    def force_exit(
        self, contract: Contract, position: Position
    ) -> ExitDecision | None:
        """적대적(반대방향 trend) 전환 → 즉시 청산 (스펙 §3.3). 그 외 None(유지)."""
        if not contract.is_trend:
            return None
        adverse = (
            position.side == PositionSide.LONG
            and contract.direction == RegimeDirection.SHORT
        ) or (
            position.side == PositionSide.SHORT
            and contract.direction == RegimeDirection.LONG
        )
        if adverse:
            return ExitDecision(
                reason=ExitReason.REGIME_EXIT, note="adverse trend regime"
            )
        return None
