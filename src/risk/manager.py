"""범용 리스크 매니저.

엔진 전역 안전장치만 담당:
- 포지션 사이징 (전략이 자기 risk_per_trade_pct, max_leverage를 인자로 전달)
- 일일 손실 한도
- 드로우다운 락 (수동 unlock)
- 동시 포지션 수 한도

전략별 사이징 차등은 owner 분기가 아닌, 전략 모듈이 자기 파라미터를 넘기는 방식으로 일원화.
SL/TP 가격 산정은 전략(StrategyModule.compute_stop_loss/take_profit)으로 이관.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict[str, Any]) -> None:
        rc = config.get("risk", {}) or {}
        self.max_daily_loss_pct = float(rc.get("max_daily_loss_pct", 0.05))
        self.max_drawdown_pct = float(rc.get("max_drawdown_pct", 0.35))
        self.max_position_size = float(rc.get("max_position_size_btc", 1.0))
        self.max_concurrent_positions = int(rc.get("max_concurrent_positions", 1))

        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.initial_balance: float = 0.0

        self._dd_locked: bool = False
        self._unlock_baseline: float | None = None

    # ----- 잔액 / equity -----

    def set_initial_balance(self, balance: float) -> None:
        self.initial_balance = balance
        if self.peak_equity == 0:
            self.peak_equity = balance

    def update_equity(self, current_equity: float) -> None:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self._unlock_baseline is not None and current_equity >= self.peak_equity:
            self._unlock_baseline = None

    def reset_daily_pnl(self) -> None:
        self.daily_pnl = 0.0

    def add_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl

    # ----- 드로우다운 락 -----

    @property
    def is_drawdown_locked(self) -> bool:
        return self._dd_locked

    @property
    def unlock_baseline(self) -> float | None:
        return self._unlock_baseline

    def current_drawdown_pct(self, balance: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - balance) / self.peak_equity

    def _effective_baseline(self) -> float:
        if self._unlock_baseline is not None:
            return self._unlock_baseline
        return self.peak_equity

    def _enter_drawdown_lock(self, balance: float) -> None:
        self._dd_locked = True
        baseline = self._effective_baseline()
        dd_pct = (baseline - balance) / baseline * 100 if baseline > 0 else 0.0
        logger.warning(
            "EMERGENCY BRAKE: drawdown %.1f%% >= %.1f%% from baseline $%.2f "
            "(all-time peak $%.2f). Trading halted. Balance=$%.2f. "
            "Manual unlock_drawdown() required.",
            dd_pct,
            self.max_drawdown_pct * 100,
            baseline,
            self.peak_equity,
            balance,
        )

    def unlock_drawdown(self, current_balance: float) -> None:
        self._dd_locked = False
        self._unlock_baseline = current_balance
        logger.info(
            "Drawdown unlocked (manual). unlock_baseline=$%.2f, peak=$%.2f. "
            "New trigger: balance below $%.2f.",
            current_balance,
            self.peak_equity,
            current_balance * (1 - self.max_drawdown_pct),
        )

    # ----- 사이징 -----

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        balance: float,
        *,
        risk_per_trade_pct: float,
        max_leverage: float,
        volatility_factor: float = 1.0,
    ) -> float:
        """전략 모듈이 자기 risk 파라미터를 명시적으로 전달.

        엔진 전역 한도(max_position_size_btc)와 leverage 클램프만 RiskManager가 적용.

        volatility_factor (BP-2-2 동적 사이징, 사안 J 가):
        - factor = current_atr_pct / target_atr_pct
        - factor > 1.0 (변동성 평소 이상): size 축소 (size *= 1/factor)
        - factor ≤ 1.0 (평소 또는 잔잔): size 유지 (축소만, 증가 없음)
        - default 1.0이면 비활성과 동일
        """
        if entry_price <= 0 or balance <= 0:
            return 0.0

        risk_amount = balance * risk_per_trade_pct
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0:
            logger.warning("Invalid stop distance, returning 0 size")
            return 0.0

        raw_size = risk_amount / price_risk
        adjustment = min(1.0, 1.0 / volatility_factor) if volatility_factor > 0 else 1.0
        raw_size *= adjustment
        max_size_by_leverage = (balance * max_leverage) / entry_price
        size = min(raw_size, max_size_by_leverage, self.max_position_size)
        return max(size, 0.0)

    # ----- 진입 검증 -----

    def validate_order(self, balance: float, current_position_count: int) -> bool:
        """신규 진입 허용 여부. 전략·방향 무관, 엔진 전역 안전장치만 검사.

        동방향 진입 차단 같은 전략·포지션 단위 검사는 엔진이 담당
        (StrategyContext.is_slot_occupied 및 ReverseSignalPolicy 사용).
        """
        if self._dd_locked:
            return False

        if balance > 0 and self.daily_pnl <= -(balance * self.max_daily_loss_pct):
            logger.warning(
                "Daily loss limit reached: PnL=$%.2f, limit=-$%.2f",
                self.daily_pnl,
                balance * self.max_daily_loss_pct,
            )
            return False

        baseline = self._effective_baseline()
        if baseline > 0:
            drawdown = (baseline - balance) / baseline
            if drawdown >= self.max_drawdown_pct:
                self._enter_drawdown_lock(balance)
                return False

        if current_position_count >= self.max_concurrent_positions:
            return False

        return True
