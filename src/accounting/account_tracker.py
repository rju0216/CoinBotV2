"""계좌 상태 추적기 (tracking 전용 — 정책 enforcement 없음).

equity/peak/일일 PnL 을 추적·계측한다. 진입 게이트·사이징·DD/일일손실 *정책*은
전략(StrategyModule.allow_entry / compute_position_size)이 소유한다 — 이 클래스는
그 판단에 필요한 숫자(AccountState)를 제공하는 계측기일 뿐이다.

재사용 가능한 리스크 게이트·사이징 공식은 src/strategy/helpers/ 에 opt-in 으로 있다.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


class AccountTracker:
    def __init__(self) -> None:
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.initial_balance: float = 0.0
        # 자정 경계 인식용 마지막 reset date (UTC)
        self.last_reset_date: date | None = None

    # ----- 잔액 / equity -----

    def set_initial_balance(self, balance: float) -> None:
        self.initial_balance = balance
        if self.peak_equity == 0:
            self.peak_equity = balance

    def update_equity(self, current_equity: float) -> None:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def add_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl

    def reset_daily_pnl(self) -> None:
        self.daily_pnl = 0.0

    def maybe_reset_for_new_day(self, now: datetime) -> bool:
        """UTC date 경계 인식 시 daily_pnl 자동 reset.

        엔진 봉 마감 entry(evaluate_strategies_on_bar)에서 매 봉 호출.
        백테/페이퍼/라이브 일관 적용.

        - 첫 호출: base date 설정만 (no-op)
        - 같은 UTC date: no-op
        - 다른 UTC date: daily_pnl reset

        Returns: True if reset 수행, 아니면 False.
        """
        current_date = (
            now.astimezone(timezone.utc).date()
            if now.tzinfo is not None else now.date()
        )
        if self.last_reset_date is None:
            self.last_reset_date = current_date
            return False
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            return True
        return False

    def drawdown_pct(self, equity: float) -> float:
        """peak 대비 현재 equity 의 낙폭 비율(>=0). 정책 판단은 모델 몫."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - equity) / self.peak_equity)
