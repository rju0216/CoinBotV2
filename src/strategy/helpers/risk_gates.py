"""진입 게이트(리스크) 공식 (opt-in). 모델의 allow_entry 에서 골라 쓴다.

ctx.account (AccountState) 를 받아 진입을 막을지 판단한다. 임계값은 모델의
params 에서 가져오는 것이 보통이다.
"""

from __future__ import annotations

from src.core.types import AccountState


def drawdown_exceeded(account: AccountState, max_drawdown_pct: float) -> bool:
    """peak 대비 낙폭이 한도 이상이면 True (진입 차단 신호)."""
    return account.drawdown_pct >= max_drawdown_pct


def daily_loss_exceeded(account: AccountState, max_daily_loss_pct: float) -> bool:
    """당일 손실이 잔액 대비 한도 이상이면 True (진입 차단 신호)."""
    if account.balance <= 0:
        return False
    return account.daily_pnl <= -(account.balance * max_daily_loss_pct)
