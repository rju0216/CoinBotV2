"""평균회귀 매매 로직 (type=range) — 셋업/트리거 2단계 상태기계 (S1).

스펙 §4. 추세매매의 거울상: 위치 복귀에 베팅, 높은 승률 + 작은 손익비, 트레일링 없음.
TrendLogic 과 **코드 분리**(독립 진화, 스펙 §5.1).

진입 상태기계 (스펙 §4.1):
  [IDLE 셋업 대기]
     LONG 셋업: close < 하단밴드 AND RSI 과매도  → [ARMED_LONG]
     SHORT 셋업: close > 상단밴드 AND RSI 과매수  → [ARMED_SHORT]
  [ARMED]
     트리거(밴드 안 복귀) → 진입(다음 봉 시초가) + reset
     레짐 range 이탈      → 중단(reset)
     RSI 과매도/과매수 해제(가격 여전히 밖) → IDLE 리셋
     거리 추가 이탈(만료)  → IDLE 리셋

청산 (스펙 §4.3): TP=이동 BB 중심선 / SL=진입가∓k_range_sl·ATR /
                 적대 전환(반대방향 trend)→force_exit, 우호 전환→유지.

I-008: 진입 판정 지표(BB·RSI)를 **직전 indicator_window 봉(bounded)** 으로 계산.
RSI(Wilder RMA)·BB 가 캔들 히스토리 길이에 의존하면 백테≠라이브(H4 위반) — RegimeService
가 ATR 을 bounded 로 처리한 것과 동일 이유. bounded 슬라이스로 길이 무관 보장.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import pandas as pd

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import ExitDecision, Position, Signal
from src.strategy.helpers.sizing import risk_based_size
from src.strategy.indicators import compute_bbands, compute_rsi
from src.strategy.regime.contract import Contract, RegimeDirection


class _RangeState(Enum):
    IDLE = auto()  # 셋업 대기
    ARMED_LONG = auto()  # LONG 셋업 충족, 트리거(복귀) 대기
    ARMED_SHORT = auto()  # SHORT 셋업 충족


class RangeLogic:
    """평균회귀 정책 + 진입 상태기계.

    상태(_state)는 **flat 일 때만** 전진(plugin 이 generate_signal 에서 advance 호출).
    포지션 오픈 시 plugin 이 reset() — 단일 포지션 원칙(스펙 §5.2)상 보유 중 셋업
    추적은 행사 불가하므로 불필요.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        p = params or {}
        self.theta_range = float(p.get("theta_range", 0.6))
        self.k_range_sl = float(p.get("k_range_sl", 1.5))
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_oversold = float(p.get("rsi_oversold", 30.0))
        self.rsi_overbought = float(p.get("rsi_overbought", 70.0))
        self.bb_period = int(p.get("bb_period", 20))
        self.bb_std = float(p.get("bb_std", 2.0))
        # armed 상태 추가 이탈 만료 폭 = expiry_atr_mult·ATR (스펙 §6 스윕 대상)
        self.expiry_atr_mult = float(p.get("expiry_atr_mult", 1.0))
        self.risk_per_trade_pct = float(p.get("risk_per_trade_pct", 0.01))
        self.max_leverage = float(p.get("max_leverage", 5.0))
        # I-008: 지표 bounded 윈도우 (히스토리 길이 무관 → 백테=라이브). RSI/BB 수렴에
        # 충분(rsi_period·bb_period 의 수 배). RegimeService.window 와 맞추는 게 자연스럽다.
        self.indicator_window = int(p.get("indicator_window", 100))
        self._state = _RangeState.IDLE

    def reset(self) -> None:
        """상태기계 초기화 (진입 성공·포지션 청산 시 plugin 이 호출)."""
        self._state = _RangeState.IDLE

    @property
    def state_name(self) -> str:
        """현재 상태 라벨 (모니터링/테스트용)."""
        return self._state.name

    # ---- 지표 (I-008 bounded) ----

    def _bounded(self, candles: pd.DataFrame) -> pd.DataFrame | None:
        """직전 indicator_window 봉 슬라이스. 최소 지표 길이 미충족 시 None.

        pandas_ta 는 길이 부족 시 None 을 반환(예외 아님)하므로 호출 전에 막는다.
        """
        n = self.indicator_window
        window = candles.iloc[-n:] if len(candles) > n else candles
        if len(window) < max(self.bb_period, self.rsi_period) + 1:
            return None
        return window

    def _indicators(
        self, candles: pd.DataFrame
    ) -> tuple[float, float, float, float] | None:
        """직전 indicator_window 봉으로 (lower, mid, upper, rsi) 마지막 값 산출.

        warmup 미충족(길이 부족 또는 NaN)이면 None.
        """
        window = self._bounded(candles)
        if window is None:
            return None
        bb = compute_bbands(window, self.bb_period, self.bb_std)
        rsi = compute_rsi(window, self.rsi_period)
        lower = bb["lower"].iloc[-1]
        mid = bb["mid"].iloc[-1]
        upper = bb["upper"].iloc[-1]
        last_rsi = rsi.iloc[-1]
        if pd.isna(lower) or pd.isna(mid) or pd.isna(upper) or pd.isna(last_rsi):
            return None
        return float(lower), float(mid), float(upper), float(last_rsi)

    def _bb_mid(self, candles: pd.DataFrame) -> float | None:
        """이동 중심선(BB mid) 마지막 값 (TP 추적용). warmup 미충족·NaN 이면 None."""
        window = self._bounded(candles)
        if window is None:
            return None
        mid = compute_bbands(window, self.bb_period, self.bb_std)["mid"].iloc[-1]
        return None if pd.isna(mid) else float(mid)

    # ---- 진입 (flat 일 때 매 봉 advance) ----

    def advance(
        self, contract: Contract, candles: pd.DataFrame
    ) -> Signal | None:
        """상태기계 1봉 전진. 트리거 충족 시 Signal, 그 외 None.

        레짐이 range 이탈(type≠range 또는 conf<θ_range)이면 즉시 중단(reset).
        판정은 직전 마감 봉(candles.iloc[-1]) 기준 — 진입은 plugin 이 다음 봉 시초가로.
        """
        if not contract.is_range or contract.confidence < self.theta_range:
            self.reset()
            return None
        ind = self._indicators(candles)
        if ind is None:
            return None  # warmup 미충족
        lower, _mid, upper, rsi = ind
        close = float(candles["close"].iloc[-1])
        atr = contract.volatility

        if self._state == _RangeState.IDLE:
            if close < lower and rsi <= self.rsi_oversold:
                self._state = _RangeState.ARMED_LONG
            elif close > upper and rsi >= self.rsi_overbought:
                self._state = _RangeState.ARMED_SHORT
            return None

        if self._state == _RangeState.ARMED_LONG:
            if close >= lower:  # 트리거: 하단 밴드 안으로 복귀
                self.reset()
                return Signal(
                    side=SignalSide.LONG,
                    confidence=contract.confidence,
                    meta={"volatility": atr},
                )
            if close < lower - self.expiry_atr_mult * atr:  # 거리 만료
                self.reset()
                return None
            if rsi > self.rsi_oversold:  # 모멘텀 소강(가격은 여전히 밖) → 셋업 리셋
                self.reset()
                return None
            return None

        # ARMED_SHORT (대칭)
        if close <= upper:  # 트리거: 상단 밴드 안으로 복귀
            self.reset()
            return Signal(
                side=SignalSide.SHORT,
                confidence=contract.confidence,
                meta={"volatility": atr},
            )
        if close > upper + self.expiry_atr_mult * atr:  # 거리 만료
            self.reset()
            return None
        if rsi < self.rsi_overbought:  # 모멘텀 소강 → 셋업 리셋
            self.reset()
            return None
        return None

    def initial_stop_loss(
        self, signal: Signal, entry_price: float, volatility: float
    ) -> float:
        """SL = 진입가 ∓ k_range_sl·ATR (= "range 가 아니라 돌파였다" 거리)."""
        dist = self.k_range_sl * volatility
        if signal.side == SignalSide.LONG:
            return entry_price - dist
        return entry_price + dist

    def take_profit(self, candles: pd.DataFrame) -> float | None:
        """진입 시 TP = BB 중심선 (이후 update_take_profit 가 매 봉 추적). NaN→None."""
        return self._bb_mid(candles)

    def position_size(
        self,
        signal: Signal,
        entry_price: float,
        stop_loss: float,
        balance: float,
    ) -> float:
        """size = risk_based_size × confidence (추세와 같은 형태나 별도 메서드 — 독립 진화)."""
        base = risk_based_size(
            entry_price,
            stop_loss,
            balance,
            risk_per_trade_pct=self.risk_per_trade_pct,
            max_leverage=self.max_leverage,
        )
        return base * signal.confidence

    # ---- 보유 중 (매 봉) ----

    def update_take_profit(
        self, contract: Contract, position: Position, candles: pd.DataFrame
    ) -> float | None:
        """이동 중심선 TP 추적: 매 봉 BB mid 로 갱신 (스펙 §4.3). NaN→None(유지)."""
        return self._bb_mid(candles)

    def force_exit(
        self, contract: Contract, position: Position
    ) -> ExitDecision | None:
        """적대적 전환(반대방향 trend)→즉시 청산. 우호(같은방향 trend)·range 유지→None.

        스펙 §4.3: range-LONG 인데 하방 trend / range-SHORT 인데 상방 trend → 적대.
        우호적 전환은 무시(기존 TP/SL 유지) — range 유지 시 TP/SL 이 처리.
        """
        if not contract.is_trend:
            return None  # range 유지 또는 무신호 → 보유 (TP/SL 이 처리)
        adverse = (
            position.side == PositionSide.LONG
            and contract.direction == RegimeDirection.SHORT
        ) or (
            position.side == PositionSide.SHORT
            and contract.direction == RegimeDirection.LONG
        )
        if adverse:
            return ExitDecision(
                reason=ExitReason.REGIME_EXIT, note="adverse trend vs range position"
            )
        return None
