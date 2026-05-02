"""PPO (강화학습) 기반 독립 전략 플러그인.

ML/DL 플러그인과 본질적으로 다른 접근:
- 분류 모델이 아닌 정책(policy) — 직접 액션(HOLD/LONG/SHORT)을 출력
- `should_force_exit` hook으로 직접 청산 결정 (§4.7)
- SL/TP는 안전장치로만 (atr_sl_mult=5.0, RR=10.0)

config 예시:
  strategies:
    active: ["rl_ppo"]
  rl_ppo:
    risk_per_trade_pct: 0.01
    max_leverage: 5
    model_path: "models/ppo/latest"
    entry_timeframe: "15m"
    required_timeframes: ["15m", "1h", "4h"]
    lookback: 60
    atr_period: 14
    atr_sl_mult: 5.0
    reward_risk_ratio: 10.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.core.enums import ExitReason, PositionSide, SignalSide
from src.core.types import ExitDecision, Position, Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.features import get_features_for_ctx
from src.strategy.indicators import compute_atr
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class RLPPO(StrategyModule):
    name = "rl_ppo"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.entry_timeframe = params.get("entry_timeframe", "15m")
        self.required_timeframes = params.get(
            "required_timeframes", [self.entry_timeframe]
        )
        if self.entry_timeframe not in self.required_timeframes:
            self.required_timeframes = [self.entry_timeframe] + self.required_timeframes

        self._model = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._lookback = int(params.get("lookback", 60))
        self._model_path = params.get("model_path", "models/ppo/latest")

    def _resolve_model_dir(self) -> Path:
        """model_path → 실제 모델 디렉토리 해석. latest.json 간접 참조 지원."""
        model_dir = Path(self._model_path)

        latest_json = model_dir.parent / "latest.json"
        if model_dir.name == "latest" and latest_json.exists():
            with open(latest_json) as f:
                model_dir = Path(json.load(f)["path"])
        elif (model_dir / "latest.json").exists():
            with open(model_dir / "latest.json") as f:
                model_dir = Path(json.load(f)["path"])

        return model_dir

    def _ensure_model(self) -> None:
        """PPO 정책·scaler·메타 lazy 로드."""
        if self._model is not None:
            return

        import joblib
        from stable_baselines3 import PPO

        model_dir = self._resolve_model_dir()

        with open(model_dir / "feature_names.json") as f:
            self._feature_names = json.load(f)

        # SB3는 .zip 확장자 자동 추가/인식
        self._model = PPO.load(str(model_dir / "model"), device="cpu")
        self._scaler = joblib.load(str(model_dir / "scaler.joblib"))

        logger.info(
            "PPO 모델 로드 완료: %s (%d 피처, lookback=%d)",
            model_dir,
            len(self._feature_names),
            self._lookback,
        )

    def _build_observation(self, ctx: StrategyContext) -> np.ndarray | None:
        """ctx.candles → (lookback × n_features,) 1D obs. 부족하거나 NaN이면 None."""
        features = get_features_for_ctx(ctx, self.entry_timeframe)
        if len(features.dropna()) < self._lookback:
            return None

        available = [c for c in self._feature_names if c in features.columns]
        if len(available) != len(self._feature_names):
            logger.warning(
                "피처 불일치: 기대 %d, 가용 %d",
                len(self._feature_names),
                len(available),
            )
            return None

        features_sub = features[self._feature_names]
        seq = features_sub.iloc[-self._lookback:].values  # (L, F)
        if np.any(np.isnan(seq)):
            return None

        seq_scaled = self._scaler.transform(seq)  # (L, F)
        return seq_scaled.flatten().astype(np.float32)

    def _predict_action(self, obs: np.ndarray) -> int:
        """deterministic action 예측 (0=HOLD, 1=LONG, 2=SHORT)."""
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        self._ensure_model()

        obs = self._build_observation(ctx)
        if obs is None:
            return Signal(side=SignalSide.HOLD)

        action = self._predict_action(obs)

        if action == 1:
            return Signal(side=SignalSide.LONG, confidence=1.0)
        if action == 2:
            return Signal(side=SignalSide.SHORT, confidence=1.0)
        return Signal(side=SignalSide.HOLD)

    def should_force_exit(
        self, ctx: StrategyContext, position: Position
    ) -> ExitDecision | None:
        """RL 정책이 현재 포지션 방향과 다른 액션을 선호하면 청산 (§4.7).

        - 현재 LONG + action ≠ 1 (LONG) → 청산
        - 현재 SHORT + action ≠ 2 (SHORT) → 청산
        - 같은 방향 유지 → None (유지)
        """
        self._ensure_model()

        obs = self._build_observation(ctx)
        if obs is None:
            return None  # 데이터 부족 시 보수적으로 유지

        action = self._predict_action(obs)

        if position.side == PositionSide.LONG and action != 1:
            return ExitDecision(
                reason=ExitReason.FORCE_EXIT,
                note=f"PPO action={action} (LONG 유지 안 함)",
            )
        if position.side == PositionSide.SHORT and action != 2:
            return ExitDecision(
                reason=ExitReason.FORCE_EXIT,
                note=f"PPO action={action} (SHORT 유지 안 함)",
            )
        return None

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        """SL은 안전장치로만 (atr_sl_mult=5.0, 매우 넓게)."""
        df = ctx.candles[self.entry_timeframe]
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_sl_mult", 5.0))

        if len(df) < atr_period + 1:
            offset = ctx.current_price * 0.01
            if signal.side == SignalSide.LONG:
                return ctx.current_price - offset
            return ctx.current_price + offset

        atr_val = float(compute_atr(df, atr_period).iloc[-1])
        if signal.side == SignalSide.LONG:
            return ctx.current_price - atr_val * atr_mult
        return ctx.current_price + atr_val * atr_mult

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        """TP도 안전장치로만 (RR=10.0)."""
        rr = float(self.params.get("reward_risk_ratio", 10.0))
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr
