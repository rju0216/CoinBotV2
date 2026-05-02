"""XGBoost 기반 독립 전략 플러그인.

3-class 분류 모델(SHORT/HOLD/LONG)의 확률 출력을 매매 신호로 변환.
SL/TP는 ATR 기반으로 모델과 독립적으로 산정.

LightGBM 플러그인(ml_lightgbm.py)과 구조 동일 — XGBoost API만 차이:
- 모델 로드: xgb.Booster(model_file=...)
- 추론: xgb.DMatrix(row) wrapping 필요 (numpy 직접 입력 불가)

config 예시:
  strategies:
    active: ["ml_xgboost"]
  ml_xgboost:
    risk_per_trade_pct: 0.01
    max_leverage: 5
    model_path: "models/xgboost/latest"
    confidence_threshold: 0.55
    entry_timeframe: "15m"
    required_timeframes: ["15m", "1h", "4h"]
    atr_period: 14
    atr_sl_mult: 2.0
    reward_risk_ratio: 2.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.features import get_features_for_ctx
from src.strategy.indicators import compute_atr
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class MLXGBoost(StrategyModule):
    name = "ml_xgboost"
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
        self._feature_names: list[str] = []
        self._model_path = params.get("model_path", "models/xgboost/latest")
        self._confidence_threshold = float(
            params.get("confidence_threshold", 0.55)
        )

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
        """모델 lazy 로드. 첫 generate_signal 호출 시 1회 실행."""
        if self._model is not None:
            return
        import xgboost as xgb

        model_dir = self._resolve_model_dir()
        self._model = xgb.Booster()
        self._model.load_model(str(model_dir / "model.json"))

        with open(model_dir / "feature_names.json") as f:
            self._feature_names = json.load(f)

        logger.info(
            "XGBoost 모델 로드 완료: %s (%d 피처)",
            model_dir,
            len(self._feature_names),
        )

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        import xgboost as xgb

        self._ensure_model()

        features = get_features_for_ctx(ctx, self.entry_timeframe)
        if len(features.dropna()) < 1:
            return Signal(side=SignalSide.HOLD)

        available = [c for c in self._feature_names if c in features.columns]
        if len(available) != len(self._feature_names):
            logger.warning(
                "피처 불일치: 기대 %d, 가용 %d",
                len(self._feature_names),
                len(available),
            )
            return Signal(side=SignalSide.HOLD)

        row = features.iloc[-1][self._feature_names].values.reshape(1, -1)
        if np.any(np.isnan(row)):
            return Signal(side=SignalSide.HOLD)

        # XGBoost는 numpy 직접 입력 불가 — DMatrix wrapping 필요
        dmatrix = xgb.DMatrix(row, feature_names=self._feature_names)
        probs = self._model.predict(dmatrix)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        meta = {"probs": [round(float(p), 4) for p in probs]}

        if pred_class == 2 and confidence >= self._confidence_threshold:
            return Signal(side=SignalSide.LONG, confidence=confidence, meta=meta)
        if pred_class == 0 and confidence >= self._confidence_threshold:
            return Signal(side=SignalSide.SHORT, confidence=confidence, meta=meta)
        return Signal(side=SignalSide.HOLD, confidence=confidence, meta=meta)

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        df = ctx.candles[self.entry_timeframe]
        atr_period = int(self.params.get("atr_period", 14))
        atr_mult = float(self.params.get("atr_sl_mult", 2.0))

        if len(df) < atr_period + 1:
            offset = ctx.current_price * 0.005
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
        rr = float(self.params.get("reward_risk_ratio", 2.0))
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr
