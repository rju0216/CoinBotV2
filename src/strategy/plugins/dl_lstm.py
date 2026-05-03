"""LSTM 기반 독립 전략 플러그인.

3-class 분류 모델(SHORT/HOLD/LONG)의 확률 출력을 매매 신호로 변환.
SL/TP는 ATR 기반으로 모델과 독립적으로 산정.

ML 플러그인(B-1)과 다른 점:
- 시퀀스 입력: lookback개 봉의 피처 윈도우를 모델에 넘김
- StandardScaler로 피처 정규화 (joblib로 직렬화된 scaler 로드)
- LSTMClassifier 인스턴스 + state_dict 로드 (CPU 추론)

config 예시:
  strategies:
    active: ["dl_lstm"]
  dl_lstm:
    risk_per_trade_pct: 0.01
    max_leverage: 5
    model_path: "models/lstm/latest"
    confidence_threshold: 0.55
    entry_timeframe: "15m"
    required_timeframes: ["15m", "1h", "4h"]
    lookback: 60
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
class DLLSTM(StrategyModule):
    name = "dl_lstm"
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
        self._model_path = params.get("model_path", "models/lstm/latest")
        self._confidence_threshold = float(
            params.get("confidence_threshold", 0.55)
        )
        # Phase E-2-3 Step 2 (I-B009)
        self._calibration_method = str(
            params.get("calibration_method", "none")
        ).lower()
        self._calibrator = None

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
        """모델·scaler·메타 lazy 로드. 첫 generate_signal 호출 시 1회 실행."""
        if self._model is not None:
            return

        import joblib
        import torch

        from src.ml.models import LSTMClassifier

        model_dir = self._resolve_model_dir()

        with open(model_dir / "feature_names.json") as f:
            self._feature_names = json.load(f)

        with open(model_dir / "train_meta.json") as f:
            meta = json.load(f)
        arch = meta.get("model_arch", {})

        model = LSTMClassifier(
            n_features=int(arch["n_features"]),
            hidden_size=int(arch.get("hidden_size", 64)),
            num_layers=int(arch.get("num_layers", 1)),
            dropout=float(arch.get("dropout", 0.3)),
        )
        state_dict = torch.load(
            str(model_dir / "model.pth"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model

        self._scaler = joblib.load(str(model_dir / "scaler.joblib"))

        # I-B009 calibrator lazy 로드
        if self._calibration_method in ("platt", "isotonic"):
            cal_path = model_dir / f"calibrator_{self._calibration_method}.joblib"
            if cal_path.exists():
                self._calibrator = joblib.load(cal_path)
                logger.info("Calibrator 로드 완료: %s (%s)", cal_path, self._calibration_method)
            else:
                logger.warning("Calibrator 파일 없음: %s — raw 확률 사용", cal_path)

        logger.info(
            "LSTM 모델 로드 완료: %s (%d 피처, lookback=%d)",
            model_dir,
            len(self._feature_names),
            self._lookback,
        )

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        import torch

        from src.ml.sequence_utils import make_sequence_from_recent

        self._ensure_model()

        features = get_features_for_ctx(ctx, self.entry_timeframe)
        if len(features.dropna()) < self._lookback:
            return Signal(side=SignalSide.HOLD)

        available = [c for c in self._feature_names if c in features.columns]
        if len(available) != len(self._feature_names):
            logger.warning(
                "피처 불일치: 기대 %d, 가용 %d",
                len(self._feature_names),
                len(available),
            )
            return Signal(side=SignalSide.HOLD)

        # 학습과 동일 컬럼 순서로 정렬
        features_sub = features[self._feature_names]
        seq = make_sequence_from_recent(features_sub, self._lookback)
        if seq is None:
            return Signal(side=SignalSide.HOLD)

        # Scaler transform: (1, L, F) → reshape (L, F) → transform → reshape back
        seq_2d = seq.reshape(-1, seq.shape[-1])
        seq_scaled = self._scaler.transform(seq_2d).reshape(seq.shape)

        with torch.no_grad():
            x = torch.from_numpy(seq_scaled).float()
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        # I-B009 calibrator 적용
        if self._calibrator is not None:
            probs = self._calibrator.transform(probs.reshape(1, -1))[0]
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
