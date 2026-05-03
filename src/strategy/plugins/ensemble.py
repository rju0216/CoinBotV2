"""Ensemble 전략 플러그인 (BP-3-2).

여러 ML/DL plugin을 sub-model로 두고 신호를 결합한다. 각 sub-model의
generate_signal 결과 Signal.meta["probs"]에서 raw probabilities를 추출해
soft voting (확률 평균) 후 argmax + confidence_threshold 검사.

설계 원칙 (사안 결정):
- F=(가) Soft voting (확률 평균)
- R=(가) PPO 제외 (정책 모델, raw probs 무의미)
- S=(가) sub-model 로드 실패 → skip + warning (이후 영구 미사용)
- S-2=(가) min_models=2 default → 2개 미만 살아있으면 HOLD
- V=(가) SL/TP는 단일 모델과 동일 패턴 (ATR 기반, ensemble 자체 config)

DRY:
- sub-model의 모델 로드/추론/calibration 코드는 sub-plugin에 위임
- ensemble은 신호 결합 + min_models 보호만 담당
- BP-3-1 calibration_method 자동 적용 (sub-plugin 자체 calibrator 사용)

config 예시 (ensemble.yaml):
  strategies:
    active: ["ensemble"]
  ensemble:
    sub_models: ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer"]
    min_models: 2
    confidence_threshold: 0.55
    entry_timeframe: "15m"
    required_timeframes: ["15m", "1h", "4h"]
    atr_period: 14
    atr_sl_mult: 2.0
    reward_risk_ratio: 2.0
    risk_per_trade_pct: 0.01
    max_leverage: 5
  ml_lightgbm: { ... }    # 각 sub-model 자체 config 섹션
  ml_xgboost:  { ... }
  dl_lstm:     { ... }
  dl_transformer: { ... }
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.indicators import compute_atr
from src.strategy.registry import get_strategy_class, register_strategy

logger = logging.getLogger(__name__)


@register_strategy
class Ensemble(StrategyModule):
    name = "ensemble"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.entry_timeframe = params.get("entry_timeframe", "15m")
        self.required_timeframes = params.get(
            "required_timeframes", [self.entry_timeframe]
        )
        if self.entry_timeframe not in self.required_timeframes:
            self.required_timeframes = (
                [self.entry_timeframe] + self.required_timeframes
            )

        self._sub_model_names: list[str] = list(
            params.get("sub_models", ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer"])
        )
        self._min_models: int = int(params.get("min_models", 2))
        self._confidence_threshold: float = float(
            params.get("confidence_threshold", 0.55)
        )

        # sub-plugin 인스턴스 — lazy 로드 시점에 sub-plugin 생성자에 sub_params 주입
        self._sub_instances: dict[str, StrategyModule] = {}
        self._failed_models: set[str] = set()
        # 마지막으로 로그한 살아있는 모델 수 (degradation 알림용)
        self._last_alive_count: int | None = None

    def _ensure_sub_models(self, full_config: dict[str, Any] | None) -> None:
        """첫 호출 시 sub-plugin 인스턴스 생성. 각 sub-model의 config 섹션 필요.

        full_config는 ctx.params로 받지 못함 — params는 ensemble 자기 섹션만.
        대신 sub-model용 params는 ensemble 자체 섹션에 sub_params 인라인하거나,
        load_active_strategies 시 inject. 본 플러그인은 self.params["sub_params"]
        형태로 sub-model 섹션 dict를 받음.
        """
        if self._sub_instances or self._failed_models:
            return  # 이미 1회 시도함
        sub_params_map: dict[str, dict[str, Any]] = self.params.get(
            "sub_params", {}
        )
        for name in self._sub_model_names:
            try:
                cls = get_strategy_class(name)
                sub_params = sub_params_map.get(name)
                if sub_params is None:
                    logger.warning(
                        "Ensemble: sub-model '%s'의 sub_params 누락 — skip",
                        name,
                    )
                    self._failed_models.add(name)
                    continue
                self._sub_instances[name] = cls(sub_params)
                logger.info("Ensemble: sub-model '%s' 로드 완료", name)
            except Exception as e:
                logger.warning(
                    "Ensemble: sub-model '%s' 로드 실패 → skip (%s)",
                    name, e,
                )
                self._failed_models.add(name)
        self._log_active_count_change()

    def _log_active_count_change(self) -> None:
        alive = len(self._sub_instances) - len(
            [n for n in self._sub_instances if n in self._failed_models]
        )
        # alive는 sub_instances 중 실제 사용 가능한 수 — _failed_models는 instances에 없음
        alive = len(self._sub_instances)
        if self._last_alive_count != alive:
            level = (
                logger.warning if alive < len(self._sub_model_names)
                else logger.info
            )
            level(
                "ENSEMBLE STATUS: %d/%d models active (skipped: %s)",
                alive,
                len(self._sub_model_names),
                sorted(self._failed_models) or "none",
            )
            self._last_alive_count = alive

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        self._ensure_sub_models(None)

        # min_models 보호
        active_names = [
            n for n in self._sub_model_names
            if n in self._sub_instances and n not in self._failed_models
        ]
        if len(active_names) < self._min_models:
            logger.error(
                "Ensemble: active models %d < min_models %d → HOLD",
                len(active_names), self._min_models,
            )
            return Signal(side=SignalSide.HOLD)

        # 각 sub-plugin에서 raw probs 수집
        probs_list: list[np.ndarray] = []
        contributors: list[str] = []
        runtime_failed: list[str] = []
        for name in active_names:
            sub = self._sub_instances[name]
            try:
                sub_signal = sub.generate_signal(ctx)
                probs = sub_signal.meta.get("probs") if sub_signal.meta else None
                if probs is None or len(probs) == 0:
                    # early HOLD (features 부족 등) — 이번 봉만 skip, 영구 disable 아님
                    continue
                probs_arr = np.asarray(probs, dtype=np.float64)
                if probs_arr.shape[0] != 3:
                    continue
                probs_list.append(probs_arr)
                contributors.append(name)
            except Exception as e:
                logger.warning(
                    "Ensemble: sub-model '%s' 추론 실패 → 영구 skip (%s)",
                    name, e,
                )
                self._failed_models.add(name)
                runtime_failed.append(name)
        if runtime_failed:
            self._log_active_count_change()

        # 활성 sub-model 모두 이번 봉 추론 실패 시 HOLD
        if len(probs_list) == 0:
            return Signal(side=SignalSide.HOLD)
        if len(probs_list) < self._min_models:
            # 봉별 min_models 미달 (영구 skip + 이번 봉 추론 실패 조합)
            return Signal(side=SignalSide.HOLD)

        # Soft voting: 확률 평균
        avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        # 합계 1로 재정규화 (calibrator clip + 평균 후 미세 편차 방지)
        avg_probs = avg_probs / max(avg_probs.sum(), 1e-12)
        pred_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_class])
        meta = {
            "probs": [round(float(p), 4) for p in avg_probs],
            "contributors": contributors,
        }

        if pred_class == 2 and confidence >= self._confidence_threshold:
            return Signal(side=SignalSide.LONG, confidence=confidence, meta=meta)
        if pred_class == 0 and confidence >= self._confidence_threshold:
            return Signal(side=SignalSide.SHORT, confidence=confidence, meta=meta)
        return Signal(side=SignalSide.HOLD, confidence=confidence, meta=meta)

    # ---- SL/TP — 단일 모델과 동일 패턴 (사안 V 가) ----

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

    # ---- 외부 조회 (테스트/모니터링용) ----

    def get_active_models_count(self) -> int:
        return len(self._sub_instances) - sum(
            1 for n in self._sub_instances if n in self._failed_models
        )

    def get_failed_models(self) -> list[str]:
        return sorted(self._failed_models)
