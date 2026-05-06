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
from datetime import datetime
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
        # I-BL007 Phase 1+3-C: sub-plugin 실패 사유 + 진단 정보 집계 (이번 봉 한정)
        # 실패 case: {name: {"reason": ..., "nan_by_tf": ..., "available_rows": ...}}
        # 정상 case의 진단 (rows_dropped, used_row_ts)는 sub-plugin meta에 들어있음
        unavailable_subs: dict[str, dict] = {}
        # 정상 contributors의 sub meta — ensemble propagate용 (모든 sub 동일 features
        # 사용하므로 dropped/used_row_ts는 동일. 첫 번째만 추출)
        contrib_diag: dict | None = None
        for name in active_names:
            sub = self._sub_instances[name]
            try:
                sub_signal = sub.generate_signal(ctx)
                sub_meta = sub_signal.meta or {}
                probs = sub_meta.get("probs")
                if probs is None or len(probs) == 0:
                    # early HOLD — 진단 정보 집계
                    unavailable_subs[name] = {
                        "reason": sub_meta.get("fail_reason", "unknown"),
                        "nan_by_tf": sub_meta.get("nan_by_tf"),
                        "available_rows": sub_meta.get("available_rows"),
                        "required_lookback": sub_meta.get("required_lookback"),
                    }
                    continue
                probs_arr = np.asarray(probs, dtype=np.float64)
                if probs_arr.shape[0] != 3:
                    unavailable_subs[name] = {"reason": "probs_shape_invalid"}
                    continue
                probs_list.append(probs_arr)
                contributors.append(name)
                # 정상 contributors의 진단 정보 (첫 번째만 사용 — 모두 동일 features)
                if contrib_diag is None:
                    contrib_diag = {
                        "gap_to_latest": sub_meta.get("gap_to_latest", 0),
                        "used_row_ts": sub_meta.get("used_row_ts"),
                    }
            except Exception as e:
                logger.warning(
                    "Ensemble: sub-model '%s' 추론 실패 → 영구 skip (%s)",
                    name, e,
                )
                self._failed_models.add(name)
                runtime_failed.append(name)
                unavailable_subs[name] = {"reason": f"runtime_exception:{type(e).__name__}"}
        if runtime_failed:
            self._log_active_count_change()

        # 활성 sub-model 모두 이번 봉 추론 실패 시 HOLD
        if len(probs_list) == 0:
            return Signal(
                side=SignalSide.HOLD,
                meta={"unavailable_subs": unavailable_subs} if unavailable_subs else {},
            )
        if len(probs_list) < self._min_models:
            # 봉별 min_models 미달 (영구 skip + 이번 봉 추론 실패 조합)
            return Signal(
                side=SignalSide.HOLD,
                meta={
                    "unavailable_subs": unavailable_subs,
                    "contributors": contributors,
                },
            )

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
        # I-BL007 Phase 3-C: 정상 contributors의 진단 정보 propagate (gap > 0 시만 의미)
        if contrib_diag and contrib_diag.get("gap_to_latest", 0) > 0:
            meta["gap_to_latest"] = contrib_diag.get("gap_to_latest")
            meta["used_row_ts"] = contrib_diag.get("used_row_ts")

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

    def get_sub_instances(self) -> dict[str, StrategyModule]:
        """Sub-plugin 인스턴스 dict 반환. 미초기화 시 lazy init 호출."""
        if not self._sub_instances and not self._failed_models:
            self._ensure_sub_models(None)
        return dict(self._sub_instances)

    # ---- BL-2 OOS warm-up hook (I-BL003 fix) ----

    def extract_train_meta(self) -> tuple[datetime | None, float | None]:
        """Sub-plugin들의 train_meta 집계 — cutoff=min(보수적), acc=mean.

        ensemble 자체엔 model_path 없음. paper 운영 중 record_prediction되는
        buffer key는 "ensemble"이므로 ensemble buffer 사전 채움 baseline 산출.
        """
        sub_instances = self.get_sub_instances()
        cutoffs: list[datetime] = []
        accs: list[float] = []
        for sub in sub_instances.values():
            sub_cutoff, sub_acc = sub.extract_train_meta()
            if sub_cutoff is not None:
                cutoffs.append(sub_cutoff)
            if sub_acc is not None:
                accs.append(sub_acc)
        if not cutoffs:
            return None, None
        cutoff_dt = min(cutoffs)
        learned_acc = sum(accs) / len(accs) if accs else None
        return cutoff_dt, learned_acc
