"""RegimeService — 추론 시 Contract 생성 (D4-2d).

매 봉 causal 캔들(직전 마감 봉까지)을 받아 Contract 를 낸다:
  ts → 해당 모델 로드 → feature(causal)+zscore → warmup None
  → **직전 W봉 sliding-window filter**(forward-only) → γ 집계 contract* + ATR.

설계 정정(D4-2d): 백테=라이브 일관(H4)을 위해 incremental 상태 carry 대신
**sliding-window**. 둘 다 "현재 봉 직전 W봉을 filter" → 같은 causal 데이터 →
같은 γ. (incremental 은 라이브 재시작 시 validity 시작부터 재현 불가 → 불일치.)
contract 는 봉당 timestamp 캐시 (한 봉에 여러 번 호출돼도 1회 계산).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.indicators import compute_atr
from src.strategy.regime.artifact import RegimeModel
from src.strategy.regime.contract import Contract
from src.strategy.regime.features import compute_raw_features


class RegimeService:
    """walk-forward 모델 집합을 받아 ts 에 맞는 모델로 Contract 산출."""

    def __init__(
        self,
        models: list[RegimeModel],
        window: int = 100,
        atr_period: int = 24,
    ) -> None:
        if not models:
            raise ValueError("models 가 비어 있음")
        self.models = sorted(models, key=lambda m: self._naive(m.valid_period[0]))
        self.window = window
        self.atr_period = atr_period
        self._last_ts = None
        self._cached: Contract | None = None

    @staticmethod
    def _naive(t) -> pd.Timestamp:
        """tz 무관 비교용으로 tz-naive 화 (모든 ts 는 UTC wall-clock 가정)."""
        t = pd.Timestamp(t)
        return t.tz_localize(None) if t.tzinfo is not None else t

    def _model_for(self, ts) -> RegimeModel | None:
        t = self._naive(ts)
        for m in self.models:
            if self._naive(m.valid_period[0]) <= t < self._naive(m.valid_period[1]):
                return m
        return None

    def get_contract(self, candles: pd.DataFrame) -> Contract | None:
        """causal 1h 캔들(직전 마감 봉까지) → Contract 또는 None(warmup/모델없음).

        candles.index[-1] = 현재 평가 봉. 같은 ts 반복 호출은 캐시 반환.
        """
        if candles is None or len(candles) == 0:
            return None
        ts = candles.index[-1]
        if ts == self._last_ts:
            return self._cached

        contract = self._compute(candles, ts)
        self._last_ts = ts
        self._cached = contract
        return contract

    def _compute(self, candles: pd.DataFrame, ts) -> Contract | None:
        model = self._model_for(ts)
        if model is None:
            return None  # 이 ts 를 커버하는 모델 없음 → 무신호

        # 성능·일관: 직전 (window + warmup) 봉만으로 feature·filter·ATR 계산.
        # feature 는 causal 이라 마지막 window 행은 전체 계산과 동일 → O(T²)→O(T).
        # bounded ATR 도 같은 슬라이스라 백테=라이브 동일(H4). (전체 캔들 RMA-ATR 은
        # 히스토리 길이 의존 → 백테/라이브 불일치.)
        need = self.window + model.feature_config.warmup
        recent = candles.iloc[-need:] if len(candles) > need else candles

        raw = compute_raw_features(recent, model.feature_config)
        z = model.zscore.transform(raw)
        valid = z.dropna()
        if len(valid) == 0:
            return None  # warmup 미충족
        X = valid.to_numpy()
        Xw = X[-self.window:] if len(X) > self.window else X

        gamma = model.hmm.filter(Xw)[-1]  # 현재 봉의 filtered posterior (forward-only)

        atr_last = compute_atr(recent, self.atr_period).iloc[-1]
        vol = float(atr_last) if not pd.isna(atr_last) else 0.0

        return model.mapping.aggregate(gamma, vol)

    def reset(self) -> None:
        """캐시 초기화 (라이브 재시작 등). sliding-window 라 상태 carry 는 없음."""
        self._last_ts = None
        self._cached = None
