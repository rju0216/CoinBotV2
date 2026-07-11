"""기준선 모델 (MasterPlan §11 컴포넌트 E, 설계 §10 관문1 "무작위·순진모델").

fit/predict/predict_proba 계약을 구현 — 이게 곧 harness(Step 0.5)·2층 모델(Phase 3)이
쓸 **모델 콜러블 인터페이스**다. baseline 은 X(피처)를 무시하고 y 분포만 쓰며,
0.5 end-to-end 시연의 첫 '모델' 역할을 한다.

- PriorBaseline: train 클래스 사전확률(상수). 로그손실이 넘어야 할 가장 강한 순진 기준.
- UniformBaseline: 1/K 균등(무정보). 무작위 기준.
둘 다 결정적(RNG 없음 — 재현성). 확률적 stratified-random 은 시드 필요 → 후순위.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaselineModel(ABC):
    """fit(X, y) / predict_proba(X) / predict(X) 계약. classes_ 는 정렬 고정."""

    classes_: list

    @abstractmethod
    def fit(self, X, y) -> "BaselineModel":
        ...

    @abstractmethod
    def predict_proba(self, X) -> pd.DataFrame:
        ...

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        cols = np.asarray(proba.columns)
        return cols[proba.to_numpy().argmax(axis=1)]

    @staticmethod
    def _index(X) -> pd.Index:
        return X.index if hasattr(X, "index") else pd.RangeIndex(len(X))


class PriorBaseline(BaselineModel):
    """train 클래스 빈도를 상수 확률로. predict = 최빈 클래스."""

    def fit(self, X, y) -> "PriorBaseline":
        vc = pd.Series(y).value_counts(normalize=True)
        self.classes_ = sorted(vc.index)
        self.priors_ = np.array([float(vc.get(c, 0.0)) for c in self.classes_])
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        idx = self._index(X)
        data = np.tile(self.priors_, (len(idx), 1))
        return pd.DataFrame(data, index=idx, columns=self.classes_)


class UniformBaseline(BaselineModel):
    """1/K 균등 확률 (무정보 기준)."""

    def fit(self, X, y) -> "UniformBaseline":
        self.classes_ = sorted(pd.Series(y).unique())
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        idx = self._index(X)
        k = len(self.classes_)
        data = np.full((len(idx), k), 1.0 / k)
        return pd.DataFrame(data, index=idx, columns=self.classes_)
