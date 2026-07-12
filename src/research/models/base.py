"""학습 모델 계약 base (Phase 3, MasterPlan §2 모델 콜러블).

``ProbaModel`` = harness 가 요구하는 duck-typed 계약(fit/predict_proba/predict/classes_).
``baselines.BaselineModel`` 과 **동일한 계약**이나, baseline 은 "참조·null 족", 이쪽은
"학습 모델 족"으로 나눈다(MasterPlan §2: baseline 이 참조 구현). harness 는 상속을 요구하지
않고 duck-typing 하므로, 여기 base 는 공유 ``predict``(argmax) 편의만 제공한다 —
Phase 0 커밋 코드(baselines)를 안 건드리려 argmax 소량 중복을 감수(D-021).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ProbaModel(ABC):
    """fit(X, y) / predict_proba(X)→DataFrame[classes] / predict(X) / classes_ 계약.

    - ``predict_proba`` 반환 = index=X.index, columns=``classes_``(정렬 고정) DataFrame.
    - ``predict`` = proba argmax(기본 구현). 하위 클래스가 override 가능.
    - ``classes_`` = fit 에서 확정하는 정렬된 클래스 목록.
    """

    classes_: list

    @abstractmethod
    def fit(self, X, y) -> "ProbaModel":
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
