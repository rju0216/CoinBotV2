"""트리 벤치 (Phase 3, 설계 §6 "트리 벤치마크").

배리어결과 단일태스크 GBDT 하한선 — MLP 가 넘어야 정당(설계 §6). lightgbm 단일 채택
(D-022, xgboost 보류). 창·하이퍼는 R1 placeholder(D-023), R3 에서 튜닝.

계약: ``ProbaModel``(fit/predict_proba→DataFrame[classes]/classes_). harness 가 폴드
train-only 로 정규화·NaN 드롭한 X 를 받는다(모델은 폴드 무지, D-019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.research.models.base import ProbaModel

# R1 placeholder 하이퍼 (D-023, R3 R2/R3 튜닝). 작은 폴드 과적합 억제로 보수적.
_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


class TreeBench(ProbaModel):
    """lightgbm 다중분류 벤치. 단일클래스 train 폴드는 상수확률로 폴백(fit 크래시 회피)."""

    def __init__(self, seed: int = 0, params: dict | None = None) -> None:
        self.seed = seed
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.model_: LGBMClassifier | None = None
        self.classes_: list = []
        self.features_: list | None = None

    def _as_df(self, X) -> pd.DataFrame:
        # fit·predict 에 일관된 DataFrame(피처명 동일) → lightgbm feature-name 경고 제거.
        if isinstance(X, pd.DataFrame):
            return X
        cols = self.features_ if self.features_ is not None else range(np.asarray(X).shape[1])
        return pd.DataFrame(np.asarray(X), columns=list(cols))

    def fit(self, X, y) -> "TreeBench":
        y = pd.Series(y)
        self.classes_ = sorted(y.unique())
        Xdf = self._as_df(X)
        self.features_ = list(Xdf.columns)
        if len(self.classes_) < 2:
            # 단일클래스 폴드: 트리 학습 불가 → 상수확률(그 클래스=1). PriorBaseline 동형.
            self.model_ = None
            return self
        self.model_ = LGBMClassifier(
            **self.params,
            random_state=self.seed,
            n_jobs=1,             # 재현성 (스레드 비결정성 제거)
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        self.model_.fit(Xdf, y.to_numpy())
        return self

    def predict_proba(self, X) -> pd.DataFrame:
        idx = self._index(X)
        if self.model_ is None:
            # 폴백: 유일 클래스에 확률 1
            data = np.ones((len(idx), 1))
            return pd.DataFrame(data, index=idx, columns=self.classes_)
        proba = self.model_.predict_proba(self._as_df(X)[self.features_])
        df = pd.DataFrame(proba, index=idx, columns=list(self.model_.classes_))
        # classes_ 순서로 정렬(누락 없음 — 트리 classes_ ⊆ self.classes_)
        return df.reindex(columns=self.classes_, fill_value=0.0)
