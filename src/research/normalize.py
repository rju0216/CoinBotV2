"""정규화 — z-score train-only (MasterPlan §11 컴포넌트 D, 설계 §7).

fit 통계는 **train fold 에서만** 계산하고 transform 에 재사용한다(인과성). val
통계를 쓰면 누수. splitter 의 Fold(train_index/val_index)와 결합해 fold 단위로
적용한다(seam 통합 테스트, 규칙 16).

**교체 가능 인터페이스**(BaseScaler): F-6 에 따라 Phase 2 에서 robust scaling
(median/MAD)이 필요하면 드롭인으로 교체할 수 있게 fit/transform 계약만 공유한다.
train-only 인과 계약은 어느 scaler 든 동일하게 성립한다.

**F-6 (Phase 2 Step 2.4 진단·해소)**: 실 1h 8축 진단 결과 팻테일 축(semi_dev·
kf_uncertainty 초과첨도 ~73, kf_slope ~8.4)에서 z-score std 가 소수 극단에 부풀어
(kf_uncertainty std/robustScale≈5.0) 본체가 압축됨 → ``RobustScaler``(median/MAD)를
드롭인으로 제공. **z vs robust 최종 선택은 Phase 3 예측력**(F-1 비교예산). 기본은 z-score
유지, robust 는 harness normalizer_factory 로 교체 가능(둘 다 train-only 인과 동일).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseScaler(ABC):
    """fit(train)/transform 계약. Phase 2 robust scaler 도 이 인터페이스를 구현."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseScaler":
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


class ZScoreNormalizer(BaseScaler):
    """열별 (x - mean) / std. 통계는 fit(train) 에서만. NaN 은 스킵(fit)·보존(transform).

    상수/영분산 열은 std=1 로 두어 통과(inf 방지).
    """

    def __init__(self, ddof: int = 0, eps: float = 1e-12) -> None:
        self.ddof = ddof
        self.eps = eps
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame) -> "ZScoreNormalizer":
        self.columns_ = list(train_df.columns)
        self.mean_ = train_df.mean(skipna=True)
        std = train_df.std(ddof=self.ddof, skipna=True)
        # 영분산(상수) 열 → 1 로 대체해 통과(0-중심만)
        self.std_ = std.where(std > self.eps, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None or self.columns_ is None:
            raise RuntimeError("fit 이전에 transform 호출됨")
        sub = df[self.columns_]
        return (sub - self.mean_) / self.std_


class RobustScaler(BaseScaler):
    """열별 (x - median) / (c·MAD). 통계는 fit(train) 에서만 (train-only 인과, F-6).

    MAD = median(|x - median|), c=1.4826 로 정규분포서 std 와 일치시킴(robust scale).
    소수 극단이 median·MAD 를 거의 안 흔들어(중앙 50%만 반영) 팻테일 축(semi_dev·
    kf_uncertainty 등)의 본체 압축을 막는다 — z-score 대비 핵심 이점. NaN 은 스킵(fit)·
    보존(transform). 영-MAD(상수/이산) 열은 scale=1 로 통과(inf 방지, ZScore 와 동형).
    """

    def __init__(self, c: float = 1.4826, eps: float = 1e-12) -> None:
        self.c = c
        self.eps = eps
        self.median_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, train_df: pd.DataFrame) -> "RobustScaler":
        self.columns_ = list(train_df.columns)
        self.median_ = train_df.median(skipna=True)
        mad = (train_df - self.median_).abs().median(skipna=True)
        scale = self.c * mad
        # 영-MAD(상수/이산) 열 → 1 로 대체해 통과(0-중심만)
        self.scale_ = scale.where(scale > self.eps, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None or self.scale_ is None or self.columns_ is None:
            raise RuntimeError("fit 이전에 transform 호출됨")
        sub = df[self.columns_]
        return (sub - self.median_) / self.scale_
