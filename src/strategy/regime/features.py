"""HMM 입력 feature 파이프라인 (발견층 입력).

3 코어 feature (스펙 §1.2, S1-1=실현변동성):
  - log_return        : ln(P_t/P_{t-1})           (1봉)   방향+모멘텀
  - realized_vol      : std(log_return, 24봉)      (24봉)  변동성
  - efficiency_ratio  : Kaufman ER                 (48봉)  추세성

원칙:
  - 정상성(가격 절대수준 금지) — 전부 수익률·비율.
  - causal: 현재 마감 봉(close_t) **포함**해 계산. Donchian 식 shift 미적용
    (봉이 닫혔으므로 그 봉의 수익률에 쓰는 게 causal). (체크리스트 H2)
  - first-valid index = max(24,48) = 48. 앞 48봉 NaN (학습 제외). (H1)
  - z-score 표준화 파라미터(mean/std)는 **train 구간에서만** 산출하고 artifact 에
    동행시켜 test/live 에 그대로 적용 (재계산 금지 = 미래 누수 차단).

주의: 여기 realized_vol(정규화, HMM 분류 입력) 은 contract.volatility(ATR, 가격
단위, SL/TP 용)와 **별개**다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.strategy.indicators import (
    compute_efficiency_ratio,
    compute_log_returns,
    compute_realized_vol,
)

#: feature 컬럼 순서 (HMM observation 벡터 차원 순서와 일치)
FEATURE_NAMES: tuple[str, ...] = ("log_return", "realized_vol", "efficiency_ratio")


@dataclass(frozen=True)
class FeatureConfig:
    """feature 윈도우 설정 (artifact 동행). log_return 은 항상 1봉."""

    vol_window: int = 24
    er_window: int = 48

    @property
    def warmup(self) -> int:
        """첫 유효 feature 행까지 필요한 선행 봉 수 (= 최대 윈도우). 앞 warmup 봉 NaN."""
        return max(self.vol_window, self.er_window)

    def to_dict(self) -> dict:
        return {"vol_window": self.vol_window, "er_window": self.er_window}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(vol_window=d["vol_window"], er_window=d["er_window"])


def compute_raw_features(
    df: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """3 코어 feature (raw, 표준화 전) 계산.

    Returns:
        index=df.index, columns=FEATURE_NAMES 의 DataFrame. 앞 config.warmup 봉은 NaN.
    """
    return pd.DataFrame(
        {
            "log_return": compute_log_returns(df),
            "realized_vol": compute_realized_vol(df, config.vol_window),
            "efficiency_ratio": compute_efficiency_ratio(df, config.er_window),
        },
        index=df.index,
    )[list(FEATURE_NAMES)]


@dataclass(frozen=True)
class ZScoreParams:
    """train 구간에서 산출한 표준화 파라미터 (artifact 동행).

    z = (x - mean) / std. std=0(상수 feature) 이면 1로 보호.
    """

    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, raw_features: pd.DataFrame) -> "ZScoreParams":
        """train raw features 의 유효행(NaN 제외)에서 mean/std(ddof=0) 산출.

        ddof=0(모표준편차): 표준화 스케일용. (feature 정의의 rolling std(ddof=1)
        와는 별개 — 그건 realized_vol 의 값 자체, 이건 정규화 스케일.)
        """
        valid = raw_features.dropna()
        mean = {c: float(valid[c].mean()) for c in raw_features.columns}
        std = {c: float(valid[c].std(ddof=0)) for c in raw_features.columns}
        return cls(mean=mean, std=std)

    def transform(self, raw_features: pd.DataFrame) -> pd.DataFrame:
        """저장된 train params 로 표준화 (재계산 없음 = causal). NaN 행은 NaN 유지."""
        out = pd.DataFrame(index=raw_features.index)
        for c in raw_features.columns:
            s = self.std.get(c, 1.0)
            denom = s if s > 0 else 1.0
            out[c] = (raw_features[c] - self.mean.get(c, 0.0)) / denom
        return out[list(raw_features.columns)]

    def to_dict(self) -> dict:
        return {"mean": dict(self.mean), "std": dict(self.std)}

    @classmethod
    def from_dict(cls, d: dict) -> "ZScoreParams":
        return cls(mean=dict(d["mean"]), std=dict(d["std"]))
