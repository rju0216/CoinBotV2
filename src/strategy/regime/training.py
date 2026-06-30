"""레짐 모델 학습 오케스트레이션 (offline) — build_model + walk_forward (D4-2d).

build_model: 한 train 윈도우 → 완성 RegimeModel.
walk_forward: anchored 확장 윈도우로 구간별 모델 생성.

실 학습(전체 데이터·K선택 스윕)은 D4-5 에서 사용자 실행. 여기는 재사용 파이프라인.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.regime.artifact import RegimeModel
from src.strategy.regime.features import (
    FeatureConfig,
    ZScoreParams,
    compute_raw_features,
)
from src.strategy.regime.hmm import GaussianEmission, HMM
from src.strategy.regime.mapping import StateMapping


def build_model(
    train_candles: pd.DataFrame,
    k: int,
    tau: float,
    valid_period: tuple[str, str],
    *,
    feature_config: FeatureConfig = FeatureConfig(),
    n_init: int = 5,
    seed: int = 0,
    meta: dict | None = None,
    **fit_kw,
) -> RegimeModel:
    """한 train 윈도우 → RegimeModel (feature→zscore→HMM fit→mapping)."""
    raw = compute_raw_features(train_candles, feature_config)
    zscore = ZScoreParams.fit(raw)
    z = zscore.transform(raw)
    valid_idx = z.dropna().index
    X = z.loc[valid_idx].to_numpy()
    raw_log = raw["log_return"].loc[valid_idx].to_numpy()  # X 와 정렬

    hmm = HMM(k, GaussianEmission(k, X.shape[1])).fit(
        X, n_init=n_init, seed=seed, **fit_kw
    )
    gamma = hmm.smoothed_posterior(X)  # offline 특성화 (smoothed OK)
    mapping = StateMapping.fit(raw_log, gamma, tau)

    full_meta = {
        "train_start": str(train_candles.index[0]),
        "train_end": str(train_candles.index[-1]),
        "emission_kind": hmm.emission.kind,
        "k": k,
    }
    full_meta.update(meta or {})
    return RegimeModel(
        hmm=hmm,
        mapping=mapping,
        zscore=zscore,
        feature_config=feature_config,
        valid_period=valid_period,
        meta=full_meta,
    )


def make_anchored_windows(
    valid_start, valid_end, retrain_months: int
) -> list[tuple[str, str, str]]:
    """anchored 확장 walk-forward 윈도우 생성.

    [valid_start, valid_end) 를 retrain_months 마다 잘라 valid 블록을 만들고, 각 블록의
    train 은 데이터 시작부터 valid_start 직전(train_end = valid_start − 1h)까지(anchored).
    train_end 를 valid_start 보다 1봉 앞에 둬 train/valid 경계 누수를 막는다.

    Returns: [(train_end, valid_start, valid_end), ...] (ISO 문자열, walk_forward 입력 형식).
    """
    if retrain_months <= 0:
        raise ValueError("retrain_months 는 양수여야 함")

    def _utc(t) -> pd.Timestamp:
        # 실데이터 candles 는 tz-aware UTC index → 경계 문자열도 tz-aware 로 맞춰
        # candles.loc[:train_end] 의 tz-naive/aware 비교 에러를 막는다.
        ts = pd.Timestamp(t)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    windows: list[tuple[str, str, str]] = []
    cur = _utc(valid_start)
    end = _utc(valid_end)
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=retrain_months), end)
        train_end = cur - pd.Timedelta(hours=1)
        windows.append((str(train_end), str(cur), str(nxt)))
        cur = nxt
    return windows


def walk_forward(
    candles: pd.DataFrame,
    windows: list[tuple],
    k: int,
    tau: float,
    **build_kw,
) -> list[RegimeModel]:
    """anchored 확장 walk-forward.

    windows: [(train_end, valid_start, valid_end), ...] (ts). train 은 데이터 시작부터
    train_end 까지(anchored). valid_period = [valid_start, valid_end).
    """
    models = []
    for train_end, valid_start, valid_end in windows:
        train = candles.loc[:train_end]
        models.append(
            build_model(
                train, k, tau, (str(valid_start), str(valid_end)), **build_kw
            )
        )
    return models
