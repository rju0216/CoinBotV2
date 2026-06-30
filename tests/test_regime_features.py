"""레짐 feature 파이프라인 단위 테스트 (D4-2a).

윈도우 경계·인덱싱 정확성 (I-005 체크리스트):
  H1 first-valid 인덱스 = 48 (최대 윈도우), 앞 NaN.
  H2 causal — 현재 마감 봉 포함, 미래 데이터가 과거 feature 에 영향 없음.
  H5 realized_vol = log_return 의 rolling std (일관).
+ z-score train-only / transform stateless (누수 차단).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.regime.features import (
    FEATURE_NAMES,
    FeatureConfig,
    ZScoreParams,
    compute_raw_features,
)


def _make_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 50000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": 1.0,
        },
        index=idx,
    )


# ---- H1: warmup / first-valid 인덱스 ----

def test_warmup_first_valid_index_is_max_window():
    cfg = FeatureConfig()  # vol=24, er=48
    feats = compute_raw_features(_make_df(200), cfg)
    valid = feats.dropna()
    # 첫 유효 행은 정확히 위치 48 (= max(24,48)) (0-based)
    assert feats.index.get_loc(valid.index[0]) == cfg.warmup == 48
    # 앞 48행은 어떤 feature 든 NaN 포함 (학습 제외 구간)
    assert feats.iloc[:48].isna().any(axis=1).all()
    # 48행 이후는 전부 유효
    assert not feats.iloc[48:].isna().any().any()


def test_columns_order_matches_feature_names():
    feats = compute_raw_features(_make_df(100))
    assert tuple(feats.columns) == FEATURE_NAMES


# ---- H2: causal (미래 데이터 누출 없음) ----

def test_causal_no_lookahead():
    df = _make_df(200)
    full = compute_raw_features(df)
    # index 100 이후를 교란 (미래)
    df2 = df.copy()
    df2.iloc[100:] = df2.iloc[100:].to_numpy() * 1.5
    perturbed = compute_raw_features(df2)
    # index 0~99 feature 는 불변이어야 한다 (현재까지의 데이터만 사용)
    pd.testing.assert_frame_equal(full.iloc[:100], perturbed.iloc[:100])


# ---- H5: realized_vol = log_return 의 rolling std (일관) ----

def test_realized_vol_matches_rolling_std():
    df = _make_df(120)
    feats = compute_raw_features(df, FeatureConfig(vol_window=24))
    log_ret = np.log(df["close"] / df["close"].shift(1))
    expected = log_ret.rolling(24).std()
    pd.testing.assert_series_equal(
        feats["realized_vol"], expected, check_names=False
    )


# ---- z-score: train-only 산출 + stateless transform ----

def test_zscore_standardizes_train():
    raw = compute_raw_features(_make_df(500))
    train = raw.iloc[:300]
    params = ZScoreParams.fit(train)
    z = params.transform(train).dropna()
    for c in FEATURE_NAMES:
        assert abs(z[c].mean()) < 1e-9          # 평균 0
        assert abs(z[c].std(ddof=0) - 1.0) < 1e-9  # 표준편차 1


def test_zscore_transform_uses_stored_params_not_recompute():
    raw = compute_raw_features(_make_df(500))
    train, test = raw.iloc[:300], raw.iloc[300:]
    params = ZScoreParams.fit(train)
    # test 를 train params 로 변환 = (x - train_mean)/train_std (재계산 아님)
    z_test = params.transform(test)
    # transform 이 정확히 (x - train_mean)/train_std 임을 컬럼별로 증명
    # → 저장된 train params 사용(재계산 없음 = 누수 차단). 이 단언이 완전 증명이다.
    for c in FEATURE_NAMES:
        expected = (test[c] - params.mean[c]) / params.std[c]
        pd.testing.assert_series_equal(z_test[c], expected, check_names=False)


def test_zscore_constant_feature_guarded():
    # std=0 인 상수 feature 도 0 나눗셈 없이 처리
    idx = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    raw = pd.DataFrame({"log_return": [1.0] * 10}, index=idx)
    params = ZScoreParams.fit(raw)
    out = params.transform(raw)
    assert (out["log_return"] == 0.0).all()  # (1-1)/1 = 0, inf/nan 없음
