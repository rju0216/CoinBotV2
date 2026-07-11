"""정규화 회귀 테스트 — z-score·RobustScaler: 수식·영분산·NaN·train-only·outlier 강건."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.normalize import BaseScaler, RobustScaler, ZScoreNormalizer


def _df(**cols):
    return pd.DataFrame(cols)


def test_zscore_formula():
    df = _df(a=[1.0, 2.0, 3.0, 4.0])
    norm = ZScoreNormalizer(ddof=0)
    out = norm.fit_transform(df)
    mean, std = 2.5, np.std([1, 2, 3, 4])  # ddof=0
    expected = (df["a"] - mean) / std
    assert np.allclose(out["a"].to_numpy(), expected.to_numpy())


def test_zero_variance_column_passthrough():
    df = _df(const=[5.0, 5.0, 5.0], var=[1.0, 2.0, 3.0])
    out = ZScoreNormalizer().fit_transform(df)
    # 상수 열 → std=1 대체, (x-mean)/1 = 0, inf/nan 없음
    assert np.isfinite(out["const"].to_numpy()).all()
    assert np.allclose(out["const"].to_numpy(), 0.0)


def test_nan_skipped_in_fit_preserved_in_transform():
    df = _df(a=[1.0, np.nan, 3.0, 5.0])
    norm = ZScoreNormalizer(ddof=0).fit(df)
    # mean/std 는 NaN 스킵 (1,3,5)
    assert norm.mean_["a"] == pytest.approx(3.0)
    out = norm.transform(df)
    assert np.isnan(out["a"].iloc[1])          # NaN 보존
    assert not np.isnan(out["a"].iloc[0])


def test_is_base_scaler():
    assert isinstance(ZScoreNormalizer(), BaseScaler)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ZScoreNormalizer().transform(_df(a=[1.0, 2.0]))


def test_train_only_stats_used_for_val():
    train = _df(a=[0.0, 10.0])       # mean 5, std 5
    val = _df(a=[5.0, 15.0])
    norm = ZScoreNormalizer(ddof=0).fit(train)
    out = norm.transform(val)
    # val 은 train 통계로 변환: (5-5)/5=0, (15-5)/5=2
    assert np.allclose(out["a"].to_numpy(), [0.0, 2.0])


# ---- RobustScaler (F-6 드롭인) ----

def test_robust_formula():
    df = _df(a=[1.0, 2.0, 3.0, 4.0, 5.0])
    out = RobustScaler().fit_transform(df)
    med = 3.0
    mad = np.median(np.abs(np.array([1, 2, 3, 4, 5]) - med))  # =1
    expected = (df["a"] - med) / (1.4826 * mad)
    assert np.allclose(out["a"].to_numpy(), expected.to_numpy())


def test_robust_is_base_scaler():
    assert isinstance(RobustScaler(), BaseScaler)


def test_robust_resists_outliers_unlike_zscore():
    """핵심 이점(F-6): 소수 극단이 robust 스케일을 거의 안 흔든다.

    본체 [−2..2] 에 극단 1e6 하나 주입 → z-score 는 std 폭발로 본체를 0 근처로 압축,
    robust 는 본체 스프레드를 보존.
    """
    body = np.arange(-2.0, 3.0)                     # -2,-1,0,1,2
    df = _df(a=np.concatenate([body, [1e6]]))
    z = ZScoreNormalizer(ddof=0).fit_transform(df)["a"].to_numpy()[:-1]
    r = RobustScaler().fit_transform(df)["a"].to_numpy()[:-1]
    # z-score 는 본체가 뭉갬(극단 std), robust 는 유의미한 스프레드 유지
    assert np.ptp(z) < 0.01          # 본체 range ≈ 0 (압축)
    assert np.ptp(r) > 1.0           # 본체 range 보존
    # robust 스케일이 극단에 안 끌림 → 본체가 제대로 퍼짐


def test_robust_zero_mad_passthrough():
    df = _df(const=[5.0, 5.0, 5.0], var=[1.0, 2.0, 3.0])
    out = RobustScaler().fit_transform(df)
    assert np.isfinite(out["const"].to_numpy()).all()
    assert np.allclose(out["const"].to_numpy(), 0.0)   # 상수 열 → 0


def test_robust_nan_skipped_in_fit_preserved():
    df = _df(a=[1.0, np.nan, 3.0, 5.0, 7.0])
    norm = RobustScaler().fit(df)
    assert norm.median_["a"] == pytest.approx(4.0)      # median(1,3,5,7)
    out = norm.transform(df)
    assert np.isnan(out["a"].iloc[1])
    assert not np.isnan(out["a"].iloc[0])


def test_robust_train_only():
    train = _df(a=[0.0, 2.0, 4.0, 6.0, 8.0])           # median 4, MAD 2 → scale 2.9652
    val = _df(a=[4.0, 10.0])
    norm = RobustScaler().fit(train)
    out = norm.transform(val)
    expected = (np.array([4.0, 10.0]) - 4.0) / (1.4826 * 2.0)
    assert np.allclose(out["a"].to_numpy(), expected)


def test_robust_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        RobustScaler().transform(_df(a=[1.0, 2.0]))
