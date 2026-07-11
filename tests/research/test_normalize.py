"""z-score 정규화 회귀 테스트 — 수식·영분산·NaN·train-only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.normalize import BaseScaler, ZScoreNormalizer


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
