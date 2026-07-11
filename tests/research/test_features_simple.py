"""단순 피처 4축 회귀 — 인과성(핵심)·정상성/스케일·워밍업·엣지·조립기 (Phase 2 Step 2.1).

핵심: 각 피처가 **그 시점 정보만** 쓰는지 ``assert_causal`` 로 강제(기둥 2). 정규화는
harness 소유라 여기서 안 보고, 대신 구성상 정상성(로그비율 대칭·종가위치 유계)을 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import assert_causal
from src.research.features import simple
from src.research.features.build import DEFAULT_PARAMS, build_features
from tests.research.synthetic import make_ohlcv


# ---- 인과성 (핵심) ----

def test_vol_change_causal():
    df = make_ohlcv()
    report = assert_causal(lambda d: simple.vol_change(d, window=48, lag=24), df)
    assert not report.leaked


def test_relative_volume_causal():
    df = make_ohlcv()
    report = assert_causal(lambda d: simple.relative_volume(d, window=48), df)
    assert not report.leaked


def test_semi_dev_causal():
    df = make_ohlcv()
    report = assert_causal(lambda d: simple.rolling_semi_deviation(d, window=48), df)
    assert not report.leaked


def test_close_position_causal():
    df = make_ohlcv()
    report = assert_causal(simple.close_position, df)
    assert not report.leaked


# ---- index 정렬 ----

@pytest.mark.parametrize(
    "fn",
    [
        lambda d: simple.vol_change(d, window=48, lag=24),
        lambda d: simple.relative_volume(d, window=48),
        lambda d: simple.rolling_semi_deviation(d, window=48),
        simple.close_position,
    ],
)
def test_index_aligned(fn):
    df = make_ohlcv()
    s = fn(df)
    assert isinstance(s, pd.Series)
    assert s.index.equals(df.index)


# ---- 정상성 / 스케일 / 유계 ----

def test_close_position_bounded():
    """종가위치 ∈[0,1] — 유계(정상성 구성)."""
    s = simple.close_position(make_ohlcv()).dropna()
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_close_position_flat_bar_is_neutral():
    """무변동 봉(high==low) → 중립 0.5."""
    idx = pd.date_range("2020-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=idx,
    )
    assert (simple.close_position(df) == 0.5).all()


def test_close_position_extremes():
    """종가=고가 →1, 종가=저가 →0."""
    idx = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [110.0, 110.0],
            "low": [90.0, 90.0],
            "close": [110.0, 90.0],   # 고가마감 / 저가마감
            "volume": [1.0, 1.0],
        },
        index=idx,
    )
    s = simple.close_position(df)
    assert s.iloc[0] == pytest.approx(1.0)
    assert s.iloc[1] == pytest.approx(0.0)


def test_vol_change_symmetric_around_zero():
    """로그비율 → 팽창(+)·수축(-) 대칭, 0 중심 근방. 정상성 구성 확인."""
    s = simple.vol_change(make_ohlcv(n=500), window=48, lag=24).dropna()
    assert s.abs().median() < 1.0            # 봉당 변동성 변화는 보통 작음
    assert s.min() < 0 < s.max()             # 양·음 모두 존재


def test_relative_volume_zero_centered():
    """상대거래량 로그 → 정상 대비 0 근방(급증/한산 대칭)."""
    s = simple.relative_volume(make_ohlcv(n=500), window=48).dropna()
    assert abs(s.median()) < 1.0
    assert s.min() < 0 < s.max()


# ---- 워밍업 NaN ----

def test_warmup_nan():
    df = make_ohlcv()
    # vol_change: σ 창(48) + lag(24) 만큼 워밍업
    assert simple.vol_change(df, window=48, lag=24).iloc[:60].isna().all()
    # relative_volume: window(48) 기저 + shift(1)
    assert simple.relative_volume(df, window=48).iloc[:48].isna().all()
    # semi_dev: 수익률 window(48)
    assert simple.rolling_semi_deviation(df, window=48).iloc[:47].isna().all()
    # 끝은 유효
    assert not np.isnan(simple.vol_change(df, window=48, lag=24).iloc[-1])


# ---- 엣지: 입력 검증 ----

def test_invalid_params():
    df = make_ohlcv()
    with pytest.raises(ValueError):
        simple.vol_change(df, window=0, lag=24)
    with pytest.raises(ValueError):
        simple.relative_volume(df, window=0)
    with pytest.raises(ValueError):
        simple.rolling_semi_deviation(df, window=-1)
    with pytest.raises(ValueError):
        simple.vol_change(df, window=48, lag=24, estimator="nope")


# ---- 조립기 (seam, 규칙 16) ----

def test_build_features_shape_and_columns():
    df = make_ohlcv()
    X = build_features(df)
    assert X.index.equals(df.index)
    assert list(X.columns) == [
        "vol_change", "relative_volume", "semi_dev", "close_position",
        "er", "hurst", "kf_slope", "kf_uncertainty",
    ]


def test_build_features_causal():
    """조립기 전체가 인과 — 각 열이 그 시점 정보만."""
    df = make_ohlcv()
    report = assert_causal(build_features, df)
    assert not report.leaked


def test_build_features_param_override():
    """params 로 기본창 덮어쓰기(Phase 3 창 탐색 대비)."""
    df = make_ohlcv()
    X = build_features(df, params={"relative_volume": {"window": 24}})
    # 창이 짧아지면 워밍업 NaN 도 짧아짐 (24 vs 48)
    assert not np.isnan(X["relative_volume"].iloc[30])
    # 기본 그대로인 열은 영향 없음
    assert "vol_change" in X.columns
    assert DEFAULT_PARAMS["relative_volume"]["window"] == 48   # 원본 불변(사본 수정)
