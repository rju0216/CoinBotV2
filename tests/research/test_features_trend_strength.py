"""추세강도 피처 2축 회귀 — ER · Hurst(R/S) (Phase 2 Step 2.2).

핵심: 인과성(``assert_causal``, 기둥 2) + 의미 검증(ER 유계·추세/노이즈 극단, Hurst 가
추세추종>무작위>평균회귀 순서를 잡는가). 성과는 안 봄(소프트웨어 정합성만).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import assert_causal
from src.research.features import trend_strength as ts
from tests.research.synthetic import make_ohlcv


def _df_from_close(close: np.ndarray) -> pd.DataFrame:
    """close 배열 → 유효 OHLCV df (ER·Hurst 는 close 만 쓰지만 유효 프레임 구성)."""
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )


def _ar1_close(n: int, phi: float, seed: int) -> pd.DataFrame:
    """AR(1) 수익률 → close. phi>0 지속(추세추종), phi<0 반지속(평균회귀)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    close = 10000 * np.exp(np.cumsum(r))
    return _df_from_close(close)


# ---- 인과성 (핵심) ----

def test_er_causal():
    df = make_ohlcv()
    assert not assert_causal(lambda d: ts.efficiency_ratio(d, window=48), df).leaked


def test_hurst_causal():
    df = make_ohlcv(n=280)
    assert not assert_causal(lambda d: ts.hurst(d, window=128), df).leaked


# ---- index 정렬 ----

def test_index_aligned():
    df = make_ohlcv()
    for s in (ts.efficiency_ratio(df, 48), ts.hurst(df, 128)):
        assert isinstance(s, pd.Series)
        assert s.index.equals(df.index)


# ---- ER 의미 ----

def test_er_bounded():
    s = ts.efficiency_ratio(make_ohlcv(n=400), 48).dropna()
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_er_straight_line_is_one():
    """직선 추세(일정 스텝) → ER=1 (순변화=경로)."""
    close = 100.0 + np.arange(200) * 0.5
    er = ts.efficiency_ratio(_df_from_close(close), 48).dropna()
    assert er.iloc[-1] == pytest.approx(1.0)


def test_er_zigzag_near_zero():
    """제자리 지그재그 → ER≈0 (순변화≈0, 경로 큼)."""
    base = 100.0 + np.tile([0.0, 1.0], 100)   # 100,101,100,101,...
    er = ts.efficiency_ratio(_df_from_close(base), 48).dropna()
    assert er.iloc[-1] < 0.1


# ---- Hurst 의미 (추세추종 > 무작위 > 평균회귀) ----

def test_hurst_orders_persistence():
    """지속(phi=+0.6) > 무작위(phi=0) > 반지속(phi=-0.6) — R/S 가 순서를 잡는다."""
    w = 128
    h_pers = ts.hurst(_ar1_close(600, 0.6, seed=1), w).dropna().median()
    h_rand = ts.hurst(_ar1_close(600, 0.0, seed=1), w).dropna().median()
    h_anti = ts.hurst(_ar1_close(600, -0.6, seed=1), w).dropna().median()
    assert h_pers > h_rand > h_anti


def test_hurst_bounded():
    s = ts.hurst(make_ohlcv(n=400), 128).dropna()
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_hurst_random_walk_near_half():
    """무작위 → H 가 0.5 근방 밴드(R/S 소표본 편향 감안 느슨히)."""
    h = ts.hurst(_ar1_close(800, 0.0, seed=7), 128).dropna().median()
    assert 0.35 < h < 0.75


# ---- 워밍업 NaN ----

def test_warmup_nan():
    df = make_ohlcv(n=400)
    assert ts.efficiency_ratio(df, 48).iloc[:48].isna().all()
    assert ts.hurst(df, 128).iloc[:127].isna().all()
    assert not np.isnan(ts.efficiency_ratio(df, 48).iloc[-1])
    assert not np.isnan(ts.hurst(df, 128).iloc[-1])


# ---- 엣지 ----

def test_invalid_window():
    df = make_ohlcv()
    with pytest.raises(ValueError):
        ts.efficiency_ratio(df, 0)
    with pytest.raises(ValueError):
        ts.hurst(df, -1)


def test_hurst_short_window_nan():
    """R/S 하위창 분할 불가(너무 짧은 창) → NaN, 크래시 없음."""
    df = make_ohlcv(n=100)
    # window 10 < 2*min_n(16) → 각 창 NaN
    assert ts.hurst(df, 10).dropna().empty
