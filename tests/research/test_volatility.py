"""변동성 추정기 회귀 테스트 — 인과성(핵심)·분수 스케일·워밍업·레지스트리.

핵심: 배리어 폭 재료인 σ_i 가 **그 시점 정보만** 쓰는지 ``assert_causal`` 로 강제
(설계 §4 대전제·기둥 2). 라벨 자체의 forward 성은 여기 대상이 아니다(triple_barrier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import assert_causal
from src.research.labeling.volatility import (
    VOL_ESTIMATORS,
    atr,
    yang_zhang,
)


def _ohlc(n=250, seed=0):
    """합성 OHLCV — 랜덤워크 종가 + 유효 OHLC(high≥max(o,c), low≤min(o,c))."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, 0.01, n)
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    hi_extra = np.abs(rng.normal(0, 0.005, n))
    lo_extra = np.abs(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + hi_extra)
    low = np.minimum(open_, close) * (1 - lo_extra)
    vol = rng.uniform(1, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


@pytest.mark.parametrize("name", ["atr", "yz"])
def test_causal_no_future_leak(name):
    """배리어 폭(σ)이 미래를 안 본다 — 절단불변 + 미래교란 둘 다 통과."""
    df = _ohlc()
    fn = VOL_ESTIMATORS[name]
    report = assert_causal(lambda d: fn(d, 30), df)   # 누수 시 LeakageError
    assert not report.leaked


@pytest.mark.parametrize("name", ["atr", "yz"])
def test_index_aligned_and_nonneg(name):
    df = _ohlc()
    s = VOL_ESTIMATORS[name](df, 30)
    assert isinstance(s, pd.Series)
    assert s.index.equals(df.index)
    assert (s.dropna() >= 0).all()


@pytest.mark.parametrize("name", ["atr", "yz"])
def test_fractional_scale(name):
    """분수 정규화 → 가격 무관 스케일(수천불 BTC 에도 σ 는 작은 소수)."""
    s = VOL_ESTIMATORS[name](_ohlc(), 30).dropna()
    assert s.median() < 0.5   # 정상 시장이면 봉당 변동성 분수는 «1


@pytest.mark.parametrize("name", ["atr", "yz"])
def test_warmup_is_nan(name):
    df = _ohlc()
    s = VOL_ESTIMATORS[name](df, 30)
    assert s.iloc[:20].isna().all()   # 창 미달 구간은 NaN
    assert s.iloc[-1] == s.iloc[-1]   # 끝은 유효(non-NaN)


def test_constant_ohlc_zero_vol():
    """변동 없는 시장 → σ=0 (범위·수익 모두 0). 결정론 정상성 검사."""
    idx = pd.date_range("2020-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=idx,
    )
    assert atr(df, 20).dropna().abs().max() == pytest.approx(0.0, abs=1e-12)
    assert yang_zhang(df, 20).dropna().abs().max() == pytest.approx(0.0, abs=1e-9)


def test_atr_captures_range_expansion():
    """ATR 는 봉 범위 확장을 반영 — 큰 range 봉 뒤 σ 상승."""
    df = _ohlc()
    calm = atr(df, 20)
    df2 = df.copy()
    # 중간 한 봉 range 를 크게 부풀림
    df2.iloc[120, df2.columns.get_loc("high")] *= 1.10
    df2.iloc[120, df2.columns.get_loc("low")] *= 0.90
    expanded = atr(df2, 20)
    # 그 봉 이후(창 안) ATR 이 올라감
    assert expanded.iloc[121] > calm.iloc[121]


def test_registry_covers_both():
    assert set(VOL_ESTIMATORS) == {"atr", "yz"}
    assert VOL_ESTIMATORS["atr"] is atr
    assert VOL_ESTIMATORS["yz"] is yang_zhang


def test_invalid_window():
    df = _ohlc()
    with pytest.raises(ValueError):
        atr(df, 0)
    with pytest.raises(ValueError):
        yang_zhang(df, 1)   # YZ 는 분산에 최소 2봉 필요
