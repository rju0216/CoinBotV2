"""누수 하네스 회귀 테스트 — causal 통과 / leaky 포착 / regime 인과 경계."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import (
    LeakageError,
    assert_causal,
    check_future_leak,
    check_truncation_invariance,
)
from src.research.validation.regime import (
    TREND_COL,
    VOL_COL,
    RegimeParams,
    tag_regimes,
)


def _ohlcv(closes):
    c = np.asarray(closes, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(c), freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


# ---- 하네스 자체 검증: causal 통과 ----

def test_causal_rolling_mean_passes():
    df = _ohlcv(100 + np.arange(60) * 0.5)
    fn = lambda d: d["close"].rolling(3).mean()  # noqa: E731  (과거만)
    assert_causal(fn, df)  # 누수 없음 → 예외 없음
    assert not check_truncation_invariance(fn, df).leaked
    assert not check_future_leak(fn, df).leaked


# ---- leaky 포착 ----

def test_global_mean_subtract_is_caught():
    df = _ohlcv(100 + np.arange(60) * 0.5)
    fn = lambda d: d["close"] - d["close"].mean()  # noqa: E731 (전역 통계 = 미래 포함)
    assert check_truncation_invariance(fn, df).leaked
    assert check_future_leak(fn, df).leaked
    with pytest.raises(LeakageError):
        assert_causal(fn, df)


def test_lookahead_shift_is_caught():
    df = _ohlcv(100 + np.arange(60) * 0.5)
    fn = lambda d: d["close"].shift(-1)  # noqa: E731 (명시적 미래참조)
    with pytest.raises(LeakageError):
        assert_causal(fn, df)


def test_nan_warmup_not_falsely_flagged():
    df = _ohlcv(100 + np.arange(40) * 0.3)
    fn = lambda d: d["close"].rolling(5).mean()  # noqa: E731 (앞 4개 NaN)
    r = check_truncation_invariance(fn, df)
    assert not r.leaked  # NaN==NaN 은 동일 취급


def test_low_quantile_leak_caught():
    # d - d.min() (전역 최소가 pos 0) — 이전 단방향 교란은 위음성 → 양방향으로 잡혀야 함
    vals = np.concatenate([[0.0], np.arange(1, 60, dtype=float)])  # 최소가 맨 앞
    df = pd.DataFrame({"x": vals})
    fn = lambda d: d["x"] - d["x"].min()  # noqa: E731 (전역 min = 미래 포함, 비인과)
    assert check_future_leak(fn, df).leaked


def test_short_series_raises_not_vacuous_green():
    df = pd.DataFrame({"x": [1.0]})   # 체크포인트를 못 만드는 길이 → vacuous green 금지
    with pytest.raises(ValueError):
        check_future_leak(lambda d: d["x"], df)  # noqa: E731


# ---- regime: 인과 경계 못박기 ----

_P = RegimeParams(ma_len=5, slope_k=3, deadband=0.001, vol_len=5, vol_hi_pct=0.5, min_duration=1)


def test_regime_trend_is_causal():
    df = _ohlcv(100 * (1.01 ** np.arange(60)))  # 단조 상승
    trend_fn = lambda d: tag_regimes(d, _P)[TREND_COL]  # noqa: E731
    assert_causal(trend_fn, df)  # 추세 = MA·기울기·히스테리시스(전부 backward) → 인과


def test_regime_vol_is_ex_post_and_flagged():
    # 전반부 저변동성 + 후반부 고변동성 → 전구간 quantile 문턱이 절단/교란에 따라 바뀜
    lo = 100 + np.cumsum(np.tile([0.1, -0.1], 15))
    hi = lo[-1] + np.cumsum(np.tile([5.0, -5.0], 15))
    df = _ohlcv(np.concatenate([lo, hi]))
    vol_fn = lambda d: tag_regimes(d, _P)[VOL_COL]  # noqa: E731 (전구간 백분위 = 사후)
    # 하네스가 vol 의 사후(비인과)성을 정확히 플래그해야 함 (경계 못박기)
    assert check_truncation_invariance(vol_fn, df).leaked
