"""국면 태깅 회귀 테스트 — 추세 판정·히스테리시스·완성봉 인과성·방화벽 계약."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data.loader import forward_fill_completed
from src.research.validation.regime import (
    LABEL_COL,
    REGIME_PREFIX,
    TREND_COL,
    VOL_COL,
    RegimeParams,
    _apply_min_duration,
    tag_regimes,
)


def _price_df(closes, freq="D"):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq=freq, tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1.0},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


# 테스트용 작은 창 (hysteresis off)
_P = RegimeParams(ma_len=5, slope_k=3, deadband=0.001, vol_len=5, vol_hi_pct=0.5, min_duration=1)


def test_uptrend_tagged_up():
    df = _price_df([100 * (1.02 ** i) for i in range(40)])  # 꾸준한 상승
    out = tag_regimes(df, _P)
    tail = out[TREND_COL].dropna()
    assert (tail == "up").all()


def test_downtrend_tagged_down():
    df = _price_df([100 * (0.98 ** i) for i in range(40)])  # 꾸준한 하락
    out = tag_regimes(df, _P)
    tail = out[TREND_COL].dropna()
    assert (tail == "down").all()


def test_flat_tagged_flat():
    df = _price_df([100.0] * 40)  # 완전 횡보
    out = tag_regimes(df, _P)
    tail = out[TREND_COL].dropna()
    assert (tail == "flat").all()


def test_warmup_is_nan():
    df = _price_df([100 * (1.02 ** i) for i in range(40)])
    out = tag_regimes(df, _P)
    # ma_len+slope_k 이전 구간은 추세 NaN
    assert out[TREND_COL].iloc[0] is None or pd.isna(out[TREND_COL].iloc[0])


def test_min_duration_holds_regime_through_blip():
    # down 지속 중 2봉짜리 up 블립 → min_duration=3 이면 전환 안 됨
    states = pd.Series(
        ["down"] * 6 + ["up", "up"] + ["down"] * 6,
        index=pd.RangeIndex(14),
        dtype=object,
    )
    committed = _apply_min_duration(states, min_duration=3)
    assert (committed == "down").all()


def test_min_duration_switches_when_persistent():
    states = pd.Series(
        ["down"] * 5 + ["up"] * 5,
        index=pd.RangeIndex(10),
        dtype=object,
    )
    committed = _apply_min_duration(states, min_duration=3)
    # 앞 5개 down, 이후 up 이 3봉 지속되면 전환 → 마지막은 up
    assert committed.iloc[0] == "down"
    assert committed.iloc[-1] == "up"


def test_forward_fill_completed_uses_only_completed_bar():
    # 1d 값 A(01-01) B(01-02) C(01-03)
    higher = pd.Series(
        ["A", "B", "C"],
        index=pd.DatetimeIndex(
            ["2020-01-01", "2020-01-02", "2020-01-03"], tz="UTC", name="timestamp"
        ),
    )
    lower = pd.DatetimeIndex(
        ["2020-01-02 00:00", "2020-01-02 05:00", "2020-01-03 00:00"],
        tz="UTC", name="timestamp",
    )
    got = forward_fill_completed(higher, lower, "1d")
    # 01-02 00:00: 01-01 봉만 완성 → A (진행 중 01-02 봉 배제)
    # 01-02 05:00: 여전히 A
    # 01-03 00:00: 01-02 봉 완성 → B (진행 중 01-03 배제)
    assert list(got.values) == ["A", "A", "B"]


def test_firewall_all_columns_namespaced():
    df = _price_df([100 * (1.01 ** i) for i in range(40)])
    out = tag_regimes(df, _P)
    # 방화벽 계약: 모든 출력 컬럼이 regime 네임스페이스 → feature 조립부가 배제 가능
    assert all(c.startswith(REGIME_PREFIX) for c in out.columns)
    assert set(out.columns) == {TREND_COL, VOL_COL, LABEL_COL}
