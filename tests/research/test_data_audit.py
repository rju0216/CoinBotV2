"""연속성·정합성 감사 회귀 테스트 (합성 데이터)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.data.audit import AuditError, audit_continuity


def _clean(n=48, freq="1h"):
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.5] * n,
            "volume": [100.0] * n,
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


def test_clean_no_corruption_full_coverage():
    rep = audit_continuity(_clean(), "1h")
    assert not rep.has_corruption
    assert rep.n_duplicates == 0
    assert rep.monotonic
    assert len(rep.gaps) == 0
    assert rep.n_missing_bars == 0
    assert rep.coverage == pytest.approx(1.0)
    rep.raise_if_corrupt()  # 예외 없어야 함


def test_gap_detected_and_counted():
    df = _clean(48)
    # 인덱스 10~12(2봉)을 제거해 갭 생성
    df = df.drop(df.index[10:12])
    rep = audit_continuity(df, "1h")
    assert not rep.has_corruption          # 갭은 손상 아님
    assert len(rep.gaps) == 1
    assert rep.gaps[0].missing == 2
    assert rep.n_missing_bars == 2
    assert rep.coverage < 1.0


def test_duplicate_is_corruption():
    df = _clean(10)
    df = pd.concat([df, df.iloc[[5]]])     # 한 timestamp 중복
    rep = audit_continuity(df, "1h")
    assert rep.n_duplicates == 1
    assert rep.has_corruption
    with pytest.raises(AuditError):
        rep.raise_if_corrupt()


def test_non_monotonic_is_corruption():
    df = _clean(10)
    df = df.iloc[[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]]  # 순서 뒤섞음
    rep = audit_continuity(df, "1h")
    assert not rep.monotonic
    assert rep.has_corruption


def test_ohlc_violation_detected():
    df = _clean(10)
    df.iloc[3, df.columns.get_loc("high")] = 9.5   # high < close(10.5)
    rep = audit_continuity(df, "1h")
    assert rep.n_ohlc_violations == 1
    assert rep.has_corruption


def test_nan_detected():
    df = _clean(10)
    df.iloc[4, df.columns.get_loc("close")] = float("nan")
    rep = audit_continuity(df, "1h")
    assert rep.n_nan_rows == 1
    assert rep.has_corruption


def test_nonpositive_price_detected():
    df = _clean(10)
    df.iloc[2, df.columns.get_loc("low")] = 0.0
    rep = audit_continuity(df, "1h")
    assert rep.n_nonpositive_price == 1
    assert rep.has_corruption


def test_offgrid_bar_is_irregular_not_gap():
    # 1d 그리드에 16:00 오프-그리드 봉 삽입 (정수배 아님 → irregular, gap 아님)
    df = _clean(6, freq="D")
    extra = df.iloc[[2]].copy()
    extra.index = pd.DatetimeIndex([df.index[2] + pd.Timedelta(hours=16)], name="timestamp")
    df = pd.concat([df, extra]).sort_index()
    rep = audit_continuity(df, "1d")
    assert rep.n_irregular > 0
    assert len(rep.gaps) == 0           # 오프-그리드는 갭으로 오분류되지 않음
    assert rep.has_corruption


def test_clean_gap_is_multiple_not_irregular():
    # 정확히 2일 결측(정수배) → gap, irregular 아님
    df = _clean(8, freq="D")
    df = df.drop(df.index[3:5])         # 2봉 제거
    rep = audit_continuity(df, "1d")
    assert rep.n_irregular == 0
    assert len(rep.gaps) == 1
    assert rep.gaps[0].missing == 2
