"""research 데이터 로더 회귀 테스트 (합성 CSV — 실데이터 비의존)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data.audit import audit_continuity
from src.research.data.loader import (
    OHLCV_COLUMNS,
    csv_filename,
    load_ohlcv,
    resample_ohlcv,
)


def _write_csv(path, index, **cols):
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(index, name="timestamp"))
    df.to_csv(path)


def test_csv_filename_convention():
    assert csv_filename("BTC/USDT:USDT", "1h") == "BTC_USDT_USDT_1h.csv"
    assert csv_filename("ETH/USDT:USDT", "4h") == "ETH_USDT_USDT_4h.csv"


def test_load_ohlcv_normalizes(tmp_path):
    idx = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    path = tmp_path / csv_filename("BTC/USDT:USDT", "1h")
    _write_csv(
        path, idx,
        open=[1, 2, 3, 4, 5], high=[2, 3, 4, 5, 6], low=[0.5, 1, 2, 3, 4],
        close=[1.5, 2.5, 3.5, 4.5, 5.5], volume=[10, 10, 10, 10, 10],
    )
    df = load_ohlcv("BTC/USDT:USDT", "1h", candle_dir=str(tmp_path))

    assert list(df.columns) == OHLCV_COLUMNS
    assert df.index.tz is not None          # UTC tz-aware
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "timestamp"
    assert df.dtypes.map(lambda d: d == np.float64).all()
    assert len(df) == 5


def test_load_ohlcv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ohlcv("BTC/USDT:USDT", "1h", candle_dir=str(tmp_path))


def test_load_ohlcv_unsupported_timeframe(tmp_path):
    with pytest.raises(ValueError):
        load_ohlcv("BTC/USDT:USDT", "7m", candle_dir=str(tmp_path))


def test_load_ohlcv_missing_column(tmp_path):
    idx = pd.date_range("2020-01-01", periods=3, freq="1h", tz="UTC")
    path = tmp_path / csv_filename("BTC/USDT:USDT", "1h")
    # volume 누락
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [0.5, 1, 2], "close": [1.5, 2.5, 3.5]},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    df.to_csv(path)
    with pytest.raises(ValueError):
        load_ohlcv("BTC/USDT:USDT", "1h", candle_dir=str(tmp_path))


# ---- I-001 remediation: 1h→1d 재표본 헬퍼 ----

def _hourly(n_days, start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))  # high ≥ o,c
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))  # low ≤ o,c
    return pd.DataFrame({
        "open": open_, "high": hi, "low": lo, "close": close,
        "volume": np.abs(rng.normal(1000, 200, n)),
    }, index=idx)


def test_resample_1h_to_1d_ohlcv_agg():
    h = _hourly(3)
    d = resample_ohlcv(h, "1h", "1d")
    assert len(d) == 3
    # 첫 일봉 = 그날 24봉 집계
    day0 = h.iloc[:24]
    assert d["open"].iloc[0] == pytest.approx(day0["open"].iloc[0])
    assert d["high"].iloc[0] == pytest.approx(day0["high"].max())
    assert d["low"].iloc[0] == pytest.approx(day0["low"].min())
    assert d["close"].iloc[0] == pytest.approx(day0["close"].iloc[-1])
    assert d["volume"].iloc[0] == pytest.approx(day0["volume"].sum())
    assert (d.index.hour == 0).all()          # 00:00 UTC 그리드


def test_resample_drops_partial_buckets():
    # 마지막 부분일(12봉)은 완전 버킷 아님 → 드롭
    h = _hourly(2).iloc[: 24 + 12]            # 2일 + 반나절
    d = resample_ohlcv(h, "1h", "1d")
    assert len(d) == 1                        # 완전한 1일만


def test_resample_derived_1d_passes_audit():
    # 파생 1d 는 감사 통과(off-grid/부분봉 corruption 없음) — I-001 핵심
    h = _hourly(40)
    d = resample_ohlcv(h, "1h", "1d")
    assert not audit_continuity(d, "1d").has_corruption


def test_resample_rejects_non_multiple_or_downscale():
    h = _hourly(2)
    with pytest.raises(ValueError):
        resample_ohlcv(h, "1d", "1h")         # 하향 불가
    with pytest.raises(ValueError):
        resample_ohlcv(h, "1h", "7m")         # 미지원 TF
