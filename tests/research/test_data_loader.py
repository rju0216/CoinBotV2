"""research 데이터 로더 회귀 테스트 (합성 CSV — 실데이터 비의존)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import OHLCV_COLUMNS, csv_filename, load_ohlcv


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
