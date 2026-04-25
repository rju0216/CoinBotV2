"""공통 지표 라이브러리. 전략들이 공유하는 기술적 지표 계산 함수 모음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def compute_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return ta.ema(df[col], length=period)


def compute_sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return ta.sma(df[col], length=period)


def compute_ma(df: pd.DataFrame, period: int, ma_type: str = "ema") -> pd.Series:
    if ma_type == "sma":
        return compute_sma(df, period)
    return compute_ema(df, period)


def compute_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    macd.columns = ["macd", "histogram", "signal"]
    return macd


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    adx = ta.adx(df["high"], df["low"], df["close"], length=period)
    result = pd.DataFrame(index=df.index)
    result["adx"] = adx.iloc[:, 0]
    result["plus_di"] = adx.iloc[:, 1]
    result["minus_di"] = adx.iloc[:, 2]
    return result


def compute_bbands(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.DataFrame:
    bb = ta.bbands(df["close"], length=period, std=std)
    result = pd.DataFrame(index=df.index)
    result["lower"] = bb.iloc[:, 0]
    result["mid"] = bb.iloc[:, 1]
    result["upper"] = bb.iloc[:, 2]
    return result


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.atr(df["high"], df["low"], df["close"], length=period)


def compute_choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index: 높을수록 횡보, 낮을수록 추세."""
    atr_1 = ta.atr(df["high"], df["low"], df["close"], length=1)
    atr_sum = atr_1.rolling(period).sum()
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    hl_range = high_max - low_min
    hl_range = hl_range.replace(0, np.nan)
    return 100 * np.log10(atr_sum / hl_range) / np.log10(period)


def compute_efficiency_ratio(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio: 높을수록 추세, 낮을수록 횡보."""
    direction = (df["close"] - df["close"].shift(period)).abs()
    volatility = df["close"].diff().abs().rolling(period).sum()
    volatility = volatility.replace(0, np.nan)
    return direction / volatility


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return ta.rsi(df["close"], length=period)


def compute_bb_width(
    df: pd.DataFrame, period: int = 20, std: float = 2.0
) -> pd.Series:
    """Bollinger Band Width (%): 변동성 확장/수축 측정."""
    bb = compute_bbands(df, period, std)
    mid = bb["mid"].replace(0, np.nan)
    return (bb["upper"] - bb["lower"]) / mid * 100
