"""단순 피처 4축 — 변동성변화·상대거래량·semi-dev·종가위치 (Phase 2 Step 2.1).

robust KF(계열1)·ER/Hurst(계열2)를 제외한 나머지 4축. 전부 **OHLCV 만** 입력하고
그 시점까지 정보만 쓴다(인과, 기둥 2). 자기정규화 없음(harness 소유) — 대신 정상성을
**구성으로** 확보한다: 로그비율(변동성변화·상대거래량)·유계값(종가위치 ∈[0,1])·
분수 수익률 기반(semi-dev). ``assert_causal`` 로 인과 회귀(test_features_simple).

**되풀이(정보누수) 방지 (설계 §8)**: 변동성 *수준* 은 배리어 폭과 되풀이라 입력서 제외,
**변화(수축팽창)만** 준다 — 배리어가 안 쓰는 독립 정보.

**DRY**: 변동성 σ 는 ``labeling.volatility.VOL_ESTIMATORS`` 재사용, 하방 준편차는
``validation.metrics.semi_deviation`` 단일출처 재사용(규칙 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.labeling.volatility import VOL_ESTIMATORS
from src.research.validation.metrics import semi_deviation

_OHLC = ("open", "high", "low", "close")


def vol_change(
    df: pd.DataFrame,
    window: int,
    lag: int,
    estimator: str = "atr",
) -> pd.Series:
    """변동성 **변화**(수축팽창) = ``ln(σ_t / σ_{t-lag})`` — 계열3.

    σ 는 인과 추정기(VOL_ESTIMATORS, 분수 정규화). 로그비율이라 대칭·정상적:
    >0 팽창, <0 수축, 0 불변. **수준이 아닌 변화**만 주어 배리어 폭과의 되풀이를
    끊는다(설계 §8). σ<=0(무변동 창)·워밍업은 NaN.

    인과: σ 는 i 까지 창(volatility 가 보장), lag 는 과거 shift → 미래 불참.
    """
    if window <= 0 or lag <= 0:
        raise ValueError("window, lag 는 양수여야 함")
    if estimator not in VOL_ESTIMATORS:
        raise ValueError(f"미지원 estimator: {estimator} ({list(VOL_ESTIMATORS)})")
    sigma = VOL_ESTIMATORS[estimator](df, window)
    # 로그 정의역 밖(σ<=0) → NaN (ln 경고·-inf 방지)
    pos = sigma.where(sigma > 0.0)
    return (np.log(pos) - np.log(pos.shift(lag))).rename("vol_change")


def relative_volume(df: pd.DataFrame, window: int) -> pd.Series:
    """상대거래량 = ``ln(vol_t / MA_past(window))`` — 계열4.

    기저 = **과거 window 봉 평균**(현재 봉 제외, ``shift(1)``) → "현재가 최근 정상
    대비 얼마나". 로그라 정상적(0 정상, >0 급증, <0 한산). 크립토 거래량 신뢰성
    이슈는 하류에서 문턱 높게 소비(설계 §8) — 여기선 측정만. 기저<=0·워밍업은 NaN.

    인과: 기저는 과거 봉만(shift(1)), vol_t 는 봉마감 시점 기지 → 미래 불참.
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    if "volume" not in df.columns:
        raise ValueError("volume 컬럼 누락")
    vol = df["volume"]
    baseline = vol.rolling(window).mean().shift(1)   # 현재 봉 제외한 과거 기저
    pos = baseline.where(baseline > 0.0)
    return (np.log(vol.where(vol > 0.0)) - np.log(pos)).rename("relative_volume")


def rolling_semi_deviation(df: pd.DataFrame, window: int) -> pd.Series:
    """하방 준편차 = 수익률 window 창의 ``semi_deviation`` — 계열5 (팻테일 측정).

    로그수익률 ``ln(close_t/close_{t-1})`` 의 창별 하방 준편차(target=창평균). KF 가
    *제거* 한 팻테일의 **크기** 를 측정 → KF(추세)와 상보(설계 §8). semi_deviation 은
    ``metrics`` 단일출처 재사용(DRY). 분수 수익률 기반이라 스케일 무관·정상적.

    인과: window 는 t 에서 끝나는 과거 수익률만 → 미래 불참.
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    ret = np.log(df["close"] / df["close"].shift(1))
    sd = ret.rolling(window).apply(
        lambda w: semi_deviation(w), raw=True
    )
    return sd.rename("semi_dev")


def close_position(df: pd.DataFrame) -> pd.Series:
    """종가위치 = ``(close - low) / (high - low)`` ∈[0,1] — 계열6 (봉 압력).

    봉 범위 내 종가 위치: 1=고가 마감(매수압), 0=저가 마감(매도압), 0.5=중립.
    range<=0(무변동 봉)은 중립 0.5. **같은 봉 OHLC 만** 쓰므로 자명하게 인과
    (창 없음, 워밍업 없음). 최하 우선순위 축(설계 §8).
    """
    missing = [c for c in _OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC 컬럼 누락 {missing}")
    rng = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / rng.where(rng > 0.0)
    return pos.fillna(0.5).rename("close_position")
