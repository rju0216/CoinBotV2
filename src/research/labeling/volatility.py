"""변동성 추정기 — ATR·Yang-Zhang (삼중배리어 폭 재료, MasterPlan Phase 1 / 설계 §4).

**공통 콜러블 인터페이스**: ``fn(df, window) -> pd.Series`` — 봉별 변동성 σ_i 를 반환.
estimator 를 콜러블로 통일해 삼중배리어 스캔·R0 스윕이 종류를 모른 채 교체·순회한다
(3번째 추정기 추가는 함수 1개 + 레지스트리 등록으로 끝 — 미래확장성).

**분수(상대) 정규화**: 두 추정기 모두 **가격 대비 분수**(dimensionless)로 σ 를 낸다.
그래서 배리어는 하류에서 ``close_i · (1 ± x·σ_i)`` 로 그어지고, 배수 x 가 추정기 간
대략 비교 가능해진다(설계 §7 상대량화 → 정상성; x 는 여전히 추정기별 탐색축).

**인과성 (기둥 2)**: σ_i 는 **i 에서 끝나는 창**의 과거 봉만 쓴다(미래 불참). YZ 는
overnight 항에 close_{i-1} 이 필요해 유효 시작이 ATR 보다 1봉 늦다. 워밍업은 NaN.
이 인과성은 ``causality.leakage.assert_causal`` 로 회귀 강제한다(test_volatility).

주의: **배리어 폭**은 인과여야 하지만(여기서 보장), **라벨 자체**는 forward 다
(미래 경로로 어느 배리어를 먼저 쳤나 판정) — 그건 삼중배리어의 관심사다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

VolEstimator = Callable[[pd.DataFrame, int], pd.Series]

_OHLC = ("open", "high", "low", "close")


def _require_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in _OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC 컬럼 누락 {missing}")


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """ATR(평균 True Range)을 **분수(가격 대비)** 로 반환.

    True Range = max(high-low, |high-close_prev|, |low-close_prev|) — 갭 내성(설계 §4).
    각 봉 TR 을 그 봉 close 로 정규화(``tr_frac``)한 뒤 window 이동평균 → 창 내 가격이
    크게 추세여도 정상성 유지(설계 §7). 인과: 창은 i 에서 끝나고 close_prev 는 과거.
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    _require_ohlc(df)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr_frac = tr / close                      # 봉별 상대 True Range
    return tr_frac.rolling(window).mean().rename("atr")


def yang_zhang(df: pd.DataFrame, window: int) -> pd.Series:
    """Yang-Zhang 변동성을 **분수(봉당 로그수익 표준편차)** 로 반환.

    σ²_YZ = σ²_overnight + k·σ²_open + σ²_RS  (설계 §4: OHLC 최대 활용·갭/추세 강건).
    - overnight o = ln(open_i / close_{i-1}), open-close c = ln(close_i / open_i)
    - Rogers-Satchell RS = ln(hi/cl)·ln(hi/op) + ln(lo/cl)·ln(lo/op)  (드리프트 독립)
    - k = 0.34 / (1.34 + (w+1)/(w-1))
    σ_overnight·σ_open 은 표본분산(ddof=1), σ²_RS 는 창 평균. 인과: 전부 i 까지 창.
    """
    if window <= 1:
        raise ValueError("yang_zhang window 는 2 이상이어야 함")
    _require_ohlc(df)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    prev_c = c.shift(1)

    overnight = np.log(o / prev_c)            # 갭 (close_{i-1} → open_i)
    open_close = np.log(c / o)                # 개장 (open_i → close_i)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    var_overnight = overnight.rolling(window).var(ddof=1)
    var_open = open_close.rolling(window).var(ddof=1)
    var_rs = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_yz = var_overnight + k * var_open + var_rs
    # 수치 오차로 미세 음수가 나올 수 있어 하한 0 (sqrt NaN 방지)
    return np.sqrt(var_yz.clip(lower=0.0)).rename("yz")


VOL_ESTIMATORS: dict[str, VolEstimator] = {
    "atr": atr,
    "yz": yang_zhang,
}
