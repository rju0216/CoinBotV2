"""추세강도/레짐 피처 2축 — ER · Hurst (Phase 2 Step 2.2, 설계 §8 계열2).

두 축은 **보완**한다(설계 §8): ER 은 "지금 움직임이 얼마나 깨끗한가(추세 강도)",
Hurst 는 "이 시장이 밀고 가는 성질인가 되돌아오는 성질인가(추세추종 vs 평균회귀)".
둘 다 **가격(close)에서만** 계산 — regime 태그(``validation.regime``, 사후·방화벽)와
무관하다. 이 모듈이 "regime" 단어를 안 쓰는 이유(태그가 피처로 새면 즉시 누수, 기둥 2).

**Hurst 방식 = R/S (Rescaled Range)** — 사용자 확정(세션). 표준·단순·해석명확. 추세오염
강건성(DFA)이 Phase 2 에서 이점 작음(로그수익 기반 + harness 정규화 + KF 추세분리로 이미
커버) → 기본축엔 과함. Phase 3 에서 예측력이 요구하면 DFA/VR 로 격상(장식금지·설계 §8).

인과: 두 축 모두 **창이 t 에서 끝나는 과거 정보만** → ``assert_causal`` 회귀.
정상성: ER∈[0,1] 유계, Hurst R/S 는 [0,1] 로 클립(이론 범위, 유계화).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def efficiency_ratio(df: pd.DataFrame, window: int) -> pd.Series:
    """Kaufman ER = |순변화| / |경로총길이| ∈[0,1] — 추세/노이즈 비율.

    분자 = ``|close_t - close_{t-w}|`` (창 순 방향 변화), 분모 = 창 내 봉별 변화 절대합
    (실제 지나온 경로). 1=완전 추세(직선), 0=제자리 왕복(순수 노이즈). 무변동 창
    (경로 0)은 NaN. 인과: 분자·분모 모두 창 t 까지 정보.
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    close = df["close"]
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (net / path.where(path > 0.0)).rename("er")


def _rs_hurst(x: np.ndarray, min_n: int = 8) -> float:
    """R/S 슬로프로 Hurst 추정. x = 창 내 (로그)수익률 배열.

    창을 여러 하위길이 n(창→창/2→…≥min_n)으로 쪼개, 각 n 에서 청크별 rescaled range
    R/S 평균을 구하고 log(R/S) ~ log(n) 회귀 기울기 = H. 랜덤워크≈0.5, >0.5 추세추종,
    <0.5 평균회귀. NaN(첫 봉 수익률)은 제외. 표본 부족·퇴화 창은 NaN.
    """
    x = x[~np.isnan(x)]
    w = len(x)
    if w < 2 * min_n:
        return np.nan
    # 하위 창 길이 (창 → 절반 → … ≥ min_n)
    ns: list[int] = []
    n = w
    while n >= min_n:
        ns.append(n)
        n //= 2
    ns = sorted(set(ns))
    if len(ns) < 2:
        return np.nan

    log_n: list[float] = []
    log_rs: list[float] = []
    for n in ns:
        k = w // n                      # 비겹침 청크 수
        rs_vals: list[float] = []
        for j in range(k):
            chunk = x[j * n:(j + 1) * n]
            z = np.cumsum(chunk - chunk.mean())
            r = z.max() - z.min()
            s = chunk.std()             # 모표준편차
            if s > 0.0 and r > 0.0:
                rs_vals.append(r / s)
        if rs_vals:
            log_n.append(np.log(n))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_n) < 2:
        return np.nan
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))   # 이론 범위 [0,1] 로 유계화


def hurst(df: pd.DataFrame, window: int, min_n: int = 8) -> pd.Series:
    """Hurst 지수(R/S) — 창 내 로그수익률의 장기의존성 ∈[0,1] (설계 §8 계열2).

    로그수익률 ``ln(close_t/close_{t-1})`` 의 window 창에 R/S 회귀. >0.5 추세추종,
    0.5 무작위, <0.5 평균회귀. 창은 t 에서 끝남(인과). window 는 하위창 분할이 가능하도록
    ``min_n`` 의 배수 이상 권장(부족 시 NaN).
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    ret = np.log(df["close"] / df["close"].shift(1))
    h = ret.rolling(window).apply(lambda w: _rs_hurst(w, min_n), raw=True)
    return h.rename("hurst")
