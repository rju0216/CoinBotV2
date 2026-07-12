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
from numpy.lib.stride_tricks import sliding_window_view


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

    **벡터화(F-13)**: ``sliding_window_view`` 로 전 창을 (M, window) 배열화해 sub-window별
    R/S 를 전 창 동시 계산 → ``rolling.apply``(창당 Python 호출) 대비 대폭 가속. 결과는
    스칼라 ``_rs_hurst`` 와 **수치 동치**(회귀 검증). NaN 포함 창(첫 창의 ret[0]=NaN) 및 퇴화
    창(어떤 sub-window 길이의 유효 청크 0)은 ``_rs_hurst`` 로 폴백해 정확 매칭.
    """
    if window <= 0:
        raise ValueError("window 는 양수여야 함")
    close = df["close"]
    ret = np.log(close / close.shift(1)).to_numpy()
    n_total = len(ret)
    out = np.full(n_total, np.nan)
    if n_total < window:
        return pd.Series(out, index=df.index, name="hurst")

    # sub-window 길이 (window → 절반 → … ≥ min_n) — _rs_hurst 와 동일 규칙.
    # len(ns)<2 ⟺ window<2·min_n (첫 절반이 min_n 미만) → _rs_hurst 의 하한과 동형.
    ns: list[int] = []
    n = window
    while n >= min_n:
        ns.append(n)
        n //= 2
    ns = sorted(set(ns))
    if len(ns) < 2:
        return pd.Series(out, index=df.index, name="hurst")

    W = sliding_window_view(ret, window)      # (M, window), 행 i → 출력 index i+window-1
    m = W.shape[0]
    nan_rows = np.isnan(W).any(axis=1)          # NaN 포함 창(주로 ret[0] 포함 첫 창)

    log_n = np.log(np.asarray(ns, dtype=float))
    log_rs = np.full((m, len(ns)), np.nan)
    for ni, nlen in enumerate(ns):
        k = window // nlen                       # 비겹침 청크 수
        sum_rs = np.zeros(m)
        cnt = np.zeros(m)
        for j in range(k):
            chunk = W[:, j * nlen:(j + 1) * nlen]
            z = np.cumsum(chunk - chunk.mean(axis=1, keepdims=True), axis=1)
            r = z.max(axis=1) - z.min(axis=1)
            s = chunk.std(axis=1)                # 모표준편차(ddof=0), _rs_hurst 동일
            valid = (s > 0.0) & (r > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = np.where(valid, r / np.where(s > 0.0, s, 1.0), 0.0)
            sum_rs += rs
            cnt += valid
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_rs = np.where(cnt > 0, sum_rs / np.where(cnt > 0, cnt, 1.0), np.nan)
            log_rs[:, ni] = np.log(mean_rs)

    # 정상 경로: 모든 sub-window 유효(대부분) → 벡터화 최소제곱 기울기.
    finite = np.isfinite(log_rs)
    simple = (finite.sum(axis=1) == len(ns)) & (~nan_rows)
    xm = log_n - log_n.mean()
    denom = float((xm ** 2).sum())
    with np.errstate(invalid="ignore"):
        ym = log_rs - log_rs.mean(axis=1, keepdims=True)
        slope = (ym * xm).sum(axis=1) / denom    # polyfit(deg=1) 기울기와 동일
    res = np.where(simple, np.clip(slope, 0.0, 1.0), np.nan)

    # NaN 포함 창은 NaN 유지 — pandas rolling(min_periods=window)이 유효관측<window 를
    # 미계산 NaN 처리하는 것과 매칭(폴백 금지). 폴백은 **유한하나 퇴화한** 행만(드묾).
    degenerate = (~simple) & (~nan_rows)
    for i in np.where(degenerate)[0]:
        res[i] = _rs_hurst(W[i], min_n)

    out[window - 1:] = res
    return pd.Series(out, index=df.index, name="hurst")
