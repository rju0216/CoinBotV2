"""국면(regime) 태깅 — 규칙 기반, 분석 전용.

MasterPlan D-003/D-006/§11. 매크로 국면을 1d 에서 계산하고, 추세 3종
(up/flat/down)을 1차 국면으로, 변동성 2종(high/low)을 오버레이로 둔다.

**방화벽 (기둥 2 인과성)**: regime 태그는 결과를 슬라이스하는 **분석 전용** 축이다.
모델 입력이 아니므로 사후(전구간) 통계 사용이 허용된다(변동성 백분위 문턱).
그러나 **feature 파이프라인으로 유입되면 즉시 누수**다. 이를 강제하기 위해 출력
컬럼은 전부 ``REGIME_PREFIX`` 로 네임스페이스한다 — Phase 2 feature 조립부는
이 접두어 컬럼을 반드시 배제한다(회귀 테스트로 강제).

수치(ma_len·slope_k·deadband·vol_len·vol_hi_pct·min_duration)는 6.5년 태깅
세그먼테이션을 리뷰해 확정한다(캘리브레이션 — 기둥 5 데이터가 정함).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# regime 출력 네임스페이스 (feature 방화벽의 계약): 모든 출력 컬럼이 이 접두어로
# 시작한다. Phase 2 feature 조립부는 이 접두어 컬럼을 전부 배제한다(회귀 테스트).
REGIME_PREFIX = "regime"
TREND_COL = "regime_trend"      # up | flat | down
VOL_COL = "regime_vol"          # high | low
LABEL_COL = "regime"            # 결합 라벨 (예: "up|high")

TREND_STATES = ("up", "flat", "down")
VOL_STATES = ("high", "low")


@dataclass(frozen=True)
class RegimeParams:
    """국면 태깅 파라미터 (기본값 = Step 0.1 캘리브레이션 확정값).

    2026-07-11 6.5년 BTC 1d 태깅 세그먼테이션 리뷰로 확정. 추세 분포 up/down/flat =
    1042/689/521(붕괴 없음), 28 세부 세그먼트가 ~7-8 매크로 에라로 뭉침. regime 은
    선별 도구가 아니라 분석 축이므로 과도 튜닝(=재량 주입)을 피해 확정.
    """

    ma_len: int = 100          # 추세 MA 길이(봉)
    slope_k: int = 20          # MA 기울기 측정 지평(봉)
    deadband: float = 0.02     # |MA 변화율| 이 이하이면 flat
    vol_len: int = 30          # 실현변동성 창(봉)
    vol_hi_pct: float = 0.5    # 이 백분위 초과이면 high (사후·전구간)
    min_duration: int = 14     # 히스테리시스: 새 추세가 이만큼 지속돼야 전환


def _apply_min_duration(states: pd.Series, min_duration: int) -> pd.Series:
    """히스테리시스: 새 상태가 min_duration 봉 연속될 때까지 기존 상태를 유지.

    NaN(워밍업) 은 그대로 통과시키고, 첫 유효 상태를 초기 committed 로 삼는다.
    전환은 확정 시점부터 반영(직전 잔진동 구간은 소급 relabel 하지 않음 — 인과 친화).
    """
    if min_duration <= 1:
        return states.copy()

    out = states.copy()
    committed: str | None = None
    run_val: str | None = None
    run_len = 0

    for i in range(len(states)):
        s = states.iloc[i]
        if s is None or (isinstance(s, float) and np.isnan(s)):
            out.iloc[i] = committed  # 워밍업 이후 갭 방어; 시작 워밍업은 그대로 None
            if committed is None:
                out.iloc[i] = s
            continue
        if committed is None:
            committed = s
            out.iloc[i] = committed
            run_val, run_len = s, 1
            continue
        if s == committed:
            run_val, run_len = s, 0
            out.iloc[i] = committed
            continue
        # s != committed → 지속성 카운트
        if s == run_val:
            run_len += 1
        else:
            run_val, run_len = s, 1
        if run_len >= min_duration:
            committed = s
            run_len = 0
        out.iloc[i] = committed
    return out


def _trend_states(close: pd.Series, p: RegimeParams) -> pd.Series:
    """추세 상태 시리즈(up/flat/down + 히스테리시스). **인과** — 과거 MA slope 만 사용,
    ``_apply_min_duration`` 은 소급 relabel 하지 않음(전환은 확정 시점부터). tag_regimes(분석)
    과 causal_tag_regimes(정책) 공용 — vol 축만 두 함수가 다르다(DRY)."""
    ma = close.rolling(p.ma_len).mean()
    ma_prev = ma.shift(p.slope_k)
    slope = (ma - ma_prev) / ma_prev  # 분모 인과: 과거 MA
    trend_raw = pd.Series(index=close.index, dtype=object)
    trend_raw[slope > p.deadband] = "up"
    trend_raw[slope < -p.deadband] = "down"
    trend_raw[(slope >= -p.deadband) & (slope <= p.deadband)] = "flat"
    # slope NaN(워밍업) → NaN 유지
    return _apply_min_duration(trend_raw, p.min_duration)


def tag_regimes(
    df: pd.DataFrame, params: RegimeParams | None = None
) -> pd.DataFrame:
    """OHLCV(권장 1d) → 국면 태그 DataFrame.

    반환: index=입력 index, 컬럼 = regime_trend / regime_vol / regime.
    워밍업(창 부족) 구간은 NaN.
    """
    p = params or RegimeParams()
    close = df["close"].astype(float)

    # --- 추세: 공유 _trend_states (인과 MA slope + 히스테리시스) ---
    trend = _trend_states(close, p)

    # --- 변동성: 로그수익률 실현변동성 → 사후 백분위 문턱(분석 전용) ---
    logret = np.log(close / close.shift(1))
    rv = logret.rolling(p.vol_len).std()
    thr = rv.quantile(p.vol_hi_pct)  # 전구간 백분위 (ex-post 허용)
    vol = pd.Series(index=df.index, dtype=object)
    vol[rv > thr] = "high"
    vol[rv <= thr] = "low"
    # rv NaN → NaN

    out = pd.DataFrame(index=df.index)
    out[TREND_COL] = trend
    out[VOL_COL] = vol
    # 결합 라벨: 둘 다 유효할 때만
    both = trend.notna() & vol.notna()
    label = pd.Series(index=df.index, dtype=object)
    label[both] = trend[both].astype(str) + "|" + vol[both].astype(str)
    out[LABEL_COL] = label
    return out


# Phase 6 causal 국면 (정책층 매매필터용). 사전등록·무튜닝 값.
CAUSAL_VOL_WINDOW_BARS = 365   # rolling 백분위 창(일봉 기준 1년). 2026-07 확정, 채점 중 불변.


def causal_tag_regimes(
    df: pd.DataFrame,
    params: RegimeParams | None = None,
    vol_window_bars: int = CAUSAL_VOL_WINDOW_BARS,
) -> pd.DataFrame:
    """OHLCV(권장 1d) → **인과** 국면 태그 (Phase 6 정책층 매매필터용, I-005 해소).

    ``tag_regimes`` 와의 유일한 차이 = **vol 문턱**. tag_regimes 는 전구간 백분위(사후·
    분석전용)를 쓰지만, 이 함수는 **rolling(vol_window_bars) 백분위**로 과거 창만 사용해
    lookahead 를 제거한다. 추세축은 공유 ``_trend_states`` (이미 인과). ``assert_causal``
    (절단불변+미래교란) 통과가 계약이다.

    ★방화벽★: 이 출력은 **정책층(3층) 진입게이트 입력**이지 **feature(모델 입력)가 아니다.**
    분석전용 tag_regimes 와 달리 매매에 실제로 쓰이므로 인과가 필수. feature 파이프라인엔
    절대 유입 금지(모델 재fit 아님 — 얼린 예측 위 정책 소비, 기둥3).

    vol_window_bars: rolling 백분위 창(봉). 일봉 365=1년(사전등록·무튜닝).
    반환: tag_regimes 와 동일 스키마(regime_trend/regime_vol/regime), 워밍업 NaN.
    """
    p = params or RegimeParams()
    close = df["close"].astype(float)

    # --- 추세: 공유 _trend_states (인과) ---
    trend = _trend_states(close, p)

    # --- 변동성: 실현변동성 → **rolling 백분위 문턱**(인과, 과거 창만) ---
    logret = np.log(close / close.shift(1))
    rv = logret.rolling(p.vol_len).std()
    thr = rv.rolling(vol_window_bars, min_periods=vol_window_bars).quantile(p.vol_hi_pct)
    vol = pd.Series(index=df.index, dtype=object)
    valid = rv.notna() & thr.notna()
    vol[valid & (rv > thr)] = "high"
    vol[valid & (rv <= thr)] = "low"
    # rv/thr NaN(워밍업) → vol NaN 유지

    out = pd.DataFrame(index=df.index)
    out[TREND_COL] = trend
    out[VOL_COL] = vol
    both = trend.notna() & vol.notna()
    label = pd.Series(index=df.index, dtype=object)
    label[both] = trend[both].astype(str) + "|" + vol[both].astype(str)
    out[LABEL_COL] = label
    return out


# forward_fill_completed 는 data/loader.py 로 승격됨(TF정렬 공용 헬퍼, MTF·regime 공용).
