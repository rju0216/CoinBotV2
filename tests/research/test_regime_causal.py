"""causal_tag_regimes 인과성 seam 테스트 (Phase 6 Step 6.0, I-005 해소 박제).

계약: causal_tag_regimes 는 vol 문턱을 **rolling 백분위**(과거 창만)로 계산해
``assert_causal``(절단불변 + 미래교란)을 통과한다. 대조로 tag_regimes(분석·전구간
quantile)의 vol 축은 절단 시 바뀜을 박제 — 왜 인과 버전이 필요했나(I-005).
추세축은 두 함수가 공유(_trend_states)라 동일해야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.causality.leakage import assert_causal, check_truncation_invariance
from src.research.validation.regime import (
    LABEL_COL,
    TREND_COL,
    VOL_COL,
    RegimeParams,
    causal_tag_regimes,
    tag_regimes,
)

_P = RegimeParams()  # 확정 기본값 (ma_len100·vol_len30·vol_hi_pct0.5 …)


def _vol_switch_df(n: int = 520, seed: int = 0) -> pd.DataFrame:
    """변동성 국면 스위칭 일봉 (저변동 0.01 ↔ 고변동 0.05, 60봉 주기).

    전구간 quantile(비인과)이 절단에 민감하도록 vol 이 시대별로 크게 다른 시계열.
    n≥ vol_len(30)+vol_window(365) 여야 인과 vol 이 유효 구간을 가진다.
    """
    rng = np.random.default_rng(seed)
    vols = np.where((np.arange(n) // 60) % 2 == 0, 0.01, 0.05)
    rets = rng.normal(0.0, vols)
    close = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2019-06-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1.0},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


def _encode(out: pd.DataFrame) -> pd.DataFrame:
    """카테고리 국면 → 수치 (assert_causal 의 rtol/atol 비교용). 워밍업 NaN 유지."""
    tmap = {"up": 1.0, "flat": 0.0, "down": -1.0}
    vmap = {"high": 1.0, "low": -1.0}
    return pd.DataFrame(
        {"trend": out[TREND_COL].map(tmap), "vol": out[VOL_COL].map(vmap)},
        index=out.index,
    )


def test_causal_regime_passes_assert_causal():
    """causal_tag_regimes 전체(추세+vol) 절단불변+미래교란 통과 (I-005 해소 핵심)."""
    df = _vol_switch_df()
    report = assert_causal(
        lambda d: _encode(causal_tag_regimes(d, _P, vol_window_bars=365)), df
    )
    assert not report.leaked


def test_noncausal_vol_fails_truncation():
    """대조 박제: tag_regimes vol 축 = 전구간 quantile → 절단 불변 위반(=비인과)."""
    df = _vol_switch_df()
    report = check_truncation_invariance(
        lambda d: _encode(tag_regimes(d, _P))[["vol"]], df
    )
    assert report.leaked  # 비인과 확인 — causal 버전으로 고친 근거


def test_causal_trend_matches_analysis():
    """추세축은 공유 _trend_states → tag_regimes 와 완전 동일."""
    df = _vol_switch_df()
    a = tag_regimes(df, _P)[TREND_COL]
    c = causal_tag_regimes(df, _P)[TREND_COL]
    assert a.equals(c)


def test_causal_schema_and_warmup():
    """스키마 = tag_regimes 동일. vol 은 rolling(365) 워밍업 만큼 앞부분 NaN."""
    df = _vol_switch_df()
    out = causal_tag_regimes(df, _P)
    assert set(out.columns) == {TREND_COL, VOL_COL, LABEL_COL}
    # 첫 (vol_len+vol_window) 이전은 vol NaN, 이후 유효값 존재
    assert out[VOL_COL].iloc[:_P.vol_len + 365 - 1].isna().all()
    assert out[VOL_COL].notna().any()
