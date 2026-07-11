"""Robust KF 피처 회귀 — 인과성 + **행동 검증**(의미대로 작동하는가) (Phase 2 Step 2.3).

인과·정상성만으론 죽은 피처도 통과하므로, 설계 의도가 **소프트웨어 동작으로** 실현되는지
직접 검증한다(성과 아님, 기둥 5·6):
  1. 일시 스파이크 vs 지속 이동 — robust 가 일시엔 덜 반응·지속엔 추종, Gaussian 보다 강건
  2. 불확실성 비상수성 — 격변 구간 kf_uncertainty↑ (죽은 피처 아님, 장식금지 가드)
  3. dof→∞ → Gaussian KF 복원 (재가중 일반화 정당성)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import assert_causal
from src.research.features.kalman import kf_trend
from tests.research.synthetic import make_ohlcv

# 행동 테스트용 명료 파라미터 (기본값은 Phase3 튜닝 placeholder라, 알고리즘 동작을
# 가시화하는 값으로 메커니즘 자체를 검증한다).
_CLR = dict(q_level=1e-4, q_slope=1e-5, r=1e-4, dof=4.0, p0=1e-2, warmup=5)


def _df_from_logprice_series(price: np.ndarray) -> pd.DataFrame:
    n = len(price)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    open_ = np.concatenate([[price[0]], price[:-1]])
    high = np.maximum(open_, price) * 1.0
    low = np.minimum(open_, price) * 1.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price, "volume": 1.0},
        index=idx,
    )


def _reference_gaussian_kf(y: np.ndarray, q_level, q_slope, r, p0):
    """비robust 표준 KF slope (dof→∞ 극한 대조용, u≡1)."""
    n = len(y)
    slope = np.full(n, np.nan)
    x0, x1 = y[0], 0.0
    p00, p01, p11 = p0, 0.0, p0
    for t in range(1, n):
        lvl_pred, slp_pred = x0 + x1, x1
        p00_p = p00 + 2 * p01 + p11 + q_level
        p01_p = p01 + p11
        p11_p = p11 + q_slope
        eps = y[t] - lvl_pred
        s = p00_p + r
        k0, k1 = p00_p / s, p01_p / s
        x0 = lvl_pred + k0 * eps
        x1 = slp_pred + k1 * eps
        p00 = (1 - k0) * p00_p
        p01 = (1 - k0) * p01_p
        p11 = p11_p - k1 * p01_p
        slope[t] = x1
    return slope


# ---- 인과성 (핵심) ----

def test_kf_causal():
    df = make_ohlcv(n=300)
    report = assert_causal(kf_trend, df)      # DataFrame 2열 전체 인과
    assert not report.leaked


# ---- 산출 형태 ----

def test_output_shape_and_columns():
    df = make_ohlcv(n=200)
    out = kf_trend(df)
    assert list(out.columns) == ["kf_slope", "kf_uncertainty"]
    assert out.index.equals(df.index)


def test_warmup_nan():
    df = make_ohlcv(n=200)
    out = kf_trend(df, warmup=20)
    assert out.iloc[:20].isna().all().all()
    assert out.iloc[-1].notna().all()


def test_uncertainty_nonneg():
    out = kf_trend(make_ohlcv(n=200))
    assert (out["kf_uncertainty"].dropna() >= 0).all()


# ---- 행동 1: 일시 스파이크 vs 지속 이동 (robust 실체, 기둥6) ----

def test_robust_damps_transient_more_than_gaussian():
    """동일 일시 스파이크에 robust slope 가 Gaussian 보다 덜 흔들린다."""
    base = np.full(120, 100.0)
    spike = base.copy()
    spike[60] = 110.0                          # 한 봉만 튐 → 즉시 복귀
    df = _df_from_logprice_series(spike)
    robust = kf_trend(df, **_CLR)["kf_slope"].to_numpy()
    gauss = _reference_gaussian_kf(
        np.log(spike), _CLR["q_level"], _CLR["q_slope"], _CLR["r"], _CLR["p0"]
    )
    # 스파이크 직후 몇 봉의 slope 크기: robust < gaussian
    win = slice(61, 66)
    assert np.nanmax(np.abs(robust[win])) < np.nanmax(np.abs(gauss[win]))


def test_follows_sustained_shift():
    """지속 이동(레벨 영구 상승)엔 robust slope 가 결국 추종(양수화)."""
    price = np.full(200, 100.0)
    price[80:] = 110.0                         # 80봉부터 영구 상승
    df = _df_from_logprice_series(price)
    slope = kf_trend(df, **_CLR)["kf_slope"]
    # 이동 직후엔 신중(작음), 충분히 지나면 양의 추세로 추종
    assert slope.iloc[120] > 0.0
    # 그리고 잔잔해진 뒤(말미)엔 다시 0 근방으로 수렴
    assert abs(slope.iloc[-1]) < abs(slope.iloc[95])


def test_sustained_response_exceeds_transient():
    """지속 이동의 최대 slope 반응 > 일시 스파이크의 최대 반응(신중↔추종 구분)."""
    trans = np.full(200, 100.0); trans[80] = 110.0
    sust = np.full(200, 100.0); sust[80:] = 110.0
    s_trans = kf_trend(_df_from_logprice_series(trans), **_CLR)["kf_slope"]
    s_sust = kf_trend(_df_from_logprice_series(sust), **_CLR)["kf_slope"]
    peak_trans = s_trans.iloc[80:130].abs().max()
    peak_sust = s_sust.iloc[80:130].abs().max()
    assert peak_sust > peak_trans


# ---- 행동 2: 불확실성 비상수성 (장식금지 가드, 기둥5) ----

def test_uncertainty_is_not_constant():
    """robust 재가중이 P 를 데이터 의존으로 → 불확실성이 상수 아님."""
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.005, 300)
    ret[120:140] += rng.normal(0, 0.08, 20)    # 격변 구간 주입
    price = 100 * np.exp(np.cumsum(ret))
    unc = kf_trend(_df_from_logprice_series(price), **_CLR)["kf_uncertainty"].dropna()
    # 상수라면 (max-min)/mean ≈ 0 — 유의미한 변동 존재를 요구
    assert (unc.max() - unc.min()) / unc.mean() > 0.1


def test_uncertainty_rises_in_turbulence():
    """격변 구간의 불확실성 > 잔잔한 구간(이상치가 P 를 덜 줄임)."""
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.003, 400)
    ret[200:230] += rng.choice([-1, 1], 30) * 0.09   # 이상치 클러스터
    price = 100 * np.exp(np.cumsum(ret))
    unc = kf_trend(_df_from_logprice_series(price), **_CLR)["kf_uncertainty"]
    calm = unc.iloc[60:130].median()
    turb = unc.iloc[205:245].median()
    assert turb > calm


# ---- 행동 3: dof→∞ = Gaussian 복원 (재가중 정당성, 기둥6) ----

def test_large_dof_recovers_gaussian():
    """dof→∞ 이면 u→1 → 표준 Gaussian KF slope 와 일치."""
    df = make_ohlcv(n=250)
    y = np.log(df["close"].to_numpy())
    robust_bigdof = kf_trend(
        df, q_level=1e-4, q_slope=1e-5, r=1e-4, dof=1e12, p0=1e-2, warmup=5
    )["kf_slope"].to_numpy()
    gauss = _reference_gaussian_kf(y, 1e-4, 1e-5, 1e-4, 1e-2)
    m = ~np.isnan(robust_bigdof)
    np.testing.assert_allclose(robust_bigdof[m], gauss[m], rtol=1e-6, atol=1e-12)


def test_small_dof_differs_from_gaussian():
    """작은 dof 는 (이상치 있는 데이터에서) Gaussian 과 유의미하게 달라야 한다."""
    price = np.full(150, 100.0); price[75] = 115.0
    df = _df_from_logprice_series(price)
    robust = kf_trend(df, dof=3.0, q_level=1e-4, q_slope=1e-5, r=1e-4,
                      p0=1e-2, warmup=5)["kf_slope"].to_numpy()
    gauss = _reference_gaussian_kf(np.log(price), 1e-4, 1e-5, 1e-4, 1e-2)
    m = ~np.isnan(robust)
    assert not np.allclose(robust[m], gauss[m], rtol=1e-3)


# ---- 엣지 ----

def test_invalid_params():
    df = make_ohlcv(n=50)
    with pytest.raises(ValueError):
        kf_trend(df, dof=0)
    with pytest.raises(ValueError):
        kf_trend(df, r=0)
    with pytest.raises(ValueError):
        kf_trend(df, p0=0)
