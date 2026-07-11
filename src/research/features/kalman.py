"""Robust Kalman 추세 필터 — 계열1 slope + 불확실성 (Phase 2 Step 2.3, 설계 §7·§8).

로그가격을 **선형 local linear trend** 상태공간으로 필터링해 봉당 추세(slope)와 그
불확실성을 낸다. 팻테일(거대 점프)에 **Student-t 측정노이즈**로 강건 — "한 번은 신중,
지속되면 추종"(설계 §7). 두 산출물이 하류로 흐른다: slope=방향/강도, 불확실성=사이징
사슬 연결(설계 §8·§12).

**상태공간 (D-016, S/기둥4·6)** — 선형 2차원(비선형 미도입 → EKF/UKF 불요, 설계 §7):
    상태 x=[level, slope],  전이 F=[[1,1],[0,1]],  관측 y=log(close)=level+v,  H=[1,0]
    과정노이즈 Q=diag([q_level,q_slope]),  측정노이즈 분산 R.

**Robust 실현 (D-017, S/기둥6)** — Student-t 를 가우시안 스케일혼합으로 본 **1-step 재가중**:
    예측 x⁻=Fx, P⁻=FPFᵀ+Q ;  혁신 ε=y−Hx⁻, S₀=HP⁻Hᵀ+R
    가중 d²=ε²/S₀,  u=(dof+1)/(dof+d²)   ← 이상치일수록 u↓
    유효 R_eff=R/u,  S=HP⁻Hᵀ+R_eff       ← 이상치면 R 팽창→게인↓(신중)
    갱신 K=P⁻Hᵀ/S, x=x⁻+Kε, P=(I−KH)P⁻
  지속 불일치 → 게인 작음 → P 가 Q 만큼 팽창 → S₀↑ → d²↓ → u→1 → 게인 복원(추종).
  **dof→∞ 이면 u→1, R_eff→R 로 Gaussian KF 복원**(회귀 테스트로 검증).

**Q/R/dof (D-018, S/교차-Phase)** — Phase 2 는 **고정 기본값**(다른 피처 window 가 Phase 2
고정·Phase 3 R2 튜닝인 것과 평행; KF 는 창 그룹서 예외, Q/R 독립 튜닝, 설계 §7). Phase 2
는 fitting 없음 → 자명 인과. **F-12**: Phase 3 에서 Q/R/dof 를 어떻게 인과적합할지(폴드
train-only MLE vs config 하이퍼 coordinate descent) 는 Phase 3 R2 착수 전 확정(Phase 2 무관).

**온라인 전방 필터 (D-019, S)** — 각 t 는 y_{0..t} 만(폴드무관 순수 인과). ``assert_causal``
회귀. features=OHLCV→X 순수함수, 폴드로직은 harness 소유(D-002 계약 유지).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kf_trend(
    df: pd.DataFrame,
    *,
    q_level: float = 1e-5,
    q_slope: float = 1e-6,
    r: float = 1e-4,
    dof: float = 4.0,
    p0: float = 1e-2,
    warmup: int = 20,
) -> pd.DataFrame:
    """로그가격 robust KF → DataFrame[kf_slope, kf_uncertainty] (df.index 정렬).

    - ``kf_slope`` = 필터링된 봉당 로그추세(정상적·스케일무관).
    - ``kf_uncertainty`` = √(P[slope,slope]) = slope 사후 표준편차. robust 재가중이
      P 를 데이터 의존으로 만들어 격변 구간엔↑·잔잔한 추세엔↓(비상수 → 정보 有).

    기본값은 Phase 2 placeholder(로그수익 ~1% 스케일 기준) — Phase 3 R2 튜닝(F-12).
    초기 공분산 P₀=diag([p0,p0])(p0=1e-2 = 완만히 느슨한 prior; slope엔 사실상 diffuse,
    level엔 중간 수준), 전이구간 ``warmup`` 봉은 전이 안정화 위해 NaN.
    """
    if dof <= 0 or r <= 0 or q_level < 0 or q_slope < 0 or p0 <= 0:
        raise ValueError("dof·r·p0>0, q_level·q_slope>=0 이어야 함")
    if warmup < 0:
        raise ValueError("warmup 은 음수 불가")
    if "close" not in df.columns:
        raise ValueError("close 컬럼 누락")

    y = np.log(df["close"].to_numpy(dtype=float))
    n = len(y)
    slope_out = np.full(n, np.nan)
    unc_out = np.full(n, np.nan)
    if n == 0:
        return pd.DataFrame(
            {"kf_slope": slope_out, "kf_uncertainty": unc_out}, index=df.index
        )

    # 초기 상태: level=첫 관측, slope=0. 완만히 느슨한 prior P₀=diag([p0,p0]).
    x0, x1 = (y[0] if np.isfinite(y[0]) else 0.0), 0.0
    p00, p01, p11 = p0, 0.0, p0

    for t in range(1, n):
        # 예측: x⁻=Fx,  P⁻=FPFᵀ+Q  (F=[[1,1],[0,1]] 수식 전개)
        lvl_pred = x0 + x1
        slp_pred = x1
        p00_p = p00 + 2.0 * p01 + p11 + q_level
        p01_p = p01 + p11
        p11_p = p11 + q_slope

        yt = y[t]
        if not np.isfinite(yt):
            # 관측 결측 → 갱신 없이 예측 유지(감사 데이터엔 없음; 안전).
            x0, x1 = lvl_pred, slp_pred
            p00, p01, p11 = p00_p, p01_p, p11_p
            continue

        # 혁신 + Student-t 1-step 재가중
        eps = yt - lvl_pred                 # H=[1,0] → Hx⁻=lvl_pred
        s0 = p00_p + r
        d2 = (eps * eps) / s0
        u = (dof + 1.0) / (dof + d2)         # dof→∞ ⇒ u→1 (Gaussian)
        r_eff = r / u
        s = p00_p + r_eff

        # 게인 K = P⁻Hᵀ/S = [p00_p, p01_p]/S
        k0 = p00_p / s
        k1 = p01_p / s
        x0 = lvl_pred + k0 * eps
        x1 = slp_pred + k1 * eps
        # P=(I−KH)P⁻, KH=[[k0,0],[k1,0]] → 대칭 보존 형태
        p00 = (1.0 - k0) * p00_p
        p01 = (1.0 - k0) * p01_p
        p11 = p11_p - k1 * p01_p

        if t >= warmup:
            slope_out[t] = x1
            unc_out[t] = np.sqrt(p11) if p11 > 0 else 0.0

    return pd.DataFrame(
        {"kf_slope": slope_out, "kf_uncertainty": unc_out}, index=df.index
    )
