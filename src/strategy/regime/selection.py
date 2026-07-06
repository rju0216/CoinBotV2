"""K 선택 — holdout LL + BIC + 커버리지 진단 (D4-2c).

스펙 §1.5 의 세 렌즈: ① holdout log-likelihood(과적합) ② BIC(복잡도) ③ 커버리지
(매매에 필요한 trend/long·trend/short·range/none 이 다 나오나).

D3 method/value 분리: 여기는 **진단표 + 추천 K** 만 산출한다. 최종 K 는 D4 실행에서
사람이 표를 보고 확정(빈 조합 억지로 안 채움 — 스펙 §1.7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.strategy.regime.contract import RegimeDirection, RegimeType
from src.strategy.regime.hmm import GaussianEmission, HMM
from src.strategy.regime.mapping import StateMapping

#: 매매에 필요한 코어 3조합 (스펙 §1.7)
CORE_COMBOS = frozenset(
    {
        (RegimeType.TREND, RegimeDirection.LONG),
        (RegimeType.TREND, RegimeDirection.SHORT),
        (RegimeType.RANGE, RegimeDirection.NONE),
    }
)


@dataclass(frozen=True)
class KDiagnostic:
    """단일 K 의 진단 결과."""

    k: int
    holdout_ll: float
    bic: float
    covered: frozenset
    has_core_coverage: bool


def evaluate_k(
    X_train: np.ndarray,
    raw_log_train: np.ndarray,
    X_holdout: np.ndarray,
    k: int,
    tau: float,
    n_init: int = 5,
    seed: int = 0,
    **fit_kw,
) -> tuple[KDiagnostic, HMM, StateMapping]:
    """단일 K 평가: train 적합 → holdout LL, BIC(train), 매핑→커버리지.

    emission 은 **Gaussian 고정** (I-013): gen1 K선택은 미배선(select_k 빌드 미사용).
    t emission K선택(BIC 에 ν 반영, D3 §157)은 D4-5 에서 make_emission 으로 배선.
    """
    hmm = HMM(k, GaussianEmission(k, X_train.shape[1])).fit(
        X_train, n_init=n_init, seed=seed, **fit_kw
    )
    holdout_ll = hmm.log_likelihood(X_holdout)
    bic = hmm.bic(X_train)
    gamma = hmm.smoothed_posterior(X_train)  # offline 특성화 (smoothed OK)
    mapping = StateMapping.fit(raw_log_train, gamma, tau)
    covered = frozenset(mapping.covered_combos())
    diag = KDiagnostic(
        k=k,
        holdout_ll=holdout_ll,
        bic=bic,
        covered=covered,
        has_core_coverage=CORE_COMBOS.issubset(covered),
    )
    return diag, hmm, mapping


def recommend_k(diagnostics: list[KDiagnostic]) -> int:
    """추천 K (순수 로직): 커버리지 통과 중 BIC 최소. 없으면 holdout LL 최대.

    최종 결정이 아니라 추천 — 사람이 진단표로 확인 (D3 method/value 분리).
    """
    if not diagnostics:
        raise ValueError("진단 결과가 비어 있음")
    passing = [d for d in diagnostics if d.has_core_coverage]
    if passing:
        return min(passing, key=lambda d: d.bic).k
    return max(diagnostics, key=lambda d: d.holdout_ll).k


def select_k(
    X_train: np.ndarray,
    raw_log_train: np.ndarray,
    X_holdout: np.ndarray,
    tau: float,
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6),
    n_init: int = 5,
    seed: int = 0,
    **fit_kw,
) -> tuple[list[KDiagnostic], int]:
    """k_range 전체 평가 → (진단표, 추천 K)."""
    diagnostics = [
        evaluate_k(
            X_train, raw_log_train, X_holdout, k, tau,
            n_init=n_init, seed=seed, **fit_kw,
        )[0]
        for k in k_range
    ]
    return diagnostics, recommend_k(diagnostics)
