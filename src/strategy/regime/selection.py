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
from src.strategy.regime.hmm import HMM, make_emission
from src.strategy.regime.mapping import StateMapping


def _core_combos(enable_range: bool) -> frozenset:
    """gen1 매매에 필요한 코어 조합 (스펙 §1.7).

    추세단독(enable_range=False, O-7 range 드롭)이면 trend long/short 만 필요.
    range 활성(gen2 revival)이면 (RANGE,NONE) 추가.
    """
    combos = {
        (RegimeType.TREND, RegimeDirection.LONG),
        (RegimeType.TREND, RegimeDirection.SHORT),
    }
    if enable_range:
        combos.add((RegimeType.RANGE, RegimeDirection.NONE))
    return frozenset(combos)


#: 코어 조합 (range 포함 — 하위호환·gen2 대비). 추세단독은 `_core_combos(False)`.
CORE_COMBOS = _core_combos(True)


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
    emission_kind: str = "gaussian",
    emission_params: dict | None = None,
    enable_range: bool = True,
    **fit_kw,
) -> tuple[KDiagnostic, HMM, StateMapping]:
    """단일 K 평가: train 적합 → holdout LL, BIC(train), 매핑→커버리지.

    emission 은 `make_emission` 으로 배선 (I-013 해소, D4-5 S2) — gaussian/student_t
    둘 다 K선택 가능. BIC 는 `hmm.n_params`(emission 자동, t 는 ν 포함) 로 복잡도 반영.
    core 커버리지는 `enable_range` 에 따라 동적. **주의(C-1)**: 커버리지는 τ 의존이라
    K 확정의 **주 기준이 아님** — holdout LL(elbow)·BIC 가 주, 커버리지는 보조/veto.
    """
    emission = make_emission(
        emission_kind, k, X_train.shape[1], **(emission_params or {})
    )
    hmm = HMM(k, emission).fit(X_train, n_init=n_init, seed=seed, **fit_kw)
    holdout_ll = hmm.log_likelihood(X_holdout)
    bic = hmm.bic(X_train)
    gamma = hmm.smoothed_posterior(X_train)  # offline 특성화 (smoothed OK)
    mapping = StateMapping.fit(raw_log_train, gamma, tau, enable_range=enable_range)
    covered = frozenset(mapping.covered_combos())
    diag = KDiagnostic(
        k=k,
        holdout_ll=holdout_ll,
        bic=bic,
        covered=covered,
        has_core_coverage=_core_combos(enable_range).issubset(covered),
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
    emission_kind: str = "gaussian",
    emission_params: dict | None = None,
    enable_range: bool = True,
    **fit_kw,
) -> tuple[list[KDiagnostic], int]:
    """k_range 전체 평가 → (진단표, 추천 K). emission_kind 로 gaussian/student_t 선택."""
    diagnostics = [
        evaluate_k(
            X_train, raw_log_train, X_holdout, k, tau,
            n_init=n_init, seed=seed,
            emission_kind=emission_kind, emission_params=emission_params,
            enable_range=enable_range, **fit_kw,
        )[0]
        for k in k_range
    ]
    return diagnostics, recommend_k(diagnostics)
