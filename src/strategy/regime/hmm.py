"""레짐 발견용 HMM — EM 학습 + forward-only filtering (D4-2b).

설계 (D2/D3 결정):
  - **emission 추상화**: HMM 은 `Emission` 인터페이스(log_prob·m_step·init_params·
    get/set_params·to_dict·n_params·n_states·n_dim)만 통해 상호작용한다. GaussianEmission
    (D4-2) 과 StudentTEmission(D4-4, fat tail) 을 클래스 교체만으로 갈아끼운다(O-6 단계화).
  - **emission.m_step 자기완결**: HMM E-step 은 상태 책임도 γ(T,K) 만 제공하고,
    emission 이 자기 M-step 을 내부에서 완수 — t 의 추가 latent(스케일 u)도
    emission 안에서 처리(γ 만 받으면 됨). → t 복잡도가 emission 에 격리.
  - **학습 EM = forward-backward**(offline, 미래참조 아님). **추론 = forward-only
    filter**(causal). 전부 log-space(logsumexp) 수치안정.

filtered posterior γ_t = P(상태_t | o_1:t) 는 매핑 레이어(D4-2c)가 contract 로 번역.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import digamma, gammaln, logsumexp


def _chol_delta2_logdet(
    X: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, float]:
    """Cholesky 로 δ²=(x−μ)ᵀΣ⁻¹(x−μ) (T,) 와 logdet(Σ) 동시 산출.

    Gaussian·t emission 의 log_prob·mahalanobis 공통 경로 — scipy logpdf 직접 대체
    (I-007 최적화: 매 EM iter 의 scipy 입력검증·cov 재분해 제거). cov 는 reg 로
    항상 PD → cholesky 안전(특이화 조기감지). logdet = 2·Σ log(diag L).
    """
    L = np.linalg.cholesky(cov)
    z = solve_triangular(L, (X - mean).T, lower=True)  # (d, T)
    delta2 = (z ** 2).sum(axis=0)  # (T,)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    return delta2, logdet


class Emission(ABC):
    """상태별 관측 분포. HMM 은 이 인터페이스만 통해 emission 과 상호작용."""

    @abstractmethod
    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """log p(x_t | 상태=k) → (T, K)."""

    @abstractmethod
    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """M-step: 상태 책임도 γ(T,K)만 받아 자기 파라미터를 in-place 갱신.

        emission-특화 latent(t 의 스케일 u 등)는 여기 내부에서 자체 계산한다.
        """

    @abstractmethod
    def init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """데이터 기반 무작위 초기화 (다중 init 용)."""

    @abstractmethod
    def get_params(self) -> Any:
        """현재 파라미터 스냅샷 (best-LL 보존용)."""

    @abstractmethod
    def set_params(self, params: Any) -> None:
        """스냅샷 복원."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """파라미터 개수 (BIC 용)."""

    @property
    @abstractmethod
    def n_states(self) -> int: ...

    @property
    @abstractmethod
    def n_dim(self) -> int: ...

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-직렬화 가능 dict(배열은 list, "kind" 포함). artifact 동행."""


class GaussianEmission(Emission):
    """다변량 Gaussian emission (full 공분산 + diagonal 정규화)."""

    kind = "gaussian"

    def __init__(self, n_states: int, n_dim: int, reg: float = 1e-6) -> None:
        self.K = n_states
        self.d = n_dim
        self.reg = reg
        self.means = np.zeros((n_states, n_dim))
        self.covs = np.tile(np.eye(n_dim), (n_states, 1, 1))

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        out = np.empty((T, self.K))
        const = self.d * np.log(2.0 * np.pi)
        for k in range(self.K):
            delta2, logdet = _chol_delta2_logdet(X, self.means[k], self.covs[k])
            out[:, k] = -0.5 * (const + logdet + delta2)
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        Nk = gamma.sum(axis=0)  # (K,)
        for k in range(self.K):
            if Nk[k] < 1e-8:
                continue  # 할당 거의 없음 → 기존 파라미터 유지
            w = gamma[:, k]
            mean_k = (w[:, None] * X).sum(axis=0) / Nk[k]
            diff = X - mean_k
            cov_k = (diff.T * w) @ diff / Nk[k] + self.reg * np.eye(self.d)
            self.means[k] = mean_k
            self.covs[k] = cov_k

    def init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        idx = rng.choice(len(X), self.K, replace=False)
        self.means = X[idx].astype(float).copy()
        global_cov = np.atleast_2d(np.cov(X.T)) + self.reg * np.eye(self.d)
        self.covs = np.tile(global_cov, (self.K, 1, 1))

    def get_params(self) -> Any:
        return (self.means.copy(), self.covs.copy())

    def set_params(self, params: Any) -> None:
        self.means, self.covs = params[0].copy(), params[1].copy()

    @property
    def n_params(self) -> int:
        # state 별: 평균 d + 공분산 d(d+1)/2
        return self.K * (self.d + self.d * (self.d + 1) // 2)

    @property
    def n_states(self) -> int:
        return self.K

    @property
    def n_dim(self) -> int:
        return self.d

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_states": self.K,
            "n_dim": self.d,
            "reg": self.reg,
            "means": self.means.tolist(),
            "covs": self.covs.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianEmission":
        em = cls(d["n_states"], d["n_dim"], d.get("reg", 1e-6))
        em.means = np.asarray(d["means"], dtype=float)
        em.covs = np.asarray(d["covs"], dtype=float)
        return em


class StudentTEmission(Emission):
    """다변량 Student-t emission (scale 행렬 Σ + 상태별 자유도 ν) — fat tail (스펙 §1.4).

    Gaussian scale-mixture 표현:
      x | u ~ N(μ_k, Σ_k/u),  u ~ Gamma(ν_k/2, ν_k/2)  → 주변분포 t(μ_k, Σ_k, ν_k).
    m_step 이 상태책임도 γ 만 받아 스케일 latent u 의 E-step 을 내부에서 수행한다
    (Emission ABC 계약 — t 의 추가 latent 를 emission 안에 격리). ν 는 상태별 추정
    (share_nu=True 면 공통 ν 하나), Q-함수(complete-data 기대우도) 최대화의 digamma
    방정식 root-find (표준 EM/ECM — 관측우도 직접 최대화인 ECME 변형 아님).

    **명명 주의**: `scales` = **scale 행렬 Σ (공분산 아님)**. 실제 cov = Σ·ν/(ν−2) (ν>2).
    GaussianEmission.covs 와 의미가 달라 이름을 분리했다 (shape/cov 혼동 차단). `nus` = ν.
    """

    kind = "student_t"

    def __init__(
        self,
        n_states: int,
        n_dim: int,
        reg: float = 1e-6,
        nu_init: float = 10.0,
        nu_min: float = 2.0,
        nu_max: float = 200.0,
        share_nu: bool = False,
    ) -> None:
        self.K = n_states
        self.d = n_dim
        self.reg = reg
        self.nu_init = float(nu_init)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.share_nu = share_nu
        self.means = np.zeros((n_states, n_dim))
        self.scales = np.tile(np.eye(n_dim), (n_states, 1, 1))
        self.nus = np.full(n_states, float(nu_init))

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        out = np.empty((T, self.K))
        d = self.d
        for k in range(self.K):
            nu = self.nus[k]
            delta2, logdet = _chol_delta2_logdet(X, self.means[k], self.scales[k])
            out[:, k] = (
                gammaln((nu + d) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * logdet
                - 0.5 * (nu + d) * np.log1p(delta2 / nu)
            )
        return out

    def m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        Nk = gamma.sum(axis=0)  # (K,)
        # E-step(u): 갱신 전 params 로 상태별 산출·저장 (단일 E-step 표준 EM).
        E_u_all = np.zeros_like(gamma)  # (T, K)
        E_logu_all = np.zeros_like(gamma)
        updated = np.zeros(self.K, dtype=bool)  # μ,Σ 를 실제 갱신한 상태 (ν 일관용)
        for k in range(self.K):
            if Nk[k] < 1e-8:
                continue  # 빈 상태 → 기존 파라미터 유지
            nu = self.nus[k]
            delta2, _ = _chol_delta2_logdet(X, self.means[k], self.scales[k])
            E_u_all[:, k] = (nu + self.d) / (nu + delta2)
            E_logu_all[:, k] = digamma((nu + self.d) / 2.0) - np.log(
                (nu + delta2) / 2.0
            )
            # M-step μ, Σ(scale): μ = Σγu·x / Σγu, Σ = Σγu(·)(·)ᵀ / Σγ (분모 Nk).
            w = gamma[:, k]
            gu = w * E_u_all[:, k]
            sum_gu = gu.sum()
            if sum_gu < 1e-12:
                continue  # 병리적(E[u]≈0) → μ,Σ,ν 모두 유지 (일관 skip)
            mean_k = (gu[:, None] * X).sum(axis=0) / sum_gu
            d2 = X - mean_k
            scale_k = (d2.T * gu) @ d2 / Nk[k] + self.reg * np.eye(self.d)
            self.means[k] = mean_k
            self.scales[k] = scale_k
            updated[k] = True
        # ν 업데이트 (정확형 E[logu] → 보정항 없는 canonical 방정식). μ,Σ 갱신 상태만.
        if self.share_nu:
            if updated.any():
                g = gamma[:, updated]
                den = float(g.sum())
                c = (
                    float((g * (E_logu_all[:, updated] - E_u_all[:, updated])).sum()
                          / den)
                    if den > 1e-12
                    else 0.0
                )
                self.nus[:] = self._solve_nu(c)
        else:
            for k in range(self.K):
                if not updated[k]:
                    continue
                c = float(
                    (gamma[:, k] * (E_logu_all[:, k] - E_u_all[:, k])).sum() / Nk[k]
                )
                self.nus[k] = self._solve_nu(c)

    def _solve_nu(self, c: float) -> float:
        """g(ν)=1−ψ(ν/2)+log(ν/2)+c=0 의 근. g 는 ν 에 단조감소.

        c=(1/N)Σγ(E[logu]−E[u]) (정확형·보정항 없음). MLE 가 [nu_min,nu_max] 밖이면
        경계로 clamp (brentq 동부호 crash 방지).
        """

        def g(nu: float) -> float:
            return 1.0 - digamma(nu / 2.0) + np.log(nu / 2.0) + c

        glo, ghi = g(self.nu_min), g(self.nu_max)
        if glo == 0.0:
            return self.nu_min
        if ghi == 0.0:
            return self.nu_max
        if glo * ghi > 0.0:
            # 동부호 = 근이 구간 밖. g 단조감소 → 둘 다 양수면 근>max, 음수면 근<min.
            return self.nu_max if ghi > 0.0 else self.nu_min
        return float(brentq(g, self.nu_min, self.nu_max))

    def init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        idx = rng.choice(len(X), self.K, replace=False)
        self.means = X[idx].astype(float).copy()
        global_cov = np.atleast_2d(np.cov(X.T)) + self.reg * np.eye(self.d)
        self.scales = np.tile(global_cov, (self.K, 1, 1))
        self.nus = np.full(self.K, self.nu_init)

    def get_params(self) -> Any:
        return (self.means.copy(), self.scales.copy(), self.nus.copy())

    def set_params(self, params: Any) -> None:
        self.means = params[0].copy()
        self.scales = params[1].copy()
        self.nus = params[2].copy()

    @property
    def n_params(self) -> int:
        # state 별: 평균 d + scale d(d+1)/2 ; ν 는 share_nu 면 1, 아니면 K.
        base = self.K * (self.d + self.d * (self.d + 1) // 2)
        return base + (1 if self.share_nu else self.K)

    @property
    def n_states(self) -> int:
        return self.K

    @property
    def n_dim(self) -> int:
        return self.d

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_states": self.K,
            "n_dim": self.d,
            "reg": self.reg,
            "nu_init": self.nu_init,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "share_nu": self.share_nu,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "nus": self.nus.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StudentTEmission":
        em = cls(
            d["n_states"],
            d["n_dim"],
            reg=d.get("reg", 1e-6),
            nu_init=d.get("nu_init", 10.0),
            nu_min=d.get("nu_min", 2.0),
            nu_max=d.get("nu_max", 200.0),
            share_nu=d.get("share_nu", False),
        )
        em.means = np.asarray(d["means"], dtype=float)
        em.scales = np.asarray(d["scales"], dtype=float)
        em.nus = np.asarray(d["nus"], dtype=float)
        return em


def make_emission(kind: str, n_states: int, n_dim: int, **params) -> Emission:
    """순방향 생성: kind + params → 새 Emission (emission_from_dict 의 대칭).

    build_model·selection(D4-5)·향후 확장이 emission 생성을 공유한다 (DRY). kind 별로
    쓰는 params 키가 다르므로 dict(kwargs) 로 받고, 안 쓰는 키는 무시한다.
    """
    if kind == "gaussian":
        return GaussianEmission(n_states, n_dim, reg=params.get("reg", 1e-6))
    if kind == "student_t":
        return StudentTEmission(
            n_states,
            n_dim,
            reg=params.get("reg", 1e-6),
            nu_init=params.get("nu_init", 10.0),
            nu_min=params.get("nu_min", 2.0),
            nu_max=params.get("nu_max", 200.0),
            share_nu=params.get("share_nu", False),
        )
    raise ValueError(f"unknown emission kind: {kind!r}")


def emission_from_dict(d: dict) -> Emission:
    """직렬화 dict → Emission (kind 로 dispatch)."""
    if d["kind"] == "gaussian":
        return GaussianEmission.from_dict(d)
    if d["kind"] == "student_t":
        return StudentTEmission.from_dict(d)
    raise ValueError(f"unknown emission kind: {d['kind']!r}")


class HMM:
    """이산 상태 HMM. transition + emission(추상). EM 학습, forward-only filter 추론."""

    def __init__(self, n_states: int, emission: Emission) -> None:
        if emission.n_states != n_states:
            raise ValueError("emission.n_states 가 n_states 와 다름")
        self.K = n_states
        self.emission = emission
        self.log_pi = np.log(np.full(n_states, 1.0 / n_states))
        self.log_A = np.log(np.full((n_states, n_states), 1.0 / n_states))
        self.log_likelihood_: float = -np.inf

    # ---- forward-backward (log-space) ----

    def _forward_log(self, log_obs: np.ndarray) -> tuple[np.ndarray, float]:
        T = log_obs.shape[0]
        log_alpha = np.empty((T, self.K))
        log_alpha[0] = self.log_pi + log_obs[0]
        log_A = self.log_A
        for t in range(1, T):
            # logsumexp(log_alpha[t-1][:,None] + log_A, axis=0) 인라인 (I-007:
            # 스텝당 scipy 호출 제거, max-shift 동일 알고리즘 → 수치 등가).
            m = log_alpha[t - 1][:, None] + log_A  # (K_prev, K_next)
            mx = m.max(axis=0)
            log_alpha[t] = log_obs[t] + mx + np.log(np.exp(m - mx).sum(axis=0))
        mx_last = log_alpha[-1].max()
        ll = float(mx_last + np.log(np.exp(log_alpha[-1] - mx_last).sum()))
        return log_alpha, ll

    def _backward_log(self, log_obs: np.ndarray) -> np.ndarray:
        T = log_obs.shape[0]
        log_beta = np.zeros((T, self.K))
        log_A = self.log_A
        for t in range(T - 2, -1, -1):
            # logsumexp(log_A + (log_obs[t+1]+log_beta[t+1])[None,:], axis=1) 인라인.
            m = log_A + (log_obs[t + 1] + log_beta[t + 1])[None, :]  # (K_j, K_k)
            mx = m.max(axis=1, keepdims=True)
            log_beta[t] = mx[:, 0] + np.log(np.exp(m - mx).sum(axis=1))
        return log_beta

    # ---- EM 학습 ----

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        seed: int = 0,
    ) -> "HMM":
        rng = np.random.default_rng(seed)
        best_ll = -np.inf
        best_snapshot = None
        for _ in range(n_init):
            self._init_params(X, rng)
            prev_ll = -np.inf
            ll = -np.inf
            for it in range(n_iter):
                log_obs = self.emission.log_prob(X)
                log_alpha, ll = self._forward_log(log_obs)
                log_beta = self._backward_log(log_obs)
                log_gamma = log_alpha + log_beta - ll
                gamma = np.exp(log_gamma)
                self._m_step_transition(log_alpha, log_beta, log_obs, ll, gamma)
                self.emission.m_step(X, gamma)
                if it > 0 and ll - prev_ll < tol * max(abs(prev_ll), 1.0):
                    break
                prev_ll = ll
            if ll > best_ll:
                best_ll = ll
                best_snapshot = self._snapshot()
        if best_snapshot is not None:
            self._restore(best_snapshot)
            # 복원된 best 파라미터의 정확한 LL 재계산. 학습 루프의 ll 은 매 iter
            # 의 M-step 직전 값이라 수렴 시점 params 보다 1스텝 stale (best-init
            # 비교엔 차이<tol 이라 무방하나, 저장값은 정확히 맞춘다).
            self.log_likelihood_ = self.log_likelihood(X)
        return self

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        self.emission.init_params(X, rng)
        self.log_pi = np.log(np.full(self.K, 1.0 / self.K))
        if self.K == 1:
            self.log_A = np.zeros((1, 1))
            return
        A = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(A, 0.9)
        self.log_A = np.log(A)

    def _m_step_transition(
        self,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_obs: np.ndarray,
        ll: float,
        gamma: np.ndarray,
    ) -> None:
        # log_xi[t,j,k] = log_alpha[t,j] + log_A[j,k] + log_obs[t+1,k]
        #                 + log_beta[t+1,k] - ll  (t = 0..T-2)
        log_xi = (
            log_alpha[:-1][:, :, None]
            + self.log_A[None, :, :]
            + (log_obs[1:] + log_beta[1:])[:, None, :]
            - ll
        )
        A_num = np.exp(log_xi).sum(axis=0)  # (K, K)
        A_den = gamma[:-1].sum(axis=0)[:, None]  # sum_t γ[t,j], t=0..T-2
        A = A_num / np.maximum(A_den, 1e-300)
        A = A / A.sum(axis=1, keepdims=True)
        self.log_A = np.log(np.maximum(A, 1e-300))
        self.log_pi = np.log(np.maximum(gamma[0], 1e-300))

    def _snapshot(self) -> Any:
        return (self.log_pi.copy(), self.log_A.copy(), self.emission.get_params())

    def _restore(self, snap: Any) -> None:
        self.log_pi = snap[0].copy()
        self.log_A = snap[1].copy()
        self.emission.set_params(snap[2])

    # ---- forward-only filtering (추론, causal) ----

    def filter(self, X: np.ndarray) -> np.ndarray:
        """filtered posterior γ_t = P(상태_t | o_1:t) → (T, K). 미래 미사용(causal)."""
        log_obs = self.emission.log_prob(X)
        log_alpha, _ = self._forward_log(log_obs)
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def filter_init(self, log_obs_0: np.ndarray) -> tuple[np.ndarray, float]:
        """첫 봉: 정규화 log_alpha + log-likelihood 기여분 반환."""
        a = self.log_pi + log_obs_0
        c = float(logsumexp(a))
        return a - c, c

    def filter_step(
        self, log_alpha_norm_prev: np.ndarray, log_obs_t: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """incremental 한 스텝 전진 (RegimeService 용). 정규화 log_alpha 유지(bounded).

        Returns: (정규화 log_alpha_t, c_t). posterior = exp(정규화 log_alpha_t).
        """
        a = log_obs_t + logsumexp(log_alpha_norm_prev[:, None] + self.log_A, axis=0)
        c = float(logsumexp(a))
        return a - c, c

    def smoothed_posterior(self, X: np.ndarray) -> np.ndarray:
        """smoothed γ_t = P(상태_t | o_1:T) (forward-backward) → (T, K).

        **offline 특성화 전용**(매핑 통계). 미래 정보 사용 → 추론 신호(confidence)
        에는 절대 쓰지 말 것 — 그건 filter(forward-only) 를 쓴다.
        """
        log_obs = self.emission.log_prob(X)
        log_alpha, ll = self._forward_log(log_obs)
        log_beta = self._backward_log(log_obs)
        return np.exp(log_alpha + log_beta - ll)

    # ---- 점수 (K 선택용) ----

    def log_likelihood(self, X: np.ndarray) -> float:
        log_obs = self.emission.log_prob(X)
        _, ll = self._forward_log(log_obs)
        return ll

    @property
    def n_params(self) -> int:
        """BIC 용 파라미터 수: emission + transition K(K-1) + initial (K-1)."""
        return self.emission.n_params + self.K * (self.K - 1) + (self.K - 1)

    def bic(self, X: np.ndarray) -> float:
        """BIC = -2·LL + n_params·ln(T). 낮을수록 좋음."""
        ll = self.log_likelihood(X)
        return -2.0 * ll + self.n_params * np.log(len(X))

    def to_dict(self) -> dict:
        return {
            "n_states": self.K,
            "log_pi": self.log_pi.tolist(),
            "log_A": self.log_A.tolist(),
            "emission": self.emission.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HMM":
        hmm = cls(d["n_states"], emission_from_dict(d["emission"]))
        hmm.log_pi = np.asarray(d["log_pi"], dtype=float)
        hmm.log_A = np.asarray(d["log_A"], dtype=float)
        return hmm
