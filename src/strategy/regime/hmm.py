"""레짐 발견용 HMM — EM 학습 + forward-only filtering (D4-2b).

설계 (D2/D3 결정):
  - **emission 추상화**: HMM 은 `emission.log_prob` / `emission.m_step` /
    `n_params` / `get_params`/`set_params` 만 안다. GaussianEmission 먼저,
    D4-4 에서 StudentTEmission 한 클래스 추가로 교체(O-6 단계화).
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
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


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
        for k in range(self.K):
            out[:, k] = multivariate_normal.logpdf(
                X, mean=self.means[k], cov=self.covs[k]
            )
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


def emission_from_dict(d: dict) -> Emission:
    """직렬화 dict → Emission (kind 로 dispatch). D4-4 에서 student_t 추가."""
    if d["kind"] == "gaussian":
        return GaussianEmission.from_dict(d)
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
        for t in range(1, T):
            log_alpha[t] = log_obs[t] + logsumexp(
                log_alpha[t - 1][:, None] + self.log_A, axis=0
            )
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def _backward_log(self, log_obs: np.ndarray) -> np.ndarray:
        T = log_obs.shape[0]
        log_beta = np.zeros((T, self.K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                self.log_A + log_obs[t + 1][None, :] + log_beta[t + 1][None, :],
                axis=1,
            )
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
