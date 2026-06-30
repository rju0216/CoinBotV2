"""레짐 HMM 단위 테스트 (D4-2b).

게이트:
  - 합성복원: 알려진 HMM 생성 → fit → 파라미터·filtered posterior 복원.
  - analytic spot-check: 잘 분리된 2상태에서 filtered posterior 를 손으로 유도한
    값과 대조 (forward 재귀 독립 검증).
  - filter_step(incremental) == filter(batch) (라이브-백테 동일 관측열의 토대 H4).
  - filter(forward-only) == forward 정규화 (일관).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from src.strategy.regime.hmm import GaussianEmission, HMM


def _sample_hmm(pi, A, means, covs, T, rng):
    K = len(pi)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.array([rng.multivariate_normal(means[s], covs[s]) for s in states])
    return X, states


def _make_hmm(means, covs, log_pi, log_A):
    K, d = means.shape
    em = GaussianEmission(K, d)
    em.means = np.asarray(means, dtype=float)
    em.covs = np.asarray(covs, dtype=float)
    hmm = HMM(K, em)
    hmm.log_pi = np.asarray(log_pi, dtype=float)
    hmm.log_A = np.asarray(log_A, dtype=float)
    return hmm


# ---- 합성복원 ----

def test_synthetic_recovery_means_and_states():
    rng = np.random.default_rng(42)
    K, d = 2, 2
    true_means = np.array([[0.0, 0.0], [6.0, 6.0]])
    true_covs = np.array([np.eye(2), np.eye(2)])
    true_pi = np.array([0.5, 0.5])
    true_A = np.array([[0.95, 0.05], [0.05, 0.95]])
    X, states = _sample_hmm(true_pi, true_A, true_means, true_covs, 3000, rng)

    hmm = HMM(K, GaussianEmission(K, d))
    hmm.fit(X, n_iter=200, tol=1e-6, n_init=6, seed=0)

    # 평균 복원 (첫 좌표로 정렬해 permutation 해소)
    est_sorted = hmm.emission.means[np.argsort(hmm.emission.means[:, 0])]
    true_sorted = true_means[np.argsort(true_means[:, 0])]
    np.testing.assert_allclose(est_sorted, true_sorted, atol=0.5)

    # filtered posterior 가 실제 상태를 잘 맞춤 (K=2 → 두 permutation 중 best)
    pred = hmm.filter(X).argmax(axis=1)
    acc = max((pred == states).mean(), (1 - pred == states).mean())
    assert acc > 0.95


def test_synthetic_recovery_transition_persistence():
    rng = np.random.default_rng(7)
    K, d = 2, 2
    true_means = np.array([[-5.0, 0.0], [5.0, 0.0]])
    true_covs = np.array([np.eye(2), np.eye(2)])
    true_A = np.array([[0.9, 0.1], [0.2, 0.8]])
    X, _ = _sample_hmm(np.array([0.5, 0.5]), true_A, true_means, true_covs, 4000, rng)

    hmm = HMM(K, GaussianEmission(K, d)).fit(X, n_init=6, seed=1)
    # 대각(self-transition) 우세 복원 (지속성)
    A = np.exp(hmm.log_A)
    assert A[0, 0] > 0.5 and A[1, 1] > 0.5


# ---- analytic spot-check ----

def test_analytic_filter_well_separated():
    # 2상태 1D, 평균 0/10 (잘 분리). x=0 → 상태0 거의 확실, x=10 → 상태1.
    hmm = _make_hmm(
        means=np.array([[0.0], [10.0]]),
        covs=np.array([[[1.0]], [[1.0]]]),
        log_pi=np.log([0.5, 0.5]),
        log_A=np.log([[0.8, 0.2], [0.2, 0.8]]),
    )
    gamma = hmm.filter(np.array([[0.0], [10.0]]))
    # 손유도: x=0 에서 N(0;0,1) ≫ N(0;10,1) → γ_0 ≈ [1,0]; x=10 대칭 → γ_1 ≈ [0,1]
    assert gamma[0, 0] > 0.999
    assert gamma[1, 1] > 0.999


def test_analytic_filter_exact_two_step():
    # 작은 분리에서 손계산 값과 정확 대조 (forward 재귀 독립 검증).
    means = np.array([[0.0], [2.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    pi = np.array([0.5, 0.5])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    hmm = _make_hmm(means, covs, np.log(pi), np.log(A))
    X = np.array([[0.5], [1.5]])

    # 독립 손계산 (정규화 forward, 선형 공간)
    def npdf(x, m):
        return np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
    b0 = np.array([npdf(0.5, 0.0), npdf(0.5, 2.0)])
    a0 = pi * b0
    g0 = a0 / a0.sum()
    b1 = np.array([npdf(1.5, 0.0), npdf(1.5, 2.0)])
    pred1 = A.T @ g0
    a1 = pred1 * b1
    g1 = a1 / a1.sum()
    expected = np.array([g0, g1])

    np.testing.assert_allclose(hmm.filter(X), expected, atol=1e-9)


# ---- filter 일관성 ----

def test_filter_step_matches_batch_filter():
    rng = np.random.default_rng(3)
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covs = np.array([np.eye(2), np.eye(2)])
    hmm = _make_hmm(means, covs, np.log([0.5, 0.5]), np.log([[0.9, 0.1], [0.1, 0.9]]))
    X, _ = _sample_hmm(
        np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]), means, covs, 200, rng
    )

    batch = hmm.filter(X)
    # incremental
    log_obs = hmm.emission.log_prob(X)
    a, _ = hmm.filter_init(log_obs[0])
    inc = [np.exp(a)]
    for t in range(1, len(X)):
        a, _ = hmm.filter_step(a, log_obs[t])
        inc.append(np.exp(a))
    np.testing.assert_allclose(np.array(inc), batch, atol=1e-10)


def test_filter_loglik_matches_forward():
    # incremental c_t 합 == forward 전체 log-likelihood
    rng = np.random.default_rng(5)
    means = np.array([[0.0], [3.0]])
    covs = np.array([[[1.0]], [[1.0]]])
    hmm = _make_hmm(means, covs, np.log([0.5, 0.5]), np.log([[0.8, 0.2], [0.2, 0.8]]))
    X, _ = _sample_hmm(
        np.array([0.5, 0.5]), np.array([[0.8, 0.2], [0.2, 0.8]]), means, covs, 150, rng
    )
    log_obs = hmm.emission.log_prob(X)
    a, c = hmm.filter_init(log_obs[0])
    total = c
    for t in range(1, len(X)):
        a, c = hmm.filter_step(a, log_obs[t])
        total += c
    assert abs(total - hmm.log_likelihood(X)) < 1e-8


# ---- n_params / bic ----

def test_n_params_gaussian_full_cov():
    hmm = HMM(3, GaussianEmission(3, 2))
    # emission: 3*(2 + 3) = 15 ; transition 3*2=6 ; initial 2 → 23
    assert hmm.emission.n_params == 3 * (2 + 3)
    assert hmm.n_params == 15 + 6 + 2


# ---- M-step 직접 검증 (참조구현 대조) ----

def test_m_step_transition_matches_brute_force():
    rng = np.random.default_rng(0)
    means = np.array([[0.0, 0.0], [4.0, 4.0]])
    covs = np.array([np.eye(2), np.eye(2)])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    hmm = _make_hmm(means, covs, np.log([0.5, 0.5]), np.log(A))
    X, _ = _sample_hmm(np.array([0.5, 0.5]), A, means, covs, 100, rng)

    log_obs = hmm.emission.log_prob(X)
    log_alpha, ll = hmm._forward_log(log_obs)
    log_beta = hmm._backward_log(log_obs)
    gamma = np.exp(log_alpha + log_beta - ll)

    # 참조: 3중 루프로 xi 누적 (벡터화 코드와 독립)
    T, K = log_obs.shape
    A_num = np.zeros((K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                A_num[i, j] += np.exp(
                    log_alpha[t, i] + hmm.log_A[i, j]
                    + log_obs[t + 1, j] + log_beta[t + 1, j] - ll
                )
    A_ref = A_num / A_num.sum(axis=1, keepdims=True)

    hmm._m_step_transition(log_alpha, log_beta, log_obs, ll, gamma)
    np.testing.assert_allclose(np.exp(hmm.log_A), A_ref, atol=1e-10)
    # π = gamma[0]
    np.testing.assert_allclose(np.exp(hmm.log_pi), gamma[0], atol=1e-12)


def test_emission_m_step_recovers_weighted_stats():
    # 모든 가중치를 한 상태에 (one-hot) → 그 상태 평균/공분산 = 데이터 경험값
    rng = np.random.default_rng(1)
    X = rng.multivariate_normal([3.0, -2.0], [[2.0, 0.5], [0.5, 1.0]], 3000)
    em = GaussianEmission(2, 2, reg=1e-9)
    gamma = np.zeros((3000, 2))
    gamma[:, 0] = 1.0
    em.m_step(X, gamma)
    np.testing.assert_allclose(em.means[0], X.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(em.covs[0], np.cov(X.T, bias=True), atol=1e-6)
