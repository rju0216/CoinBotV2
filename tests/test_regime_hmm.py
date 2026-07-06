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

from scipy.special import digamma
from scipy.stats import multivariate_normal, multivariate_t

from src.strategy.regime.hmm import (
    GaussianEmission,
    HMM,
    StudentTEmission,
    make_emission,
)


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


# ==================== Student-t emission (D4-4 S1) ====================


def _sample_t_hmm(pi, A, means, scales, nus, T, rng):
    """t-HMM 샘플. **shape=scales(scale 행렬)** — cov 아님 (blocker#2)."""
    K = len(pi)
    states = np.empty(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    X = np.array(
        [
            multivariate_t.rvs(
                loc=means[s], shape=scales[s], df=nus[s], random_state=rng
            )
            for s in states
        ]
    )
    return X, states


# ---- 배선 / 정의 ----

def test_student_t_log_prob_matches_scipy():
    em = StudentTEmission(2, 2)
    em.means = np.array([[0.0, 0.0], [3.0, -1.0]])
    em.scales = np.array([np.eye(2), [[2.0, 0.3], [0.3, 1.0]]])
    em.nus = np.array([4.0, 8.0])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    lp = em.log_prob(X)
    for k in range(2):
        ref = multivariate_t.logpdf(
            X, loc=em.means[k], shape=em.scales[k], df=em.nus[k]
        )
        np.testing.assert_allclose(lp[:, k], ref, atol=1e-12)


def test_student_t_reduces_to_gaussian_large_nu():
    # ν→∞ 에서 t(μ,Σ,ν) → N(μ,Σ) (cov=scale). log_prob 이 Gaussian 과 수렴.
    means = np.array([[0.0, 0.0], [3.0, -1.0]])
    covs = np.array([np.eye(2), [[2.0, 0.3], [0.3, 1.0]]])
    em_t = StudentTEmission(2, 2)
    em_t.means, em_t.scales, em_t.nus = means.copy(), covs.copy(), np.array([1e7, 1e7])
    em_g = GaussianEmission(2, 2)
    em_g.means, em_g.covs = means.copy(), covs.copy()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 2))
    np.testing.assert_allclose(em_t.log_prob(X), em_g.log_prob(X), atol=1e-3)


def test_n_params_student_t():
    em = StudentTEmission(3, 2)  # per-state ν
    # scale/mean: 3*(2 + 3)=15 ; ν per-state: +3 → 18
    assert em.n_params == 3 * (2 + 3) + 3
    em_share = StudentTEmission(3, 2, share_nu=True)
    assert em_share.n_params == 3 * (2 + 3) + 1


# ---- m_step 참조대조 (one-step, 손계산식) ----

def test_student_t_m_step_one_step_reference():
    rng = np.random.default_rng(2)
    X = rng.multivariate_normal([1.0, -1.0], [[1.5, 0.2], [0.2, 0.8]], 500)
    # soft γ (한 상태만 활성, 나머지 0 → 단일상태 참조 단순화)
    gamma = np.zeros((500, 2))
    gamma[:, 0] = rng.uniform(0.3, 1.0, size=500)

    em = StudentTEmission(2, 2, reg=1e-9)
    mu0 = np.array([1.2, -0.8])
    S0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    nu0 = 6.0
    em.means[0], em.scales[0], em.nus[0] = mu0, S0, nu0

    # 독립 손계산 (갱신 전 params 기준 E-step)
    diff = X - mu0
    Sinv = np.linalg.inv(S0)
    delta2 = np.einsum("ti,ij,tj->t", diff, Sinv, diff)
    E_u = (nu0 + 2) / (nu0 + delta2)
    E_logu = digamma((nu0 + 2) / 2) - np.log((nu0 + delta2) / 2)
    w = gamma[:, 0]
    gu = w * E_u
    mean_ref = (gu[:, None] * X).sum(0) / gu.sum()
    d2 = X - mean_ref
    scale_ref = (d2.T * gu) @ d2 / w.sum() + 1e-9 * np.eye(2)
    c = (w * (E_logu - E_u)).sum() / w.sum()
    from scipy.optimize import brentq
    nu_ref = brentq(lambda nu: 1 - digamma(nu / 2) + np.log(nu / 2) + c, 2.0, 200.0)

    em.m_step(X, gamma)
    np.testing.assert_allclose(em.means[0], mean_ref, atol=1e-10)
    np.testing.assert_allclose(em.scales[0], scale_ref, atol=1e-10)
    np.testing.assert_allclose(em.nus[0], nu_ref, atol=1e-8)


def test_student_t_nu_matches_q_maximization():
    # ν 를 **코드 경로(g(ν) brentq)와 무관하게** complete-data Q-함수 직접 최대화로
    # 독립 재검증 (one-step 참조가 같은 g(ν)를 재사용하는 tautology 보강, 리뷰 B6).
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln

    rng = np.random.default_rng(11)
    X = rng.standard_t(df=4, size=(600, 2)) * 0.8 + np.array([1.0, -0.5])
    gamma = np.zeros((600, 1))
    gamma[:, 0] = rng.uniform(0.4, 1.0, size=600)

    em = StudentTEmission(1, 2, reg=1e-9)
    em.means[0], em.scales[0], em.nus[0] = np.array([0.9, -0.4]), np.eye(2), 5.0
    # 갱신 전 params 로 E-step (코드와 동일 시점)
    diff = X - em.means[0]
    Sinv = np.linalg.inv(em.scales[0])
    delta2 = np.einsum("ti,ij,tj->t", diff, Sinv, diff)
    E_u = (5.0 + 2) / (5.0 + delta2)
    E_logu = digamma((5.0 + 2) / 2) - np.log((5.0 + delta2) / 2)
    w = gamma[:, 0]

    em.m_step(X, gamma)
    nu_code = em.nus[0]

    # 독립: Q_ν(ν) = Σγ[(ν/2)log(ν/2) − gammaln(ν/2) + (ν/2−1)E_logu − (ν/2)E_u] 최대화
    def negQ(nu):
        return -float(
            (w * ((nu / 2) * np.log(nu / 2) - gammaln(nu / 2)
                  + (nu / 2 - 1) * E_logu - (nu / 2) * E_u)).sum()
        )

    res = minimize_scalar(negQ, bounds=(2.0, 200.0), method="bounded")
    assert abs(nu_code - res.x) < 1e-4


def test_student_t_m_step_downweights_outliers():
    # outlier 주입 시 t 평균이 Gaussian 평균보다 bulk(0)에 근접 (로버스트).
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, size=(400, 1))
    X = np.vstack([X, np.full((20, 1), 30.0)])  # 극단 outlier
    gamma = np.ones((len(X), 1))

    em_t = StudentTEmission(1, 1, reg=1e-9)
    em_t.means[0], em_t.scales[0], em_t.nus[0] = np.array([0.0]), np.array([[1.0]]), 4.0
    em_t.m_step(X, gamma)

    em_g = GaussianEmission(1, 1, reg=1e-9)
    em_g.m_step(X, gamma)

    assert abs(em_t.means[0, 0]) < abs(em_g.means[0, 0])


# ---- 가드 (blocker#1 bracket / 주의#6 빈상태) ----

def test_student_t_solve_nu_bracket_guard():
    # blocker#1: g(ν)=1−ψ(ν/2)+log(ν/2)+c 의 근이 [nu_min,nu_max] 밖이면 clamp
    # (brentq 동부호 crash 없이). c 로 결정론적 검증.
    em = StudentTEmission(1, 1, nu_min=2.0, nu_max=200.0)
    # c=0 → g(2)=1.577>0, g(200)=1.005>0 (둘 다 양수) → 근>max → clamp nu_max.
    assert em._solve_nu(0.0) == 200.0
    # c=-2 → g(2)=-0.42<0, g(200)=-0.995<0 (둘 다 음수) → 근<min → clamp nu_min.
    assert em._solve_nu(-2.0) == 2.0
    # c=-1.3 → 부호 반대 → 내부 근 (brentq).
    nu = em._solve_nu(-1.3)
    assert 2.0 < nu < 200.0


def test_student_t_nu_finite_light_tail():
    # Gaussian data fit → ν 유한·crash 없음·무거운꼬리(ν=3)보다 큰 추정 (상대 비교).
    rng = np.random.default_rng(4)
    X = rng.multivariate_normal([0.0, 0.0], np.eye(2), 2000)
    em = StudentTEmission(1, 2)
    HMM(1, em).fit(X, n_init=1, n_iter=30, seed=0)
    assert np.isfinite(em.nus).all()
    assert em.nus[0] > 8.0  # 무거운꼬리 테스트(ν<20)보다 확실히 가벼운 꼬리


def test_student_t_nu_finite_for_heavy_tail():
    # 무거운꼬리(ν=3) data → 유한 ν 추정 (상한 미도달).
    rng = np.random.default_rng(5)
    X = multivariate_t.rvs(loc=[0.0, 0.0], shape=np.eye(2), df=3, size=3000,
                           random_state=rng)
    em = StudentTEmission(1, 2)
    hmm = HMM(1, em)
    hmm.fit(X, n_init=2, n_iter=50, seed=0)
    assert 2.0 < em.nus[0] < 20.0


def test_student_t_empty_state_guard():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(300, 2))
    em = StudentTEmission(2, 2)
    saved = em.get_params()
    gamma = np.zeros((300, 2))
    gamma[:, 0] = 1.0  # 상태1 은 책임도 0 (빈 상태)
    em.m_step(X, gamma)
    # 빈 상태(1)의 파라미터 불변
    np.testing.assert_allclose(em.means[1], saved[0][1])
    np.testing.assert_allclose(em.scales[1], saved[1][1])
    assert em.nus[1] == saved[2][1]


def test_student_t_log_prob_finite_after_fit():
    # blocker#3: fit 후 log_obs 전부 유한 (NaN 원천 차단 확인).
    rng = np.random.default_rng(7)
    X, _ = _sample_t_hmm(
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[0.0, 0.0], [4.0, 4.0]]),
        np.array([np.eye(2), np.eye(2)]),
        np.array([5.0, 5.0]),
        1000,
        rng,
    )
    hmm = HMM(2, StudentTEmission(2, 2)).fit(X, n_init=2, n_iter=50, seed=0)
    assert np.isfinite(hmm.emission.log_prob(X)).all()


# ---- share_nu ----

def test_student_t_share_nu_pools_to_single():
    rng = np.random.default_rng(8)
    X, _ = _sample_t_hmm(
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[-3.0, 0.0], [3.0, 0.0]]),
        np.array([np.eye(2), np.eye(2)]),
        np.array([5.0, 5.0]),
        1500,
        rng,
    )
    em = StudentTEmission(2, 2, share_nu=True)
    HMM(2, em).fit(X, n_init=2, n_iter=50, seed=0)
    assert np.isclose(em.nus[0], em.nus[1])  # 공통 ν


# ---- 합성복원 (blocker#2 shape 파라미터화) ----

def test_student_t_synthetic_recovery():
    rng = np.random.default_rng(9)
    K, d = 2, 2
    true_means = np.array([[0.0, 0.0], [6.0, 6.0]])
    true_scales = np.array([np.eye(2), np.eye(2)])
    true_nus = np.array([5.0, 5.0])
    true_pi = np.array([0.5, 0.5])
    true_A = np.array([[0.95, 0.05], [0.05, 0.95]])
    X, states = _sample_t_hmm(
        true_pi, true_A, true_means, true_scales, true_nus, 4000, rng
    )

    hmm = HMM(K, StudentTEmission(K, d)).fit(X, n_iter=200, n_init=6, seed=0)

    # 평균(=loc) 복원 (첫 좌표 정렬로 permutation 해소)
    order = np.argsort(hmm.emission.means[:, 0])
    est_means = hmm.emission.means[order]
    true_sorted = true_means[np.argsort(true_means[:, 0])]
    np.testing.assert_allclose(est_means, true_sorted, atol=0.6)

    # filtered posterior 가 실제 상태 잘 맞춤
    pred = hmm.filter(X).argmax(axis=1)
    acc = max((pred == states).mean(), (1 - pred == states).mean())
    assert acc > 0.9

    # ν 미붕괴 (상한/하한에 안 붙고 참값 5 근방; T=4000 이라 적당히 조임)
    assert np.all((hmm.emission.nus > 3.0) & (hmm.emission.nus < 12.0))


# ---- make_emission factory (S2 배선) ----

def test_make_emission_gaussian():
    em = make_emission("gaussian", 3, 2, reg=1e-5)
    assert isinstance(em, GaussianEmission)
    assert em.K == 3 and em.d == 2 and em.reg == 1e-5


def test_make_emission_student_t_passes_params():
    # 기본값과 다른 값을 주고 전 파라미터가 생성자로 위임되는지 (nu_min·reg 포함).
    em = make_emission(
        "student_t", 4, 3,
        reg=1e-4, nu_init=7.0, nu_min=3.0, nu_max=150.0, share_nu=True,
    )
    assert isinstance(em, StudentTEmission)
    assert em.K == 4 and em.d == 3
    assert em.reg == 1e-4
    assert em.nu_init == 7.0 and em.nu_min == 3.0 and em.nu_max == 150.0
    assert em.share_nu is True
    assert np.allclose(em.nus, 7.0)  # nu_init 로 초기화


def test_make_emission_ignores_unused_params():
    # gaussian 은 t 전용 키(nu_init 등)를 무시 (kind별 params 상이).
    em = make_emission("gaussian", 2, 2, nu_init=3.0, share_nu=True)
    assert isinstance(em, GaussianEmission)


def test_make_emission_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        make_emission("laplace", 2, 2)
