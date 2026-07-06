"""모델 artifact save/load round-trip 테스트 (D4-2d). 수동 모델 (fit 없음, 빠름)."""

from __future__ import annotations

import numpy as np

from src.strategy.regime.artifact import RegimeModel, load_model, save_model
from src.strategy.regime.contract import RegimeDirection, RegimeType
from src.strategy.regime.features import FeatureConfig, ZScoreParams
from src.strategy.regime.hmm import GaussianEmission, HMM, StudentTEmission
from src.strategy.regime.mapping import StateMapping, StateStats


def _manual_model() -> RegimeModel:
    em = GaussianEmission(2, 3)
    em.means = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    em.covs = np.array([np.eye(3), np.eye(3) * 1.5])
    hmm = HMM(2, em)
    hmm.log_pi = np.log([0.4, 0.6])
    hmm.log_A = np.log([[0.9, 0.1], [0.2, 0.8]])
    mapping = StateMapping(
        types=(RegimeType.TREND, RegimeType.RANGE),
        directions=(RegimeDirection.LONG, RegimeDirection.NONE),
        tau=1.5,
        stats=(StateStats(0.01, 0.002), StateStats(0.0, 0.01)),
    )
    zscore = ZScoreParams(
        mean={"log_return": 0.001, "realized_vol": 0.01, "efficiency_ratio": 0.3},
        std={"log_return": 0.005, "realized_vol": 0.003, "efficiency_ratio": 0.2},
    )
    return RegimeModel(
        hmm=hmm,
        mapping=mapping,
        zscore=zscore,
        feature_config=FeatureConfig(),
        valid_period=("2022-01-01", "2030-01-01"),
        meta={"k": 2, "emission_kind": "gaussian"},
    )


def test_artifact_round_trip(tmp_path):
    m = _manual_model()
    path = tmp_path / "model.json"
    save_model(m, path)
    loaded = load_model(path)

    # HMM
    np.testing.assert_allclose(loaded.hmm.log_pi, m.hmm.log_pi)
    np.testing.assert_allclose(loaded.hmm.log_A, m.hmm.log_A)
    np.testing.assert_allclose(loaded.hmm.emission.means, m.hmm.emission.means)
    np.testing.assert_allclose(loaded.hmm.emission.covs, m.hmm.emission.covs)
    assert loaded.hmm.emission.kind == "gaussian"
    assert loaded.hmm.K == m.hmm.K

    # mapping (enum 복원 포함)
    assert loaded.mapping.types == m.mapping.types
    assert loaded.mapping.directions == m.mapping.directions
    assert loaded.mapping.tau == m.mapping.tau
    assert loaded.mapping.stats == m.mapping.stats

    # zscore / feature_config / valid_period / meta
    assert loaded.zscore.mean == m.zscore.mean
    assert loaded.zscore.std == m.zscore.std
    assert loaded.feature_config.vol_window == m.feature_config.vol_window
    assert loaded.feature_config.er_window == m.feature_config.er_window
    assert loaded.valid_period == m.valid_period
    assert loaded.meta == m.meta


def _manual_model_student_t(share_nu: bool = False) -> RegimeModel:
    em = StudentTEmission(2, 3, share_nu=share_nu)
    em.means = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    em.scales = np.array([np.eye(3), np.eye(3) * 1.5])
    em.nus = np.array([7.0, 7.0]) if share_nu else np.array([4.0, 12.0])
    hmm = HMM(2, em)
    hmm.log_pi = np.log([0.4, 0.6])
    hmm.log_A = np.log([[0.9, 0.1], [0.2, 0.8]])
    mapping = StateMapping(
        types=(RegimeType.TREND, RegimeType.NONE),
        directions=(RegimeDirection.LONG, RegimeDirection.NONE),
        tau=0.05,
        stats=(StateStats(0.01, 0.002), StateStats(0.0, 0.01)),
    )
    zscore = ZScoreParams(
        mean={"log_return": 0.001, "realized_vol": 0.01, "efficiency_ratio": 0.3},
        std={"log_return": 0.005, "realized_vol": 0.003, "efficiency_ratio": 0.2},
    )
    return RegimeModel(
        hmm=hmm,
        mapping=mapping,
        zscore=zscore,
        feature_config=FeatureConfig(),
        valid_period=("2022-01-01", "2030-01-01"),
        meta={"k": 2, "emission_kind": "student_t"},
    )


def test_artifact_round_trip_student_t(tmp_path):
    m = _manual_model_student_t()
    path = tmp_path / "model_t.json"
    save_model(m, path)
    loaded = load_model(path)

    em = loaded.hmm.emission
    assert em.kind == "student_t"
    np.testing.assert_allclose(em.means, m.hmm.emission.means)
    np.testing.assert_allclose(em.scales, m.hmm.emission.scales)
    np.testing.assert_allclose(em.nus, m.hmm.emission.nus)
    assert em.share_nu is False
    # fit 재현에 필요한 하이퍼파라미터도 보존 (round-trip 무손실)
    assert em.nu_init == m.hmm.emission.nu_init
    assert em.nu_min == m.hmm.emission.nu_min
    assert em.nu_max == m.hmm.emission.nu_max
    assert em.reg == m.hmm.emission.reg
    # log_prob 동일성 (직렬화 전후 완전 복원)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    np.testing.assert_allclose(em.log_prob(X), m.hmm.emission.log_prob(X), atol=1e-12)


def test_artifact_round_trip_student_t_share_nu(tmp_path):
    m = _manual_model_student_t(share_nu=True)
    path = tmp_path / "model_t_share.json"
    save_model(m, path)
    loaded = load_model(path)
    em = loaded.hmm.emission
    assert em.share_nu is True
    np.testing.assert_allclose(em.nus, m.hmm.emission.nus)
    assert em.n_params == 2 * (3 + 6) + 1  # share_nu → +1


def test_emission_unknown_kind_raises():
    from src.strategy.regime.hmm import emission_from_dict
    import pytest
    with pytest.raises(ValueError):
        emission_from_dict({"kind": "totally_unknown_emission"})
