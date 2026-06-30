"""모델 artifact save/load round-trip 테스트 (D4-2d). 수동 모델 (fit 없음, 빠름)."""

from __future__ import annotations

import numpy as np

from src.strategy.regime.artifact import RegimeModel, load_model, save_model
from src.strategy.regime.contract import RegimeDirection, RegimeType
from src.strategy.regime.features import FeatureConfig, ZScoreParams
from src.strategy.regime.hmm import GaussianEmission, HMM
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


def test_emission_unknown_kind_raises():
    from src.strategy.regime.hmm import emission_from_dict
    import pytest
    with pytest.raises(ValueError):
        emission_from_dict({"kind": "student_t_not_yet"})
