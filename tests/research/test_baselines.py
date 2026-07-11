"""baseline 회귀 테스트 — fit/predict 계약·prior·uniform."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.validation.baselines import (
    BaselineModel,
    PriorBaseline,
    UniformBaseline,
)


def _X(n):
    return pd.DataFrame({"f": np.zeros(n)}, index=pd.RangeIndex(n))


def test_prior_baseline_frequencies_and_majority():
    y = ["A"] * 7 + ["B"] * 3
    m = PriorBaseline().fit(_X(len(y)), y)
    assert m.classes_ == ["A", "B"]
    assert np.allclose(m.priors_, [0.7, 0.3])
    proba = m.predict_proba(_X(5))
    assert proba.shape == (5, 2)
    assert np.allclose(proba.iloc[0].to_numpy(), [0.7, 0.3])
    # predict = 최빈 클래스
    assert (m.predict(_X(5)) == "A").all()


def test_uniform_baseline():
    y = ["A", "B", "C"]
    m = UniformBaseline().fit(_X(3), y)
    assert m.classes_ == ["A", "B", "C"]
    proba = m.predict_proba(_X(4))
    assert np.allclose(proba.to_numpy(), 1.0 / 3)


def test_is_baseline_model():
    assert isinstance(PriorBaseline(), BaselineModel)
    assert isinstance(UniformBaseline(), BaselineModel)


def test_predict_proba_index_preserved():
    idx = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    X = pd.DataFrame({"f": np.zeros(6)}, index=idx)
    m = PriorBaseline().fit(X, ["A", "A", "B", "A", "B", "A"])
    proba = m.predict_proba(X)
    assert (proba.index == idx).all()
