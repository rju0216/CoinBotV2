"""지표 회귀 테스트 — 불균형 보정·log_loss·pinball·semi_dev."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.validation.metrics import (
    FoldPrediction,
    aggregate_metrics,
    balanced_accuracy,
    mcc,
    multiclass_log_loss,
    pinball,
    semi_deviation,
)


def test_balanced_accuracy_penalizes_majority_guess():
    y_true = ["A"] * 9 + ["B"]
    y_pred = ["A"] * 10          # 정확도 0.9 지만 소수 클래스 못 맞춤
    assert balanced_accuracy(y_true, y_pred) == 0.5   # (recall_A=1 + recall_B=0)/2


def test_mcc_perfect_and_wrong():
    y = ["A", "B", "C", "A", "B"]
    assert mcc(y, y) == 1.0
    assert mcc(y, ["B", "A", "A", "C", "C"]) < 1.0


def test_log_loss_perfect_vs_uniform():
    y = ["A", "B"]
    perfect = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert multiclass_log_loss(y, perfect, ["A", "B"]) < 1e-6
    uniform = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert multiclass_log_loss(y, uniform, ["A", "B"]) == np.log(2)


def test_log_loss_nonsorted_labels_alignment():
    # 회귀: 비정렬 labels(예: LABEL_CLASSES=('up','down','expire'))에서 proba-클래스
    # 오정렬 버그 방지. sklearn 은 정렬 순서를 가정 → 정렬 안 하면 오답확률 참조.
    labels = ["up", "down", "expire"]            # 비정렬(정렬시 down,expire,up)
    y_true = ["up", "up", "down"]
    # 각 행이 정답 클래스에 0.8 → 정답 log_loss = -ln(0.8)
    proba = pd.DataFrame(
        [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], columns=labels
    )
    assert multiclass_log_loss(y_true, proba, labels) == pytest.approx(-np.log(0.8))
    # 순열 불변: labels 순서를 바꿔도(같은 proba 매핑) 동일 값
    perm = ["expire", "up", "down"]
    proba_perm = proba[perm]
    assert multiclass_log_loss(y_true, proba_perm, perm) == pytest.approx(-np.log(0.8))


def test_pinball_basic():
    assert pinball([1, 2, 3], [1, 2, 3], alpha=0.5) == 0.0
    # y-yhat=2, alpha=0.5 → max(0.5*2, -0.5*2)=1
    assert pinball([2.0], [0.0], alpha=0.5) == 1.0


def test_semi_deviation():
    # x=[1,2,3,4,5], target=3 → downside=[-2,-1,0,0,0], sqrt(5/5)=1
    assert semi_deviation([1, 2, 3, 4, 5], target=3.0) == 1.0
    # target=None → mean=3 → 동일
    assert semi_deviation([1, 2, 3, 4, 5]) == 1.0


def test_semi_deviation_skips_nan():
    a = semi_deviation([1, 2, 4, 5], target=3.0)
    b = semi_deviation([1, 2, np.nan, 4, 5], target=3.0)
    assert a == b   # NaN 스킵


def test_pooled_per_regime_heterogeneous_classes():
    # F2 회귀: 폴드마다 클래스 집합이 다를 때 per-regime log_loss 가 크래시하지 않아야 함
    idx0 = pd.date_range("2020-01-01", periods=3, freq="h", tz="UTC")
    idx1 = pd.date_range("2020-01-01 03:00", periods=3, freq="h", tz="UTC")
    labels = ["down", "flat", "up"]
    fp0 = FoldPrediction(
        0, pd.Series(["up", "down", "flat"], index=idx0),
        pd.Series(["up", "up", "up"], index=idx0),
        pd.DataFrame({"down": [.3, .3, .3], "flat": [.3, .3, .3], "up": [.4, .4, .4]}, index=idx0),
    )
    fp1 = FoldPrediction(  # flat 열 없음 (이종 클래스)
        1, pd.Series(["up", "down", "up"], index=idx1),
        pd.Series(["up", "up", "up"], index=idx1),
        pd.DataFrame({"down": [.5, .5, .5], "up": [.5, .5, .5]}, index=idx1),
    )
    regime = pd.Series(["bull"] * 6, index=idx0.append(idx1))
    rep = aggregate_metrics([fp0, fp1], regime_tags=regime, labels=labels)
    assert "log_loss" in rep.per_regime.columns   # 크래시 없이 계산됨
    assert rep.per_regime.loc["bull", "n"] == 6
