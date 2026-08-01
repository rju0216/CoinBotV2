"""지표 회귀 테스트 — 불균형 보정·log_loss·pinball·semi_dev."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.validation.metrics import (
    FoldPrediction,
    directional_metrics,
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


# ---- 방향 편향 지표 (Phase 7 상설화) ----

def _proba(rows):
    """rows = [(up, down, expire)] → proba DataFrame."""
    return pd.DataFrame(rows, columns=["up", "down", "expire"])


def test_directional_metrics_known_answer():
    # 4봉: 예측 방향 up,up,down,up / 라벨 up,down,down,expire
    proba = _proba([(0.5, 0.2, 0.3), (0.4, 0.3, 0.3), (0.1, 0.6, 0.3), (0.45, 0.25, 0.3)])
    y = pd.Series(["up", "down", "down", "expire"])
    m = directional_metrics(y, proba)
    assert m["pred_up_share"] == pytest.approx(0.75)          # 전 봉 4 중 3봉 up 방향
    assert m["resolved_share"] == pytest.approx(0.75)         # expire 1봉 제외
    assert m["label_up_share"] == pytest.approx(1 / 3)        # 해소 3봉 중 up 1
    # ★fresh-eyes H-1★ dir_bias 는 **같은 모집단**(해소봉)끼리 빼야 한다.
    # 해소 3봉의 예측 방향 = up,up,down → 2/3
    assert m["pred_up_share_resolved"] == pytest.approx(2 / 3)
    assert m["dir_bias"] == pytest.approx(2 / 3 - 1 / 3)
    # 전 봉 기준 차이는 이력 호환용으로 병기되며 모집단이 다름을 명시
    assert m["dir_bias_allbars"] == pytest.approx(0.75 - 1 / 3)
    # 해소봉 적중: (up,up)=O (up,down)=X (down,down)=O → 2/3
    assert m["dir_hit_rate"] == pytest.approx(2 / 3)
    assert m["margin_mean"] == pytest.approx(np.mean([0.3, 0.1, -0.5, 0.2]))
    assert m["s_mean"] == pytest.approx(0.7)                  # 전 봉 up+down=0.7
    # q = max/s: .5/.7, .4/.7, .6/.7, .45/.7 → 중앙값 = (0.45+0.5)/2 / 0.7
    assert m["q_median"] == pytest.approx((0.475) / 0.7)


def test_directional_metrics_detects_long_bias():
    """라벨은 균형인데 예측이 전부 롱 → dir_bias 가 크게 양수."""
    proba = _proba([(0.4, 0.2, 0.4)] * 10)
    y = pd.Series(["up", "down"] * 5)
    m = directional_metrics(y, proba)
    assert m["pred_up_share"] == 1.0
    assert m["label_up_share"] == pytest.approx(0.5)
    assert m["dir_bias"] == pytest.approx(0.5)
    assert m["dir_hit_rate"] == pytest.approx(0.5)   # 절반만 맞음


def test_directional_metrics_tie_follows_engine_long_convention():
    """★fresh-eyes H-2★ margin==0 은 엔진과 같이 **롱**으로 센다(dumb_l3 `p_up >= p_down`).

    UniformBaseline 은 전 봉이 정확히 타이라 이 경로가 실제로 도달된다. 과거 구현은
    pred_up_share(>=)와 dir_hit_rate(>)가 같은 함수 안에서 반대로 세었다.
    """
    proba = _proba([(0.3, 0.3, 0.4)] * 4)          # 전 봉 정확히 타이
    y = pd.Series(["up", "up", "down", "down"])
    m = directional_metrics(y, proba)
    assert m["pred_up_share"] == 1.0               # 타이 → 롱
    assert m["pred_up_share_resolved"] == 1.0
    assert m["dir_hit_rate"] == pytest.approx(0.5)  # 롱 4개 중 라벨 up 2개
    assert m["dir_bias"] == pytest.approx(0.5)


def test_directional_metrics_noop_for_other_label_sets():
    """label-agnostic 계약 — up/down 열이 없으면 빈 dict (다른 라벨 집합엔 no-op)."""
    proba = pd.DataFrame([[0.6, 0.4]], columns=["A", "B"])
    assert directional_metrics(pd.Series(["A"]), proba) == {}
    assert directional_metrics(pd.Series(["A"]), None) == {}


def test_directional_metrics_skips_nan_labels():
    proba = _proba([(0.5, 0.2, 0.3), (0.1, 0.6, 0.3)])
    y = pd.Series(["up", None])
    m = directional_metrics(y, proba)
    assert m["pred_up_share"] == 1.0        # NaN 행 제외 후 1봉만, 방향 up
    assert m["dir_hit_rate"] == 1.0


def test_aggregate_metrics_auto_includes_direction():
    """상설화 확인 — aggregate_metrics 가 폴드별로 방향 지표를 자동 계측한다."""
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    proba = _proba([(0.5, 0.2, 0.3)] * 4).set_index(idx)
    y = pd.Series(["up", "down", "up", "down"], index=idx)
    pred = pd.Series(["up"] * 4, index=idx)
    rep = aggregate_metrics([FoldPrediction(0, y, pred, proba)],
                            labels=["up", "down", "expire"])
    assert "pred_up_share" in rep.per_fold.columns
    assert "dir_hit_rate" in rep.per_fold.columns
    assert rep.per_fold.loc[0, "pred_up_share"] == 1.0
    assert rep.per_fold.loc[0, "dir_bias"] == pytest.approx(0.5)
