"""Phase 7 통짜 통합 회귀 (규칙 17) — 신규 3 모듈이 **함께 돌 때**의 불변식 박제.

단위 테스트는 격리돼 있고 전체 회귀는 비파괴만 본다. 여기서는 이번 Phase 에 추가된
`metrics.directional_metrics` · `feature_audit` · `alpha_decomp` 가 **같은 라벨/예측 위에서
서로 모순되지 않는지**를 관통 검증한다. 실데이터 없이 합성으로 성립하는 불변식만 다룬다.

박제하는 6 불변식:
  I1 방향 지표 ↔ 알파 분해: "전부 롱" 이면 pred_up_share=1 이고 방향 알파는 정의상 0
  I2 s/q 분해 항등식: p_dir == s × q (전 표본)
  I3 alpha 3분해 항등식: 전략 == 벤치 + 방향알파 (게이트 표본에서)
  I4 선택 알파 정의: 게이트봉 벤치 − 전체봉 벤치
  I5 feature_audit ↔ 라벨: 방향맹 피처의 AUC 는 0.5 근처, 완전정보 피처는 1.0
  I6 harness 관통: run_walk_forward 결과에 방향 지표가 폴드·국면별로 **자동** 실려온다
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.experiments.alpha_decomp import decompose
from src.research.validation.feature_audit import directional_auc
from src.research.validation.harness import run_walk_forward
from src.research.validation.metrics import directional_metrics
from src.research.validation.splitter import WalkForwardSplitter

N_H = 2
CLS = ["up", "down", "expire"]


def _fixture(n=240, seed=0):
    """합성 아티팩트 + 캔들 + X. 라벨/예측/피처가 한 인덱스에 정렬된 full-stack 입력."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    lab = np.array(["up", "down", "expire"])[rng.integers(0, 3, n)]
    # 예측: 전부 롱(up 우세) — I1 검증용
    art = pd.DataFrame({"ens_up": 0.55, "ens_down": 0.25, "ens_expire": 0.20,
                        "y_true": lab, "barrier_frac": 0.05}, index=idx)
    candles = pd.DataFrame({"close": 100.0},
                           index=pd.date_range("2024-01-01", periods=n + N_H, freq="4h", tz="UTC"))
    X = pd.DataFrame({
        "blind": rng.normal(0, 1, n),                                   # 방향맹
        "oracle": np.where(lab == "up", 1.0, np.where(lab == "down", -1.0, 0.0)),  # 완전정보
    }, index=idx)
    return art, candles, X, pd.Series(lab, index=idx)


def test_i1_all_long_zero_direction_alpha_and_full_pred_up():
    art, candles, _, y = _fixture()
    proba = art[["ens_up", "ens_down", "ens_expire"]].rename(
        columns=lambda c: c.replace("ens_", ""))
    dm = directional_metrics(y, proba)
    d = decompose(art, candles, N_H)
    assert dm["pred_up_share"] == 1.0
    assert d.loc["전체", "long_share_pct"] == 100.0
    # 두 모듈이 같은 사실을 말해야 한다: 전부 롱 → 방향 알파 0
    assert d.loc["전체", "direction_alpha_bp"] == pytest.approx(0.0, abs=1e-6)


def test_i2_sq_decomposition_identity():
    """p_dir == s × q 가 전 표본에서 성립 (s/q 분해가 항등식임을 박제)."""
    rng = np.random.default_rng(1)
    raw = rng.dirichlet([2, 2, 2], size=500)
    proba = pd.DataFrame(raw, columns=CLS)
    y = pd.Series(np.array(CLS)[rng.integers(0, 3, 500)])
    m = directional_metrics(y, proba)
    up, dn = proba["up"].to_numpy(), proba["down"].to_numpy()
    s, q = up + dn, np.maximum(up, dn) / (up + dn)
    assert np.abs(s * q - np.maximum(up, dn)).max() < 1e-12
    assert m["s_mean"] == pytest.approx(float(s.mean()))
    assert m["q_median"] == pytest.approx(float(np.median(q)))


def test_i3_alpha_three_way_identity():
    """전략 == 벤치(베타) + 방향알파 — 게이트 표본에서 항등식."""
    art, candles, _, _ = _fixture(seed=3)
    gate = np.arange(len(art)) % 3 == 0
    d = decompose(art, candles, N_H, gate=gate)
    for yr in d.index:
        assert d.loc[yr, "strategy_bp"] == pytest.approx(
            d.loc[yr, "benchmark_bp(베타)"] + d.loc[yr, "direction_alpha_bp"], abs=0.15)


def test_i4_selection_alpha_is_gate_minus_all():
    """선택 알파 == 게이트봉 벤치 − 전체봉 벤치."""
    art, candles, _, _ = _fixture(seed=4)
    gate = np.arange(len(art)) % 2 == 0
    d_gate = decompose(art, candles, N_H, gate=gate)
    d_all = decompose(art, candles, N_H)
    assert d_gate.loc["전체", "selection_alpha_bp"] == pytest.approx(
        d_gate.loc["전체", "benchmark_bp(베타)"] - d_all.loc["전체", "benchmark_bp(베타)"], abs=0.15)


def test_i5_feature_audit_separates_blind_from_oracle():
    _, _, X, y = _fixture(n=600, seed=5)
    r = directional_auc(X, y, min_n=10)
    assert r.loc["oracle", "auc"] == pytest.approx(1.0)
    assert abs(r.loc["blind", "auc"] - 0.5) < 0.08


def test_i6_harness_carries_direction_metrics_per_fold_and_regime():
    """★상설화 관통★ run_walk_forward → aggregate_metrics 가 방향 지표를 자동으로 싣는다."""
    _, _, X, y = _fixture(n=600, seed=6)
    regimes = pd.Series(np.where(np.arange(600) < 300, "A", "B"), index=X.index)

    class _Const:
        classes_ = CLS

        def fit(self, x, yy):
            return self

        def predict(self, x):
            return np.array(["up"] * len(x))

        def predict_proba(self, x):
            return pd.DataFrame(np.tile([0.6, 0.2, 0.2], (len(x), 1)),
                                index=x.index, columns=CLS)

    res = run_walk_forward(X, y, _Const, WalkForwardSplitter(200, 100, label_horizon=5),
                           regime_tags=regimes, labels=CLS)
    for col in ("pred_up_share", "dir_bias", "dir_hit_rate", "s_mean", "q_median"):
        assert col in res.report.per_fold.columns, f"per_fold 에 {col} 누락"
        assert col in res.report.per_regime.columns, f"per_regime 에 {col} 누락"
    assert (res.report.per_fold["pred_up_share"] == 1.0).all()   # 상수 롱 예측
