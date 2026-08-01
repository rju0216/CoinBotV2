"""피처 방향성·대칭성 감사 회귀 테스트 (Phase 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.validation.feature_audit import (
    auc_by_segment,
    conditional_distribution,
    directional_auc,
    fold_segments,
    sign_stability,
)
from src.research.validation.splitter import WalkForwardSplitter


def _xy(n=400, seed=0):
    """합성: informative = 라벨과 단조 관계, noise = 무관, flipped = 반대 부호."""
    rng = np.random.default_rng(seed)
    lab_up = rng.integers(0, 2, n).astype(bool)
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    X = pd.DataFrame({
        "informative": np.where(lab_up, 1.0, -1.0) + rng.normal(0, 0.3, n),
        "noise": rng.normal(0, 1, n),
        "flipped": np.where(lab_up, -1.0, 1.0) + rng.normal(0, 0.3, n),
    }, index=idx)
    y = pd.Series(np.where(lab_up, "up", "down"), index=idx)
    return X, y, idx


def test_directional_auc_ranks_informative_above_noise():
    X, y, _ = _xy()
    r = directional_auc(X, y, min_n=10)
    assert r.loc["informative", "auc"] > 0.9      # 강한 up 신호
    assert r.loc["flipped", "auc"] < 0.1          # 강한 down 신호(0.5 아래)
    assert abs(r.loc["noise", "auc"] - 0.5) < 0.1  # 방향맹
    # 정렬 = |auc-0.5| 내림차순이라 noise 가 꼴찌
    assert r.index[-1] == "noise"


def test_directional_auc_ignores_expire_bars():
    """expire 는 방향 정답이 없으므로 표본에서 빠져야 한다."""
    X, y, idx = _xy(n=200)
    y2 = y.copy()
    y2.iloc[:100] = "expire"
    r_all, r_res = directional_auc(X, y, min_n=10), directional_auc(X, y2, min_n=10)
    assert r_res.loc["informative", "n"] == 100      # 해소봉만
    assert r_all.loc["informative", "n"] == 200


def test_directional_auc_is_rank_invariant():
    """AUC 는 순위 기반 → 단조변환(정규화)에 불변. 원시 X 로 재도 정규화 후와 같다."""
    X, y, _ = _xy()
    z = (X - X.mean()) / X.std()
    a, b = directional_auc(X, y, min_n=10), directional_auc(z, y, min_n=10)
    for c in X.columns:
        assert a.loc[c, "auc"] == pytest.approx(b.loc[c, "auc"], abs=1e-9)


def test_directional_auc_min_n_guard():
    X, y, _ = _xy(n=40)
    r = directional_auc(X, y, min_n=100)
    assert r["auc"].isna().all()          # 표본 부족 → NaN (노이즈 방지)


def test_sign_stability_separates_stable_from_flipping():
    """부호 안정 피처와 세그먼트마다 뒤집히는 피처를 구분한다."""
    seg_auc = pd.DataFrame({
        "s1": [0.55, 0.62, 0.50], "s2": [0.53, 0.38, 0.50],
        "s3": [0.56, 0.64, 0.50], "s4": [0.54, 0.36, 0.50],
    }, index=pd.Index(["stable_weak", "flipping_strong", "flat"], name="feature"))
    st = sign_stability(seg_auc)
    assert st.loc["stable_weak", "pos_share"] == 1.0
    assert bool(st.loc["stable_weak", "stable"]) is True
    assert st.loc["flipping_strong", "pos_share"] == pytest.approx(0.5)
    assert bool(st.loc["flipping_strong", "stable"]) is False
    # 진폭은 flipping 이 더 크지만 안정성은 stable_weak 가 위 → 정렬 1위
    assert st.loc["flipping_strong", "sd_dev"] > st.loc["stable_weak", "sd_dev"]
    assert st.index[0] == "stable_weak"
    # ★fresh-eyes H-3★ 완전 방향맹(전 세그먼트 auc=0.50)은 pos_share=0.0 이 되어
    # 과거 구현에서 stable=True 로 분류됐다. mean_dev≠0 조건이 이를 막아야 한다.
    assert st.loc["flat", "pos_share"] == 0.0
    assert bool(st.loc["flat", "stable"]) is False


def test_sign_stability_requires_enough_segments():
    """세그먼트가 적으면 부호 일치가 우연일 수 있어 stable 로 세지 않는다."""
    seg = pd.DataFrame({"s1": [0.60], "s2": [0.58]},
                       index=pd.Index(["few_segments"], name="feature"))
    assert bool(sign_stability(seg).loc["few_segments", "stable"]) is False


def test_auc_by_segment_and_fold_segments_align():
    X, y, idx = _xy(n=600)
    sp = WalkForwardSplitter(train_min=200, val_size=100, label_horizon=5)
    segs = fold_segments(sp, X.index)
    assert segs.notna().all() and segs.index.isin(X.index).all()
    seg_auc = auc_by_segment(X, y, segs, min_n=10)
    assert list(seg_auc.index) == list(X.columns)
    assert seg_auc.shape[1] == segs.nunique()
    # informative 는 전 폴드에서 0.5 위 → 부호 안정
    st = sign_stability(seg_auc)
    assert st.loc["informative", "pos_share"] == 1.0


def test_conditional_distribution_signs_match_auc():
    X, y, _ = _xy()
    cd = conditional_distribution(X, y)
    assert cd.loc["informative", "mean_diff"] > 0      # up 일 때 값이 큼
    assert cd.loc["flipped", "mean_diff"] < 0
    assert abs(cd.loc["noise", "std_mean_diff"]) < 0.2
