"""라벨 분포 게이트 회귀 테스트 (층1 학습가능성 · 층3 국면일관성).

통제된 합성 라벨 분포로 게이트 판정을 검증한다 — **성과는 등장하지 않는다**(설계 접근 B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.labeling.distribution import (
    DistributionThresholds,
    evaluate_layer1,
    evaluate_layer3,
)
from src.research.validation.splitter import WalkForwardSplitter


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")


def _labels(seq, n):
    """길이 n 라벨 Series (seq 를 반복). seq 원소 None → NaN."""
    idx = _idx(n)
    vals = [seq[i % len(seq)] for i in range(n)]
    return pd.Series(vals, index=idx, dtype=object)


SP = WalkForwardSplitter(train_min=500, val_size=300, label_horizon=10)


def test_layer1_passes_balanced():
    labels = _labels(["up", "down", "expire"], 3000)
    r = evaluate_layer1(labels, SP)
    assert r.passed
    assert r.reasons == []
    assert abs(r.expire_fraction - 1 / 3) < 0.02
    assert r.worst_fold_minority > 100


def test_layer1_fails_expire_extreme():
    # 92% 만료, 나머지 up/down 교대
    seq = ["expire"] * 23 + ["up", "down"]
    r = evaluate_layer1(_labels(seq, 3000), SP)
    assert not r.passed
    assert any("만료 극단" in s for s in r.reasons)


def test_layer1_fails_directional_extreme():
    # 방향 96% (up/down 교대), 만료 4%
    seq = ["up", "down"] * 12 + ["expire"]
    r = evaluate_layer1(_labels(seq, 3000), SP)
    assert not r.passed
    assert any("방향 극단" in s for s in r.reasons)


def test_layer1_fails_tiny_minority():
    # up/expire 반복 + down 은 아주 드물게 → 학습창 소수클래스(down) 부족
    n = 3000
    labels = _labels(["up", "expire"], n)
    # down 을 50개만 흩뿌림
    rng = np.random.default_rng(0)
    down_pos = rng.choice(n, 50, replace=False)
    labels.iloc[down_pos] = "down"
    r = evaluate_layer1(labels, SP)
    assert not r.passed
    assert any("소수클래스" in s for s in r.reasons)
    assert r.worst_fold_minority < 100


def test_layer1_tail_nan_ignored():
    # 꼬리 NaN(미해소)이 있어도 해소분으로만 판정
    labels = _labels(["up", "down", "expire"], 3000)
    labels.iloc[-10:] = np.nan
    r = evaluate_layer1(labels, SP)
    assert r.passed


def test_layer3_consistent_when_balanced_across_regimes():
    n = 3000
    labels = _labels(["up", "down", "expire"], n)
    # 3개 국면 균등 분할, 각 국면 안에서도 균형
    reg = pd.Series(
        np.array(["bull", "bear", "flat"])[(np.arange(n) // 1000) % 3],
        index=labels.index,
    )
    r = evaluate_layer3(labels, reg)
    assert r.consistent
    assert set(r.evaluated_regimes) == {"bull", "bear", "flat"}
    assert r.violating_regimes == []


def test_layer3_inconsistent_when_one_regime_extreme():
    n = 3000
    labels = _labels(["up", "down", "expire"], n)
    reg = pd.Series(
        np.where(np.arange(n) < 1000, "bull",
                 np.where(np.arange(n) < 2000, "bear", "flat")),
        index=labels.index,
    )
    # bear 국면을 전부 만료로 → 극단 → 위배
    labels[reg == "bear"] = "expire"
    r = evaluate_layer3(labels, reg)
    assert not r.consistent
    assert "bear" in r.violating_regimes


def test_layer3_skips_small_regime():
    n = 3000
    labels = _labels(["up", "down", "expire"], n)
    reg = pd.Series("big", index=labels.index)
    reg.iloc[:50] = "tiny"          # 50봉 < min_regime_samples(200)
    r = evaluate_layer3(labels, reg, DistributionThresholds(min_regime_samples=200))
    assert "tiny" in r.skipped_regimes
    assert "big" in r.evaluated_regimes


def test_layer3_regime_nan_excluded():
    n = 3000
    labels = _labels(["up", "down", "expire"], n)
    reg = pd.Series("bull", index=labels.index)
    reg.iloc[:500] = np.nan          # 국면 워밍업
    r = evaluate_layer3(labels, reg)
    # NaN 국면은 집계에서 빠짐 (bull 만 판정)
    assert list(r.per_regime.index) == ["bull"]
    assert r.per_regime.loc["bull", "n"] == 2500
