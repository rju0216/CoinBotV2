"""레짐 학습 파이프라인 테스트 (S4) — make_anchored_windows + walk_forward + 빌드 스크립트.

build_model/EM/mapping 자체는 D4-2 테스트(test_regime_*)가 커버. 여기는 walk-forward
윈도우 생성과 tz-aware 캔들에서의 train slice 정합 + 빌드 스크립트 CLI smoke.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from src.strategy.regime.contract import RegimeType
from src.strategy.regime.training import make_anchored_windows, walk_forward
from tests.test_regime_quant_plugin import _synth_candles


# ---- make_anchored_windows ----

def test_windows_count_and_continuity():
    w = make_anchored_windows("2020-01-01", "2020-07-01", 3)
    assert len(w) == 2  # 3개월씩 → [01-01,04-01), [04-01,07-01)
    # valid 블록 연속: 앞 블록 end == 뒤 블록 start
    assert w[0][2] == w[1][1]


def test_windows_train_end_is_one_bar_before_valid_start():
    w = make_anchored_windows("2020-01-01", "2020-04-01", 3)
    train_end, valid_start, valid_end = w[0]
    # train_end = valid_start − 1h (경계 누수 차단)
    assert pd.Timestamp(train_end) == pd.Timestamp(valid_start) - pd.Timedelta(hours=1)
    assert valid_start.startswith("2020-01-01 00:00:00")
    assert valid_end.startswith("2020-04-01 00:00:00")


def test_windows_are_tz_aware():
    w = make_anchored_windows("2020-01-01", "2020-04-01", 3)
    for train_end, valid_start, valid_end in w:
        assert "+00:00" in valid_start  # candles(tz-aware UTC) 와 비교 가능
        assert "+00:00" in train_end


def test_windows_last_block_clipped_to_valid_end():
    # 2개월 retrain, 5개월 구간 → 마지막 블록이 valid_end 로 잘림
    w = make_anchored_windows("2020-01-01", "2020-06-01", 2)
    assert w[-1][2].startswith("2020-06-01")


def test_windows_reject_nonpositive_retrain():
    import pytest
    with pytest.raises(ValueError):
        make_anchored_windows("2020-01-01", "2020-06-01", 0)


def test_windows_empty_when_start_equals_end():
    assert make_anchored_windows("2020-01-01", "2020-01-01", 3) == []


def test_windows_empty_when_reversed():
    assert make_anchored_windows("2020-06-01", "2020-01-01", 3) == []


# ---- walk_forward (tz-aware candles 정합) ----

def test_walk_forward_with_anchored_windows_tz_aware():
    """tz-aware UTC 캔들 + anchored 윈도우 → train slice tz 에러 없이 학습."""
    candles = _synth_candles(400)  # 2020-01-01~ 1h, tz-aware UTC
    windows = make_anchored_windows("2020-01-08", "2020-01-15", 1)
    models = walk_forward(
        candles, windows, k=2, tau=0.5, n_init=1, n_iter=5, seed=0
    )
    assert len(models) == len(windows) >= 1
    m = models[0]
    assert m.valid_period[0].startswith("2020-01-08")
    assert m.hmm.emission.kind == "gaussian"


def test_walk_forward_train_excludes_valid_start_bar():
    """누수 차단을 캔들 레벨로 확인: train 마지막 봉 < valid_start (valid_start 봉 비포함).

    또한 train 이 anchored(데이터 시작부터)라 비어있지 않음을 train_start 로 확인.
    """
    candles = _synth_candles(400)  # 2020-01-01~ 1h
    windows = make_anchored_windows("2020-01-08", "2020-01-15", 1)
    models = walk_forward(
        candles, windows, k=2, tau=0.5, n_init=1, n_iter=5, seed=0
    )
    m = models[0]
    # train 마지막 봉(meta.train_end) < valid_start → valid_start 봉이 train 에 안 샘
    assert pd.Timestamp(m.meta["train_end"]) < pd.Timestamp(m.valid_period[0])
    # train 시작 = 데이터 시작 (anchored, 비공백)
    assert pd.Timestamp(m.meta["train_start"]) == candles.index[0]


# ---- enable_range 배선 (gen1 추세-단독, O-7 (가)) ----

def test_walk_forward_enable_range_false_no_range_type():
    """enable_range=False → 매핑에 RANGE 타입 전무(비trend 전부 NONE). 추세-단독."""
    candles = _synth_candles(400)
    windows = make_anchored_windows("2020-01-08", "2020-01-15", 1)
    models = walk_forward(
        candles, windows, k=2, tau=0.5, n_init=1, n_iter=5, seed=0,
        enable_range=False,
    )
    for m in models:
        assert RegimeType.RANGE not in m.mapping.types  # 비trend → NONE
        assert all(t in (RegimeType.TREND, RegimeType.NONE) for t in m.mapping.types)
        assert m.meta["enable_range"] is False


# ---- emission 배선 (S2, D4-4) ----

def test_walk_forward_default_gaussian_backward_compat():
    """emission 미지정 → gaussian 유지 (하위호환)."""
    candles = _synth_candles(400)
    windows = make_anchored_windows("2020-01-08", "2020-01-15", 1)
    models = walk_forward(
        candles, windows, k=2, tau=0.5, n_init=1, n_iter=5, seed=0
    )
    assert all(m.hmm.emission.kind == "gaussian" for m in models)


def test_walk_forward_student_t_emission():
    """emission_kind='student_t' → t emission 으로 학습 (nus 존재·share_nu 전달)."""
    candles = _synth_candles(400)
    windows = make_anchored_windows("2020-01-08", "2020-01-15", 1)
    models = walk_forward(
        candles, windows, k=2, tau=0.5, n_init=1, n_iter=5, seed=0,
        emission_kind="student_t",
        emission_params={"share_nu": True, "nu_init": 8.0},
    )
    for m in models:
        em = m.hmm.emission
        assert em.kind == "student_t"
        assert em.nus.shape == (2,) and np.isfinite(em.nus).all()
        assert em.share_nu is True  # emission_params 전달 확인
        assert m.meta["emission_kind"] == "student_t"  # meta 자동 배선


# ---- 빌드 스크립트 CLI smoke ----

def test_build_script_cli_help():
    r = subprocess.run(
        [sys.executable, "scripts/build_regime_artifacts.py", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "walk-forward" in r.stdout
    assert "--valid-start" in r.stdout
    assert "--no-range" in r.stdout  # gen1 추세-단독 플래그
    assert "--emission" in r.stdout  # D4-4 emission 선택
    assert "--share-nu" in r.stdout
