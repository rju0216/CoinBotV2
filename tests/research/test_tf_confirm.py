"""R4.1 15m 확인 러너 테스트 — 사전등록 판정규칙 단위 + seam 통합 (규칙16).

- 단위: classify_regime_consistency(축①)·classify_label_robustness(축②)·combine 합산.
- seam: 소슬라이스 실 15m 전스택 관통(load→build→label→splitter→walk_forward→per_regime→
  킬규칙) — per_regime 국면정렬·NaN 비전파·splitter 지평정합 계약 실증(데이터 부재 skip).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import csv_filename, load_audited
from src.research.experiments.tf_confirm import (
    MIN_REGIME_SAMPLES,
    classify_label_robustness,
    classify_regime_consistency,
    combine,
    run_15m_confirm,
)
from src.research.experiments.tf_expansion import TFGate1Result, run_tf_gate1
from src.research.labeling.triple_barrier import LabelParams

SYM = "BTC/USDT:USDT"
CANDLE_DIR = "data/candles"
requires_15m = pytest.mark.skipif(
    not os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, "15m"))),
    reason="BTC 15m 캔들 없음",
)
GLOBAL_MARGIN = 0.045          # 15m 전역 ll-margin (국소붕괴 임계)
BIG = MIN_REGIME_SAMPLES + 1   # 표본충분
SMALL = MIN_REGIME_SAMPLES - 1  # 소표본(판정 제외)


def _pr(rows: dict) -> pd.DataFrame:
    """{regime: (n, margin, reversal_seed_frac)} → per_regime 프레임. prior/mlp_ll 는 마진 정합."""
    data = {}
    for reg, (n, margin, rev) in rows.items():
        prior_ll = 1.09
        data[reg] = {"n": n, "prior_ll": prior_ll, "mlp_ll_median": prior_ll - margin,
                     "margin": margin, "reversal_seed_frac": rev}
    return pd.DataFrame.from_dict(data, orient="index")


def _gate(verdict: str, ll_margin: float) -> TFGate1Result:
    return TFGate1Result(tf="15m", verdict=verdict, reason="", evals={},
                         prior_ll=1.09, ll_margin=ll_margin)


# ---- 축① 국면일관성 ----

def test_regime_confirm_all_positive():
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.05, 0.0), "bear_low": (BIG, 0.03, 0.2),
             "range_mid": (BIG, 0.01, 0.4)}), GLOBAL_MARGIN)
    assert v.verdict == "CONFIRM"
    assert v.genuine_reversals == []


def test_regime_noise_negative_tolerated():
    # margin<0 이나 seed 과반 아님(0.4) → 진짜 반전 아님 → CONFIRM
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.05, 0.0), "range_mid": (BIG, -0.002, 0.4)}), GLOBAL_MARGIN)
    assert v.verdict == "CONFIRM"


def test_regime_flag_single_genuine_reversal():
    # 진짜 반전 1개(seed 0.8), |margin| < global → FLAG (자동킬 아님)
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.05, 0.0), "bear_low": (BIG, -0.01, 0.8)}), GLOBAL_MARGIN)
    assert v.verdict == "FLAG"
    assert v.genuine_reversals == ["bear_low"]


def test_regime_kill_two_reversals():
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.05, 0.0), "bear_low": (BIG, -0.01, 0.8),
             "range_mid": (BIG, -0.008, 0.6)}), GLOBAL_MARGIN)
    assert v.verdict == "KILL"
    assert len(v.genuine_reversals) == 2


def test_regime_kill_local_collapse():
    # 반전 1개이나 |margin|(0.05) ≥ global(0.045) → 국소붕괴 KILL
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.04, 0.0), "bear_low": (BIG, -0.05, 0.8)}), GLOBAL_MARGIN)
    assert v.verdict == "KILL"


def test_regime_small_sample_excluded():
    # 소표본 국면의 반전은 판정 제외 → CONFIRM
    v = classify_regime_consistency(
        _pr({"bull_high": (BIG, 0.05, 0.0), "tiny": (SMALL, -0.05, 1.0)}), GLOBAL_MARGIN)
    assert v.verdict == "CONFIRM"
    assert "tiny" not in v.evaluated


# ---- 축② 라벨강건성 ----

def test_label_robust_all_hold():
    v = classify_label_robustness(
        [("6h", _gate("PASS", 0.045)), ("12h", _gate("PASS", 0.03)),
         ("1d", _gate("PASS", 0.02))], require_all=True)
    assert v.verdict == "ROBUST"
    assert len(v.holds) == 3


def test_label_fragile_minority():
    v = classify_label_robustness(
        [("6h", _gate("PASS", 0.045)), ("12h", _gate("HOLD", -0.01)),
         ("1d", _gate("DISCARD", -0.02))], require_all=True)
    assert v.verdict == "FRAGILE"


def test_label_partial_majority_strict():
    # 2/3 유지 (엄격기준) → PARTIAL
    v = classify_label_robustness(
        [("6h", _gate("PASS", 0.045)), ("12h", _gate("PASS", 0.02)),
         ("1d", _gate("HOLD", -0.01))], require_all=True)
    assert v.verdict == "PARTIAL"


def test_label_zero_margin_not_hold():
    # ll_margin>0 엄격 — 0 은 miss
    v = classify_label_robustness([("6h", _gate("PASS", 0.0))], require_all=True)
    assert v.verdict == "FRAGILE"


# ---- 합산 ----

def test_combine_reject_on_kill():
    reg = classify_regime_consistency(
        _pr({"a": (BIG, -0.05, 0.9), "b": (BIG, -0.02, 0.8)}), GLOBAL_MARGIN)
    lab = classify_label_robustness([("6h", _gate("PASS", 0.045))], True)
    assert combine(reg, lab)[0] == "REJECT"


def test_combine_confirm_when_both_best():
    reg = classify_regime_consistency(_pr({"a": (BIG, 0.05, 0.0)}), GLOBAL_MARGIN)
    lab = classify_label_robustness([("6h", _gate("PASS", 0.045))], True)
    assert combine(reg, lab)[0] == "CONFIRM"


def test_combine_hold_on_flag():
    reg = classify_regime_consistency(
        _pr({"a": (BIG, 0.05, 0.0), "b": (BIG, -0.01, 0.8)}), GLOBAL_MARGIN)   # FLAG
    lab = classify_label_robustness([("6h", _gate("PASS", 0.045))], True)       # ROBUST
    assert combine(reg, lab)[0] == "HOLD"


def test_combine_reject_on_fragile():
    reg = classify_regime_consistency(_pr({"a": (BIG, 0.05, 0.0)}), GLOBAL_MARGIN)  # CONFIRM
    lab = classify_label_robustness(
        [("6h", _gate("PASS", 0.045)), ("12h", _gate("HOLD", -0.01)),
         ("1d", _gate("DISCARD", -0.02))], True)                                    # FRAGILE
    assert combine(reg, lab)[0] == "REJECT"


# ---- seam 통합 (실 15m 소슬라이스, 규칙16) ----

@requires_15m
def test_seam_per_regime_alignment_real_15m():
    """소슬라이스 15m 전스택 관통 → per_regime 계약(국면정렬·NaN비전파) 실증.

    ~42k봉·2seed·2폴드로 경량화. 무거운 전체 실행(226k·5seed)은 승인 후 별도.
    """
    df = load_tf_slice(45000)
    split = dict(train_min=25000, val_size=8000)
    r = run_tf_gate1("15m", label=LabelParams("atr", 24, 3.0, 24), split=split,
                     df=df, mlp_seeds=(0, 1))
    assert r is not None and r.per_regime is not None

    pr = r.per_regime
    # 계약2: Prior 국면마다 MLP 중앙값 존재(정렬) — 오정렬이면 reindex NaN
    assert set(pr.columns) == {"n", "prior_ll", "mlp_ll_median", "margin", "reversal_seed_frac"}
    assert pr["mlp_ll_median"].notna().all(), "MLP per_regime 이 Prior 국면과 오정렬(NaN)"
    # 계약4: 마진 NaN 비전파
    assert pr["margin"].notna().all()
    # 계약(F-8): regime_tags 모델-TF 정렬 → 귀속봉 존재(원시 상위TF 였다면 대부분 드롭)
    assert pr["n"].sum() > 0 and (pr["n"] >= MIN_REGIME_SAMPLES).any()
    # 킬규칙이 실 per_regime 에서 유효 판정 산출
    v = classify_regime_consistency(pr, r.ll_margin)
    assert v.verdict in {"CONFIRM", "FLAG", "KILL"}


@requires_15m
def test_seam_alternate_label_horizon_real_15m():
    """계약3: 다른 지평 대체라벨(N48)에서 splitter purge 정합 — 전스택 완주·per_regime 정상."""
    df = load_tf_slice(45000)
    split = dict(train_min=25000, val_size=8000)
    r = run_tf_gate1("15m", label=LabelParams("atr", 48, 3.0, 48), split=split,
                     df=df, mlp_seeds=(0,))
    assert r is not None and r.per_regime is not None
    assert r.per_regime["margin"].notna().all()


def load_tf_slice(n: int) -> pd.DataFrame:
    """15m 앞 n봉 (seam 경량 구동)."""
    return load_audited(SYM, "15m", CANDLE_DIR).iloc[:n]
