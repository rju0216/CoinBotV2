"""R4.2 MTF 역할1 테스트 — build_mtf_features 단위 + classify_mtf + seam 통합(규칙16).

- 단위: 열 네이밍·index·ff 위임(인과)·warmup NaN; classify_mtf 사전등록 규칙.
- seam: 실 15m+1h+4h 소슬라이스 MTF full-stack 관통 — F-4 종료일정렬·F-9 ff가드·
  교차TF 무lookahead(완성봉 위임)·per_regime 정렬(데이터 부재 skip).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import csv_filename, forward_fill_completed, load_audited
from src.research.experiments.tf_expansion import TFGate1Result, run_tf_gate1
from src.research.experiments.tf_mtf import (
    MTFConfigResult,
    NOISE_FLOOR,
    classify_mtf,
)
from src.research.features.build import build_features
from src.research.features.mtf import build_mtf_features
from src.research.labeling.triple_barrier import LabelParams

SYM = "BTC/USDT:USDT"
CANDLE_DIR = "data/candles"


def _has(tf):
    return os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, tf)))


requires_mtf = pytest.mark.skipif(
    not (_has("15m") and _has("1h") and _has("4h")),
    reason="15m/1h/4h 캔들 없음",
)


def _synth_ohlcv(n, freq, start="2021-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": np.abs(rng.normal(100, 10, n))},
                        index=idx).rename_axis("timestamp")


# ---- build_mtf_features 단위 ----

def test_mtf_columns_and_index():
    dec = _synth_ohlcv(400, "15min")
    h1 = _synth_ohlcv(120, "1h")
    X = build_mtf_features(dec, [("1h", h1)])
    base = build_features(dec)
    # index 보존
    assert X.index.equals(dec.index)
    # 결정TF 기본열 그대로 앞에 + 상위TF 접두어열
    assert list(X.columns[: base.shape[1]]) == list(base.columns)
    higher_cols = [c for c in X.columns if c.startswith("mtf1h_")]
    assert len(higher_cols) == base.shape[1]
    assert X.shape[1] == base.shape[1] * 2


def test_mtf_delegates_to_forward_fill_completed():
    # 상위TF 열이 forward_fill_completed(완성봉 인과) 위임과 정확 일치 (누수 차단 계약)
    dec = _synth_ohlcv(400, "15min")
    h1 = _synth_ohlcv(120, "1h")
    X = build_mtf_features(dec, [("1h", h1)])
    expected = forward_fill_completed(build_features(h1).add_prefix("mtf1h_"), dec.index, "1h")
    pd.testing.assert_frame_equal(X[list(expected.columns)], expected)


def test_mtf_guards_higher_ending_early():
    # F-4(fresh-eyes MEDIUM①): 상위TF가 결정TF tail 을 못 덮으면 stale-fill 대신 하드페일
    dec = _synth_ohlcv(400, "15min", start="2021-01-01")   # 100h 스팬
    h1_short = _synth_ohlcv(40, "1h", start="2021-01-01")   # 40h << 100h → tail 미달
    with pytest.raises(ValueError, match="stale"):
        build_mtf_features(dec, [("1h", h1_short)])
    # 경계: 완성봉이 tail 을 덮으면 통과
    h1_ok = _synth_ohlcv(120, "1h", start="2021-01-01")     # 120h > 100h
    build_mtf_features(dec, [("1h", h1_ok)])                 # raise 없음


def test_mtf_multi_higher_distinct_prefixes():
    dec = _synth_ohlcv(400, "15min")
    X = build_mtf_features(dec, [("1h", _synth_ohlcv(120, "1h")),
                                 ("4h", _synth_ohlcv(60, "4h"))])
    assert any(c.startswith("mtf1h_") for c in X.columns)
    assert any(c.startswith("mtf4h_") for c in X.columns)
    base_n = build_features(dec).shape[1]
    assert X.shape[1] == base_n * 3


# ---- classify_mtf 사전등록 규칙 ----

def _cfg(name, ll_margin, delta, material, regime="CONFIRM"):
    r = TFGate1Result("15m", "PASS", "", {}, 1.09, ll_margin)
    return MTFConfigResult(name, r, delta, material, regime if material else "-")


def test_classify_adopt_strongest_material():
    base = 0.045
    res = classify_mtf(base, [
        _cfg("1h", 0.045 + NOISE_FLOOR + 0.001, NOISE_FLOOR + 0.001, True),
        _cfg("4h", 0.045 + 0.0005, 0.0005, False),           # 노이즈 미만
        _cfg("1h+4h", 0.045 + NOISE_FLOOR + 0.005, NOISE_FLOOR + 0.005, True),
    ])
    assert res.verdict == "ADOPT_MTF"
    assert res.adopted == "1h+4h"          # material 중 ll_margin 최강


def test_classify_no_gain_all_below_floor():
    base = 0.045
    res = classify_mtf(base, [
        _cfg("1h", 0.045 + 0.001, 0.001, False),
        _cfg("4h", 0.045 - 0.002, -0.002, False),
    ])
    assert res.verdict == "NO_GAIN"
    assert res.adopted is None


def test_classify_material_but_regime_kill_excluded():
    base = 0.045
    res = classify_mtf(base, [
        _cfg("1h", 0.045 + NOISE_FLOOR + 0.01, NOISE_FLOOR + 0.01, True, regime="KILL"),
    ])
    assert res.verdict == "NO_GAIN"        # 국면붕괴 → 채택 제외


# ---- seam 통합 (실 15m+1h+4h 소슬라이스) ----

@requires_mtf
def test_seam_mtf_fullstack_real():
    """MTF full-stack 관통: 열정렬·ff위임(무lookahead)·per_regime 정렬·NaN 비전파."""
    dec = load_audited(SYM, "15m", CANDLE_DIR).iloc[:45000]
    h1 = load_audited(SYM, "1h", CANDLE_DIR)
    h4 = load_audited(SYM, "4h", CANDLE_DIR)
    X = build_mtf_features(dec, [("1h", h1), ("4h", h4)])
    base_n = build_features(dec).shape[1]
    assert X.shape[1] == base_n * 3 and X.index.equals(dec.index)

    # F-4: 상위TF 종료일 ≥ 15m → 슬라이스 tail 상위열 미커버 없음(내부 유한)
    tail = X.iloc[-1]
    assert tail.notna().all(), "tail 상위TF 열 NaN — 종료일 커버리지 손실(F-4)"

    # 위임 인과 = forward_fill_completed 정확 일치 (교차TF 무lookahead)
    exp1 = forward_fill_completed(build_features(h1).add_prefix("mtf1h_"), dec.index, "1h")
    pd.testing.assert_frame_equal(X[list(exp1.columns)], exp1)

    # gate1 관통 (소split·1seed) → per_regime 정렬·NaN 비전파
    r = run_tf_gate1("15m", label=LabelParams("atr", 24, 3.0, 24),
                     split=dict(train_min=25000, val_size=8000), df=dec, X=X, mlp_seeds=(0,))
    assert r is not None and r.per_regime is not None
    assert r.per_regime["margin"].notna().all()
