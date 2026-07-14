"""Phase 1 통짜(whole-Phase) full-stack 통합 + 층3↔regime ff seam (규칙 16·17).

- **seam 구멍 닫기**: `tag_regimes(1d)` → `forward_fill_completed(→모델 TF)` →
  `evaluate_layer3(labels)` 접합부(F-8 모델 TF 정렬·완성봉 인과·NaN 전파)를 커밋
  회귀로 박제. (Step 1.4 단위테스트는 합성 regime 을 직접 넣어 이 접합부를 못 봄.)
- **통짜(규칙 17)**: load_audited → compute_triple_barrier → splitter + regime ff →
  evaluate_layer1/layer3 를 한 흐름으로. **확정 R0 선택(atr/w96/x3.0/N24)이 실제
  층1∧층3 통과**함을 박제 — 향후 변경이 선택을 깨면 잡힘.
- **F-8 음성대조**: 원시 1d 태그를 (ff 없이) 모델 TF 에 주면 대부분 드롭 → 정렬 필요성 실증.

실데이터 의존 테스트는 skip-guard.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import (
    csv_filename,
    forward_fill_completed,
    load_audited,
    load_ohlcv,
)
from src.research.labeling.distribution import evaluate_layer1, evaluate_layer3
from src.research.labeling.triple_barrier import LabelParams, compute_triple_barrier
from src.research.validation.regime import (
    TREND_COL,
    RegimeParams,
    tag_regimes,
)
from src.research.validation.splitter import WalkForwardSplitter

SYM = "BTC/USDT:USDT"
CANDLE_DIR = "data/candles"
requires_1h = pytest.mark.skipif(
    not os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, "1h"))),
    reason="BTC 1h 캔들 없음",
)
requires_1d = pytest.mark.skipif(
    not os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, "1d"))),
    reason="BTC 1d 캔들 없음",
)

# 확정 R0 선택 (2026-07-11): N=24 1일 지평, ATR w96 x3.0 (3-way 균형 31/31/38).
CHOSEN = LabelParams("atr", 96, 3.0, 24)


def _synth_ohlc(n, start, freq, seed, scale=0.02):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    # 추세 스윙(사인) + 노이즈 → 국면 변동 생성
    t = np.arange(n)
    drift = 0.4 * np.sin(t / (n / 6.0))
    ret = rng.normal(0, scale, n) + np.diff(drift, prepend=drift[0])
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1, 100, n)},
        index=idx,
    )


# ---- seam: tag_regimes(1d) → forward_fill_completed → evaluate_layer3 (합성 결정적) ----

def test_regime_ff_layer3_seam_synthetic():
    # 1d 국면 (변동 있는 스윙), 작은 파라미터로 워밍업 단축
    d1 = _synth_ohlc(220, "2020-01-01", "D", seed=1, scale=0.01)
    P = RegimeParams(ma_len=10, slope_k=3, deadband=0.001, vol_len=5,
                     vol_hi_pct=0.5, min_duration=2)
    trend = tag_regimes(d1, P)[TREND_COL]
    assert trend.notna().any() and trend.dropna().nunique() >= 2   # 국면 변동 존재

    # 1h 라벨 (1d 태깅 구간 내)
    h1 = _synth_ohlc(3000, "2020-02-01", "h", seed=2)
    bl = compute_triple_barrier(h1, LabelParams("atr", 48, 2.0, 24))

    # 완성봉 ff 로 모델 TF(1h) 정렬 (F-8 계약)
    reg = forward_fill_completed(trend, h1.index, "1d")
    assert reg.index.equals(h1.index)           # 모델 TF 정렬
    assert reg.notna().mean() > 0.8             # 워밍업 후 대부분 태깅됨(살아있는 ff)

    r3 = evaluate_layer3(bl.labels, reg)
    # 접합 성립: 실제 국면 슬라이스 생성, trend 상태만 등장, NaN(워밍업)은 제외
    assert r3.evaluated_regimes                  # 비어있지 않음(vacuous 아님)
    assert set(r3.per_regime.index) <= {"up", "flat", "down"}
    # per_regime n 합 ≤ 태깅된 해소 라벨 수 (NaN 국면·NaN 라벨 제외)
    resolved_tagged = (bl.labels.notna() & reg.notna()).sum()
    assert r3.per_regime["n"].sum() <= resolved_tagged


def test_raw_1d_tags_mostly_drop_without_ff_negative_control():
    """F-8 음성대조: ff 없이 원시 1d 태그를 1h 에 주면 경계 봉만 매칭 → 대부분 드롭."""
    d1 = _synth_ohlc(220, "2020-01-01", "D", seed=1, scale=0.01)
    P = RegimeParams(ma_len=10, slope_k=3, deadband=0.001, vol_len=5,
                     vol_hi_pct=0.5, min_duration=2)
    trend = tag_regimes(d1, P)[TREND_COL]
    h1 = _synth_ohlc(3000, "2020-02-01", "h", seed=2)
    bl = compute_triple_barrier(h1, LabelParams("atr", 48, 2.0, 24))

    reg_ff = forward_fill_completed(trend, h1.index, "1d")
    reg_raw = trend.reindex(h1.index)             # 원시(자정 봉만 일치)

    # ff 는 촘촘, 원시는 하루 24봉 중 1봉만 → 커버리지 급감
    assert reg_raw.notna().sum() < reg_ff.notna().sum() / 10
    # 원시로 층3 하면 표본 급감(대부분 drop) — 모델 TF 정렬이 필수임을 실증
    n_ff = evaluate_layer3(bl.labels, reg_ff).per_regime["n"].sum()
    n_raw = evaluate_layer3(bl.labels, reg_raw).per_regime["n"].sum()
    assert n_raw < n_ff / 10


# ---- 통짜 full-stack (실 1h/1d, 확정 선택 박제) ----

@requires_1h
@requires_1d
def test_full_stack_chosen_label_passes_gates():
    """load_audited → 삼중배리어(확정 config) → splitter + regime ff → 층1∧층3.

    확정 R0 선택이 실제로 게이트를 통과함을 회귀로 박제.
    """
    df = load_audited(SYM, "1h")
    # regime: 1d(I-001 꼬리 인지 → load_ohlcv) trend → 완성봉 ff → 1h
    trend_1d = tag_regimes(load_ohlcv(SYM, "1d"))[TREND_COL]
    reg = forward_fill_completed(trend_1d, df.index, "1d")

    bl = compute_triple_barrier(df, CHOSEN)
    assert bl.labels.index.equals(df.index)           # X.index 정렬
    assert bl.label_horizon == 24                       # F-5 N 단일출처

    # T·V·S 확정값 (F-5): train_min=8760(1y), val_size=2160(3mo), N=24
    sp = WalkForwardSplitter(train_min=8760, val_size=2160, label_horizon=24)

    l1 = evaluate_layer1(bl.labels, sp)
    assert l1.passed, f"확정 라벨 층1 실패: {l1.reasons}"
    assert l1.worst_fold_minority > 100

    l3 = evaluate_layer3(bl.labels, reg)
    assert l3.consistent, f"확정 라벨 층3 실패: {l3.violating_regimes}"
    assert set(l3.evaluated_regimes) == {"up", "flat", "down"}
    # 3-way 균형 확인(회귀): expire 대략 0.38, 방향 대칭
    fr = l1.global_fractions
    assert 0.30 < fr["expire"] < 0.45
    assert abs(fr["up"] - fr["down"]) < 0.05


@requires_1h
@requires_1d
def test_full_stack_folds_and_purge():
    """확정 config 로 실제 폴드 생성·purge 정합 (통짜 배관)."""
    df = load_audited(SYM, "1h")
    bl = compute_triple_barrier(df, CHOSEN)
    sp = WalkForwardSplitter(8760, 2160, label_horizon=bl.label_horizon)
    folds = list(sp.split(bl.labels.index))
    assert len(folds) >= 20
    pos = pd.Index(df.index)
    for f in folds:
        last_train = pos.get_indexer([f.train_index[-1]])[0]
        first_val = pos.get_indexer([f.val_index[0]])[0]
        assert last_train + bl.label_horizon < first_val   # 라벨창 val 무침범
