"""라벨 ↔ splitter seam 통합 (규칙 16).

Phase 1 산출(삼중배리어 라벨)이 Phase 0 splitter 와 공유 계약대로 맞물리는지:
- **N 단일출처(F-5)**: 라벨 지평 N=``bl.label_horizon`` 이 splitter purge·꼬리예약을
  구동한다. train 표본의 라벨창 [i, i+N] 이 val 을 침범하지 않는다.
- **꼬리예약**: 미해소(NaN) 꼬리 N봉은 절대 val 로 쓰이지 않는다(usable_end=n-N 정합).
- **음성대조**: N 을 라벨보다 작게 주면 라벨창이 val 을 침범 → N 단일출처의 필요성 실증.
- **인덱스 정렬 계약**: 라벨 = X.index 정렬 클래스 Series (substrate §2).

주의: harness(모델 fit) 로의 seam(라벨 NaN 드롭)은 **Phase 3** 관심사(F-10) — R0 는
분포만 보고 모델을 학습하지 않으므로 여기서 다루지 않는다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.research.data.loader import csv_filename, load_audited
from src.research.labeling.triple_barrier import LabelParams, compute_triple_barrier
from src.research.validation.splitter import WalkForwardSplitter

CANDLE_DIR = "data/candles"
SYM = "BTC/USDT:USDT"
requires_1h = pytest.mark.skipif(
    not os.path.exists(os.path.join(CANDLE_DIR, csv_filename(SYM, "1h"))),
    reason="BTC 1h 캔들 없음",
)


def _synth_ohlc(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, 0.02, n)
    ret[rng.integers(0, n, n // 25)] += rng.normal(0, 0.1, n // 25)
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1, 100, n)},
        index=idx,
    )


def _purge_holds(idx, folds, N):
    """train 마지막 봉의 라벨창 [i, i+N] 이 val 시작 앞에서 끝나는지."""
    pos = pd.Index(idx)
    for f in folds:
        last_train = pos.get_indexer([f.train_index[-1]])[0]
        first_val = pos.get_indexer([f.val_index[0]])[0]
        if last_train + N >= first_val:
            return False
    return True


def test_label_horizon_drives_splitter_purge():
    df = _synth_ohlc()
    params = LabelParams("atr", 48, 2.0, 24)
    bl = compute_triple_barrier(df, params)

    # 라벨 = X.index 정렬 클래스 Series (substrate §2)
    assert bl.labels.index.equals(df.index)

    # 라벨의 N 을 splitter 에 단일출처로 공급
    sp = WalkForwardSplitter(train_min=300, val_size=150, label_horizon=bl.label_horizon)
    folds = list(sp.split(bl.labels.index))
    assert len(folds) >= 3
    assert all(f.n_purged == bl.label_horizon for f in folds)
    assert _purge_holds(df.index, folds, bl.label_horizon)


def test_tail_unresolved_never_in_validation():
    df = _synth_ohlc()
    N = 24
    bl = compute_triple_barrier(df, LabelParams("atr", 48, 2.0, N))
    sp = WalkForwardSplitter(train_min=300, val_size=150, label_horizon=N)
    folds = list(sp.split(bl.labels.index))

    n = len(df)
    tail_positions = set(range(n - N, n))          # 미해소 NaN 꼬리
    pos = pd.Index(df.index)
    for f in folds:
        val_pos = set(pos.get_indexer(f.val_index))
        assert val_pos.isdisjoint(tail_positions)   # 꼬리가 val 로 새지 않음
    # 그 꼬리 라벨은 실제로 전부 NaN
    assert bl.labels.iloc[n - N:].isna().all()


def test_wrong_N_leaks_negative_control():
    """N 을 라벨보다 작게 주면 라벨창이 val 을 침범 — N 단일출처가 필수임을 실증."""
    df = _synth_ohlc()
    N = 40
    bl = compute_triple_barrier(df, LabelParams("atr", 48, 2.0, N))

    # 올바른 N: purge 유지
    sp_ok = WalkForwardSplitter(300, 150, label_horizon=bl.label_horizon)
    assert _purge_holds(df.index, list(sp_ok.split(df.index)), N)

    # 잘못된(작은) N: 실제 라벨창 N 기준으로 침범
    sp_bad = WalkForwardSplitter(300, 150, label_horizon=N // 4)
    assert not _purge_holds(df.index, list(sp_bad.split(df.index)), N)


@requires_1h
def test_real_1h_labels_feed_splitter():
    """실 1h: load_audited → 삼중배리어 → splitter(N 단일출처). 실 T·V·S 규모."""
    df = load_audited(SYM, "1h")
    bl = compute_triple_barrier(df, LabelParams("atr", 48, 2.0, 24))
    assert bl.labels.index.equals(df.index)

    sp = WalkForwardSplitter(
        train_min=8760, val_size=2160, label_horizon=bl.label_horizon
    )
    folds = list(sp.split(bl.labels.index))
    assert len(folds) >= 10
    assert _purge_holds(df.index, folds, bl.label_horizon)
    # 해소 라벨이 3-class 를 실제로 포함(무거래 극단 아님)
    resolved = bl.labels.dropna()
    assert len(resolved) > 0.5 * len(df)          # 대부분 해소
    assert set(resolved.unique()) == {"up", "down", "expire"}
