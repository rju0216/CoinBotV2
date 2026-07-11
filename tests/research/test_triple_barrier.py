"""삼중배리어 회귀 테스트.

전략:
- **독립 참조 스캐너**(순진 이중루프, 명백히 정확)와 벡터화 구현을 대조(정확성).
- **배리어 레벨 인과** ``assert_causal`` (배리어 폭 진입시점 정보만, 기둥 2).
- **유한 지평 = N**: labels[:t] 는 t+N 이후 교란에 불변(라벨의 forward 성을 N 으로 제한).
- 꼬리/워밍업 NaN·first_touch 정합·동시터치 NaN·label_horizon 단일출처.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.causality.leakage import assert_causal
from src.research.labeling.triple_barrier import (
    LABEL_CLASSES,
    BarrierLabels,
    LabelParams,
    barrier_levels,
    compute_triple_barrier,
)


def _ohlc(n=400, seed=0, jumpy=False):
    """합성 OHLCV. jumpy=True 면 팻테일 점프 주입(동시터치·양방향 케이스 유도)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    scale = 0.03 if jumpy else 0.01
    ret = rng.normal(0, scale, n)
    if jumpy:
        ret[rng.integers(0, n, n // 20)] += rng.normal(0, 0.15, n // 20)
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01 if jumpy else 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01 if jumpy else 0.004, n)))
    vol = rng.uniform(1, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _ref_scan(df, params: LabelParams):
    """독립 순진 참조 스캐너 (명백히 정확, 느림). (labels, touch) numpy 반환."""
    levels = barrier_levels(df, params)
    up = levels["upper"].to_numpy()
    dn = levels["lower"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    N = params.horizon
    labels = np.full(n, None, dtype=object)
    touch = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(up[i]) or (i + N) > (n - 1):
            continue
        decided = None
        for k in range(1, N + 1):
            j = i + k
            u = high[j] >= up[i]
            d = low[j] <= dn[i]
            if u and d:
                decided = ("ambig", k)
                break
            if u:
                decided = ("up", k)
                break
            if d:
                decided = ("down", k)
                break
        if decided is None:
            decided = ("expire", N)
        name, k = decided
        if name == "ambig":
            continue  # NaN 제외
        labels[i] = name
        touch[i] = float(k)
    return labels, touch


@pytest.mark.parametrize("jumpy", [False, True])
@pytest.mark.parametrize("params", [
    LabelParams("atr", 30, 1.5, 24),
    LabelParams("yz", 20, 2.0, 12),
    LabelParams("atr", 40, 0.5, 48),   # 작은 x → 동시터치·조기터치 다수
])
def test_matches_reference_scanner(params, jumpy):
    df = _ohlc(400, jumpy=jumpy)
    bl = compute_triple_barrier(df, params)
    ref_labels, ref_touch = _ref_scan(df, params)

    got_labels = bl.labels.to_numpy()
    # 라벨 일치 (None/NaN 취급 통일)
    for i in range(len(df)):
        g = got_labels[i]
        r = ref_labels[i]
        g = None if (g is None or (isinstance(g, float) and np.isnan(g))) else g
        assert g == r, f"pos {i}: got {g!r} ref {r!r}"
    # first_touch 일치
    np.testing.assert_array_equal(
        np.nan_to_num(bl.first_touch.to_numpy(), nan=-1),
        np.nan_to_num(ref_touch, nan=-1),
    )


def test_only_valid_classes():
    bl = compute_triple_barrier(_ohlc(jumpy=True), LabelParams("atr", 30, 1.5, 24))
    assert set(bl.labels.dropna().unique()) <= set(LABEL_CLASSES)


def test_first_touch_bounds_and_alignment():
    bl = compute_triple_barrier(_ohlc(jumpy=True), LabelParams("yz", 20, 1.0, 20))
    ft = bl.first_touch.dropna()
    assert (ft >= 1).all() and (ft <= 20).all()
    # 라벨 NaN ⟺ first_touch NaN (동시터치·꼬리·워밍업 모두 정합)
    assert bl.labels.isna().equals(bl.first_touch.isna())
    # expire 라벨의 first_touch 는 정확히 N
    exp = bl.labels == "expire"
    assert (bl.first_touch[exp] == 20).all()


def test_tail_and_warmup_are_nan():
    df = _ohlc(300)
    N, w = 24, 30
    bl = compute_triple_barrier(df, LabelParams("atr", w, 1.5, N))
    # 마지막 N봉 = 미해소 NaN
    assert bl.labels.iloc[-N:].isna().all()
    # 워밍업(창 미달) NaN — ATR 은 첫 봉 TR=high-low 라 유효 시작이 w-1
    assert bl.labels.iloc[:w - 1].isna().all()
    # 중간엔 해소 라벨 존재
    assert bl.labels.iloc[w:-N].notna().any()


def test_barrier_levels_causal():
    """배리어 폭·레벨이 미래를 안 본다 (배리어 폭 = 진입시점 정보만)."""
    df = _ohlc()
    for params in (LabelParams("atr", 30, 2.0, 24), LabelParams("yz", 30, 2.0, 24)):
        report = assert_causal(lambda d, p=params: barrier_levels(d, p), df)
        assert not report.leaked


def test_bounded_lookahead_equals_N():
    """라벨은 forward 지만 지평은 정확히 N — t+N 이후를 교란해도 labels[:t] 불변."""
    df = _ohlc(400)
    params = LabelParams("atr", 30, 1.5, 20)
    N, t = 20, 200
    full = compute_triple_barrier(df, params).labels

    df2 = df.copy()
    for col in ("open", "high", "low", "close"):
        loc = df2.columns.get_loc(col)
        df2.iloc[t + N:, loc] = df2.iloc[t + N:, loc] * 3.0  # 강한 미래 교란(OHLC 순서 보존)
    perturbed = compute_triple_barrier(df2, params).labels

    pd.testing.assert_series_equal(full.iloc[:t], perturbed.iloc[:t])


def test_horizon_within_N_is_used():
    """[t, t+N) 교란은 labels[:t] 일부를 바꿔야 한다 (지평이 실제 N 까지 쓰임)."""
    df = _ohlc(400, seed=3)
    params = LabelParams("atr", 30, 2.5, 30)  # 넓은 폭 → 만료 많음 → 미래 의존 큼
    N, t = 30, 200
    full = compute_triple_barrier(df, params).labels
    df2 = df.copy()
    for col in ("high", "low", "open", "close"):
        loc = df2.columns.get_loc(col)
        # t 이후 지평 안 구간을 극단화 → t 직전 봉들의 판정이 바뀜
        df2.iloc[t:t + N, loc] = df2.iloc[t:t + N, loc] * np.array(
            [1.5 if m % 2 else 0.6 for m in range(N)]
        )
    perturbed = compute_triple_barrier(df2, params).labels
    changed = (full.iloc[:t].fillna("_") != perturbed.iloc[:t].fillna("_"))
    assert changed.any()   # 지평 N 내 미래를 실제로 참조함


def test_label_horizon_single_source():
    bl = compute_triple_barrier(_ohlc(), LabelParams("atr", 30, 1.5, 42))
    assert bl.label_horizon == 42          # splitter 로 넘길 N 단일출처
    assert isinstance(bl, BarrierLabels)


def test_class_distribution():
    bl = compute_triple_barrier(_ohlc(jumpy=True), LabelParams("atr", 30, 1.0, 24))
    dist = bl.class_distribution(normalize=True)
    assert abs(dist.sum() - 1.0) < 1e-9
    assert set(dist.index) <= set(LABEL_CLASSES)


def test_ohlc_nan_hard_fails():
    """비감사 데이터(미래 OHLC NaN)는 진입에서 하드페일 — 조용한 expire 편향 차단."""
    df = _ohlc(200)
    df.iloc[150, df.columns.get_loc("high")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_triple_barrier(df, LabelParams("atr", 30, 1.5, 24))


def test_invalid_params():
    with pytest.raises(ValueError):
        LabelParams("bogus", 30, 1.5, 24)
    with pytest.raises(ValueError):
        LabelParams("atr", 0, 1.5, 24)
    with pytest.raises(ValueError):
        LabelParams("atr", 30, -1.0, 24)
    with pytest.raises(ValueError):
        LabelParams("atr", 30, 1.5, 0)
