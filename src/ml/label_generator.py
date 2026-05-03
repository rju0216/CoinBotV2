"""레이블 생성기. 미래 수익률 기반 3-class 방향 분류.

라벨 매핑 (모든 함수 공통):
    0 = SHORT
    1 = HOLD / TIMEOUT (방향 없음)
    2 = LONG
    -1 = 미래 데이터 없음 (학습 시 제거)

함수:
- generate_direction_labels: horizon 후 수익률 임계 기준 (단순)
- generate_triple_barrier_labels: 가격이 upper/lower/time barrier 중 어디 먼저 닿는지
  (BP-3-3, López de Prado 학술 표준)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_direction_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold_pct: float = 0.3,
) -> pd.Series:
    """3-class 방향 레이블 생성.

    Args:
        df: OHLCV DataFrame (close 컬럼 필수)
        horizon: 미래 N봉 후 수익률 기준
        threshold_pct: ±threshold% 이상이면 LONG/SHORT (0.3 = 0.3%)

    Returns:
        pd.Series: 0=SHORT, 1=HOLD, 2=LONG, -1=미래없음
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon) * 100
    labels = pd.Series(1, index=df.index, dtype=int, name="label")
    labels[future_return > threshold_pct] = 2
    labels[future_return < -threshold_pct] = 0
    labels[future_return.isna()] = -1
    return labels


def generate_triple_barrier_labels(
    df: pd.DataFrame,
    upper_pct: float = 1.0,
    lower_pct: float = 1.0,
    time_barrier_bars: int = 10,
) -> pd.Series:
    """Triple-barrier 라벨 (BP-3-3, López de Prado).

    각 시점 t0에서 upper(+upper_pct%)/lower(-lower_pct%)/time(t0+N봉) barrier를
    설정하고, 가격이 어느 barrier에 가장 먼저 닿는지로 라벨 결정.

    동시 hit (한 봉에서 high가 upper 도달 + low가 lower 도달) → 사안 Y (가)
    Lower 우선 (SHORT 라벨). 백테/라이브 SL 우선 정책 (a)와 일관.

    Args:
        df: OHLCV DataFrame (close, high, low 필수)
        upper_pct: upper barrier 거리 (% 단위, 1.0 = 1%)
        lower_pct: lower barrier 거리 (% 단위, 1.0 = 1%)
        time_barrier_bars: vertical barrier (시간 만료 봉 수)

    Returns:
        pd.Series: 0=SHORT (lower hit), 1=TIMEOUT, 2=LONG (upper hit), -1=미래없음
    """
    n = len(df)
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int64)

    upper_mult = 1.0 + upper_pct / 100.0
    lower_mult = 1.0 - lower_pct / 100.0

    for t0 in range(n - time_barrier_bars):
        entry = close[t0]
        upper_target = entry * upper_mult
        lower_target = entry * lower_mult

        # 다음 봉부터 time_barrier_bars 동안 검사 (t0 봉 진입 직후 다음 봉부터)
        end_idx = t0 + 1 + time_barrier_bars  # exclusive
        if end_idx > n:
            break

        label = 1  # default TIMEOUT
        for t in range(t0 + 1, end_idx):
            hit_lower = low[t] <= lower_target
            hit_upper = high[t] >= upper_target
            if hit_lower and hit_upper:
                # 동시 hit (사안 Y 가): Lower 우선 → SHORT
                label = 0
                break
            if hit_lower:
                label = 0
                break
            if hit_upper:
                label = 2
                break
        labels[t0] = label

    return pd.Series(labels, index=df.index, dtype=int, name="label")


def build_labels_from_config(
    df: pd.DataFrame,
    train_cfg: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any], int]:
    """train_cfg의 label_method에 따라 라벨 + 메타 + effective_horizon 생성.

    BP-3-3에서 4 train 스크립트가 동일하게 호출 (DRY).

    Returns:
        labels: pd.Series 라벨 (값 -1/0/1/2)
        label_params: metadata.label_params에 저장할 dict (재현성)
        effective_horizon: walk-forward embargo + train tail 제거에 사용할 봉 수
            - direction: train_cfg["horizon"]
            - triple_barrier: train_cfg["triple_barrier"]["time_barrier_bars"]
    """
    method = str(train_cfg.get("label_method", "direction")).lower()
    if method == "direction":
        horizon = int(train_cfg.get("horizon", 10))
        threshold = float(train_cfg.get("threshold_pct", 0.3))
        labels = generate_direction_labels(
            df, horizon=horizon, threshold_pct=threshold
        )
        return (
            labels,
            {"method": "direction", "horizon": horizon, "threshold_pct": threshold},
            horizon,
        )
    if method == "triple_barrier":
        tb = train_cfg.get("triple_barrier", {}) or {}
        upper_pct = float(tb.get("upper_pct", 1.0))
        lower_pct = float(tb.get("lower_pct", 1.0))
        time_barrier = int(tb.get("time_barrier_bars", 10))
        labels = generate_triple_barrier_labels(
            df,
            upper_pct=upper_pct,
            lower_pct=lower_pct,
            time_barrier_bars=time_barrier,
        )
        return (
            labels,
            {
                "method": "triple_barrier",
                "upper_pct": upper_pct,
                "lower_pct": lower_pct,
                "time_barrier_bars": time_barrier,
            },
            time_barrier,
        )
    raise ValueError(
        f"Unknown label_method: {method!r}. Use 'direction' or 'triple_barrier'."
    )
