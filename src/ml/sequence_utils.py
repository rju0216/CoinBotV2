"""시퀀스 변환 유틸 — DL/RL 학습+추론 공통.

피처 DataFrame을 LSTM/Transformer/PPO가 입력으로 사용하는 시퀀스 텐서로 변환.
NaN이 포함된 시퀀스, 음수 라벨(-1, label_generator의 미래 부족 시점)은 자동 제외.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """학습용 시퀀스 생성.

    각 시퀀스 i는 features의 [i-L+1 ... i] 행 윈도우. 라벨은 시점 i의 값.

    Args:
        features: (N, F) DataFrame — 시간 순서로 정렬됨
        labels: (N,) Series — features와 동일 인덱스
        lookback: 시퀀스 길이 L

    Returns:
        X: (M, L, F) float32 ndarray (M = 유효 시퀀스 개수)
        y: (M,) int64 ndarray
        end_index: (M,) DatetimeIndex — 각 시퀀스의 마지막 시점
    """
    n_features = features.shape[1]
    if len(features) < lookback:
        return (
            np.empty((0, lookback, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DatetimeIndex([]),
        )

    X_full = features.values.astype(np.float32)
    y_full = labels.values
    idx_full = features.index

    # sliding_window_view 출력: (M, F, L) → transpose로 (M, L, F)
    win = np.lib.stride_tricks.sliding_window_view(
        X_full, window_shape=lookback, axis=0
    )
    sequences = win.transpose(0, 2, 1)  # view (zero-copy)

    end_idx = idx_full[lookback - 1:]
    y_seq = y_full[lookback - 1:]

    # 시퀀스 내 NaN 또는 음수 라벨 → 제외
    no_nan = ~np.isnan(sequences).any(axis=(1, 2))
    valid_label = y_seq >= 0
    mask = no_nan & valid_label

    return sequences[mask], y_seq[mask].astype(np.int64), end_idx[mask]


def make_sequence_from_recent(
    features: pd.DataFrame,
    lookback: int,
) -> np.ndarray | None:
    """추론용: 최근 lookback개 행으로 단일 시퀀스 생성.

    Args:
        features: (N, F) DataFrame
        lookback: 시퀀스 길이

    Returns:
        (1, L, F) float32 ndarray. 데이터 부족 또는 NaN 포함 시 None.
    """
    if len(features) < lookback:
        return None

    seq = features.iloc[-lookback:].values.astype(np.float32)
    if np.any(np.isnan(seq)):
        return None

    return seq[np.newaxis, ...]
