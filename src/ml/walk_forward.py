"""워크포워드 시간 기반 롤링 윈도우 분할.

학습(train) → 테스트(test)를 시간순으로 슬라이딩하며 반복.
embargo로 train 끝과 test 시작 사이 레이블 누출을 방지한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class WalkForwardFold:
    """워크포워드 1개 fold의 시간 범위."""

    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def generate_walk_forward_splits(
    index: pd.DatetimeIndex,
    train_months: int = 6,
    test_months: int = 2,
    step_months: int = 2,
    embargo_bars: int = 10,
) -> list[WalkForwardFold]:
    """시간 기반 롤링 윈도우 분할 생성.

    Args:
        index: 데이터의 DatetimeIndex
        train_months: 학습 기간 (개월)
        test_months: 테스트 기간 (개월)
        step_months: 윈도우 이동 폭 (개월)
        embargo_bars: train/test 간 제외 구간 (레이블 누출 방지, 행 수)

    Returns:
        list[WalkForwardFold]

    예시 (train=6M, test=2M, step=2M):
        Fold 0: train 2020-01~2020-06, test 2020-07~2020-08
        Fold 1: train 2020-03~2020-08, test 2020-09~2020-10
        ...
    """
    start = index.min()
    end = index.max()
    folds: list[WalkForwardFold] = []
    fold_id = 0

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        train_start += pd.DateOffset(months=step_months)
        fold_id += 1

    return folds


def apply_embargo(
    train_idx: pd.DatetimeIndex,
    embargo_bars: int,
) -> pd.DatetimeIndex:
    """Train 끝에서 embargo_bars만큼 제거하여 레이블 누출 방지.

    Returns:
        필터링된 train index
    """
    if embargo_bars <= 0 or len(train_idx) <= embargo_bars:
        return train_idx
    return train_idx[:-embargo_bars]
