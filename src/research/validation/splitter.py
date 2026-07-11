"""Walk-forward splitter — expanding(앵커) + purge/꼬리예약 (라벨 지평 N 단일출처).

MasterPlan D-005/F-5/§11. 모델을 모르는 채 시간순 폴드(train/val)만 생성한다.
fit/predict 실행 루프는 harness(Step 0.5)의 몫 — splitter 는 폴드 인덱스만 낸다.

인과성(기둥 2):
- 피처는 과거만 보므로 학습표본 피처는 val 로 새지 않는다. 오직 라벨(forward)만
  샌다 → train 꼬리 N봉 purge 로 충분(val 쪽 purge 불필요).
- 마지막 N봉은 라벨 미해소(스코어 불가)라 절대 val 로 쓰지 않는다(꼬리 예약).
- expanding 이라 폴드 k 의 val 은 그걸 학습하지 않은 폴드 k 모델이 채점한다.

단일 출처 N(``label_horizon``)이 ① train 꼬리 purge ② 꼬리 예약 을 함께 구동한다.
rolling 모드·방식1 splitter 는 실제 필요 시점(Phase 3+)에 추가한다(장식금지).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_index: pd.DatetimeIndex
    val_index: pd.DatetimeIndex
    n_purged: int  # train 끝과 val 시작 사이에서 제외된 봉 수 (= N + embargo)

    @property
    def n_train(self) -> int:
        return len(self.train_index)

    @property
    def n_val(self) -> int:
        return len(self.val_index)

    @property
    def train_span(self) -> tuple:
        if len(self.train_index):
            return (self.train_index[0], self.train_index[-1])
        return (None, None)

    @property
    def val_span(self) -> tuple:
        if len(self.val_index):
            return (self.val_index[0], self.val_index[-1])
        return (None, None)


class WalkForwardSplitter:
    """expanding walk-forward. 단위는 봉(positional) — 감사 통과(온-그리드) 데이터 전제.

    Args:
        train_min: 최소 학습창 하한(봉). 첫 폴드의 (purge 후) 학습 크기 = train_min.
        val_size: 검증창 길이(봉).
        label_horizon: 라벨 지평 N(봉). purge·꼬리예약을 함께 구동(F-5).
        step: 폴드 전진(봉). 기본 = val_size (검증창 비겹침).
        embargo: purge 외 추가 완충(봉). 기본 0 (forward WF엔 purge=N로 충분).
    """

    def __init__(
        self,
        train_min: int,
        val_size: int,
        label_horizon: int,
        step: int | None = None,
        embargo: int = 0,
    ) -> None:
        if train_min <= 0 or val_size <= 0:
            raise ValueError("train_min, val_size 는 양수여야 함")
        if label_horizon < 0 or embargo < 0:
            raise ValueError("label_horizon, embargo 는 음수 불가")
        self.train_min = int(train_min)
        self.val_size = int(val_size)
        self.label_horizon = int(label_horizon)
        self.step = int(step) if step is not None else int(val_size)
        self.embargo = int(embargo)
        if self.step <= 0:
            raise ValueError("step 은 양수여야 함")

    def split(self, index) -> Iterator[Fold]:
        idx = pd.DatetimeIndex(index)
        if not idx.is_monotonic_increasing:
            raise ValueError("index 는 단조증가여야 함 (감사 통과 데이터 전제)")
        n = len(idx)
        gap = self.label_horizon + self.embargo   # train 꼬리 제외 폭
        usable_end = n - self.label_horizon        # val 은 [.., usable_end) 안에서만

        fold_id = 0
        produced = 0
        # 첫 폴드: train 크기(=val_start-gap) 가 train_min 이 되도록 시작
        val_start = self.train_min + gap
        while True:
            val_end = val_start + self.val_size
            if val_end > usable_end:
                break
            train_end = val_start - gap            # train = idx[:train_end]
            train_index = idx[:train_end]
            val_index = idx[val_start:val_end]
            yield Fold(fold_id, train_index, val_index, n_purged=gap)
            fold_id += 1
            produced += 1
            val_start += self.step

        if produced == 0:
            raise ValueError(
                "생성된 폴드 0개 — 설정/데이터 길이 확인 "
                f"(n={n}, train_min={self.train_min}, val_size={self.val_size}, "
                f"N={self.label_horizon}, embargo={self.embargo})"
            )
