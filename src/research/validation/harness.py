"""experiment harness — walk-forward 실행 오케스트레이션 (MasterPlan §11 컴포넌트 G).

loader/audit(데이터) → splitter(폴드) → (선택)normalize → model(fit/predict) →
metrics 집계(폴드·국면 분포)를 하나로 묶는다. splitter·aggregate_metrics·baselines·
normalize 를 전부 재사용하고, harness 는 오케스트레이션만 한다.

- model_factory: 폴드마다 새 모델 생성(fit/predict 계약). baseline 지금, MLP/트리 Phase3 드롭인.
- normalizer_factory: 정규화를 harness 가 소유(폴드 경계를 아는 쪽이 train-only 인과성 강제 —
  Step 0.3 seam 을 실파이프라인에 편입). z-score 지금, robust(F-6) 드롭인.
- seed: 재현성(실모델 대비; baseline 은 결정적).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.validation.metrics import (
    AggReport,
    FoldPrediction,
    aggregate_metrics,
)


@dataclass
class WalkForwardResult:
    report: AggReport
    fold_preds: list[FoldPrediction]
    config: dict
    n_folds: int


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    splitter,
    *,
    regime_tags: pd.Series | None = None,
    labels=None,
    normalizer_factory=None,
    seed: int | None = None,
) -> WalkForwardResult:
    """X/y(모델 TF, 감사 통과) 위에서 walk-forward 평가 실행 → AggReport."""
    _set_seed(seed)
    if labels is None:
        labels = sorted(pd.Series(y).dropna().unique())

    fold_preds: list[FoldPrediction] = []
    for fold in splitter.split(X.index):
        x_tr, x_val = X.loc[fold.train_index], X.loc[fold.val_index]
        y_tr = y.loc[fold.train_index]

        if normalizer_factory is not None:
            norm = normalizer_factory().fit(x_tr)   # train-only fit (인과성)
            x_tr, x_val = norm.transform(x_tr), norm.transform(x_val)

        model = model_factory()
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_val)
        pred = pd.Series(model.predict(x_val), index=fold.val_index)
        fold_preds.append(
            FoldPrediction(fold.fold_id, y.loc[fold.val_index], pred, proba)
        )

    if not fold_preds:
        raise ValueError("생성된 폴드가 없음 (splitter/데이터 확인)")

    report = aggregate_metrics(fold_preds, regime_tags=regime_tags, labels=labels)
    config = {
        "n_folds": len(fold_preds),
        "seed": seed,
        "labels": list(labels),
        "train_min": splitter.train_min,
        "val_size": splitter.val_size,
        "label_horizon": splitter.label_horizon,
        "normalized": normalizer_factory is not None,
    }
    return WalkForwardResult(report, fold_preds, config, len(fold_preds))
