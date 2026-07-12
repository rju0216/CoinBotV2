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
    coverage: pd.DataFrame  # 폴드별 표본·NaN 드롭 계수 (F-10 커버리지, "조용한 축소" 방지)


def _drop_nan_rows(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int]:
    """X 임의 열 NaN 또는 y NaN 인 행을 **함께** 제거 (F-10).

    실모델(MLP/트리)은 NaN 입력에 fit 불가하고, y NaN(삼중배리어 동시터치·워밍업·꼬리)은
    정답이 없어 채점 불가다. baseline 은 X 를 무시하고 y NaN 을 조용히 흘려 이 문제가
    미발현이었다(F-10). 드롭은 폴드 슬라이스 직후·**정규화 fit 이전**에 수행해 scaler 가
    항상 NaN-free 입력을 받게 한다(scaler 의 complete-컬럼 전제 보장, F-9 대비).
    반환 = (드롭된 x, 드롭된 y, 드롭 행수). x·y 인덱스 정합 유지.
    """
    valid = x.notna().all(axis=1) & y.notna()
    n_dropped = int((~valid).sum())
    return x.loc[valid], y.loc[valid], n_dropped


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
    cover_rows: list[dict] = []
    for fold in splitter.split(X.index):
        x_tr, y_tr = X.loc[fold.train_index], y.loc[fold.train_index]
        x_val, y_val = X.loc[fold.val_index], y.loc[fold.val_index]

        # F-10: NaN 위생 — 정규화·fit 이전에 train·val 각각 드롭 (통계·계약 보호)
        x_tr, y_tr, n_tr_drop = _drop_nan_rows(x_tr, y_tr)
        x_val, y_val, n_val_drop = _drop_nan_rows(x_val, y_val)
        cover_rows.append({
            "fold_id": fold.fold_id,
            "train_n": len(y_tr), "train_dropped": n_tr_drop,
            "val_n": len(y_val), "val_dropped": n_val_drop,
        })
        if len(y_tr) == 0 or len(y_val) == 0:
            continue  # 폴드가 NaN 으로 비면 스킵 (커버리지엔 기록됨)

        if normalizer_factory is not None:
            norm = normalizer_factory().fit(x_tr)   # train-only fit (인과성)
            x_tr, x_val = norm.transform(x_tr), norm.transform(x_val)

        model = model_factory()
        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_val)
        pred = pd.Series(model.predict(x_val), index=x_val.index)
        fold_preds.append(
            FoldPrediction(fold.fold_id, y_val, pred, proba)
        )

    coverage = pd.DataFrame(cover_rows).set_index("fold_id") if cover_rows else pd.DataFrame()
    if not fold_preds:
        raise ValueError("생성된 폴드가 없음 (splitter/데이터/NaN 확인)")

    report = aggregate_metrics(fold_preds, regime_tags=regime_tags, labels=labels)
    config = {
        "n_folds": len(fold_preds),
        "seed": seed,
        "labels": list(labels),
        "train_min": splitter.train_min,
        "val_size": splitter.val_size,
        "label_horizon": splitter.label_horizon,
        "normalized": normalizer_factory is not None,
        "val_dropped_total": int(coverage["val_dropped"].sum()) if len(coverage) else 0,
    }
    return WalkForwardResult(report, fold_preds, config, len(fold_preds), coverage)
