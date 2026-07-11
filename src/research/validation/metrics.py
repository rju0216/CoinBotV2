"""지표 + 집계 (MasterPlan §11 컴포넌트 E, 설계 §10 관문1).

분류 지표(배리어 3-클래스)는 **sklearn 재사용**(검증된 구현, requirements-ml.txt 선언).
"적중률 아님 — 클래스 불균형 보정"(설계): 균형정확도·MCC·로그손실.
집계는 **폴드별 분포 + 봉단위 국면 귀속**(D-006), 평균 하나로 뭉개지 않는다.

지표는 label-agnostic(임의 클래스 집합). semi_deviation 은 계열5 피처(Phase 2)와
단일 출처 공유 예정.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    mean_pinball_loss,
)


# ---- 순수 지표 (sklearn 래핑 / 자체 구현) ----

def balanced_accuracy(y_true, y_pred) -> float:
    # y_true 가 단일 클래스면 균형정확도 = 그 클래스 recall = 정확도. 직접 계산해
    # sklearn 의 non-square confusion matrix 경고를 피한다(값은 동일).
    yt = pd.Series(y_true)
    if yt.nunique() < 2:
        return float((pd.Series(y_pred).to_numpy() == yt.to_numpy()).mean())
    return float(balanced_accuracy_score(yt, y_pred))


def mcc(y_true, y_pred) -> float:
    # y_true·y_pred 중 어느 한쪽이라도 단일 클래스면 MCC 정의 불가(분모 0 → sklearn 도
    # 0 + 경고). 상수 예측 baseline 이 흔히 이에 해당 → 0.0 로 처리(경고 회피).
    yt, yp = pd.Series(y_true), pd.Series(y_pred)
    if yt.nunique() < 2 or yp.nunique() < 2:
        return 0.0
    return float(matthews_corrcoef(yt, yp))


def multiclass_log_loss(y_true, y_proba: pd.DataFrame, labels) -> float:
    """y_proba: index=표본, columns=클래스. labels 순서로 정렬해 sklearn log_loss.

    train 폴드에 없던 클래스는 proba 열이 없으므로 0 으로 채운다(그 클래스에 0 확률 →
    해당 클래스가 실제로 나오면 큰 손실, 정상). 나머지 열 합이 1 이라 정합.
    """
    labels = list(labels)
    proba = y_proba.reindex(columns=labels, fill_value=0.0).to_numpy()
    return float(log_loss(np.asarray(y_true), proba, labels=labels))


def pinball(y_true, y_pred, alpha: float) -> float:
    """분위수 alpha 의 pinball loss (도달시간 분위수 회귀용).

    주의: 만료=우측검열 표본의 완전 처리(생존분석)는 Phase 1/6 소관. 여기선 기본
    pinball 만 제공하며, 검열 표본은 호출부가 제외/보정해 넣어야 한다.
    """
    return float(mean_pinball_loss(np.asarray(y_true), np.asarray(y_pred), alpha=alpha))


def semi_deviation(x, target: float | None = None, ddof: int = 0) -> float:
    """하방 준편차 = sqrt(mean(min(x - target, 0)^2)). target 기본=평균. NaN 스킵."""
    arr = pd.Series(x, dtype="float64").dropna().to_numpy()
    if len(arr) == 0:
        return float("nan")
    t = float(np.mean(arr)) if target is None else float(target)
    downside = np.minimum(arr - t, 0.0)
    n = len(arr) - ddof
    if n <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(downside ** 2) / n))


# ---- 집계 (폴드·국면별 분포) ----

@dataclass
class FoldPrediction:
    fold_id: int
    y_true: pd.Series                    # val_index 로 색인
    y_pred: pd.Series                    # 라벨 예측
    y_proba: pd.DataFrame | None = None  # index=val, columns=클래스 (log_loss 용)


@dataclass
class AggReport:
    per_fold: pd.DataFrame    # index=fold_id, cols=지표
    per_regime: pd.DataFrame  # index=국면, cols=지표(+n)
    summary: pd.DataFrame     # index=지표, cols=[mean,median,min,max,std] (폴드 분포)


def compute_classification_metrics(
    y_true, y_pred, y_proba: pd.DataFrame | None = None, labels=None
) -> dict:
    out = {
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
    }
    if y_proba is not None and labels is not None:
        out["log_loss"] = multiclass_log_loss(y_true, y_proba, labels)
    return out


def _pool(fold_preds: list[FoldPrediction], labels=None):
    """전 폴드 예측을 풀링. 이종 클래스 폴드에서 NaN 이 새지 않도록 **풀링 전에**
    각 폴드 proba 를 labels 로 정렬(누락 클래스는 0)한다(F2)."""
    yt = pd.concat([f.y_true for f in fold_preds])
    yp = pd.concat([f.y_pred for f in fold_preds])
    if not all(f.y_proba is not None for f in fold_preds):
        return yt, yp, None
    if labels is not None:
        cols = list(labels)
        proba = pd.concat(
            [f.y_proba.reindex(columns=cols, fill_value=0.0) for f in fold_preds]
        )
    else:
        proba = pd.concat([f.y_proba for f in fold_preds])
    return yt, yp, proba


def aggregate_metrics(
    fold_preds: list[FoldPrediction],
    regime_tags: pd.Series | None = None,
    labels=None,
) -> AggReport:
    """폴드별 지표(분포) + 국면별 지표(봉단위 귀속)를 집계.

    계약·주의점 (fresh-eyes 리뷰 F3~F5):
    - `regime_tags` 는 **모델 TF 로 정렬된**(예: 1d→1h forward_fill 된) 시리즈여야 한다.
      원시 상위 TF 태그를 그대로 주면 경계 봉만 매칭돼 대부분 NaN 으로 드롭된다(F4).
    - per_regime 의 `n` 은 **귀속 가능한 봉만**(태그 non-NaN) 센다 — regime 워밍업/초기
      구간은 제외되므로 "n 합 = 전 검증봉"은 태그가 gap-free 일 때만 성립한다(F5).
    - 검증창 비겹침(step≥val_size) 전제 — step<val_size(겹침)면 풀 인덱스에 중복이 생겨
      per_regime 이 이중계수된다(F7 트래커). 현재 기본은 비겹침.
    """
    if not fold_preds:
        raise ValueError("fold_preds 가 비어 있음")

    # 폴드별
    rows = {}
    for f in fold_preds:
        rows[f.fold_id] = compute_classification_metrics(
            f.y_true, f.y_pred, f.y_proba, labels
        )
    per_fold = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    per_fold.index.name = "fold_id"

    # 폴드 분포 요약 (평균 하나로 안 뭉갬)
    summary = per_fold.agg(["mean", "median", "min", "max", "std"]).T
    summary.index.name = "metric"

    # 국면별 (봉단위 귀속): 전 폴드 검증봉 풀 → 국면 태그로 그룹
    per_regime = pd.DataFrame()
    if regime_tags is not None:
        yt, yp, proba = _pool(fold_preds, labels)
        tags = regime_tags.reindex(yt.index).dropna()
        reg_rows = {}
        for reg, sub in tags.groupby(tags):
            sub_idx = sub.index
            sub_proba = proba.loc[sub_idx] if proba is not None else None
            m = compute_classification_metrics(
                yt.loc[sub_idx], yp.loc[sub_idx], sub_proba, labels
            )
            m["n"] = len(sub_idx)
            reg_rows[reg] = m
        per_regime = pd.DataFrame.from_dict(reg_rows, orient="index").sort_index()
        per_regime.index.name = "regime"

    return AggReport(per_fold=per_fold, per_regime=per_regime, summary=summary)
