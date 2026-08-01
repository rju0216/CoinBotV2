"""피처 방향성·대칭성 감사 (Phase 7).

**무엇을 재나**: 각 피처가 단독으로 "이 봉이 up 으로 끝날까 down 으로 끝날까"를 얼마나
구분하는지(단변량 AUC)와, 그 관계의 **부호가 시간에 걸쳐 유지되는지**(부호 안정성).

**왜 필요한가**: Phase 6 은 4h 예측의 롱 편향(77.7%)을 발견했으나 원인을 미검정으로 남겼다
(후보: 피처의 상승 비대칭 / 정규화 분포 이동). Phase 7 E-배치가 표본/파라미터 가설을
양방향 기각(E-a 축소서 편향 악화·E-b 확대서 공통봉 불변)했으므로, 남은 1순위 후보는
**피처↔라벨 관계의 비대칭**이다. 이 모듈이 그것을 직접 잰다.

**해석 규약**
- ``auc = 0.5`` → 방향맹(그 피처는 up/down 을 전혀 구분 못 함)
- ``auc > 0.5`` → 값이 클수록 up · ``auc < 0.5`` → 값이 클수록 down
- **세그먼트별 부호가 뒤집히면** 그 관계는 시변이라 walk-forward 모델이 **인과적으로 쓸 수
  없다**(과거 부호를 미래에 잘못 적용). 크기(|auc−0.5|)보다 **부호 안정성이 더 중요**하다.

**대상은 해소봉만** — 라벨이 expire 인 봉은 방향 정답이 없다.
AUC 는 순위 기반이라 **폴드 내에서는** 정규화·단조변환에 불변이다. 단 harness 정규화는
폴드마다 **다른 아핀 변환**이므로, 이 불변성은 **세그먼트=폴드로 쪼갠 `auc_by_segment`
에서만** 성립한다. 전 표본 pooled `directional_auc` 는 폴드별 아핀이 섞여 정확히 같지 않다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIRECTION_CLASSES = ("up", "down")
MIN_N = 100          # 세그먼트 AUC 최소 표본 (미만이면 NaN — 노이즈 방지)


def _resolved_mask(y: pd.Series, classes=DIRECTION_CLASSES) -> np.ndarray:
    return y.isin(list(classes)).to_numpy()


def directional_auc(X: pd.DataFrame, y: pd.Series, classes=DIRECTION_CLASSES,
                    min_n: int = MIN_N) -> pd.DataFrame:
    """피처별 단변량 방향 AUC (전 표본). index=피처, cols=[auc, abs_dev, n].

    X·y 는 index 로 정렬한다(교집합만 사용). 피처의 NaN 행은 그 피처 계산에서만 제외.
    """
    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]
    res = _resolved_mask(y, classes)
    lab_up = (y == classes[0]).to_numpy()
    rows = {}
    for c in X.columns:
        v = X[c].to_numpy(dtype="float64")
        m = res & ~np.isnan(v)
        if m.sum() < min_n or len(np.unique(lab_up[m])) < 2:
            rows[c] = {"auc": np.nan, "abs_dev": np.nan, "n": int(m.sum())}
            continue
        auc = float(roc_auc_score(lab_up[m], v[m]))
        rows[c] = {"auc": auc, "abs_dev": abs(auc - 0.5), "n": int(m.sum())}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "feature"
    return out.sort_values("abs_dev", ascending=False)


def auc_by_segment(X: pd.DataFrame, y: pd.Series, segments: pd.Series,
                   classes=DIRECTION_CLASSES, min_n: int = MIN_N) -> pd.DataFrame:
    """세그먼트별 AUC. index=피처, cols=세그먼트 라벨.

    ``segments`` = X.index 로 정렬된 세그먼트 라벨(연도·**폴드**·국면 등).
    폴드(22)가 연도(6)보다 표본이 3.7배라 부호 안정성 판정이 정확하다.
    """
    idx = X.index.intersection(y.index).intersection(segments.index)
    X, y, segments = X.loc[idx], y.loc[idx], segments.loc[idx]
    cols = {}
    for seg, sub in segments.groupby(segments):
        cols[seg] = directional_auc(X.loc[sub.index], y.loc[sub.index],
                                    classes, min_n)["auc"]
    # directional_auc 는 |auc-0.5| 로 정렬하므로 세그먼트마다 행 순서가 다르다.
    # X.columns 로 고정해 출력 순서를 결정적으로 만든다(값은 라벨 정렬이라 불변).
    out = pd.DataFrame(cols).reindex(X.columns)
    out.index.name = "feature"
    return out


def sign_stability(seg_auc: pd.DataFrame) -> pd.DataFrame:
    """세그먼트 AUC 의 **부호 일관성** 진단. index=피처.

    - ``pos_share``: (auc>0.5) 인 세그먼트 비율. **1.0 또는 0.0 이면 부호 완전 안정**,
      0.5 근처면 해마다 뒤집힘(= 시변 관계, 인과적으로 사용 불가).
    - ``mean_dev``/``sd_dev``: (auc−0.5) 의 평균·표준편차. 진폭과 흔들림.
    - ``stable``: 부호가 전 세그먼트에서 유지(pos_share ∈ {0,1})되고 평균 진폭이 0 이 아님.
    """
    dev = seg_auc - 0.5
    n_valid = dev.notna().sum(axis=1)
    pos = (dev > 0).sum(axis=1)
    pos_share = (pos / n_valid.replace(0, np.nan))
    out = pd.DataFrame({
        "n_segments": n_valid,
        "pos_share": pos_share,
        "mean_dev": dev.mean(axis=1),
        "sd_dev": dev.std(axis=1),
    })
    # docstring 조건("평균 진폭이 0 이 아님")을 실제로 강제. dev>0 판정이라 auc==0.5
    # (완전 방향맹)는 pos_share=0.0 이 되어 과거 구현은 **방향맹 피처를 stable=True** 로
    # 분류했다. 세그먼트 수 문턱도 22폴드 관례에 맞춰 올린다.
    out["stable"] = (out["pos_share"].isin([0.0, 1.0]) & (n_valid >= 3)
                     & (out["mean_dev"].abs() > 1e-12))
    out.index.name = "feature"
    return out.sort_values(["stable", "mean_dev"], ascending=[False, False])


def fold_segments(splitter, index: pd.Index) -> pd.Series:
    """splitter 의 val 폴드를 세그먼트 라벨(fold_id)로 변환 — auc_by_segment 입력용."""
    lab = pd.Series(np.nan, index=index, dtype="float64")
    for fold in splitter.split(index):
        lab.loc[fold.val_index] = fold.fold_id
    return lab.dropna().astype(int)


def conditional_distribution(X: pd.DataFrame, y: pd.Series,
                             classes=DIRECTION_CLASSES) -> pd.DataFrame:
    """up/down 조건부 분포 요약 — 평균·중앙값 차 (AUC 의 보조 해석).

    AUC 는 순위만 보므로 "얼마나 다른가"는 안 알려준다. 여기서 원 단위 차를 함께 본다.
    """
    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]
    up_m = (y == classes[0]).to_numpy()
    dn_m = (y == classes[1]).to_numpy()
    rows = {}
    for c in X.columns:
        v = X[c]
        u, d = v[up_m], v[dn_m]
        pooled_sd = float(v.std())
        rows[c] = {
            "mean_up": float(u.mean()), "mean_down": float(d.mean()),
            "mean_diff": float(u.mean() - d.mean()),
            "std_mean_diff": float((u.mean() - d.mean()) / pooled_sd) if pooled_sd > 0 else np.nan,
            "median_diff": float(u.median() - d.median()),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "feature"
    return out.reindex(out["std_mean_diff"].abs().sort_values(ascending=False).index)
