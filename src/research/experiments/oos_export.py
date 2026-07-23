"""Phase 4 Step 4.0 — 채택 엣지(15m+1h)의 OOS 예측 박제 (관문2 입력, D-a).

관문2(경제성)는 **관문1을 통과한 바로 그 예측**을 엔진에 재생해 비용 차감 후 생존을
본다(선별↔확인 분리, 기둥3). 그러려면 R4.2 에서 채택된 config(15m 결정TF + 1h 상위맥락,
라벨 atr/w24/x3.0/N24=6h)의 **walk-forward OOS 예측**을 시계열로 박제해야 한다.

**동일 컴포넌트 재현(D-a·D-019)**: gate1(`run_tf_gate1`)이 쓰는 것과 **정확히 같은**
build_mtf_features·splitter_for·ZScoreNormalizer·SmallMLP·regime_tags·5seed 로 `run_walk_forward`
를 돌려 폴드별 OOS proba 를 뽑는다(harness 가 폴드/정규화 인과성 소유). splitter step=val_size
라 val 폴드 비겹침 → concat 하면 timestamp 중복 없는 연속 OOS 시계열.

**재현 검증(CLAUDE 19)**: 박제 예측으로 재계산한 ll-margin 이 문서값(mtf_1h +0.0486)과
일치해야 오염 없는 baseline. 불일치 시 하드페일(하류 배선 전 차단).

**아티팩트**: `data/research/phase4/oos_predictions_15m_mtf1h.parquet`
  index = OOS 15m 봉 timestamp (그 봉 마감이 만든 예측 — 엔진은 다음봉 open 진입).
  cols  = s{k}_{up,down,expire} (seed 0..4) · ens_{up,down,expire} (seed 평균) ·
          y_true (실현 라벨) · regime (국면 태그). + json 사이드카(config·재현·커버리지).

5seed 전부 박제(소비 전략=θ 스윕은 앙상블·최종은 per-seed 분포는 Step 4.3 결정).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.experiments.tf_expansion import (
    MLP_SEEDS,
    _regime_tags_for,
    _summ,
    load_tf,
    splitter_for,
)
from src.research.experiments.tf_mtf import DECISION_TF, PRIMARY_LABEL
from src.research.features.build import build_features
from src.research.features.mtf import build_mtf_features
from src.research.labeling.triple_barrier import LABEL_CLASSES, LabelParams, compute_triple_barrier
from src.research.models import SmallMLP
from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline
from src.research.validation.harness import run_walk_forward

# 채택 config (R4.2 D-029). 상위맥락 = 1h 단독(4h 중복 제외).
ADOPTED_HIGHER_TFS = ("1h",)
# 문서 확정 재현 목표 (§12-6): mtf_1h ll-margin. 재현 오차 허용(부동소수·플랫폼).
DOC_LL_MARGIN = 0.0486
REPRO_TOL = 5e-4
CLASS_ORDER = ["up", "down", "expire"]           # 저장 열 순서(고정, 사람이 읽기 쉬움)
ARTIFACT_DIR = os.path.join("data", "research", "phase4")
ARTIFACT_NAME = "oos_predictions_15m_mtf1h"


@dataclass
class OOSExportResult:
    artifact: pd.DataFrame           # index=OOS ts, 예측·y·regime
    meta: dict = field(default_factory=dict)


def build_adopted_xy(candle_dir: str = "data/candles"):
    """채택 config 의 (결정df, X 16열, y 라벨, splitter, regime_tags, barrier_frac) — gate1 동일.

    barrier_frac = **라벨의 배리어 폭** x·σ_i (= upper_i/close_i − 1, compute_triple_barrier 의
    바로 그 배리어에서 파생). 플러그인이 진입가 기준 배리어 `entry·(1±barrier_frac)` 를 세울 때
    라벨과 **정확히 동일한 σ** 를 쓰게 한다(F-11 정합, D-b①). 재계산 아닌 라벨 경로 재사용(DRY)."""
    dec = load_tf(DECISION_TF, candle_dir)
    higher = [(tf, load_tf(tf, candle_dir)) for tf in ADOPTED_HIGHER_TFS]
    X = build_mtf_features(dec, higher)
    bl = compute_triple_barrier(dec, PRIMARY_LABEL)
    y = bl.labels
    barrier_frac = (bl.upper / dec["close"] - 1.0).rename("barrier_frac")
    sp = splitter_for(DECISION_TF, PRIMARY_LABEL.horizon)
    tags = _regime_tags_for(DECISION_TF, X.index, candle_dir)
    return dec, X, y, sp, tags, barrier_frac


def build_frontier_xy(decision_tf: str, label: LabelParams, higher_tfs=(),
                      candle_dir: str = "data/candles"):
    """일반화 (P1, Phase5 지평 프론티어): 임의 (decision_tf, label, higher_tfs) 의 (dec,X,y,sp,tags,bf).

    higher_tfs 비면 **build_features(단독TF, gate1 과 동일)**, 있으면 build_mtf_features.
    barrier_frac·splitter·regime_tags 는 채택 경로(build_adopted_xy)와 동일 규율.
    higher_tfs=("1h",)+PRIMARY_LABEL 이면 build_adopted_xy 와 동치(등가성 테스트로 박제)."""
    dec = load_tf(decision_tf, candle_dir)
    if higher_tfs:
        higher = [(tf, load_tf(tf, candle_dir)) for tf in higher_tfs]
        X = build_mtf_features(dec, higher)
    else:
        X = build_features(dec)
    bl = compute_triple_barrier(dec, label)
    y = bl.labels
    barrier_frac = (bl.upper / dec["close"] - 1.0).rename("barrier_frac")
    sp = splitter_for(decision_tf, label.horizon)
    tags = _regime_tags_for(decision_tf, X.index, candle_dir)
    return dec, X, y, sp, tags, barrier_frac


def _oos_proba(fold_preds) -> pd.DataFrame:
    """폴드별 y_proba concat → 연속 OOS proba (CLASS_ORDER 정렬, 누락 클래스 0)."""
    frames = [fp.y_proba.reindex(columns=CLASS_ORDER, fill_value=0.0) for fp in fold_preds]
    out = pd.concat(frames, axis=0).sort_index()
    if out.index.has_duplicates:                 # splitter 비겹침 계약 위반 방어
        raise ValueError("OOS 예측에 중복 timestamp — 폴드 val 겹침(step<val_size?)")
    return out


def _oos_ytrue(fold_preds) -> pd.Series:
    return pd.concat([fp.y_true for fp in fold_preds], axis=0).sort_index()


def export_oos_predictions(
    candle_dir: str = "data/candles",
    mlp_seeds=MLP_SEEDS,
    out_dir: str | None = ARTIFACT_DIR,
    artifact_name: str = ARTIFACT_NAME,
    repro_target: float | None = DOC_LL_MARGIN,
    repro_tol: float = REPRO_TOL,
    config_meta: dict | None = None,
    X=None, y=None, splitter=None, regime_tags=None, barrier_frac=None,
) -> OOSExportResult:
    """채택/임의 config 를 5seed walk-forward → OOS 예측 박제 + gate1 ll-margin 재현 검증.

    X/y/splitter/regime_tags/barrier_frac 주입 시 build 를 건너뜀(seam 테스트·프론티어).
    repro_target = 이 config 의 gate1 ll-margin(기본=채택 문서값 0.0486). 불일치 시 하드페일.
    artifact_name = config별 파일명. out_dir=None 이면 persist 안 함(검증 전용)."""
    if X is None:
        _, X, y, splitter, regime_tags, barrier_frac = build_adopted_xy(candle_dir)
    labels = list(LABEL_CLASSES)

    def _wf(factory, seed):
        return run_walk_forward(X, y, factory, splitter, regime_tags=regime_tags,
                                labels=labels, normalizer_factory=ZScoreNormalizer, seed=seed)

    # Prior baseline (재현 검증용 prior_ll) — MLP 와 동일 val·NaN드롭 정렬
    prior_res = _wf(PriorBaseline, None)
    prior_ll = _summ(prior_res)[0]

    seed_series: dict[int, pd.DataFrame] = {}
    seed_ll: list[float] = []
    y_true = None
    for s in mlp_seeds:
        res = _wf(lambda: SmallMLP(seed=s), s)
        seed_series[s] = _oos_proba(res.fold_preds)
        seed_ll.append(_summ(res)[0])
        if y_true is None:
            y_true = _oos_ytrue(res.fold_preds)

    mlp_ll_median = float(np.median(seed_ll))
    ll_margin = prior_ll - mlp_ll_median

    # ---- 재현 검증 (CLAUDE 19): 박제가 gate1 ll-margin 과 일치? ----
    # repro_target=None → 탐색 모드(사전 gate1 값 없음, 재현할 기준 없음)로 스킵.
    repro_ok = True if repro_target is None else abs(ll_margin - repro_target) <= repro_tol
    if not repro_ok:
        raise ValueError(
            f"재현 실패: OOS ll-margin {ll_margin:+.4f} vs gate1 목표 {repro_target:+.4f} "
            f"(허용 {repro_tol}). 박제 예측이 gate1 과 다름 — 하류 배선 차단(오염 방지)."
        )

    # ---- 아티팩트 조립 ----
    idx = seed_series[mlp_seeds[0]].index
    cols = {}
    for s in mlp_seeds:
        ps = seed_series[s].reindex(idx)           # 전 seed 동일 index(동일 NaN드롭) 전제 정합
        for c in CLASS_ORDER:
            cols[f"s{s}_{c}"] = ps[c].to_numpy()
    ens = sum(seed_series[s].reindex(idx)[CLASS_ORDER] for s in mlp_seeds) / len(mlp_seeds)
    for c in CLASS_ORDER:
        cols[f"ens_{c}"] = ens[c].to_numpy()
    artifact = pd.DataFrame(cols, index=idx)
    artifact.index.name = "timestamp"
    artifact["y_true"] = y_true.reindex(idx)
    artifact["regime"] = regime_tags.reindex(idx)
    if barrier_frac is not None:
        bf = barrier_frac.reindex(idx)
        n_bf_nan = int(bf.isna().sum())
        if n_bf_nan:            # σ 워밍업은 OOS 시작 前(24봉)이라 OOS idx 엔 NaN 없어야 함
            raise ValueError(f"barrier_frac 에 OOS 구간 NaN {n_bf_nan}개 — 배리어 폭 결손(F-11 배선 차단)")
        artifact["barrier_frac"] = bf
    # MTF 역할2 게이트용 상위맥락 = 1h 추세 부호(mtf1h_kf_slope 원시값, 비정규화 — 부호만 사용).
    # Phase4 스윕에서 "신호방향과 1h추세 일치시만 진입" 게이트에 쓰인다(F-1 사전등록).
    if X is not None and "mtf1h_kf_slope" in getattr(X, "columns", []):
        artifact["mtf1h_kf_slope"] = X["mtf1h_kf_slope"].reindex(idx)

    cfg_meta = config_meta or {
        "decision_tf": DECISION_TF, "higher_tfs": list(ADOPTED_HIGHER_TFS),
        "label": {"estimator": PRIMARY_LABEL.estimator, "window": PRIMARY_LABEL.window,
                  "x": PRIMARY_LABEL.x, "horizon": PRIMARY_LABEL.horizon},
    }
    meta = {
        "config": {**cfg_meta, "n_features": int(X.shape[1]),
                   "mlp_seeds": list(mlp_seeds), "class_order": CLASS_ORDER},
        "reproduction": {
            "prior_ll": prior_ll, "mlp_ll_median": mlp_ll_median,
            "ll_margin": ll_margin, "repro_target": repro_target,
            "seed_ll": seed_ll, "repro_ok": bool(repro_ok), "tol": repro_tol,
        },
        "coverage": {
            "n_oos_rows": int(len(idx)),
            "oos_start": str(idx[0]), "oos_end": str(idx[-1]),
            "n_folds": int(prior_res.n_folds),
        },
    }
    if "barrier_frac" in artifact.columns:
        bf = artifact["barrier_frac"]
        meta["barrier_frac"] = {           # D-b①: 진입가 기준 배리어 폭 x·σ (라벨 동일)
            "median": float(bf.median()), "min": float(bf.min()), "max": float(bf.max()),
            "desc": "TP/SL = entry*(1 +/- barrier_frac). = x*sigma_i (라벨 upper/close-1).",
        }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        pq = os.path.join(out_dir, artifact_name + ".parquet")
        artifact.to_parquet(pq)
        with open(os.path.join(out_dir, artifact_name + "_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        meta["artifact_path"] = pq

    return OOSExportResult(artifact=artifact, meta=meta)


def export_frontier(decision_tf: str, label: LabelParams, repro_target: float,
                    higher_tfs=(), repro_tol: float = REPRO_TOL,
                    artifact_name: str | None = None, out_dir: str | None = ARTIFACT_DIR,
                    candle_dir: str = "data/candles", mlp_seeds=MLP_SEEDS) -> OOSExportResult:
    """일반화 편의 래퍼 (P1): 임의 config 의 OOS 박제. repro_target = 그 config 의 gate1 ll-margin.

    artifact_name 미지정 시 config 파생명. build_frontier_xy 로 X/y/bf 구성 후 export_oos_predictions."""
    _, X, y, sp, tags, bf = build_frontier_xy(decision_tf, label, higher_tfs, candle_dir)
    name = artifact_name or (
        f"oos_{decision_tf}_N{label.horizon}_w{label.window}_x{label.x}"
        + ("_mtf" + "".join(higher_tfs) if higher_tfs else "")
    )
    cfg_meta = {
        "decision_tf": decision_tf, "higher_tfs": list(higher_tfs),
        "label": {"estimator": label.estimator, "window": label.window,
                  "x": label.x, "horizon": label.horizon},
    }
    return export_oos_predictions(
        candle_dir=candle_dir, mlp_seeds=mlp_seeds, out_dir=out_dir, artifact_name=name,
        repro_target=repro_target, repro_tol=repro_tol, config_meta=cfg_meta,
        X=X, y=y, splitter=sp, regime_tags=tags, barrier_frac=bf,
    )


def _print(res: OOSExportResult) -> None:
    m = res.meta
    print("=" * 68)
    print("[Step 4.0] OOS 예측 박제 (채택 15m+1h, 관문2 입력)")
    r = m["reproduction"]
    print(f"  재현: ll-margin {r['ll_margin']:+.4f} vs 목표 {r['repro_target']:+.4f} "
          f"→ {'OK' if r['repro_ok'] else 'FAIL'}  (prior_ll={r['prior_ll']:.4f}, "
          f"mlp_ll_med={r['mlp_ll_median']:.4f})")
    print(f"  seed_ll={[round(x,4) for x in r['seed_ll']]}")
    c = m["coverage"]
    print(f"  OOS: {c['n_oos_rows']:,}행  {c['oos_start']} ~ {c['oos_end']}  ({c['n_folds']}폴드)")
    print(f"  피처={m['config']['n_features']}열  아티팩트={m.get('artifact_path','(미저장)')}")
    print("=" * 68)


if __name__ == "__main__":
    _print(export_oos_predictions())
