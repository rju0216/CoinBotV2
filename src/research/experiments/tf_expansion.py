"""R4 TF/MTF 확장 인프라 (MasterPlan §5 R4, 설계 §9).

1h 튜닝저항 → 다른 시간척도 탐색. TF마다 봉의 시간의미가 달라 **splitter·라벨을 TF별로
스케일**해야 한다(결정1·3):
- **splitter**: 봉 기반 유지(D-005 무변경) + 시간→봉 환산 헬퍼(1yr/3mo). 갭0이라 정확.
- **라벨**: N=24봉이 TF마다 경제지평 상이 → **분포게이트로 학습가능 N 선정**(R0 TF별 축소판).
- **데이터**: 1d 손상→1h 파생(I-001), 나머지 clean 캐시. coverage 갭가드.

MTF(설계 §9)는 상위TF 피처를 완성봉 ff → 결정TF X 에 concat (R4.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.historical import TF_MS
from src.research.data.audit import audit_continuity
from src.research.data.loader import load_audited, resample_ohlcv
from src.research.experiments.r1_smoke import (
    ModelEval,
    _consistency_vs_prior,
    _eval_flags,
    classify_gate1,
)
from src.research.features.build import build_features
from src.research.labeling.distribution import evaluate_layer1
from src.research.labeling.triple_barrier import (
    LABEL_CLASSES,
    LabelParams,
    compute_triple_barrier,
)
from src.research.models import SmallMLP, TreeBench
from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline, UniformBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.ledger import RunLedger
from src.research.validation.regime import LABEL_COL, forward_fill_completed, tag_regimes
from src.research.validation.splitter import WalkForwardSplitter

SYMBOL = "BTC/USDT:USDT"
_DAY_MS = 86_400_000
TRAIN_DAYS, VAL_DAYS = 365, 90        # walk-forward 시간 고정 (전 TF 공통)


def bars(tf: str, days: float) -> int:
    """달력 일수 → 해당 TF 봉 수 (갭0 전제, 정확). 예: bars('4h',365)=2190."""
    return int(days * _DAY_MS // TF_MS[tf])


def splitter_for(tf: str, label_horizon: int) -> WalkForwardSplitter:
    """TF 스케일 walk-forward splitter (train 1yr / val 3mo 봉환산). splitter 코드 무변경."""
    return WalkForwardSplitter(
        train_min=bars(tf, TRAIN_DAYS), val_size=bars(tf, VAL_DAYS),
        label_horizon=label_horizon,
    )


def load_tf(tf: str, candle_dir: str = "data/candles") -> pd.DataFrame:
    """TF OHLCV(감사 clean). 1d 는 손상(I-001)→1h 파생. 갭0 가드(시간→봉 환산 정확성)."""
    if tf == "1d":
        df = resample_ohlcv(load_audited(SYMBOL, "1h", candle_dir), "1h", "1d")
        audit_continuity(df, "1d").raise_if_corrupt()
    else:
        df = load_audited(SYMBOL, tf, candle_dir)
    rep = audit_continuity(df, tf)
    if rep.coverage < 0.9999:            # 시간→봉 환산은 갭0 전제 (결정3 주의)
        raise ValueError(f"{tf} coverage {rep.coverage:.5f} < 1 — 갭 있어 시간환산 부정확")
    return df


@dataclass
class LabelPick:
    params: LabelParams
    passed: bool
    fractions: dict          # up/down/expire 비율
    worst_fold_minority: int
    horizon_days: float


def sweep_labels_for_tf(tf: str, df: pd.DataFrame,
                        n_days_candidates=(1, 2, 3, 5),
                        vol_windows=(24, 48, 96),
                        x: float = 3.0, estimator: str = "atr") -> list[LabelPick]:
    """TF별 라벨 후보 스윕 → 분포게이트 결과. N 은 경제지평(일)로 지정→봉환산. atr/x3.0 고정."""
    picks = []
    for w in vol_windows:
        for nd in n_days_candidates:
            N = bars(tf, nd)
            if N < 2:
                continue
            params = LabelParams(estimator=estimator, window=w, x=x, horizon=N)
            bl = compute_triple_barrier(df, params)
            res = evaluate_layer1(bl.labels, splitter_for(tf, N))
            picks.append(LabelPick(
                params=params, passed=res.passed,
                fractions={k: round(float(v), 3) for k, v in res.global_fractions.items()},
                worst_fold_minority=res.worst_fold_minority, horizon_days=nd,
            ))
    return picks


# ---- R4.1: TF별 관문1 (단독 TF) ----

MLP_SEEDS = (0, 1, 2, 3, 4)
# 분포게이트로 선정(각 TF에서 1h 균형 0.31/0.31/0.38 근접). atr/x3.0 고정.
# 1d 는 데이터부족(2370봉)으로 게이트 전멸(minMinor<100) → 신뢰 관문1 불가, 제외.
TF_LABELS = {
    "4h": LabelParams(estimator="atr", window=24, x=3.0, horizon=24),    # 4일, 0.34/0.30/0.36
    "15m": LabelParams(estimator="atr", window=24, x=3.0, horizon=24),   # 6시간, 0.30/0.31/0.38
}


def _summ(res) -> tuple:
    """(log_loss, mcc, balanced_accuracy) 폴드-중앙값 튜플."""
    s = res.report.summary
    return (float(s.loc["log_loss", "median"]), float(s.loc["mcc", "median"]),
            float(s.loc["balanced_accuracy", "median"]))


def _regime_tags_for(tf: str, index: pd.Index, candle_dir: str) -> pd.Series:
    """1d(파생) 국면을 결정TF index 로 완성봉 ff. tf==1d 면 직접."""
    d1d = resample_ohlcv(load_audited(SYMBOL, "1h", candle_dir), "1h", "1d")
    reg = tag_regimes(d1d)[LABEL_COL]
    if tf == "1d":
        return reg.reindex(index)
    tags = forward_fill_completed(reg, index, "1d")
    if isinstance(tags, pd.DataFrame):
        tags = tags.iloc[:, 0]
    return tags.reindex(index)


@dataclass
class TFGate1Result:
    tf: str
    verdict: str
    reason: str
    evals: dict            # name -> ModelEval
    prior_ll: float
    ll_margin: float       # prior_ll - mlp_ll (엣지 크기, TF 간 비교)


def run_tf_gate1(tf: str, candle_dir: str = "data/candles", ledger_path: str | None = None,
                 mlp_seeds=MLP_SEEDS, label: LabelParams | None = None,
                 split: dict | None = None) -> TFGate1Result | None:
    """단독 TF 관문1 (R1과 동일 규칙). 1h 대비 엣지 강도 비교용 지표 반환."""
    if not os.path.exists(os.path.join(candle_dir,
                          SYMBOL.replace("/", "_").replace(":", "_") + f"_{tf}.csv")) and tf != "1d":
        return None
    label = label or TF_LABELS[tf]
    df = load_tf(tf, candle_dir)
    X = build_features(df)                              # 기본창(R2), 봉기준
    y = compute_triple_barrier(df, label).labels
    sp = (WalkForwardSplitter(label_horizon=label.horizon, **split) if split
          else splitter_for(tf, label.horizon))
    regime_tags = _regime_tags_for(tf, X.index, candle_dir)
    labels = list(LABEL_CLASSES)

    def _wf(factory, seed):
        return run_walk_forward(X, y, factory, sp, regime_tags=regime_tags, labels=labels,
                                normalizer_factory=ZScoreNormalizer, seed=seed)

    prior_res = _wf(PriorBaseline, None)
    uniform_res = _wf(UniformBaseline, None)
    tree_res = _wf(lambda: TreeBench(seed=0), 0)
    prior_ll = _summ(prior_res)[0]

    mlp_runs = [_wf(lambda: SmallMLP(seed=s), s) for s in mlp_seeds]
    mlp_ll = [_summ(r)[0] for r in mlp_runs]
    mlp_mcc = [_summ(r)[1] for r in mlp_runs]
    mlp_ba = [_summ(r)[2] for r in mlp_runs]
    mlp_cons = [_consistency_vs_prior(r, prior_res) for r in mlp_runs]

    evals = {}
    for name, res in (("prior", prior_res), ("uniform", uniform_res), ("tree", tree_res)):
        m = _summ(res)
        evals[name] = _eval_flags(ModelEval(name, m[0], m[1], m[2],
                                            _consistency_vs_prior(res, prior_res)), prior_ll)
    mlp_ev = ModelEval("mlp", float(np.median(mlp_ll)), float(np.median(mlp_mcc)),
                       float(np.median(mlp_ba)), float(np.median(mlp_cons)),
                       seed_spread={"ll": (min(mlp_ll), max(mlp_ll))})
    evals["mlp"] = _eval_flags(mlp_ev, prior_ll)
    verdict, reason = classify_gate1(evals, prior_ll)

    if ledger_path:
        led = RunLedger(ledger_path)
        camp = f"R4_{tf}"
        if led.budget(camp) is None:
            led.register_budget(camp, 2)
        led.record({"tf": tf, "model": "tree"}, {"ll": evals["tree"].ll_median},
                   campaign=camp, is_comparison=True)
        led.record({"tf": tf, "model": "mlp"}, {"ll": mlp_ev.ll_median, "mcc": mlp_ev.mcc_median},
                   campaign=camp, is_comparison=True)

    return TFGate1Result(tf=tf, verdict=verdict, reason=reason, evals=evals,
                         prior_ll=prior_ll, ll_margin=prior_ll - mlp_ev.ll_median)


def _print_tf_report(r: TFGate1Result) -> None:
    print("=" * 64)
    print(f"[{r.tf}] 관문1: {r.verdict} — {r.reason}")
    print(f"  Prior_ll={r.prior_ll:.4f} | MLP ll-margin(엣지크기)={r.ll_margin:+.4f}")
    print(f"  {'model':8} {'ll':>8} {'mcc':>8} {'ba':>7} {'consist':>8} flag")
    for name in ("uniform", "prior", "tree", "mlp"):
        e = r.evals[name]
        flag = "STRONG" if e.strong else ("weak" if e.weak else "-")
        print(f"  {name:8} {e.ll_median:8.4f} {e.mcc_median:8.3f} {e.ba_median:7.3f} "
              f"{e.consistency:8.2f} {flag}")
    print(f"  --- 1h 기준: mcc 0.128 · ba 0.413 · ll-margin +0.0284 ---")
    print("=" * 64)


if __name__ == "__main__":
    ledger = os.path.join("data", "research", "r4_ledger.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    res = run_tf_gate1("4h", ledger_path=ledger)
    if res is None:
        print("SKIP: 데이터 부재.")
    else:
        _print_tf_report(res)
