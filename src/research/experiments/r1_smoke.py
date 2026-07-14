"""R1 스모크 — Phase 3 관문1 첫 성과 측정 (MasterPlan §5 R1, 설계 §11 관문1).

**수직 슬라이스**(D-001): 1h · 확정 라벨 atr/w96/x3.0/N24 · 기본창 · 기본 하이퍼.
탐색·선택 없음(그건 R2+). 트리 벤치·작은 MLP 를 Prior/Uniform 기준선과 walk-forward
비교해 "엣지 냄새"의 이진 신호를 최속 확인한다.

규율:
- **비교 예산 = 2**({트리, MLP}), 사전등록(F-1). Prior/Uniform 은 바닥 기준선(무해).
- **판정규칙은 이 파일에 박제 = 실행 전 사전등록**(``classify_gate1``, 사후합리화 방지).
- **MLP 다중 seed**(안정성 측정, 선택 아님 → 예산 무관).
- **borderline → HOLD**(즉시 폐기는 명백 무엣지만, A-1 방어책): R3 멀티태스크·튜닝까지 유보.
- 인과 self-check(build_features ``assert_causal``) + NaN 커버리지 로깅.

실 데이터(1h) 부재 시 ``run_r1_smoke`` 는 ``None`` 반환(skip-guard).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.causality.leakage import LeakageError, assert_causal
from src.research.data.audit import audit_continuity
from src.research.data.loader import (
    csv_filename,
    forward_fill_completed,
    load_audited,
    resample_ohlcv,
)
from src.research.features.build import build_features
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
from src.research.validation.regime import LABEL_COL, tag_regimes
from src.research.validation.splitter import WalkForwardSplitter

SYMBOL = "BTC/USDT:USDT"
CAMPAIGN = "R1_smoke"

# 확정 라벨 (Phase 1 R0): atr/w96/x3.0/N24
LABEL = LabelParams(estimator="atr", window=96, x=3.0, horizon=24)
# splitter T·V·S (F-5 확정): 1h → 22 폴드
SPLIT = dict(train_min=8760, val_size=2160)   # label_horizon = LABEL.horizon
# MLP 다중 seed (보강1 안정성). 트리는 결정적이라 단일 seed.
MLP_SEEDS = (0, 1, 2, 3, 4)

# ---- 사전등록 판정규칙 (실행 전 잠금, 보강3) ----
# 슬림화(사용자 확정): 게이트 3조건(ll<Prior ∧ mcc>0 ∧ 과반 일관성). BA 게이트 제거·
# 60%→과반(임의 상수 제거, 마법의 절대값 0). 비대칭(엄격PASS/관대DISCARD/HOLD완충) 유지.
PREREGISTERED_RULE = (
    "R1 관문1 (엣지 냄새, 학습모델=트리·MLP): "
    "edge(model) = [ll_median < Prior_ll_median] ∧ [mcc_median > 0] ∧ "
    "[폴드의 과반(>50%)에서 ll < Prior_ll]. "
    "STRONG=세 조건 전부, WEAK=앞 두 조건(ll<Prior ∧ mcc>0)만. "
    "PASS(R2 진행)=학습모델 중 하나라도 STRONG. "
    "DISCARD(폐기)=학습모델 중 WEAK 조차 없음(동전던지기). "
    "HOLD(유보→R3)=WEAK 있으나 STRONG 없음. "
    "균형정확도(BA)는 보고 지표(게이트 아님, 설계 §11). "
    "MLP 지표=seed 중앙값(seed 스프레드는 해석 보조, HOLD 근거로만). "
    "MLP-vs-트리 우열은 R1 게이트 아님(R2 방향 기록용)."
)


@dataclass
class ModelEval:
    name: str
    ll_median: float
    mcc_median: float
    ba_median: float
    consistency: float          # 폴드 중 ll < Prior_ll 비율
    strong: bool = False
    weak: bool = False
    seed_spread: dict = field(default_factory=dict)   # MLP: 지표별 (min,max) over seeds


@dataclass
class R1Result:
    verdict: str                # PASS | HOLD | DISCARD
    reason: str
    evals: dict                 # name -> ModelEval
    prior_ll_median: float
    coverage_val_dropped: int
    causal_ok: bool
    per_model_summary: dict     # name -> summary DataFrame (보고용)


def _fold_metric(res, col: str) -> pd.Series:
    return res.report.per_fold[col]


def _consistency_vs_prior(model_res, prior_res) -> float:
    """폴드별 log_loss 가 Prior 보다 낮은(우세) 폴드 비율."""
    m = _fold_metric(model_res, "log_loss")
    p = _fold_metric(prior_res, "log_loss")
    common = m.index.intersection(p.index)
    if len(common) == 0:
        return 0.0
    return float((m.loc[common] < p.loc[common]).mean())


def _eval_flags(ev: ModelEval, prior_ll: float) -> ModelEval:
    ll_beats = ev.ll_median < prior_ll
    mcc_pos = ev.mcc_median > 0.0
    consistent = ev.consistency > 0.5          # 과반 (BA 는 게이트 아님·보고만)
    ev.weak = bool(ll_beats and mcc_pos)
    ev.strong = bool(ll_beats and mcc_pos and consistent)
    return ev


def classify_gate1(evals: dict, prior_ll_median: float) -> tuple[str, str]:
    """사전등록 규칙(``PREREGISTERED_RULE``) → (verdict, reason). 학습모델만 대상."""
    learned = [e for n, e in evals.items() if n in ("tree", "mlp")]
    strongs = [e.name for e in learned if e.strong]
    weaks = [e.name for e in learned if e.weak]
    if strongs:
        return "PASS", f"STRONG 학습모델: {strongs} → R2 진행"
    if not weaks:
        return "DISCARD", "학습모델 중 WEAK 조차 없음(Prior 미초과/동전던지기) → 폐기"
    return "HOLD", f"WEAK={weaks} 있으나 STRONG 없음 → R3(멀티태스크·튜닝)까지 유보"


def _build_inputs(candle_dir: str):
    """1h 감사통과 로드 → X(8열)·y(확정라벨)·regime_tags(1d→1h ff). 정렬은 index 교집합."""
    df = load_audited(SYMBOL, "1h", candle_dir)
    X = build_features(df)
    bl = compute_triple_barrier(df, LABEL)
    y = bl.labels

    # I-001: 손상된 캐시 1d(off-grid + audit-미검출 값손상 14일) 대신 감사-clean 1h 에서
    # 1d 재도출. 파생 1d 가 audit 를 통과함을 확인(belt-and-suspenders, "감사 통과만 하류로").
    df1d = resample_ohlcv(df, "1h", "1d")
    audit_continuity(df1d, "1d").raise_if_corrupt()
    reg1d = tag_regimes(df1d)[LABEL_COL]
    regime_tags = forward_fill_completed(reg1d, X.index, "1d")
    if isinstance(regime_tags, pd.DataFrame):
        regime_tags = regime_tags.iloc[:, 0]
    regime_tags = regime_tags.reindex(X.index)
    return df, X, y, regime_tags


def _summ(res) -> dict:
    s = res.report.summary
    return {
        "ll_median": float(s.loc["log_loss", "median"]),
        "mcc_median": float(s.loc["mcc", "median"]),
        "ba_median": float(s.loc["balanced_accuracy", "median"]),
    }


def run_r1_smoke(candle_dir: str = "data/candles", ledger_path: str | None = None,
                 mlp_seeds=MLP_SEEDS) -> R1Result | None:
    """R1 스모크 실행 → R1Result. 데이터 부재 시 None(skip-guard)."""
    def _exists(tf: str) -> bool:
        return os.path.exists(os.path.join(candle_dir, csv_filename(SYMBOL, tf)))
    if not (_exists("1h") and _exists("1d")):
        return None

    df, X, y, regime_tags = _build_inputs(candle_dir)

    # 보강4: 인과 self-check (build_features 미래교란 불변). 실 df 슬라이스 표본으로.
    # 누수 시 assert_causal 은 raise → 전체 중단 대신 causal_ok=False 로 보고(견고).
    try:
        assert_causal(lambda d: build_features(d), df.iloc[:1500])
        causal_ok = True
    except LeakageError:
        causal_ok = False

    sp = WalkForwardSplitter(label_horizon=LABEL.horizon, **SPLIT)
    common_labels = list(LABEL_CLASSES)

    def _run(factory, seed=None):
        return run_walk_forward(
            X, y, factory, sp, regime_tags=regime_tags, labels=common_labels,
            normalizer_factory=ZScoreNormalizer, seed=seed,
        )

    base_runs = {"prior": _run(PriorBaseline), "uniform": _run(UniformBaseline)}
    prior_res = base_runs["prior"]
    tree_res = _run(lambda: TreeBench(seed=0), seed=0)

    prior_ll = float(prior_res.report.summary.loc["log_loss", "median"])

    # MLP 다중 seed → seed별 fold-median + Prior 대비 일관성, 그 seed 중앙값을 게이트에.
    mlp_runs = [_run(lambda: SmallMLP(seed=s), seed=s) for s in mlp_seeds]
    mlp_ll = [_summ(r)["ll_median"] for r in mlp_runs]
    mlp_mcc = [_summ(r)["mcc_median"] for r in mlp_runs]
    mlp_ba = [_summ(r)["ba_median"] for r in mlp_runs]
    mlp_cons = [_consistency_vs_prior(r, prior_res) for r in mlp_runs]

    evals: dict = {}
    for name, res in (("prior", prior_res), ("uniform", base_runs["uniform"]), ("tree", tree_res)):
        m = _summ(res)
        ev = ModelEval(name, m["ll_median"], m["mcc_median"], m["ba_median"],
                       _consistency_vs_prior(res, prior_res))
        evals[name] = _eval_flags(ev, prior_ll)
    mlp_ev = ModelEval(
        "mlp", float(np.median(mlp_ll)), float(np.median(mlp_mcc)),
        float(np.median(mlp_ba)), float(np.median(mlp_cons)),
        seed_spread={"ll": (min(mlp_ll), max(mlp_ll)), "mcc": (min(mlp_mcc), max(mlp_mcc)),
                     "ba": (min(mlp_ba), max(mlp_ba))},
    )
    evals["mlp"] = _eval_flags(mlp_ev, prior_ll)

    verdict, reason = classify_gate1(evals, prior_ll)

    # 사전등록 예산 + 비교 기록 (F-1). Prior/Uniform 은 기준선(is_comparison=False).
    if ledger_path:
        led = RunLedger(ledger_path)
        if led.budget(CAMPAIGN) is None:
            led.register_budget(CAMPAIGN, 2)   # 규칙은 PREREGISTERED_RULE 상수로 박제(사전등록)
        for name in ("prior", "uniform"):
            led.record({"model": name}, _summ(base_runs[name]),
                       campaign=CAMPAIGN, is_comparison=False)
        led.record({"model": "tree", "label": "atr/w96/x3.0/N24"},
                   _summ(tree_res), campaign=CAMPAIGN, is_comparison=True)
        led.record({"model": "mlp", "seeds": list(mlp_seeds)},
                   {"ll_median": mlp_ev.ll_median, "mcc_median": mlp_ev.mcc_median,
                    "ba_median": mlp_ev.ba_median}, campaign=CAMPAIGN, is_comparison=True)

    return R1Result(
        verdict=verdict, reason=reason, evals=evals, prior_ll_median=prior_ll,
        coverage_val_dropped=int(tree_res.config.get("val_dropped_total", 0)),
        causal_ok=causal_ok,
        per_model_summary={
            "prior": prior_res.report.summary,
            "uniform": base_runs["uniform"].report.summary,
            "tree": tree_res.report.summary,
            "mlp_per_regime_tree": tree_res.report.per_regime,
        },
    )


def _print_report(r: R1Result) -> None:
    print("=" * 64)
    print(f"R1 스모크 관문1 판정: {r.verdict}")
    print(f"  근거: {r.reason}")
    print(f"  Prior log_loss 중앙값(ln3={np.log(3):.4f}): {r.prior_ll_median:.4f}")
    print(f"  인과 self-check(build_features): {'OK' if r.causal_ok else 'LEAK!'}")
    print(f"  val NaN 드롭 합: {r.coverage_val_dropped}")
    print("-" * 64)
    print(f"  {'model':8} {'ll_med':>8} {'mcc_med':>8} {'ba_med':>7}  {'consist':>7}  flags")
    for name in ("uniform", "prior", "tree", "mlp"):
        e = r.evals[name]
        flag = "STRONG" if e.strong else ("weak" if e.weak else "-")
        print(f"  {name:8} {e.ll_median:8.4f} {e.mcc_median:8.4f} "
              f"{e.ba_median:7.4f}  {e.consistency:7.2f}  {flag}")
    if r.evals["mlp"].seed_spread:
        print(f"  MLP seed 스프레드: {r.evals['mlp'].seed_spread}")
    print("=" * 64)


if __name__ == "__main__":
    ledger = os.path.join("data", "research", "r1_ledger.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    result = run_r1_smoke(ledger_path=ledger)
    if result is None:
        print("SKIP: 1h/1d 캔들 부재 (data/candles).")
    else:
        _print_report(result)
