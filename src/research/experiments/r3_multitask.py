"""R3 모델 하이퍼 — 멀티태스크 co-training 가설 + 정규화 (MasterPlan §5 R3, 설계 §5·§6).

R1·R2(약엣지·창 무gain) 위에서 **모델**을 튜닝해 엣지가 강해지나 확인. 최대 지렛대 =
**멀티태스크**(도달시간 co-training, 기둥7 실현 — A-1 스테이징 회수). 창 고정(R2 기본창).

규율(F-1): 예산 사전등록 + 선택 config 국면분리 재확인. 선택 = MLP 5seed seed-중앙 log_loss.
**단일태스크 기준선 = reach=None**(R1 바이트 동일 모델; λ=0-with-head 아님, 초기화 RNG 불변).

reach 타깃(러너 소유, 결정2 보강): **first_touch/N, 단 expire·라벨NaN → NaN**(검열 제외 마스킹).
모델은 generic(NaN 샘플 손실 제외)라 도메인 지식은 여기 있음.

데이터 부재 시 None(skip-guard). 실행: ``python -m ...r3_multitask``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.data.audit import audit_continuity
from src.research.data.loader import (
    csv_filename,
    forward_fill_completed,
    load_audited,
    resample_ohlcv,
)
from src.research.features.build import build_features
from src.research.labeling.triple_barrier import LABEL_CLASSES, LabelParams, compute_triple_barrier
from src.research.models import SmallMLP
from src.research.normalize import RobustScaler, ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline, UniformBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.ledger import RunLedger
from src.research.validation.regime import LABEL_COL, tag_regimes
from src.research.validation.splitter import WalkForwardSplitter

SYMBOL = "BTC/USDT:USDT"
CAMPAIGN = "R3_multitask"
LABEL = LabelParams(estimator="atr", window=96, x=3.0, horizon=24)
SPLIT = dict(train_min=8760, val_size=2160)
MLP_SEEDS = (0, 1, 2, 3, 4)

# 평가할 config: (이름, 멀티태스크?, λ, 정규화). 단일태스크-z = R1 기준(reach=None).
CONFIGS = [
    ("single-task/z", False, 0.0, "z"),
    ("multitask λ0.1/z", True, 0.1, "z"),
    ("multitask λ0.3/z", True, 0.3, "z"),
    ("single-task/robust", False, 0.0, "robust"),
    ("multitask λ0.1/robust", True, 0.1, "robust"),
]
_NORM = {"z": ZScoreNormalizer, "robust": RobustScaler}


@dataclass
class CfgEval:
    name: str
    ll: float
    mcc: float
    ba: float
    ll_spread: tuple


@dataclass
class R3Result:
    evals: list = field(default_factory=list)
    prior_ll: float = 0.0
    uniform_ll: float = 0.0
    baseline_ll: float = 0.0   # single-task/z (R1 기준)
    best: str = ""
    n_comparisons: int = 0


def _load_base_with_reach(candle_dir: str):
    """df(1h)·y(라벨)·reach(도달시간 타깃)·regime_tags. reach = first_touch/N, expire→NaN(검열)."""
    df = load_audited(SYMBOL, "1h", candle_dir)
    bl = compute_triple_barrier(df, LABEL)
    y = bl.labels
    N = LABEL.horizon
    reach = (bl.first_touch.astype(float) / N).where(y != "expire")  # expire·라벨NaN → NaN
    # sanity: reach 는 up/down 에서만 유효
    assert reach[y == "expire"].isna().all()
    assert reach[y.isin(["up", "down"])].notna().all()

    df1d = resample_ohlcv(df, "1h", "1d")           # I-001: 감사-clean 파생
    audit_continuity(df1d, "1d").raise_if_corrupt()
    reg = tag_regimes(df1d)[LABEL_COL]
    tags = forward_fill_completed(reg, df.index, "1d")
    if isinstance(tags, pd.DataFrame):
        tags = tags.iloc[:, 0]
    return df, y, reach, tags.reindex(df.index)


def _summ(res) -> tuple:
    s = res.report.summary
    return (float(s.loc["log_loss", "median"]), float(s.loc["mcc", "median"]),
            float(s.loc["balanced_accuracy", "median"]))


def run_r3(candle_dir: str = "data/candles", ledger_path: str | None = None,
           mlp_seeds=MLP_SEEDS, configs=None, split: dict | None = None):
    """R3 멀티태스크·정규화 평가 → R3Result. 1h 부재 시 None."""
    if not os.path.exists(os.path.join(candle_dir, csv_filename(SYMBOL, "1h"))):
        return None
    configs = configs if configs is not None else CONFIGS
    df, y, reach, regime_tags = _load_base_with_reach(candle_dir)
    X = build_features(df)                            # R2 확정 기본창
    sp = WalkForwardSplitter(label_horizon=LABEL.horizon, **(split or SPLIT))
    labels = list(LABEL_CLASSES)

    def _wf(factory, seed, norm):
        return run_walk_forward(X, y, factory, sp, regime_tags=regime_tags, labels=labels,
                                normalizer_factory=norm, seed=seed)

    prior_ll = _summ(_wf(PriorBaseline, None, ZScoreNormalizer))[0]
    uniform_ll = _summ(_wf(UniformBaseline, None, ZScoreNormalizer))[0]

    led = RunLedger(ledger_path) if ledger_path else None
    if led and led.budget(CAMPAIGN) is None:
        led.register_budget(CAMPAIGN, 12)

    evals = []
    for name, mt, lam, norm_key in configs:
        norm = _NORM[norm_key]
        rseach = reach if mt else None
        summ = []
        for s in mlp_seeds:
            r = _wf(lambda: SmallMLP(seed=s, params={"lam": lam}, reach=rseach), s, norm)
            summ.append(_summ(r))
        lls = [a[0] for a in summ]
        ev = CfgEval(name, float(np.median(lls)), float(np.median([a[1] for a in summ])),
                     float(np.median([a[2] for a in summ])), (min(lls), max(lls)))
        evals.append(ev)
        if led:
            led.record({"config": name, "lam": lam, "norm": norm_key},
                       {"ll": ev.ll, "mcc": ev.mcc}, campaign=CAMPAIGN, is_comparison=True)

    baseline_ll = next(e.ll for e in evals if e.name == "single-task/z")
    best = min(evals, key=lambda e: e.ll).name
    return R3Result(evals=evals, prior_ll=prior_ll, uniform_ll=uniform_ll,
                    baseline_ll=baseline_ll, best=best, n_comparisons=len(evals))


def _print_report(r: R3Result) -> None:
    print("=" * 66)
    print(f"R3 멀티태스크·정규화 — 비교 {r.n_comparisons}판 | Prior={r.prior_ll:.4f} Uniform={r.uniform_ll:.4f}")
    print(f"  단일태스크/z 기준(=R1): ll={r.baseline_ll:.4f}")
    print("-" * 66)
    print(f"  {'config':24} {'ll':>8} {'Δ vs R1':>8} {'mcc':>7} {'ba':>7}")
    for e in r.evals:
        print(f"  {e.name:24} {e.ll:8.4f} {e.ll - r.baseline_ll:+8.4f} {e.mcc:7.3f} {e.ba:7.3f}")
    print("-" * 66)
    print(f"  최소 ll config: {r.best}")
    print("=" * 66)


if __name__ == "__main__":
    ledger = os.path.join("data", "research", "r3_ledger.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    result = run_r3(ledger_path=ledger)
    if result is None:
        print("SKIP: 1h 캔들 부재.")
    else:
        _print_report(result)
