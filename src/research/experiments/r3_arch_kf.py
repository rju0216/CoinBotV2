"""R3.2 구조·KF 소그리드 (MasterPlan §5 R3, 조건부 확장 — 사용자 결정).

R3.1(멀티태스크·정규화 무gain) 후 미검증 지렛대 확인: MLP **깊이/너비** + **KF Q/R/dof**
(config 하이퍼, D-025). 단일태스크·z 고정(R3.1 결과). 사전등록 그리드(F-1, 낚시 방지):

- 구조: (d2w32=기준)·d2w64·d3w32·d3w64 — 용량↑ 과적합 여부. X 불변, 모델만.
- KF: dof{2,8}·r{1e-3,1e-5} — 팻테일 강건성·측정노이즈. **X 재빌드**(kf 는 build 파라미터).

선택 = MLP 5seed seed-중앙 log_loss 최소. 기준선 뚜렷이 이기면 국면재확인(별도). 데이터
부재 시 None. 실행: ``python -m ...r3_arch_kf``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from src.research.experiments.r3_multitask import (
    LABEL,
    MLP_SEEDS,
    SPLIT,
    _load_base_with_reach,
    _summ,
)
from src.research.features.build import DEFAULT_PARAMS, build_features
from src.research.labeling.triple_barrier import LABEL_CLASSES
from src.research.models import SmallMLP
from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.ledger import RunLedger
from src.research.validation.splitter import WalkForwardSplitter

SYMBOL = "BTC/USDT:USDT"
CAMPAIGN = "R3_arch_kf"

# 사전등록 그리드: (이름, kf_override|None, model_override|None). 단일태스크·z.
CONFIGS = [
    ("base d2w32", None, None),                       # = R1 기준
    ("d2w64", None, {"trunk_width": 64}),
    ("d3w32", None, {"trunk_depth": 3}),
    ("d3w64", None, {"trunk_depth": 3, "trunk_width": 64}),
    ("KF dof2", {"dof": 2.0}, None),
    ("KF dof8", {"dof": 8.0}, None),
    ("KF r1e-3", {"r": 1e-3}, None),
    ("KF r1e-5", {"r": 1e-5}, None),
]


@dataclass
class R32Result:
    evals: list = field(default_factory=list)   # (name, ll, mcc, ba, spread)
    prior_ll: float = 0.0
    baseline_ll: float = 0.0
    best: str = ""
    n_comparisons: int = 0


def _build_key(kf_override):
    return tuple(sorted((kf_override or {}).items()))


def run_r32(candle_dir: str = "data/candles", ledger_path: str | None = None,
            mlp_seeds=MLP_SEEDS, configs=None, split: dict | None = None):
    if not os.path.exists(os.path.join(candle_dir,
                          SYMBOL.replace("/", "_").replace(":", "_") + "_1h.csv")):
        return None
    configs = configs if configs is not None else CONFIGS
    df, y, _reach, regime_tags = _load_base_with_reach(candle_dir)
    sp = WalkForwardSplitter(label_horizon=LABEL.horizon, **(split or SPLIT))
    labels = list(LABEL_CLASSES)

    x_cache: dict = {}

    def _get_X(kf_override):
        k = _build_key(kf_override)
        if k not in x_cache:
            params = {"kf": {**DEFAULT_PARAMS["kf"], **(kf_override or {})}} if kf_override else None
            x_cache[k] = build_features(df, params)
        return x_cache[k]

    def _wf(factory, seed, X):
        return run_walk_forward(X, y, factory, sp, regime_tags=regime_tags, labels=labels,
                                normalizer_factory=ZScoreNormalizer, seed=seed)

    prior_ll = _summ(_wf(PriorBaseline, None, _get_X(None)))[0]

    led = RunLedger(ledger_path) if ledger_path else None
    if led and led.budget(CAMPAIGN) is None:
        led.register_budget(CAMPAIGN, 12)

    evals = []
    for name, kf_ov, model_ov in configs:
        X = _get_X(kf_ov)
        summ = [_summ(_wf(lambda: SmallMLP(seed=s, params=(model_ov or {})), s, X))
                for s in mlp_seeds]
        lls = [a[0] for a in summ]
        ev = (name, float(np.median(lls)), float(np.median([a[1] for a in summ])),
              float(np.median([a[2] for a in summ])), (min(lls), max(lls)))
        evals.append(ev)
        if led:
            led.record({"config": name, "kf": kf_ov, "model": model_ov},
                       {"ll": ev[1], "mcc": ev[2]}, campaign=CAMPAIGN, is_comparison=True)

    baseline_ll = next(e[1] for e in evals if e[0] == "base d2w32")
    best = min(evals, key=lambda e: e[1])[0]
    return R32Result(evals=evals, prior_ll=prior_ll, baseline_ll=baseline_ll,
                     best=best, n_comparisons=len(evals))


def _print_report(r: R32Result) -> None:
    print("=" * 62)
    print(f"R3.2 구조·KF — 비교 {r.n_comparisons}판 | Prior={r.prior_ll:.4f}")
    print(f"  기준 base d2w32(=R1): ll={r.baseline_ll:.4f}")
    print("-" * 62)
    print(f"  {'config':16} {'ll':>8} {'Δ vs R1':>9} {'mcc':>7} {'ba':>7}")
    for name, ll, mcc, ba, _sp in r.evals:
        print(f"  {name:16} {ll:8.4f} {ll - r.baseline_ll:+9.4f} {mcc:7.3f} {ba:7.3f}")
    print("-" * 62)
    print(f"  최소 ll config: {r.best}")
    print("=" * 62)


if __name__ == "__main__":
    ledger = os.path.join("data", "research", "r3_arch_kf_ledger.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    result = run_r32(ledger_path=ledger)
    if result is None:
        print("SKIP: 1h 캔들 부재.")
    else:
        _print_report(result)
