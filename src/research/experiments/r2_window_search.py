"""R2 창 순차탐색 — 피처 창 coordinate descent (MasterPlan §5 R2, 설계 §7).

R1(관문1 PASS) 위에서 **피처 창**을 좌표하강으로 탐색해 예측력을 높인다. 1h·확정라벨
·단일태스크 고정. **KF Q/R/dof·정규화·멀티태스크는 R3**(결정1-가). 규율(F-1):

- **선택 = MLP(5seed) seed-중앙값 log_loss 최소.** 각 config = 1 비교(사전등록 예산).
- **트리는 관찰용**(`is_comparison=False`) — "창=정보량, 정보량은 모델무관" 가설을
  MLP 랭킹과 대조 검증(가설 성립시 R3/R4 트리 프록시 정당화). **선택을 바꾸지 않음.**
- **Prior/Uniform 은 config 무관**(X 무시, y만) → 1회 참조선.
- coordinate descent: 한 피처 창을 스윕(나머지 고정) → MLP 최소 채택 → 다음. 1~2 사이클.
  config 캐시로 재방문 중복 제거.

데이터 부재 시 ``run_r2`` 는 None(skip-guard). 실행: ``python -m ...r2_window_search``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.research.data.audit import audit_continuity
from src.research.data.loader import csv_filename, load_audited, resample_ohlcv
from src.research.features.build import DEFAULT_PARAMS, build_features
from src.research.labeling.triple_barrier import LABEL_CLASSES, LabelParams, compute_triple_barrier
from src.research.models import SmallMLP, TreeBench
from src.research.normalize import ZScoreNormalizer
from src.research.validation.baselines import PriorBaseline, UniformBaseline
from src.research.validation.harness import run_walk_forward
from src.research.validation.ledger import RunLedger
from src.research.validation.regime import LABEL_COL, forward_fill_completed, tag_regimes
from src.research.validation.splitter import WalkForwardSplitter

SYMBOL = "BTC/USDT:USDT"
CAMPAIGN = "R2_window_search"
LABEL = LabelParams(estimator="atr", window=96, x=3.0, horizon=24)
SPLIT = dict(train_min=8760, val_size=2160)
MLP_SEEDS = (0, 1, 2, 3, 4)

# 탐색 차원: (피처키, 파라미터키, 후보 그리드). 초기값 = R1 기본창(DEFAULT_PARAMS).
# KF·close_position 제외(창 아님/R3). 1h 봉 기준 시간척도(24=1d·48=2d·168=1w·336=2w).
SEARCH_DIMS: list[tuple[str, str, list]] = [
    ("er", "window", [24, 48, 96, 168, 336]),
    ("hurst", "window", [64, 128, 256]),
    ("vol_change", "window", [24, 48, 96, 168]),
    ("vol_change", "lag", [12, 24, 48]),
    ("vol_change", "estimator", ["atr", "yz"]),
    ("relative_volume", "window", [24, 48, 96, 168]),
    ("semi_dev", "window", [24, 48, 96, 168]),
]


@dataclass
class ConfigEval:
    mlp_ll: float          # seed-중앙값 fold-중앙값 log_loss (선택 목적함수)
    mlp_mcc: float
    mlp_ba: float
    mlp_ll_spread: tuple   # (min,max) over seeds
    tree_ll: float         # 참고
    tree_mcc: float
    tree_ba: float


@dataclass
class R2Result:
    selected: dict                 # 최종 선택 파라미터
    default_eval: ConfigEval       # R1 기본창 기준
    selected_eval: ConfigEval
    prior_ll: float
    uniform_ll: float
    history: list = field(default_factory=list)   # 스윕별 곡선
    rank_agreement: list = field(default_factory=list)  # 트리 vs MLP 순위상관
    n_comparisons: int = 0


def _load_base(candle_dir: str):
    """param 무관 입력: df(1h) · y(확정라벨) · regime_tags(파생 1d ff). 1회 로드."""
    df = load_audited(SYMBOL, "1h", candle_dir)
    y = compute_triple_barrier(df, LABEL).labels
    df1d = resample_ohlcv(df, "1h", "1d")           # I-001: 감사-clean 1h→1d 파생
    audit_continuity(df1d, "1d").raise_if_corrupt()
    reg = tag_regimes(df1d)[LABEL_COL]
    tags = forward_fill_completed(reg, df.index, "1d")
    if isinstance(tags, pd.DataFrame):
        tags = tags.iloc[:, 0]
    return df, y, tags.reindex(df.index)


def _summ(res) -> tuple:
    s = res.report.summary
    return (float(s.loc["log_loss", "median"]),
            float(s.loc["mcc", "median"]),
            float(s.loc["balanced_accuracy", "median"]))


def _key(params: dict) -> tuple:
    return tuple(sorted((f, k, str(v)) for f, d in params.items() for k, v in d.items()))


def run_r2(candle_dir: str = "data/candles", ledger_path: str | None = None,
           cycles: int = 1, mlp_seeds=MLP_SEEDS, dims=None, max_comparisons: int = 60,
           split: dict | None = None):
    """R2 창 순차탐색 실행 → R2Result. 1h 부재 시 None (1d는 1h에서 파생, I-001)."""
    if not os.path.exists(os.path.join(candle_dir, csv_filename(SYMBOL, "1h"))):
        return None

    dims = dims if dims is not None else SEARCH_DIMS
    df, y, regime_tags = _load_base(candle_dir)
    sp = WalkForwardSplitter(label_horizon=LABEL.horizon, **(split or SPLIT))
    labels = list(LABEL_CLASSES)

    def _wf(factory, seed=None, X=None):
        return run_walk_forward(X, y, factory, sp, regime_tags=regime_tags, labels=labels,
                                normalizer_factory=ZScoreNormalizer, seed=seed)

    # config 무관 참조선 (X 무시): 1회
    X0 = build_features(df, {k: dict(v) for k, v in DEFAULT_PARAMS.items()})
    prior_ll = _summ(_wf(PriorBaseline, X=X0))[0]
    uniform_ll = _summ(_wf(UniformBaseline, X=X0))[0]

    led = RunLedger(ledger_path) if ledger_path else None
    if led and led.budget(CAMPAIGN) is None:
        led.register_budget(CAMPAIGN, max_comparisons)

    cache: dict = {}
    n_comp = [0]

    def evaluate(params: dict) -> ConfigEval:
        key = _key(params)
        if key in cache:
            return cache[key]
        X = build_features(df, params)
        seeds_summ = [_summ(_wf(lambda: SmallMLP(seed=s), seed=s, X=X)) for s in mlp_seeds]
        lls = [s[0] for s in seeds_summ]
        mlp_ll = float(np.median(lls))
        mlp_mcc = float(np.median([s[1] for s in seeds_summ]))
        mlp_ba = float(np.median([s[2] for s in seeds_summ]))
        t_ll, t_mcc, t_ba = _summ(_wf(lambda: TreeBench(seed=0), seed=0, X=X))
        ev = ConfigEval(mlp_ll, mlp_mcc, mlp_ba, (min(lls), max(lls)), t_ll, t_mcc, t_ba)
        cache[key] = ev
        n_comp[0] += 1
        if led:
            led.record({"params_key": list(key)}, {"mlp_ll": mlp_ll, "mlp_mcc": mlp_mcc},
                       campaign=CAMPAIGN, is_comparison=True)           # MLP = 선택 비교
            led.record({"params_key": list(key), "model": "tree"},
                       {"tree_ll": t_ll, "tree_mcc": t_mcc},
                       campaign=CAMPAIGN, is_comparison=False)          # 트리 = 관찰용
        return ev

    current = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    default_eval = evaluate(current)
    history, rank_agreement = [], []

    for cyc in range(cycles):
        for (fkey, pkey, grid) in dims:
            if n_comp[0] >= max_comparisons:
                break
            sweep = []
            for cand in grid:
                trial = copy.deepcopy(current)
                trial[fkey][pkey] = cand
                ev = evaluate(trial)
                sweep.append((cand, ev))
            best_cand = min(sweep, key=lambda r: r[1].mlp_ll)[0]
            current[fkey][pkey] = best_cand
            history.append({"cycle": cyc, "feature": fkey, "param": pkey,
                            "curve": [(c, e.mlp_ll, e.tree_ll) for c, e in sweep],
                            "selected": best_cand})
            # 트리 vs MLP 순위 상관 (가설검증): 후보들의 ll 랭킹 일치도
            if len(sweep) >= 2:
                mlp_rank = pd.Series([e.mlp_ll for _, e in sweep]).rank()
                tree_rank = pd.Series([e.tree_ll for _, e in sweep]).rank()
                rho = float(mlp_rank.corr(tree_rank, method="spearman"))
                rank_agreement.append({"feature": fkey, "param": pkey, "spearman": rho})

    selected_eval = evaluate(current)
    return R2Result(selected=current, default_eval=default_eval, selected_eval=selected_eval,
                    prior_ll=prior_ll, uniform_ll=uniform_ll, history=history,
                    rank_agreement=rank_agreement, n_comparisons=n_comp[0])


def _print_report(r: R2Result) -> None:
    print("=" * 68)
    print(f"R2 창 순차탐색 — 비교 {r.n_comparisons}판 | Prior_ll={r.prior_ll:.4f} Uniform_ll={r.uniform_ll:.4f}")
    print(f"  기본창 MLP: ll={r.default_eval.mlp_ll:.4f} mcc={r.default_eval.mlp_mcc:.3f} ba={r.default_eval.mlp_ba:.3f}")
    print(f"  선택창 MLP: ll={r.selected_eval.mlp_ll:.4f} mcc={r.selected_eval.mlp_mcc:.3f} ba={r.selected_eval.mlp_ba:.3f}")
    print(f"  개선 Δll = {r.selected_eval.mlp_ll - r.default_eval.mlp_ll:+.4f}")
    print("-" * 68)
    print("  선택 파라미터:")
    for f, d in r.selected.items():
        print(f"    {f}: {d}")
    print("-" * 68)
    print("  트리 vs MLP 순위상관(Spearman) — '정보량 모델무관' 가설검증:")
    for a in r.rank_agreement:
        print(f"    {a['feature']}.{a['param']}: rho={a['spearman']:.2f}")
    if r.rank_agreement:
        rhos = [a["spearman"] for a in r.rank_agreement if a["spearman"] == a["spearman"]]
        if rhos:
            print(f"    평균 rho = {np.mean(rhos):.2f}  (높을수록 트리 프록시 정당)")
    print("=" * 68)


if __name__ == "__main__":
    ledger = os.path.join("data", "research", "r2_ledger.jsonl")
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    result = run_r2(ledger_path=ledger)
    if result is None:
        print("SKIP: 1h/1d 캔들 부재.")
    else:
        _print_report(result)
