"""R4.2 MTF 역할1 — 상위맥락이 15m 통계엣지를 강화하나 (관문1, 설계 §9).

15m 단독(R4.1 확정 ll-margin +0.045) baseline 에 상위TF(1h/4h) **완성봉** 피처를 concat 해
엣지가 **material 하게** 강해지나. 성과로 고르는 게 아니라 — 사전등록 임계(노이즈밴드)를
넘는지 판정. 넘으면 채택, 아니면 NO_GAIN(장식금지, 기둥5). 그 자체가 관문1 결론.

판정규칙은 실행 전 박제(사전등록, ``MTF_MATERIAL_RULE``). config·라벨·splitter·seed 는
baseline 과 **동일**, 오직 피처만 확장(효과 격리). F-1: ledger 예산=config수, **max 고르기
아닌 임계초과 채택**. 채택 후보는 per_regime 국면일관성(축① 규칙 재사용)도 통과해야.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.experiments.tf_confirm import classify_regime_consistency
from src.research.experiments.tf_expansion import (
    SYMBOL,
    TFGate1Result,
    load_tf,
    run_tf_gate1,
)
from src.research.features.mtf import build_mtf_features
from src.research.labeling.triple_barrier import LabelParams
from src.research.validation.ledger import RunLedger

# R2/R3 에서 확립된 seed 노이즈 floor (매직상수 아님 — 선례값 재사용). Δ 가 이보다 커야 material.
NOISE_FLOOR = 0.002
DECISION_TF = "15m"
PRIMARY_LABEL = LabelParams("atr", 24, 3.0, 24)          # 15m 확정 주라벨 6h/w24
HIGHER_TF_SETS = (("1h",), ("4h",), ("1h", "4h"))         # (다) 직교 2 + 조합 1

MTF_MATERIAL_RULE = (
    "R4.2 MTF 역할1 (상위맥락이 15m 엣지 강화하나, 관문1): "
    "baseline = 15m 단독(동일 라벨 6h/w24·splitter·5seed). config 마다 상위TF 완성봉 피처 concat. "
    "config = MATERIAL_GAIN iff [gate1 verdict==PASS] ∧ "
    "[ll_margin − baseline_ll_margin > NOISE_FLOOR(0.002, R2/R3 확립)]. 아니면 NO_GAIN(장식). "
    "채택(ADOPT_MTF) = MATERIAL_GAIN 중 ll_margin 최강 + 그 per_regime 국면일관성 KILL 아님. "
    "전부 NO_GAIN(또는 국면붕괴) → 15m 단독 유지(관문1 결론: 상위맥락 무익). "
    "F-1: max 고르기 아닌 임계초과 채택, ledger 예산=config수(사전등록)."
)


@dataclass
class MTFConfigResult:
    name: str                # 상위TF 조합 라벨 (예: "1h", "1h+4h")
    result: TFGate1Result
    delta_ll_margin: float   # ll_margin − baseline
    material: bool
    regime_verdict: str      # per_regime 국면일관성 (material 일 때만 판정)


@dataclass
class MTFResult:
    baseline_ll_margin: float
    configs: list            # [MTFConfigResult]
    verdict: str             # ADOPT_MTF | NO_GAIN
    adopted: str | None
    reason: str


def classify_mtf(baseline_ll_margin: float, configs: list) -> MTFResult:
    """사전등록 규칙(``MTF_MATERIAL_RULE``) → 판정. configs = [MTFConfigResult]."""
    ok = [c for c in configs if c.material and c.regime_verdict != "KILL"]
    if ok:
        best = max(ok, key=lambda c: c.result.ll_margin)
        return MTFResult(baseline_ll_margin, configs, "ADOPT_MTF", best.name,
                         f"{best.name} Δ{best.delta_ll_margin:+.4f} material + 국면 {best.regime_verdict}")
    return MTFResult(baseline_ll_margin, configs, "NO_GAIN", None,
                     "전 config NO_GAIN(또는 국면붕괴) → 상위맥락 무익, 15m 단독 유지")


def run_mtf_role1(
    baseline_ll_margin: float,
    label: LabelParams = PRIMARY_LABEL,
    higher_tf_sets=HIGHER_TF_SETS,
    candle_dir: str = "data/candles",
    mlp_seeds=(0, 1, 2, 3, 4),
    ledger_path: str | None = None,
    decision_df: pd.DataFrame | None = None,
    higher_dfs: dict | None = None,
    split: dict | None = None,
) -> MTFResult:
    """MTF 역할1 실행: 각 상위TF 집합마다 build_mtf_features → gate1 → baseline 대비 Δ 판정.

    baseline_ll_margin 은 15m 단독(동일 config) 확정값을 넘긴다(재실행 회피, 동일 seed·splitter
    라 직접 비교 가능). decision_df/higher_dfs/split 주입 시 seam 소슬라이스 구동."""
    dec = decision_df if decision_df is not None else load_tf(DECISION_TF, candle_dir)
    cache = dict(higher_dfs or {})

    def _hdf(tf):
        if tf not in cache:
            cache[tf] = load_tf(tf, candle_dir)
        return cache[tf]

    led = None
    if ledger_path:
        led = RunLedger(ledger_path)
        if led.budget("R4_mtf") is None:
            led.register_budget("R4_mtf", len(higher_tf_sets))

    configs = []
    for tfs in higher_tf_sets:
        higher = [(tf, _hdf(tf)) for tf in tfs]
        X = build_mtf_features(dec, higher)
        r = run_tf_gate1(DECISION_TF, candle_dir=candle_dir, label=label, df=dec, X=X,
                         split=split, mlp_seeds=mlp_seeds)
        delta = r.ll_margin - baseline_ll_margin
        material = (r.verdict == "PASS") and (delta > NOISE_FLOOR)
        rv = classify_regime_consistency(r.per_regime, r.ll_margin).verdict if material else "-"
        name = "+".join(tfs)
        configs.append(MTFConfigResult(name, r, delta, material, rv))
        if led:
            led.record({"mtf": name, "n_features": X.shape[1]},
                       {"ll_margin": r.ll_margin, "delta": delta, "material": float(material)},
                       campaign="R4_mtf", is_comparison=True)

    return classify_mtf(baseline_ll_margin, configs)


def _print_mtf(res: MTFResult) -> None:
    print("=" * 68)
    print("[R4.2 MTF 역할1] %s -- %s" % (res.verdict, res.reason))
    print("  baseline(15m 단독) ll-margin = %+.4f  | NOISE_FLOOR=%.4f" %
          (res.baseline_ll_margin, NOISE_FLOOR))
    print("  %-8s %-8s %9s %9s %8s %s" %
          ("mtf", "verdict", "ll-margin", "delta", "material", "regime"))
    for c in res.configs:
        print("  %-8s %-8s %+9.4f %+9.4f %8s %s" %
              (c.name, c.result.verdict, c.result.ll_margin, c.delta_ll_margin,
               str(c.material), c.regime_verdict))
    print("=" * 68)
