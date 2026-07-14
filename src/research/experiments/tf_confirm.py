"""R4.1 15m 확인 — pick-and-confirm 의 "확인" (MasterPlan §12-6·다음단계, F-2).

R4.1 단독TF 관문1에서 **15m 이 1h 보다 강엣지**(ll-margin +0.045)로 나왔으나, 이는
시도 TF 중 "최고"라 **winner's curse** 위험이 있다(F-2). 고른 뒤 **확인**해야 채택한다.
두 축으로 확인한다 — 성과로 고르는 게 아니라, 이미 고른 15m 이 **진짜(전반 유지)인가**:

**확인축① 국면일관성** (D-006 등급형 진단, "붕괴·부호반전 시에만 킬"):
  15m 엣지가 6국면 전반에 유지되나. per-regime 마진(Prior_ll − MLP_ll)을 보되, **노이즈성
  소폭 열세는 킬 아님** — seed 과반이 열세인 "진짜 부호반전"만 붕괴로 센다(자기보정, 매직상수
  없음, R1 "임의상수 제거" 철학 계승). 1h 자신도 일관성 0.91 이고 R2.4 에서 한 국면 소폭
  악화(+0.0032)는 킬 아니었다 — 그 선례와 정합.

**확인축② 라벨-강건성** (F-1 은 **선별**에만 적용, 여기는 **확인**이라 넓힐수록 강함):
  15m 엣지가 6h 라벨에만 있나 — **분포로만 사전선정**한 여러 라벨에서 전부/과반 유지되나.
  라벨 집합은 성과 보기 전 확정(F-1: 분포 선정, 성과 피킹 차단), 개수는 ledger 등록.

두 축 **합산** 판정(독립 자동킬 아님). 판정규칙은 **실행 전 이 파일에 박제 = 사전등록**
(``REGIME_CONSISTENCY_RULE``·``LABEL_ROBUSTNESS_RULE``, 사후합리화 방지).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.research.experiments.tf_expansion import TFGate1Result, run_tf_gate1
from src.research.labeling.distribution import DistributionThresholds
from src.research.labeling.triple_barrier import LabelParams

# ---- 사전등록 판정규칙 (실행 전 잠금) ----
REGIME_CONSISTENCY_RULE = (
    "확인축① 국면일관성 (D-006 등급형, 붕괴·부호반전만 킬): "
    "판정대상 = 표본충분 국면(n ≥ min_regime_samples=200). "
    "각 국면 margin = Prior_ll − MLP_median_ll (양수=엣지 유지). "
    "'진짜 부호반전' = [margin < 0] ∧ [5 seed 과반(>50%)이 그 국면에서 Prior 보다 나쁨] "
    "— 노이즈성 소폭 음마진(seed 갈림)은 반전 아님(자기보정). "
    "CONFIRM = 진짜 부호반전 0개(노이즈성 음마진은 허용·보고). "
    "KILL(붕괴) = 진짜 부호반전 ≥2 국면, 또는 어느 한 국면의 |margin| ≥ 전역 ll-margin "
    "(그 국면 하나가 전역 엣지를 상쇄 = 국소 붕괴). "
    "FLAG = 진짜 부호반전 정확히 1개이며 국소붕괴 아님 → 자동킬 아님, 축②와 합산 판정. "
    "margin 크기·표본수 n 은 등급형 보고(게이트 아님)."
)
LABEL_ROBUSTNESS_RULE = (
    "확인축② 라벨-강건성 (F-1 은 선별에만, 여기는 확인): "
    "라벨 집합 = 분포게이트 통과분에서 성과 보기 전 사전선정(지평·창 스팬). "
    "라벨별 edge_holds = [gate1 verdict == PASS] ∧ [ll_margin > 0]. "
    "ROBUST = 전 라벨 edge_holds (엄격 기본). "
    "PARTIAL = 과반이나 전부는 아님 → 축①과 합산 판정. "
    "FRAGILE = 과반 미만 edge_holds → 15m 엣지가 라벨 특이적(기각 신호). "
    "통과기준(전부/과반)은 집합 확정 시 사전선언, 개수는 ledger 등록."
)

# 축① 판정대상 국면 표본하한 (층3 재사용, 매직상수 회피)
MIN_REGIME_SAMPLES = DistributionThresholds().min_regime_samples


@dataclass
class RegimeConsistencyVerdict:
    verdict: str                  # CONFIRM | FLAG | KILL
    reason: str
    genuine_reversals: list       # 진짜 부호반전 국면명
    evaluated: list               # 판정대상(표본충분) 국면명
    table: pd.DataFrame           # 등급형 보고 (전 국면 margin·n·reversal_seed_frac)


def classify_regime_consistency(
    per_regime: pd.DataFrame, global_ll_margin: float,
    min_n: int = MIN_REGIME_SAMPLES,
) -> RegimeConsistencyVerdict:
    """확인축① 사전등록 규칙(``REGIME_CONSISTENCY_RULE``) → 판정. per_regime =
    ``run_tf_gate1(...).per_regime`` (cols: n·prior_ll·mlp_ll_median·margin·reversal_seed_frac)."""
    ev = per_regime[per_regime["n"] >= min_n]
    reversal = ev[(ev["margin"] < 0) & (ev["reversal_seed_frac"] > 0.5)]
    collapse = reversal[reversal["margin"].abs() >= global_ll_margin]
    names = list(reversal.index)
    evaluated = list(ev.index)
    if len(reversal) >= 2 or len(collapse) >= 1:
        why = (f"국소붕괴 {list(collapse.index)}" if len(collapse)
               else f"진짜 부호반전 {len(reversal)}국면 {names}")
        return RegimeConsistencyVerdict("KILL", f"붕괴: {why} → 15m 국면 취약", names,
                                        evaluated, per_regime)
    if len(reversal) == 1:
        return RegimeConsistencyVerdict(
            "FLAG", f"진짜 부호반전 1국면 {names}(국소붕괴 아님) → 축②와 합산", names,
            evaluated, per_regime)
    return RegimeConsistencyVerdict("CONFIRM", f"진짜 부호반전 0(판정 {len(evaluated)}국면 전반 유지)",
                                    names, evaluated, per_regime)


@dataclass
class LabelRobustnessVerdict:
    verdict: str                  # ROBUST | PARTIAL | FRAGILE
    reason: str
    holds: list                   # edge_holds 라벨명
    total: list                   # 판정 라벨명 전체
    require_all: bool             # 사전선언 통과기준 (전부/과반)


def classify_label_robustness(
    results: list[tuple[str, TFGate1Result]], require_all: bool = True,
) -> LabelRobustnessVerdict:
    """확인축② 사전등록 규칙(``LABEL_ROBUSTNESS_RULE``) → 판정. results = [(라벨명, gate1결과)].
    require_all 은 집합 확정 시 사전선언한 통과기준(전부 True / 과반 False)."""
    holds = [name for name, r in results if r is not None
             and r.verdict == "PASS" and r.ll_margin > 0]
    total = [name for name, _ in results]
    n_hold, n_tot = len(holds), len(total)
    if n_tot == 0:
        return LabelRobustnessVerdict("FRAGILE", "판정 라벨 0", holds, total, require_all)
    if n_hold == n_tot:
        return LabelRobustnessVerdict("ROBUST", f"전 라벨 유지 ({n_hold}/{n_tot})",
                                      holds, total, require_all)
    passed = (n_hold == n_tot) if require_all else (n_hold > n_tot / 2)
    if passed:                     # 과반기준에서 과반 충족
        return LabelRobustnessVerdict("PARTIAL", f"과반 유지 ({n_hold}/{n_tot}) → 축①과 합산",
                                      holds, total, require_all)
    if n_hold > n_tot / 2:         # 엄격기준인데 과반만
        return LabelRobustnessVerdict("PARTIAL", f"과반만 유지 ({n_hold}/{n_tot}, 엄격기준 미달)"
                                      " → 축①과 합산", holds, total, require_all)
    return LabelRobustnessVerdict("FRAGILE", f"과반 미만 ({n_hold}/{n_tot}) → 라벨 특이적",
                                  holds, total, require_all)


def combine(regime: RegimeConsistencyVerdict,
            label: LabelRobustnessVerdict) -> tuple[str, str]:
    """두 축 합산 최종 판정. KILL/FRAGILE 은 기각, 둘 다 최상이면 확정, 그 사이는 HOLD."""
    if regime.verdict == "KILL" or label.verdict == "FRAGILE":
        return "REJECT", f"기각 — 국면:{regime.verdict} · 라벨:{label.verdict} → 1h 로 관문2"
    if regime.verdict == "CONFIRM" and label.verdict == "ROBUST":
        return "CONFIRM", "15m 확정 — 국면 전반 유지 + 라벨 강건 → 주 엣지 후보"
    return "HOLD", (f"경계 — 국면:{regime.verdict} · 라벨:{label.verdict} "
                    "→ 합산 검토(FLAG/PARTIAL 정도·관문2 비용 함께 판단)")


@dataclass
class Confirm15mResult:
    regime: RegimeConsistencyVerdict
    label: LabelRobustnessVerdict
    final_verdict: str
    final_reason: str
    primary: TFGate1Result
    alternates: list              # [(라벨명, TFGate1Result)]


def run_15m_confirm(
    primary_label: LabelParams,
    alternate_labels: list[tuple[str, LabelParams]],
    require_all: bool = True,
    candle_dir: str = "data/candles",
    ledger_path: str | None = None,
    df: pd.DataFrame | None = None,
) -> Confirm15mResult | None:
    """15m 확인 실행: 주라벨 gate1(→축①) + 대체라벨들 gate1(→축②) → 합산.

    라벨 집합은 **분포로만 사전선정**해 넘긴다(성과 피킹 차단, F-1). require_all =
    사전선언 통과기준. 15m 캐시 부재 시 None(skip-guard)."""
    primary = run_tf_gate1("15m", candle_dir=candle_dir, ledger_path=ledger_path,
                           label=primary_label, df=df)
    if primary is None:
        return None
    regime = classify_regime_consistency(primary.per_regime, primary.ll_margin)

    alt_results = []
    for name, lab in alternate_labels:
        r = run_tf_gate1("15m", candle_dir=candle_dir, ledger_path=ledger_path,
                         label=lab, df=df)
        alt_results.append((name, r))
    label_v = classify_label_robustness(alt_results, require_all=require_all)

    final_verdict, final_reason = combine(regime, label_v)
    return Confirm15mResult(regime=regime, label=label_v, final_verdict=final_verdict,
                            final_reason=final_reason, primary=primary, alternates=alt_results)


def _print_confirm(r: Confirm15mResult) -> None:
    print("=" * 68)
    print(f"[15m 확인] 최종: {r.final_verdict} — {r.final_reason}")
    print("-" * 68)
    print(f"확인축① 국면일관성: {r.regime.verdict} — {r.regime.reason}")
    print(f"  {'regime':16} {'n':>7} {'prior_ll':>9} {'mlp_ll':>9} {'margin':>9} {'rev_sd':>7}")
    for reg, row in r.regime.table.iterrows():
        mark = " ←반전" if reg in r.regime.genuine_reversals else (
            "" if row["n"] >= MIN_REGIME_SAMPLES else " (소표본제외)")
        print(f"  {str(reg):16} {int(row['n']):7d} {row['prior_ll']:9.4f} "
              f"{row['mlp_ll_median']:9.4f} {row['margin']:+9.4f} "
              f"{row['reversal_seed_frac']:7.2f}{mark}")
    print("-" * 68)
    print(f"확인축② 라벨강건성: {r.label.verdict} — {r.label.reason} "
          f"(기준={'전부' if r.label.require_all else '과반'})")
    for name, res in r.alternates:
        if res is None:
            print(f"  {name:24} SKIP(데이터 부재)")
            continue
        hold = "HOLD" if (res.verdict == "PASS" and res.ll_margin > 0) else "miss"
        print(f"  {name:24} verdict={res.verdict:8} ll-margin={res.ll_margin:+.4f} [{hold}]")
    print("=" * 68)
