"""라벨 분포 게이트 — 층1 학습가능성 + 층3 국면일관성 (설계 §4, MasterPlan R0).

**성과를 일절 안 본다** (설계 접근 B): 라벨을 성과로 튜닝하면 라벨을 성과에 맞춰
조작하는 과적합. 그래서 배리어 파라미터(estimator·창·x·N) 선별은 오직 **분포 건강도**.

- **층1 (학습가능성 필터, 통과/탈락)**: 방향(up+down) vs 만료(expire)가 **어느 쪽도
  극단(대략 80%대 후반↑) 아니고**, **소수 클래스가 각 학습 창에서 최소 절대수** 확보.
  정밀 경계는 의도적으로 느슨(명백한 극단만 쳐냄).
- **층2 폐기** (상하 대칭성·노이즈 지속성·도달시간 형태 — 2층·검증에 맡김).
- **층3 (강건성 선택)**: 같은 파라미터가 불장·베어·횡보 넘나들며 층1(극단 회피) 유지하나.
  국면 태그로 슬라이스(**분석 전용 방화벽** — regime 은 입력 아님). 너무 작은 국면은
  표본 부족이라 제외(설계 하한 주의).

층2 는 폐기라 구현하지 않는다(장식금지).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.research.labeling.triple_barrier import LABEL_CLASSES


@dataclass(frozen=True)
class DistributionThresholds:
    """느슨한 문턱 (명백한 극단만 쳐냄, 설계 §4 층1). 데이터가 정함 — 과튜닝 금지."""

    max_side_fraction: float = 0.87    # 방향/만료 어느 쪽도 이 이상이면 극단 → 탈락
    min_fold_minority: int = 100       # 각 학습창 3-class 소수 클래스 최소 절대수
    min_regime_samples: int = 200      # 층3: 이만큼 표본 있는 국면만 일관성 판정 대상


@dataclass
class Layer1Result:
    passed: bool
    global_fractions: pd.Series          # 클래스별 비율(해소분)
    directional_fraction: float          # up+down
    expire_fraction: float
    worst_fold_minority: int             # 전 폴드 중 최소 소수클래스 수
    per_fold_minority: pd.Series         # fold_id → 소수클래스 수
    reasons: list[str] = field(default_factory=list)


@dataclass
class Layer3Result:
    consistent: bool
    per_regime: pd.DataFrame             # index=국면, cols=[up,down,expire,n,directional,extreme_ok]
    evaluated_regimes: list              # 표본 충분해 판정한 국면
    skipped_regimes: list                # 표본 부족으로 제외
    violating_regimes: list              # 극단(층1 위배) 국면


def _class_counts(labels: pd.Series) -> pd.Series:
    """3-class 카운트 (누락 클래스는 0). 해소분(dropna)만."""
    counts = labels.dropna().value_counts()
    return pd.Series({c: int(counts.get(c, 0)) for c in LABEL_CLASSES})


def evaluate_layer1(
    labels: pd.Series,
    splitter,
    thresholds: DistributionThresholds | None = None,
) -> Layer1Result:
    """층1: 방향/만료 극단 회피 + 각 학습창 소수클래스 최소 절대수.

    Args:
        labels: 삼중배리어 라벨(X.index 정렬, NaN=미해소). 해소분만 집계.
        splitter: WalkForwardSplitter — 각 폴드 **학습창**의 소수클래스 수를 잰다.
    """
    t = thresholds or DistributionThresholds()
    counts = _class_counts(labels)
    total = int(counts.sum())
    fractions = counts / total if total else counts.astype(float)
    expire_frac = float(fractions.get("expire", 0.0))
    directional_frac = float(fractions.get("up", 0.0) + fractions.get("down", 0.0))

    # 폴드별 학습창 소수클래스 절대수 (3-class 최솟값)
    per_fold = {}
    for fold in splitter.split(labels.index):
        tr_counts = _class_counts(labels.loc[fold.train_index])
        per_fold[fold.fold_id] = int(tr_counts.min())
    per_fold_s = pd.Series(per_fold, name="minority")
    per_fold_s.index.name = "fold_id"
    worst = int(per_fold_s.min()) if len(per_fold_s) else 0

    reasons = []
    if total == 0:
        reasons.append("해소 라벨 0개")
    if expire_frac >= t.max_side_fraction:
        reasons.append(f"만료 극단 {expire_frac:.3f} ≥ {t.max_side_fraction}")
    if directional_frac >= t.max_side_fraction:
        reasons.append(f"방향 극단 {directional_frac:.3f} ≥ {t.max_side_fraction}")
    if worst < t.min_fold_minority:
        reasons.append(f"학습창 소수클래스 {worst} < {t.min_fold_minority}")

    return Layer1Result(
        passed=not reasons,
        global_fractions=fractions,
        directional_fraction=directional_frac,
        expire_fraction=expire_frac,
        worst_fold_minority=worst,
        per_fold_minority=per_fold_s,
        reasons=reasons,
    )


def evaluate_layer3(
    labels: pd.Series,
    regime_tags: pd.Series,
    thresholds: DistributionThresholds | None = None,
) -> Layer3Result:
    """층3: 국면별로 층1의 극단 회피가 유지되나 (국면 일관성).

    Args:
        labels: 삼중배리어 라벨(X.index 정렬).
        regime_tags: **모델 TF 로 정렬된** 국면 태그(예: 1d→1h forward_fill). 분석 전용
            슬라이싱 축 — 라벨 입력이 아니다(방화벽). NaN 국면(워밍업)은 제외.
    """
    t = thresholds or DistributionThresholds()
    tags = regime_tags.reindex(labels.index)
    rows = {}
    evaluated, skipped, violating = [], [], []

    for reg, idx in labels.groupby(tags).groups.items():
        counts = _class_counts(labels.loc[idx])
        n = int(counts.sum())
        frac = counts / n if n else counts.astype(float)
        directional = float(frac.get("up", 0.0) + frac.get("down", 0.0))
        expire = float(frac.get("expire", 0.0))
        extreme_ok = (expire < t.max_side_fraction) and (directional < t.max_side_fraction)
        rows[reg] = {
            "up": float(frac.get("up", 0.0)),
            "down": float(frac.get("down", 0.0)),
            "expire": expire,
            "n": n,
            "directional": directional,
            "extreme_ok": bool(extreme_ok),
        }
        if n < t.min_regime_samples:
            skipped.append(reg)
        else:
            evaluated.append(reg)
            if not extreme_ok:
                violating.append(reg)

    per_regime = pd.DataFrame.from_dict(rows, orient="index")
    if len(per_regime):
        per_regime.index.name = "regime"
        per_regime = per_regime.sort_index()
    # 일관성: 판정 대상(표본 충분) 국면이 하나라도 있고, 그중 위배 국면이 없음
    consistent = bool(evaluated) and not violating
    return Layer3Result(
        consistent=consistent,
        per_regime=per_regime,
        evaluated_regimes=evaluated,
        skipped_regimes=skipped,
        violating_regimes=violating,
    )
