"""미래 누수 탐지 하네스 (MasterPlan §11 컴포넌트 C).

시점별 함수 ``fn: 시계열 → 시점별 출력(같은 index)`` 이 **그 시점까지 정보만**
쓰는지 자동 검증한다. 두 메커니즘이 상호보완한다:

- **절단 불변(truncation)**: ``fn(data[:t])`` 가 ``fn(data)[:t]`` 와 일치. 미래의
  존재가 과거 출력을 바꾸면 누수. (엔진 _slice_candles 와 같은 정신.)
- **미래 교란(perturbation)**: ``data[t:]`` 를 강하게 교란한 사본으로 재계산해도
  ``[:t]`` 불변. 길이 보존형 누수(전역 mean/quantile·bfill·centered window) 포착.

적용 대상 = **인과성을 주장하는** 함수(피처·라벨의 그 시점 정보). analysis-only
함수(예: regime_vol 의 전구간 quantile)는 의도적 비인과라 적용 대상이 아니다 —
오히려 하네스가 그 비인과를 **정확히 플래그**함을 확인해 사후 경계를 못박는 데 쓴다.

패턴은 audit(AuditReport)과 동형: 리포트 반환 + raise_if_leaked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

Frame = Union[pd.Series, pd.DataFrame]


class LeakageError(AssertionError):
    """미래 누수(인과성 위반)가 감지됐을 때."""


@dataclass
class Divergence:
    checkpoint: int      # 분기 검사 위치 t (positional)
    method: str          # "truncation" | "perturbation"
    first_bad_pos: int   # 처음으로 값이 갈린 위치 (positional)


@dataclass
class CausalityReport:
    n_checkpoints: int
    divergences: list[Divergence] = field(default_factory=list)

    @property
    def leaked(self) -> bool:
        return len(self.divergences) > 0

    def raise_if_leaked(self) -> None:
        if self.leaked:
            d = self.divergences[0]
            raise LeakageError(
                f"미래 누수 감지: method={d.method} checkpoint={d.checkpoint} "
                f"first_bad_pos={d.first_bad_pos} (총 {len(self.divergences)}건)"
            )

    def merge(self, other: CausalityReport) -> CausalityReport:
        return CausalityReport(
            self.n_checkpoints + other.n_checkpoints,
            self.divergences + other.divergences,
        )


def _to_frame(x: Frame) -> pd.DataFrame:
    return x.to_frame() if isinstance(x, pd.Series) else x


def _first_divergence(a: Frame, b: Frame, rtol: float, atol: float) -> int | None:
    """a, b 를 위치별로 비교해 처음 갈리는 positional index 반환(없으면 None).

    NaN==NaN 은 같음. 수치는 근사(rtol/atol), 그 외(object/문자열)는 정확 일치.
    """
    a = _to_frame(a).reset_index(drop=True)
    b = _to_frame(b).reset_index(drop=True)
    if list(a.columns) != list(b.columns):
        return 0
    n = min(len(a), len(b))
    a, b = a.iloc[:n], b.iloc[:n]
    for col in a.columns:
        s1, s2 = a[col], b[col]
        both_na = (s1.isna() & s2.isna()).to_numpy()
        if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
            eq = np.isclose(
                s1.to_numpy(dtype=float), s2.to_numpy(dtype=float),
                rtol=rtol, atol=atol, equal_nan=True,
            )
        else:
            eq = ((s1.astype(object) == s2.astype(object)).to_numpy()) | both_na
        bad = ~eq
        if bad.any():
            return int(np.argmax(bad))
    return None


def _default_checkpoints(n: int) -> list[int]:
    # 조밀화(앞쪽 포함) — 극단이 앞에 있어 절단이 못 잡는 경우를 교란이 보완.
    cps = sorted({int(n * f) for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)})
    return [t for t in cps if 0 < t < n]


def _perturb_future(data: Frame, t: int) -> Frame:
    """positional t 이후의 수치 값을 **극단 양/음 교대(±1e6)** 로 치환한 사본 반환.

    양방향 교란이라 미래의 min·max·mean·분위수를 모두 바꾼다 — 위로만 부풀리면
    min/저분위 기반 누수(예: ``d - d.min()``)를 놓치므로 교대 패턴을 쓴다.
    인과 함수는 미래를 안 보므로 [:t] 가 불변, 누수 함수만 [:t] 가 흔들린다.
    """
    pert = data.copy()
    n_future = len(pert) - t
    if n_future <= 0:
        return pert
    pattern = 1e6 * np.where(np.arange(n_future) % 2 == 0, 1.0, -1.0)
    if isinstance(pert, pd.Series):
        if pd.api.types.is_numeric_dtype(pert):
            pert.iloc[t:] = pattern
        return pert
    for c in pert.columns:
        if pd.api.types.is_numeric_dtype(pert[c]):
            pert.iloc[t:, pert.columns.get_loc(c)] = pattern
    return pert


def check_truncation_invariance(
    fn: Callable[[Frame], Frame],
    data: Frame,
    checkpoints: list[int] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> CausalityReport:
    full = fn(data)
    cps = checkpoints if checkpoints is not None else _default_checkpoints(len(data))
    if not cps:
        raise ValueError(f"시계열이 너무 짧아 인과성 검증 불가 (len={len(data)})")
    divs: list[Divergence] = []
    for t in cps:
        trunc_out = fn(data.iloc[:t])
        pos = _first_divergence(full.iloc[:t], trunc_out, rtol, atol)
        if pos is not None:
            divs.append(Divergence(t, "truncation", pos))
    return CausalityReport(len(cps), divs)


def check_future_leak(
    fn: Callable[[Frame], Frame],
    data: Frame,
    checkpoints: list[int] | None = None,
    perturb: Callable[[Frame, int], Frame] = _perturb_future,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> CausalityReport:
    full = fn(data)
    cps = checkpoints if checkpoints is not None else _default_checkpoints(len(data))
    if not cps:
        raise ValueError(f"시계열이 너무 짧아 인과성 검증 불가 (len={len(data)})")
    divs: list[Divergence] = []
    for t in cps:
        # 교란은 의도적 적대 입력(극단값)이라 fn 내부 수치 경고(log 음수 등)는 노이즈 →
        # 이 평가 구간만 억제. [:t] 인과 판정 자체는 영향 없음.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            pert_out = fn(perturb(data, t))
        pos = _first_divergence(full.iloc[:t], pert_out.iloc[:t], rtol, atol)
        if pos is not None:
            divs.append(Divergence(t, "perturbation", pos))
    return CausalityReport(len(cps), divs)


def assert_causal(
    fn: Callable[[Frame], Frame],
    data: Frame,
    checkpoints: list[int] | None = None,
) -> CausalityReport:
    """절단 불변 + 미래 교란 둘 다 통과해야 함. 누수 시 LeakageError."""
    report = check_truncation_invariance(fn, data, checkpoints).merge(
        check_future_leak(fn, data, checkpoints)
    )
    report.raise_if_leaked()
    return report
