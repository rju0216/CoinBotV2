"""삼중배리어 라벨 생성 (MasterPlan Phase 1 / 설계 §4·§5).

각 진입 봉 i 에서: 그 시점 변동성 σ_i(인과) 로 상단 ``close_i·(1+x·σ_i)`` / 하단
``close_i·(1-x·σ_i)`` 배리어를 긋고, **이후 N봉의 high/low 경로**로 어느 배리어를
먼저 쳤나 판정한다 → 3-class {up, down, expire}(타깃1, 설계 §5).

**인과 규율 (분리)**:
- **배리어 레벨은 인과** (σ_i·close_i 모두 i 까지 정보) → ``barrier_levels`` 에
  ``assert_causal`` 적용(test).
- **라벨 자체는 forward** (미래 경로 판정) — 절단불변 대상 아님. 대신 **유한 지평 = N**
  을 보장: labels[:t] 는 t+N 이후 교란에 불변(test). 이 N 이 splitter purge·꼬리예약
  단일출처(F-5).

**결정(사용자 확정)**:
- 동시터치(한 미래 봉이 상단·하단을 동시 터치, 봉내 선후 불명) → **NaN 제외**
  (판정불가; expire 오염·방향편향 회피. Phase 4 에서 하위 TF 인트라바 복원 여지).
- **도달시간(first_touch)** = 첫 도달까지 봉 수 ∈[1,N], expire→N (무비용 부산물,
  Phase 3 멀티태스크 타깃2 재스캔 회피). 라벨이 NaN 이면 first_touch 도 NaN.

**꼬리 예약**: 전체 N봉 지평이 확보되지 않는 마지막 N봉(및 워밍업 σ NaN)은 미해소
→ NaN. splitter 의 usable_end=n-N 과 정합(그 봉들은 train/val 어디에도 안 쓰임).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.labeling.volatility import VOL_ESTIMATORS

# 3-class 라벨 집합 (baselines/metrics 의 labels 인자로 그대로 전달)
LABEL_CLASSES = ("up", "down", "expire")
_UP, _DOWN, _EXPIRE, _AMBIG = 1, 2, 3, 4   # 내부 코드 (_AMBIG → NaN 제외)


@dataclass(frozen=True)
class LabelParams:
    estimator: str      # "atr" | "yz"
    window: int         # 변동성 창(봉)
    x: float            # 배리어 배수 (대칭). 폭 = x·σ
    horizon: int        # N = label_horizon (봉). splitter purge/꼬리예약 구동

    def __post_init__(self) -> None:
        if self.estimator not in VOL_ESTIMATORS:
            raise ValueError(f"미지원 estimator: {self.estimator} ({list(VOL_ESTIMATORS)})")
        if self.window <= 0 or self.horizon <= 0 or self.x <= 0:
            raise ValueError("window, horizon, x 는 양수여야 함")


@dataclass
class BarrierLabels:
    labels: pd.Series          # {up,down,expire}, NaN=워밍업/꼬리/동시터치
    first_touch: pd.Series     # 첫 도달 봉 수 ∈[1,N], expire→N, NaN=라벨 NaN
    upper: pd.Series           # 상단 배리어 레벨(가격) — 검수·인과 테스트용
    lower: pd.Series           # 하단 배리어 레벨(가격)
    params: LabelParams

    @property
    def label_horizon(self) -> int:
        """splitter 에 넘길 N (단일출처, F-5)."""
        return self.params.horizon

    def class_distribution(self, normalize: bool = True) -> pd.Series:
        """해소된 라벨의 클래스 분포 (NaN 제외). 층1 분포 게이트의 1차 재료."""
        counts = self.labels.value_counts(dropna=True)
        return counts / counts.sum() if normalize and counts.sum() else counts


def barrier_levels(df: pd.DataFrame, params: LabelParams) -> pd.DataFrame:
    """진입 봉별 (상단, 하단) 배리어 레벨(가격). **인과** — σ_i·close_i 모두 i 까지 정보.

    별도 함수로 분리해 ``assert_causal`` 회귀 대상이 되게 한다("배리어 폭 진입시점
    정보만", 기둥 2 / MasterPlan §2 상시검증).
    """
    sigma = VOL_ESTIMATORS[params.estimator](df, params.window)
    close = df["close"]
    upper = close * (1.0 + params.x * sigma)
    lower = close * (1.0 - params.x * sigma)
    return pd.DataFrame({"upper": upper, "lower": lower}, index=df.index)


def compute_triple_barrier(df: pd.DataFrame, params: LabelParams) -> BarrierLabels:
    """OHLCV → 삼중배리어 라벨. labels/first_touch/upper/lower(모두 df.index 정렬).

    입력은 **감사 통과 OHLCV**(``load_audited``) 전제. 스캔은 미래 봉 high/low 로
    배리어 교차를 감지하는데, 그 값에 NaN 이 있으면 ``NaN >= 상단`` 이 조용히 False 가
    돼 미터치(→expire) 로 편향된다. 이를 막기 위해 **원시 OHLC NaN 을 진입에서 하드페일**
    한다("감사 통과만 하류로"를 구성으로 강제, 규칙 16 / fresh-eyes LOW#4).
    """
    ohlc = df[["open", "high", "low", "close"]]
    if bool(ohlc.isna().to_numpy().any()):
        raise ValueError(
            "OHLC 에 NaN 존재 — 감사 통과 데이터만 라벨링 가능(load_audited 사용). "
            "미래 봉 NaN 은 배리어 미터치로 조용히 편향되므로 하드페일한다."
        )
    levels = barrier_levels(df, params)
    upper = levels["upper"].to_numpy()
    lower = levels["lower"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    N = params.horizon

    # 라벨 가능 봉: σ 유효(배리어 존재) AND 전체 N봉 지평 확보(i+N <= n-1)
    valid_barrier = ~np.isnan(upper)
    positions = np.arange(n)
    has_horizon = (positions + N) <= (n - 1)
    active = np.where(valid_barrier & has_horizon)[0]

    code = np.zeros(n, dtype=np.int8)          # 0 = 라벨 없음(NaN)
    touch = np.full(n, np.nan, dtype=float)

    if len(active):
        up_lvl = upper[active]
        dn_lvl = lower[active]
        done = np.zeros(len(active), dtype=bool)
        res_code = np.zeros(len(active), dtype=np.int8)
        res_touch = np.zeros(len(active), dtype=float)

        for k in range(1, N + 1):
            j = active + k                     # active 는 i+N<=n-1 이라 항상 in-bounds
            up_t = high[j] >= up_lvl
            dn_t = low[j] <= dn_lvl
            newly = (~done) & (up_t | dn_t)
            both = newly & up_t & dn_t
            up_only = newly & up_t & ~dn_t
            dn_only = newly & dn_t & ~up_t
            res_code[up_only] = _UP
            res_code[dn_only] = _DOWN
            res_code[both] = _AMBIG            # 동시터치 → NaN 제외(사용자 결정)
            res_touch[newly] = float(k)
            done |= newly

        # N봉 내 미터치 → expire
        res_code[~done] = _EXPIRE
        res_touch[~done] = float(N)
        code[active] = res_code
        touch[active] = res_touch

    code_to_label = {_UP: "up", _DOWN: "down", _EXPIRE: "expire"}
    labels = pd.Series(index=df.index, dtype=object)
    for c, lab in code_to_label.items():
        labels[code == c] = lab
    # _AMBIG(동시터치)·0(미해소) → NaN 유지, 그 자리 first_touch 도 NaN
    ambiguous = code == _AMBIG
    touch[ambiguous] = np.nan
    first_touch = pd.Series(touch, index=df.index, name="first_touch")

    return BarrierLabels(
        labels=labels.rename("label"),
        first_touch=first_touch,
        upper=levels["upper"],
        lower=levels["lower"],
        params=params,
    )
