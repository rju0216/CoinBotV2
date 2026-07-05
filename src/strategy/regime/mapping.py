"""상태 → Contract 매핑 (D4-2c).

HMM 이 발견한 K개 상태(번호만, 라벨 없음)를 contract 의 type/direction 으로
**규칙으로 자동 번역**한다 (스펙 §2.2 — 사람이 라벨 안 박음).

  - 매핑표(상태→type,direction)는 train 에서 **offline 산출·고정** (artifact 동행).
    상태 특성화는 **raw 로그수익률**의 per-state(smoothed γ 가중) 평균·std 로 한다
    (HMM 은 z-feature 로 적합하지만 해석은 raw 단위. 스펙 §2.1).
  - type = |mean_return| / std_return vs τ (구조적 변동성, 스펙 §2.4).
  - 추론: filtered γ 를 contract 별로 집계 → 최대 질량 Contract (confidence=γ 합).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.strategy.regime.contract import (
    Contract,
    RegimeDirection,
    RegimeType,
)


@dataclass(frozen=True)
class StateStats:
    """상태별 RAW 로그수익률 통계 (특성화·디버그용)."""

    mean_return: float
    std_return: float


@dataclass(frozen=True)
class StateMapping:
    """K개 상태 각각의 (type, direction) 고정 매핑 + 통계 (artifact 동행)."""

    types: tuple[RegimeType, ...]
    directions: tuple[RegimeDirection, ...]
    tau: float
    stats: tuple[StateStats, ...]

    @classmethod
    def fit(
        cls,
        raw_log_returns: np.ndarray,
        gamma: np.ndarray,
        tau: float,
        enable_range: bool = True,
    ) -> "StateMapping":
        """train RAW 로그수익률(1D) + 상태 책임도 γ(smoothed, T×K) → 매핑표.

        raw_log_returns 와 gamma 는 **같은 행으로 정렬**되어 있어야 한다(호출자 책임).
        per-state: γ-가중 평균·std → type=|μ|/std vs τ, direction=부호(trend)/none(비trend).

        enable_range: 비trend(|μ|/σ≤τ) 상태를 어떻게 볼지.
          - True(기본, gen2/기존): RANGE 로 매핑 → RangeLogic 이 평균회귀 매매.
          - False(gen1, O-7 (가)): NONE(무매매/관망) 으로 매핑 → 추세-단독. 저방향성
            구간은 매매 안 함. 매핑만 바뀌고 엔진·플러그인은 무변경(NONE 은 no-op).
        """
        r = np.asarray(raw_log_returns, dtype=float)
        g = np.asarray(gamma, dtype=float)
        if r.shape[0] != g.shape[0]:
            raise ValueError("raw_log_returns 와 gamma 의 행 수가 다름 (정렬 필요)")
        mask = ~np.isnan(r)  # 방어적 (호출자가 정렬·dropna 했다면 전부 True)
        r, g = r[mask], g[mask]

        types: list[RegimeType] = []
        directions: list[RegimeDirection] = []
        stats: list[StateStats] = []
        for k in range(g.shape[1]):
            w = g[:, k]
            Wk = float(w.sum())
            if Wk < 1e-8:
                # 할당 거의 없는 상태 → range/none (보수: 그 레짐은 매매 안 함에 가깝게)
                mean_k, std_k = 0.0, 1.0
            else:
                mean_k = float((w * r).sum() / Wk)
                var_k = float((w * (r - mean_k) ** 2).sum() / Wk)
                std_k = float(np.sqrt(max(var_k, 1e-24)))
            ratio = abs(mean_k) / std_k if std_k > 0 else 0.0
            if ratio > tau:
                t = RegimeType.TREND
                d = RegimeDirection.LONG if mean_k > 0 else RegimeDirection.SHORT
            else:
                t = RegimeType.RANGE if enable_range else RegimeType.NONE
                d = RegimeDirection.NONE
            types.append(t)
            directions.append(d)
            stats.append(StateStats(mean_return=mean_k, std_return=std_k))
        return cls(tuple(types), tuple(directions), float(tau), tuple(stats))

    def aggregate(self, filtered_gamma: np.ndarray, volatility: float) -> Contract:
        """추론: filtered γ(K,) 를 (type,direction) 별로 집계 → 최대 질량 Contract.

        confidence = 채택된 contract 에 매핑된 상태들의 filtered γ **합** (스펙 §2.2,
        다중상태→동일 contract 대응). volatility = ATR(24), 외부 주입(가격 단위).
        """
        g = np.asarray(filtered_gamma, dtype=float)
        if g.shape[0] != len(self.types):
            raise ValueError("filtered_gamma 길이가 상태 수와 다름")
        agg: dict[tuple[RegimeType, RegimeDirection], float] = {}
        for k in range(len(g)):
            key = (self.types[k], self.directions[k])
            agg[key] = agg.get(key, 0.0) + float(g[k])
        (best_type, best_dir), conf = max(agg.items(), key=lambda kv: kv[1])
        return Contract(
            type=best_type,
            direction=best_dir,
            confidence=conf,
            volatility=float(volatility),
        )

    def covered_combos(self) -> set[tuple[RegimeType, RegimeDirection]]:
        """매핑이 커버하는 (type, direction) 조합 (커버리지 진단용)."""
        return {
            (self.types[k], self.directions[k]) for k in range(len(self.types))
        }

    def to_dict(self) -> dict:
        return {
            "types": [t.value for t in self.types],
            "directions": [d.value for d in self.directions],
            "tau": self.tau,
            "stats": [
                {"mean_return": s.mean_return, "std_return": s.std_return}
                for s in self.stats
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateMapping":
        return cls(
            types=tuple(RegimeType(t) for t in d["types"]),
            directions=tuple(RegimeDirection(x) for x in d["directions"]),
            tau=float(d["tau"]),
            stats=tuple(
                StateStats(s["mean_return"], s["std_return"]) for s in d["stats"]
            ),
        )
