"""Phase 6 Step 6.3 — from-scratch 3층 진입 설계 (EV-gate + 로버스트 선별).

DumbL3 를 상속해 **청산은 그대로**(full 배리어 + N봉 만기 — 6.2 에서 조기청산 전멸이
확인된 뒤 남은 최선) 두고 **진입 게이트만 새로 설계**한다. 두 축을 플래그로 함께 싣되
서로 독립이다(한쪽만 켜서 단독 검증 가능).

**축 A — 비용인지 기대값 게이트 (ev_k)**
    기대총익률 = barrier_frac × (p_dir − p_opp)
    실비용률   = 2×(taker + slippage) + (보유시간h / 8) × funding_rate_per_8h
    진입 조건  = 기대총익률 ≥ ev_k × 실비용률

    동기(측정): 3-클래스에서 ``max(p_up, p_down)`` 는 대부분 "배리어를 치긴 하는가"
    (=1−p_expire)를 따라가고 **어느 쪽 벽인가와는 15m 에서 무상관**(corr 0.05, 4h 0.55).
    θ 게이트는 방향이 아니라 움직임을 골랐다 → 방향 마진 ``p_dir − p_opp`` 를 배리어 폭으로
    금액화해 **실비용과 직접 비교**한다. 배리어 폭·지평이 식에 들어가므로 저변동기·장기보유일수록
    문턱이 자동으로 높아진다(무튜닝 — ev_k 만이 자유값이고 의미는 "비용 대비 안전마진 배수").

**축 B — 로버스트 선별 (seed_unanimity / persistence_bars)**
    ① 시드 만장일치: s0..s{n-1} 의 방향(argmax(up,down))이 전원 일치 + 앙상블 방향과 동일.
    ② 지속성: 직전 K봉(현재 포함)의 방향이 연속 동일. **폴드 경계 갭 가드** — 아티팩트
       인덱스 간격이 결정TF 간격과 정확히 같은 구간만 "연속"으로 센다.
    가설: 확률 *크기*(θ)로 자르면 knife-edge 였으니 *합의·일관성*으로 자르면 더 튼튼한가.

**인과성**: 게이트 입력은 전부 완성봉 산출물이다 — 예측(그 봉 마감이 만든 값)·barrier_frac
(라벨과 동일 폭)·상수 비용. 지속성 플래그는 **과거 행만** 참조해 __init__ 에서 1회 벡터
계산한다(미래 행 불참조 — 회귀로 박제). ``generate_signal`` 은 엔진이 결정TF 봉마감에만
호출한다(engine ``_evaluate_bar_close``) → I-007 류 경로 없음.

**모델 재fit 0** — 얼린 OOS 예측 아티팩트 조회만(기둥3 선별↔확인 분리).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.data.historical import TF_MS
from src.strategy.plugins.dumb_l3 import DumbL3
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)

# 비용 상수 — config/default.yaml `accounting` 과 일치해야 한다(test_econ_l3 가 드리프트 박제).
DEFAULT_TAKER_FEE_PCT = 0.0005
DEFAULT_SLIPPAGE_PCT = 0.0
# F-3 사전등록 상수 펀딩(0.01%/8h). 방향 편향 방지를 위해 롱/숏 모두 **비용**으로 취급(보수적).
DEFAULT_FUNDING_RATE_PER_8H = 0.0001

_UNANIMOUS_COL = "_econ_unanimous"
_PERSIST_COL = "_econ_persist_ok"


@register_strategy
class EconL3(DumbL3):
    """경제성·로버스트니스 진입 게이트 3층 — DumbL3 청산 재사용, 진입만 재설계."""

    name = "econ_l3"

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # --- 축 A: EV 게이트 (0 = off → DumbL3 와 동일 동작) ---
        self.ev_k = float(self.params.get("ev_k", 0.0))
        self.taker_fee_pct = float(self.params.get("taker_fee_pct", DEFAULT_TAKER_FEE_PCT))
        self.slippage_pct = float(self.params.get("slippage_pct", DEFAULT_SLIPPAGE_PCT))
        self.funding_rate_per_8h = float(
            self.params.get("funding_rate_per_8h", DEFAULT_FUNDING_RATE_PER_8H)
        )
        # --- 축 B: 로버스트 선별 (기본 off) ---
        self.seed_unanimity = bool(self.params.get("seed_unanimity", False))
        self.n_seeds = int(self.params.get("n_seeds", 5))
        self.persistence_bars = int(self.params.get("persistence_bars", 0))

        # 실비용률 = 왕복 체결비용 + 지평 전체 펀딩(사전등록 상수). 진입 시점 무관 상수.
        hold_hours = self.horizon_bars * TF_MS[self.entry_timeframe] / 3_600_000.0
        self.cost_frac = (
            2.0 * (self.taker_fee_pct + self.slippage_pct)
            + (hold_hours / 8.0) * self.funding_rate_per_8h
        )

        if self.seed_unanimity:
            need = [f"s{i}_{c}" for i in range(self.n_seeds) for c in ("up", "down")]
            missing = [c for c in need if c not in self._pred.columns]
            if missing:
                raise ValueError(
                    f"아티팩트 {self.artifact_path} 에 시드 열 부재: {missing} "
                    f"(seed_unanimity=True, n_seeds={self.n_seeds})."
                )
        self._add_robustness_flags()
        logger.info(
            "EconL3: ev_k=%.2f cost=%.5f(=%.1fbp) unanimity=%s persist=%d",
            self.ev_k, self.cost_frac, self.cost_frac * 1e4,
            self.seed_unanimity, self.persistence_bars,
        )

    # ---- 로버스트 플래그 사전계산 (과거 행만 참조 — 인과) ----

    def _add_robustness_flags(self) -> None:
        """만장일치·지속성 플래그를 아티팩트 열로 추가 → generate_signal 은 O(1) 조회.

        DumbL3._lookup(tz 보정 포함)을 그대로 재사용하기 위해 별도 자료구조가 아니라
        ``self._pred`` 의 열로 붙인다(조회 경로 단일화)."""
        pred = self._pred
        s = self.pred_source
        dirs = np.where(
            pred[f"{s}_up"].to_numpy() >= pred[f"{s}_down"].to_numpy(), 1, -1
        )
        n = len(pred)

        if self.seed_unanimity:
            seed_dirs = np.stack([
                np.where(
                    pred[f"s{i}_up"].to_numpy() >= pred[f"s{i}_down"].to_numpy(), 1, -1
                )
                for i in range(self.n_seeds)
            ])
            unanimous = (np.abs(seed_dirs.sum(axis=0)) == self.n_seeds) & (seed_dirs[0] == dirs)
        else:
            unanimous = np.ones(n, dtype=bool)

        if self.persistence_bars > 1:
            # 온그리드 연속성: 인덱스 간격이 결정TF 간격과 정확히 같을 때만 "직전 봉"으로 인정
            # (폴드 경계·데이터 갭에서 끊긴 두 봉을 연속으로 세지 않는다).
            interval = pd.Timedelta(milliseconds=TF_MS[self.entry_timeframe])
            contiguous = np.zeros(n, dtype=bool)
            if n > 1:
                contiguous[1:] = (pred.index[1:] - pred.index[:-1]) == interval
            run = np.ones(n, dtype=np.int64)     # 현재 봉까지의 동일방향 연속 길이
            for i in range(1, n):
                if contiguous[i] and dirs[i] == dirs[i - 1]:
                    run[i] = run[i - 1] + 1
            persist_ok = run >= self.persistence_bars
        else:
            persist_ok = np.ones(n, dtype=bool)

        self._pred = pred.assign(**{_UNANIMOUS_COL: unanimous, _PERSIST_COL: persist_ok})

    # ---- 진입 게이트 ----

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        sig = super().generate_signal(ctx)          # θ·방향·기존 게이트 + meta(예측·배리어)
        if not sig.is_actionable:
            return sig

        p_up, p_down = float(sig.meta["p_up"]), float(sig.meta["p_down"])
        p_dir, p_opp = max(p_up, p_down), min(p_up, p_down)
        w = float(sig.meta["barrier_frac"])
        ev_frac = w * (p_dir - p_opp)               # 기대총익률(배리어 해소 기준, expire≈0)
        if self.ev_k > 0.0 and ev_frac < self.ev_k * self.cost_frac:
            return Signal(SignalSide.HOLD)

        if self.seed_unanimity or self.persistence_bars > 1:
            row = self._lookup(sig.meta["pred_key"])
            if row is None:
                return Signal(SignalSide.HOLD)
            if self.seed_unanimity and not bool(row[_UNANIMOUS_COL]):
                return Signal(SignalSide.HOLD)
            if self.persistence_bars > 1 and not bool(row[_PERSIST_COL]):
                return Signal(SignalSide.HOLD)

        sig.meta["ev_frac"] = ev_frac               # 진단용(채택 판정엔 미사용)
        sig.meta["cost_frac"] = self.cost_frac
        return sig
