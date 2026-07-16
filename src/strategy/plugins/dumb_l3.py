"""Phase 4 Step 4.1 — 멍청한 3층 플러그인 (관문2 경제성 배선, 설계 §2 Phase4).

Phase 3 에서 확정된 통계엣지(15m 결정TF + 1h 상위맥락, 라벨 atr/w24/x3.0/N24=6h)를
**엔진에 태워 비용 차감 후 생존하나**를 본다(관문2). "멍청한" 이유 = 새 학습·판단 없이
**θ 임계값 + 방향 + 라벨 배리어**뿐. 경제성 최적화(청산로직·사이징·지평 정책)는 Phase 5.

**예측 소싱 (D-a)**: gate1 을 통과한 그 walk-forward OOS 예측을 **박제한 아티팩트**
(`oos_export`)에서 조회만 한다(재추론 없음 = 검증된 엣지 그대로 재생, 선별↔확인 분리).
아티팩트 index = 결정TF 봉 timestamp(그 봉 마감이 만든 예측). 엔진은 다음봉 open 진입 →
인과 정합(예측은 봉 마감 정보만, 진입은 그 후).

**배리어 (D-b①·F-11)**: TP/SL = **실진입가(open) 기준** ``entry·(1±barrier_frac)``.
barrier_frac = x·σ_i 는 아티팩트에 저장된 **라벨과 정확히 동일한 폭**(재계산 안 함).
LONG → TP=위벽·SL=아래벽 / SHORT → 방향만 뒤집힘(벽 대칭). 만기(N봉)=should_force_exit.

**동시터치 (D-b②)**: sl_tp_fill_priority="sl_first" (보수적 바닥). 1m 해상 대조는 Step 4.2.

**사이징**: 고정 notional (복리·리스크기반 없음 → 비용분석 해석가능, Phase4 스코프).
**진입게이트**: θ 가 주 필터. MTF 역할2(1h 정렬 게이트)는 Step 4.3 도입.
**reverse**: 기본 False (라벨 충실 — 각 거래는 배리어/만기까지, 조기청산 없음).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.core.enums import SignalSide
from src.core.types import ExitDecision, Position, Signal, StrategyContext
from src.core.enums import ExitReason, PositionSide
from src.data.historical import TF_MS
from src.strategy.base import StrategyModule
from src.strategy.registry import register_strategy

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACT = "data/research/phase4/oos_predictions_15m_mtf1h.parquet"


@register_strategy
class DumbL3(StrategyModule):
    """관문2 배선용 멍청3층. 박제 OOS 예측 → θ 방향결정 → 라벨 배리어 청산."""

    name = "dumb_l3"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]          # MTF 역할2(1h) 게이트는 Step 4.3 에서 추가
    sl_tp_fill_priority = "sl_first"       # D-b② 보수 바닥

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        # D-b②: fill_tf 지정 시 그 TF 를 master 로 편입(더 작은 TF → 체결 해상 정밀).
        # 신호는 여전히 15m 봉마감에만 발화(entry_timeframe). None=15m master(sl_first 바닥).
        fill_tf = self.params.get("fill_tf")
        if fill_tf and fill_tf not in self.required_timeframes:
            self.required_timeframes = [*self.required_timeframes, fill_tf]
        self.artifact_path = self.params.get("artifact_path", _DEFAULT_ARTIFACT)
        self.pred_source = self.params.get("pred_source", "ens")   # ens | s0..s4
        self.theta = float(self.params.get("theta", 0.40))
        self.require_dir_gt_expire = bool(self.params.get("require_dir_gt_expire", False))
        self.notional = float(self.params.get("notional", 10_000.0))
        self.horizon_bars = int(self.params.get("horizon_bars", 24))
        self.allow_reverse = bool(self.params.get("allow_reverse", False))
        # MTF 역할2 진입게이트(사전등록): 신호방향과 1h추세(mtf1h_kf_slope) 부호 일치시만 진입.
        self.mtf_gate = bool(self.params.get("mtf_gate", False))

        pred = pd.read_parquet(self.artifact_path)
        need = [f"{self.pred_source}_{c}" for c in ("up", "down", "expire")] + ["barrier_frac"]
        if self.mtf_gate:
            need.append("mtf1h_kf_slope")
        missing = [c for c in need if c not in pred.columns]
        if missing:
            raise ValueError(f"아티팩트 {self.artifact_path} 에 열 부재: {missing} "
                             f"(pred_source={self.pred_source}). oos_export 재생성 필요.")
        self._pred = pred
        # 만기 timeout = N봉 × 봉간격 (TF 무관 파생)
        self._timeout = pd.Timedelta(milliseconds=self.horizon_bars * TF_MS[self.entry_timeframe])
        logger.info("DumbL3 loaded: %d preds, src=%s, theta=%.3f, notional=%.0f",
                    len(self._pred), self.pred_source, self.theta, self.notional)

    # ---- 예측 조회 ----

    def _lookup(self, key) -> pd.Series | None:
        """직전 완성봉 timestamp(key)로 박제 예측 조회. tz 불일치 보정. 부재 시 None(무거래)."""
        idx = self._pred.index
        if key in idx:
            return self._pred.loc[key]
        # 엔진 캔들 tz 와 아티팩트(UTC) 불일치 방어
        try:
            if key.tzinfo is None and idx.tz is not None:
                key = key.tz_localize(idx.tz)
            elif key.tzinfo is not None and idx.tz is None:
                key = key.tz_localize(None)
            if key in idx:
                return self._pred.loc[key]
        except (TypeError, AttributeError):
            pass
        return None

    # ---- 필수 6 (거래 정책 = 모델 소유) ----

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        df15 = ctx.candles.get(self.entry_timeframe)
        if df15 is None or len(df15) == 0:
            return Signal(SignalSide.HOLD)
        key = df15.index[-1]                        # 직전 완성 15m봉 = 예측 키
        row = self._lookup(key)
        if row is None:
            return Signal(SignalSide.HOLD)          # OOS 밖(예측 부재) → 무거래

        s = self.pred_source
        p_up, p_down, p_exp = float(row[f"{s}_up"]), float(row[f"{s}_down"]), float(row[f"{s}_expire"])
        if p_up >= p_down:
            side, p_dir = SignalSide.LONG, p_up
        else:
            side, p_dir = SignalSide.SHORT, p_down
        if p_dir < self.theta:                       # θ 진입 게이트
            return Signal(SignalSide.HOLD)
        if self.require_dir_gt_expire and p_dir <= p_exp:
            return Signal(SignalSide.HOLD)
        if self.mtf_gate:                            # MTF 역할2: 1h 추세 부호 일치시만
            slope = float(row["mtf1h_kf_slope"])
            if (side == SignalSide.LONG and slope <= 0) or (side == SignalSide.SHORT and slope >= 0):
                return Signal(SignalSide.HOLD)
        bf = row.get("barrier_frac")
        return Signal(side, confidence=p_dir,
                      meta={"barrier_frac": float(bf), "p_up": p_up, "p_down": p_down,
                            "p_expire": p_exp, "pred_key": key})

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float | None:
        w = signal.meta.get("barrier_frac")
        if w is None:
            return None
        entry = ctx.current_price                    # D-b①: 실진입가(open) 앵커
        return entry * (1.0 - w) if signal.side == SignalSide.LONG else entry * (1.0 + w)

    def compute_take_profit(self, ctx: StrategyContext, signal: Signal,
                            stop_loss: float | None) -> float | None:
        w = signal.meta.get("barrier_frac")
        if w is None:
            return None
        entry = ctx.current_price
        return entry * (1.0 + w) if signal.side == SignalSide.LONG else entry * (1.0 - w)

    def compute_position_size(self, ctx: StrategyContext, signal: Signal,
                              stop_loss: float | None) -> float:
        if ctx.current_price <= 0:
            return 0.0
        return self.notional / ctx.current_price     # 고정 notional (복리·리스크사이징 없음)

    def should_reverse(self, ctx: StrategyContext, position: Position,
                       new_signal: Signal) -> bool:
        if not self.allow_reverse:
            return False                             # 라벨 충실: 배리어/만기까지 보유
        held = PositionSide.LONG if new_signal.side == SignalSide.LONG else PositionSide.SHORT
        return position.side != held                 # 반대 방향 신호 시에만

    def allow_entry(self, ctx: StrategyContext) -> bool:
        return True                                  # θ 가 주 게이트 (멍청층: DD락 없음)

    # ---- 선택 훅 ----

    def should_force_exit(self, ctx: StrategyContext, position: Position) -> ExitDecision | None:
        """만기 청산 = 진입 후 N봉 경과(라벨 expire 대응)."""
        if ctx.now - position.entry_time >= self._timeout:
            return ExitDecision(reason=ExitReason.FORCE_EXIT, note="expire_timeout")
        return None
