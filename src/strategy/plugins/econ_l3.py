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

**축 C — 선택률 고정 (ev_rank_rate, Phase 7 Step 7.5c = D-047 의 G-C arm)**
    같은 점수(``barrier_frac × |p_up−p_down|``)를 쓰되 **절대 문턱 대신 causal rolling
    백분위**로 자른다 → 통과율이 config 무관하게 ``ev_rank_rate`` 로 고정된다.

    동기(측정): EV 게이트의 통과율이 config 마다 **2.74%~38.3%(14배)** 로 벌어져, config 간
    bp 비교가 "엣지 차이"인지 "표본 선택률 차이"인지 구분되지 않는다. 선택률을 묶으면 그
    교란축이 죽는다. 목표값 = **D2 의 G-B 통과율 2087/11876 = 17.573%**(러너가 주입).

    창 = **90일에 해당하는 봉수**(15m 8640 / 1h 2160 / 4h 540) — **전 config 동일**.
    (카운트 기반 rolling 이라 폴드 갭 때문에 실 span 은 약 89.8~92.0일로 미세 변동한다.)
    통제 장치가 시험 대상 축(결정TF·보유시간)을 따라 변하면 그 축의 결과를 오염시키므로
    봉 수가 아니라 **캘린더 시간**을 고정한다. 90일은 splitter ``val_size`` 3개월 선례.

    구현: 선행 게이트(θ·만기) 부적격 봉을 ``-1`` 로 눌러 랭킹에서 축출한 뒤 **전체 창**의
    ``1−rate`` 분위를 문턱으로 쓴다 → 창마다 정확히 ``rate×창`` 개가 통과하고, 적격이
    그보다 적은 창에서는 문턱이 음수로 내려가 **적격 전부 통과**(자연스러운 상한, 플래그).
    워밍업(``min_periods=창``) 구간은 문턱 NaN → **무거래**(causal_regime NaN 규약과 동일).

    ※ 절대문턱(ev_k)과 **동시 사용 금지**. 선행 게이트로 반영되지 않는 게이트
    (mtf/trend/vol)와도 동시 사용 금지 — 통과율이 조용히 목표에서 벗어나기 때문(하드페일).

**인과성**: 게이트 입력은 전부 완성봉 산출물이다 — 예측(그 봉 마감이 만든 값)·barrier_frac
(라벨과 동일 폭)·상수 비용. 지속성·백분위 플래그는 **과거 행만** 참조해 __init__ 에서 1회
벡터 계산한다(미래 행 불참조 — 회귀로 박제). rolling 창은 현재 봉에서 끝나므로 문턱 τ_t 는
t 시점 정보만 쓴다. ``generate_signal`` 은 엔진이 결정TF 봉마감에만 호출한다
(engine ``_evaluate_bar_close``) → I-007 류 경로 없음.

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
_RANK_COL = "_econ_rank_ok"

# 축 C 사전등록 상수 (D-047 적용, Step 7.5c 실행 전 고정 — 결과 보고 바꾸지 말 것).
EV_RANK_WINDOW_DAYS = 90       # rolling 창(캘린더). 전 config 동일 — 근거는 모듈 docstring.
# 점수는 |p_up−p_down|·barrier_frac ≥ 0 이므로 -1 은 어떤 실점수보다도 확실히 작다.
_RANK_SENTINEL = -1.0
_MS_PER_DAY = 86_400_000


def compute_rank_gate(
    pred: pd.DataFrame, *, rate: float, window_days: int, decision_tf: str,
    theta: float = 0.0, require_dir_gt_expire: bool = False, pred_source: str = "ens",
) -> tuple[np.ndarray, dict]:
    """축 C 게이트 벡터 + 진단 dict. **EconL3 와 policy_eval 의 단일 출처.**

    분리 이유: `policy_eval._analyze` 의 포착률 분모가 이 게이트를 모르면 **포화 셀을
    미포화로 오판**한다 — fresh-eyes H-5 가 EV/만기 게이트에서 고친 것과 정확히 같은
    버그다. 두 곳이 각자 계산하면 언제든 갈라지므로 함수 하나로 묶는다.

    반환 게이트는 **아티팩트 행 순서**와 정렬된 bool 배열이다.
    """
    # ★L-1★ rolling 창은 **행 순서**로 과거를 정의한다. 인덱스가 비정렬이면 "과거 창"이
    # 조용히 미래를 포함하고(비인과), 중복이 있으면 같은 봉이 두 번 가중된다. 온그리드
    # 가드(alpha_decomp.assert_on_grid)와 같은 이유의 방어 — 호출자를 믿지 않는다.
    if not pred.index.is_monotonic_increasing:
        raise ValueError("아티팩트 인덱스가 시간순이 아니다 - rolling 창이 비인과가 된다")
    if pred.index.has_duplicates:
        raise ValueError(
            f"아티팩트 인덱스에 중복 {int(pred.index.duplicated().sum())}개 - 창 가중 오염")
    up = pred[f"{pred_source}_up"].to_numpy(dtype="float64")
    dn = pred[f"{pred_source}_down"].to_numpy(dtype="float64")
    ex = pred[f"{pred_source}_expire"].to_numpy(dtype="float64")
    w = pred["barrier_frac"].to_numpy(dtype="float64")
    n_nan = int(np.isnan(w).sum())
    if n_nan:
        # NaN 은 비교가 전부 False 라 **조용히 표본에서 빠진다** → 실현 선택률이 목표에서
        # 이탈한다. alpha_decomp 의 y_true NaN 하드페일과 같은 성격.
        raise ValueError(f"barrier_frac 에 NaN {n_nan}개 — 선택률 고정 게이트는 "
                         f"전 봉 점수가 유효해야 한다.")

    score = np.abs(up - dn) * w              # EV 게이트와 **같은 점수**(문턱 규칙만 다름)
    p_dir = np.maximum(up, dn)
    mask = p_dir >= theta                    # θ 는 `>=`, 만기게이트는 `>` (엔진 타이규약 H-2)
    if require_dir_gt_expire:
        mask = mask & (p_dir > ex)

    tf_ms = TF_MS[decision_tf]
    if _MS_PER_DAY % tf_ms:
        raise ValueError(f"결정TF {decision_tf} 가 하루를 정수분할하지 않음")
    win = int(_MS_PER_DAY // tf_ms) * window_days
    masked = pd.Series(np.where(mask, score, _RANK_SENTINEL), index=pred.index)
    tau = masked.rolling(win, min_periods=win).quantile(1.0 - rate)
    tau_v = tau.to_numpy(dtype="float64")
    ready = ~np.isnan(tau_v)                 # 워밍업(창 미충족) → 무거래
    gate = mask & ready & (score >= np.where(ready, tau_v, np.inf))

    diag = {
        "rank_window_bars": win,
        "rank_warmup_bars": int((~ready).sum()),
        "rank_capped_bars": int((ready & (tau_v < 0.0)).sum()),   # 적격 < 목표인 창
        "rank_realized_rate": float(gate.mean()) if len(gate) else float("nan"),
        "rank_realized_rate_post_warmup": (
            float(gate[ready].mean()) if ready.any() else float("nan")),
    }
    return gate, diag


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
        # --- 축 C: 선택률 고정 (0 = off → 기존 동작 바이트 동일) ---
        self.ev_rank_rate = float(self.params.get("ev_rank_rate", 0.0))
        self.ev_rank_window_days = int(
            self.params.get("ev_rank_window_days", EV_RANK_WINDOW_DAYS)
        )
        self._validate_rank_config()

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
        self._add_ev_rank_flag()
        logger.info(
            "EconL3: ev_k=%.2f cost=%.5f(=%.1fbp) unanimity=%s persist=%d rank_rate=%.5f",
            self.ev_k, self.cost_frac, self.cost_frac * 1e4,
            self.seed_unanimity, self.persistence_bars, self.ev_rank_rate,
        )

    # ---- 축 C 구성 검증 (조용한 선택률 이탈 방지 — 하드페일) ----

    def _validate_rank_config(self) -> None:
        """``ev_rank_rate`` 사용 시 전제 위반을 **즉시 예외**로 잡는다.

        선택률 고정이 성립하려면 ``_add_ev_rank_flag`` 의 mask 가 ``generate_signal`` 이
        실제로 거는 선행 게이트와 **정확히 같아야** 한다. mask 에 없는 게이트가 뒤에서
        추가로 걸리면 통과율이 목표보다 낮아지는데, 그건 **조용히** 일어나 arm 전체를
        무의미하게 만든다(I-013 의 하드페일 선례와 같은 성격)."""
        if not self.ev_rank_rate:
            return
        if not 0.0 < self.ev_rank_rate < 1.0:
            raise ValueError(
                f"ev_rank_rate 는 (0,1) 구간이어야 함: {self.ev_rank_rate}")
        if self.ev_k > 0.0:
            raise ValueError(
                f"ev_rank_rate({self.ev_rank_rate})와 ev_k({self.ev_k})는 동시 사용 불가 — "
                f"축 C 는 절대문턱을 백분위로 **대체**한다(둘 다 걸면 G-A/G-B 도 G-C 도 아님).")
        conflict = [n for n, on in (("mtf_gate", self.mtf_gate),
                                    ("trend_gate", self.trend_gate),
                                    ("avoid_vol", self.avoid_vol is not None)) if on]
        if conflict:
            raise ValueError(
                f"ev_rank_rate 와 동시 사용 불가한 게이트: {conflict} — mask 에 반영되지 않아 "
                f"실현 선택률이 목표({self.ev_rank_rate:.4f})에서 조용히 이탈한다.")
        if self.ev_rank_window_days < 1:
            raise ValueError(
                f"ev_rank_window_days 는 1 이상이어야 함: {self.ev_rank_window_days}")

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

    # ---- 축 C 플래그 사전계산 (현재 봉까지의 창만 참조 — 인과) ----

    def _add_ev_rank_flag(self) -> None:
        """선택률 고정 게이트를 아티팩트 열로 사전계산 (G-C).

        ``mask`` 는 ``DumbL3.generate_signal`` 이 축 C 앞에서 거는 게이트와 동일해야 한다
        (θ 는 ``>=``, 만기게이트는 ``>`` — 엔진 타이 규약 H-2). 나머지 게이트는
        ``_validate_rank_config`` 가 하드페일로 막았다.
        """
        if not self.ev_rank_rate:
            return
        try:
            rank_ok, diag = compute_rank_gate(
                self._pred, rate=self.ev_rank_rate,
                window_days=self.ev_rank_window_days,
                decision_tf=self.entry_timeframe, theta=self.theta,
                require_dir_gt_expire=self.require_dir_gt_expire,
                pred_source=self.pred_source)
        except ValueError as e:
            raise ValueError(f"아티팩트 {self.artifact_path}: {e}") from e
        self.rank_window_bars = diag["rank_window_bars"]
        self.rank_warmup_bars = diag["rank_warmup_bars"]
        self.rank_capped_bars = diag["rank_capped_bars"]
        self.rank_realized_rate = diag["rank_realized_rate"]
        self.rank_realized_rate_post_warmup = diag["rank_realized_rate_post_warmup"]
        if diag["rank_warmup_bars"] >= len(self._pred):
            logger.warning("EconL3 rank: 아티팩트 %d봉 < 창 %d봉 → 전 구간 무거래",
                           len(self._pred), diag["rank_window_bars"])
        logger.info(
            "EconL3 rank: target=%.4f realized=%.4f (post-warmup %.4f) win=%d "
            "warmup=%d capped=%d",
            self.ev_rank_rate, diag["rank_realized_rate"],
            diag["rank_realized_rate_post_warmup"], diag["rank_window_bars"],
            diag["rank_warmup_bars"], diag["rank_capped_bars"],
        )
        self._pred = self._pred.assign(**{_RANK_COL: rank_ok})

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

        if self.ev_rank_rate or self.seed_unanimity or self.persistence_bars > 1:
            row = self._lookup(sig.meta["pred_key"])
            if row is None:
                return Signal(SignalSide.HOLD)
            if self.ev_rank_rate and not bool(row[_RANK_COL]):
                return Signal(SignalSide.HOLD)
            if self.seed_unanimity and not bool(row[_UNANIMOUS_COL]):
                return Signal(SignalSide.HOLD)
            if self.persistence_bars > 1 and not bool(row[_PERSIST_COL]):
                return Signal(SignalSide.HOLD)

        sig.meta["ev_frac"] = ev_frac               # 진단용(채택 판정엔 미사용)
        sig.meta["cost_frac"] = self.cost_frac
        return sig
