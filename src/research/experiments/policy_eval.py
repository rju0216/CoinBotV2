"""Phase 6 Step 6.0 — 정책 백테 하네스 + causal 국면 아티팩트 증강 (커밋).

scratchpad `frontier_gate2`(미커밋) 를 커밋 코드로 승격. 두 축:

1. **augment_artifact_causal_regime**: 얼린 OOS 예측 아티팩트에 인과 국면 열
   (``causal_regime_trend`` / ``causal_regime_vol``) 을 추가. **예측·기존 열 불변**(열만 증강,
   기둥3 선별↔확인 분리 — 모델 재추론 없음). 열 값은 test_regime_causal 로 인과 박제된
   ``causal_regime_frame_for`` 출력.

2. **evaluate_policy**: dumb_l3 + 정책스펙을 엔진(**1m fill**, I-004)으로 돌리고 **실펀딩**
   (Binance proxy) 차감, **연도별 net·승률 vs breakeven**·비용비를 계측. 채점 모드:
   - ``fixed``: 전체 OOS 1회 실행 → **연도별 파티션**. fixed(원리고정) 정책은 fit無라
     전 연도가 OOS (policy-WF (B) 의 fixed 특수화 — 사용자 확정).
   - ``wf``: DEV-init 후 연도별 fit-test (fittable 정책 — 6.2+에서 구현).

**등가성(규칙16/17)**: fixed·baseline 정책은 scratchpad frontier_gate2 수치를 재현해야 한다
(test 로 박제 — 하네스 신뢰의 근거).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.research.experiments.tf_expansion import causal_regime_frame_for
from src.utils.config_loader import load_config

DEFAULT_FUNDING = "data/funding/BTCUSDT_binance_funding.csv"
DEFAULT_INIT = 10_000.0


# ---------------------------------------------------------------------------
# 1. 아티팩트 causal 국면 증강 (예측 불변, 열만 추가)
# ---------------------------------------------------------------------------

def augment_artifact_causal_regime(
    artifact_path: str,
    decision_tf: str,
    candle_dir: str = "data/candles",
    out_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """얼린 예측 아티팩트에 causal_regime_trend/vol 열 추가 (제자리 또는 out_path).

    아티팩트 index(=결정TF OOS 봉)에 ``causal_regime_frame_for`` 로 인과 국면을 완성봉 정렬해
    붙인다. 기존 예측/라벨/analysis-regime 열은 손대지 않는다(``causal_`` 접두어로 분리).
    반환: (증강 df, 커버리지 dict: n_rows·trend/vol NaN 비율)."""
    art = pd.read_parquet(artifact_path)
    frame = causal_regime_frame_for(decision_tf, art.index, candle_dir)
    trend = frame["regime_trend"].reindex(art.index)
    vol = frame["regime_vol"].reindex(art.index)
    art["causal_regime_trend"] = trend
    art["causal_regime_vol"] = vol
    cov = {
        "n_rows": int(len(art)),
        "trend_nan": int(trend.isna().sum()),
        "vol_nan": int(vol.isna().sum()),
        "trend_nan_pct": round(float(trend.isna().mean()) * 100, 3),
        "vol_nan_pct": round(float(vol.isna().mean()) * 100, 3),
    }
    art.to_parquet(out_path or artifact_path)
    return art, cov


# ---------------------------------------------------------------------------
# 2. 정책 채점 (엔진 1m fill + 실펀딩 + 연도별)
# ---------------------------------------------------------------------------

def _max_drawdown_pct(curve: list[tuple]) -> float:
    """자본 곡선의 최대 낙폭(%). 빈 곡선이면 0."""
    peak, mdd = float("-inf"), 0.0
    for _, v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak * 100.0)
    return round(mdd, 2)


def _run_engine(policy_spec: dict, start: str, end: str, init: float,
                strategy: str = "dumb_l3", max_slots: int = 1,
                ensemble: list[tuple[str, dict]] | None = None) -> tuple[list[dict], dict]:
    """strategy + policy_spec 으로 엔진 실행 → (trades, diag). fill_tf='1m' 포함 권장.

    strategy: 활성화할 플러그인명(dumb_l3 | adaptive_l3 | econ_l3 …). 섹션 없으면 생성.
    max_slots: 엔진 용량(N-트랜치). 기본 1 = 기존 단일슬롯.
    ensemble: [(플러그인명, spec)] — 여러 전략(예: 지평 3·4·5일 슬리브)을 **동시 활성화**해
        공용 트랜치 풀을 나눠 쓰게 한다. 지정 시 strategy/policy_spec 은 메타로만 쓰인다.

    diag: 최종잔고·**손익 항등식 검증**(규칙10: Σpnl == 잔고 변화)·MDD(실현/mtm).
    항등식 위반은 즉시 예외 — N-트랜치에서 트랜치별 정산이 어긋나면 모든 수치가 무의미하다."""
    cfg = load_config("config/default.yaml")
    cfg["backtest"] = {**cfg.get("backtest", {}), "max_slots": int(max_slots)}
    if ensemble:
        # ★I-013★ 앙상블 슬리브는 notional 을 **반드시 명시**해야 한다. 누락 시 단일전략
        # 규약(notional=init)이 적용돼 슬리브마다 전액이 배정되고, max_slots=N 이면
        # 최대 노출이 N배로 뛴다(net_pct·mdd 가 그만큼 부풀려짐). 조용한 레버리지
        # 폭증을 막기 위해 하드페일한다.
        missing = [n for n, sub in ensemble if "notional" not in sub]
        if missing:
            raise ValueError(
                f"앙상블 슬리브에 notional 미지정: {missing}. 총노출 배분을 명시해야 한다 "
                f"(예: 슬리브당 init/max_slots={init / max(1, max_slots):.2f})."
            )
        cfg["strategies"]["active"] = [n for n, _ in ensemble]
        for name, sub in ensemble:
            cfg[name] = {**cfg.get(name, {}), **sub}
    else:
        cfg["strategies"]["active"] = [strategy]
        spec = {"notional": init, **policy_spec}
        cfg[strategy] = {**cfg.get(strategy, {}), **spec}  # 섹션 생성/병합(엔진 강제키 없음)
    eng = BacktestEngine(cfg, start=start, end=end)
    eng.initialize()
    eng.run()

    init_bal = eng.account_tracker.initial_balance
    total_pnl = sum(float(t.get("pnl") or 0.0) for t in eng.trades)
    drift = abs(eng.balance - (init_bal + total_pnl))
    if drift > 1e-6 * max(1.0, abs(init_bal)):
        raise AssertionError(
            f"손익 항등식 위반: 잔고 {eng.balance:.6f} != 초기 {init_bal:.6f} + Σpnl "
            f"{total_pnl:.6f} (차이 {drift:.6g}, max_slots={max_slots})"
        )
    diag = {
        "final_balance": round(eng.balance, 2),
        "pnl_identity_ok": True,
        "mdd_realized_pct": _max_drawdown_pct(eng.equity_curve),
        "mdd_mtm_pct": _max_drawdown_pct(eng.equity_curve_mtm),
    }
    return eng.trades, diag


def _apply_funding(df: pd.DataFrame, funding_csv: str) -> pd.Series:
    """실펀딩(Binance proxy) 거래별 손익. 롱=양의펀딩 지불(-), 숏=수취(+).

    ★I-008★ 펀딩은 **그 거래의 실제 노셔널**(size × entry_price)에 부과한다. 과거 구현은
    초기자본(init) 고정이었고, 고정사이징(notional==init)에서는 우연히 일치했으나
    **N-트랜치(notional=init/N)에서 N배 과대**·**conviction 사이징에서 과소** 계상됐다.
    고정사이징 셀은 size×entry_price == notional == init 이라 **수치 불변**(등가성 유지).

    파일 부재 시 0 시리즈."""
    try:
        fund = pd.read_csv(funding_csv)
    except FileNotFoundError:
        return pd.Series(0.0, index=df.index)
    fs = pd.Series(
        fund["funding_rate"].to_numpy(),
        index=pd.to_datetime(fund["timestamp"], unit="ms", utc=True),
    ).sort_index()
    fi = fs.index

    def _one(row) -> float:
        lo = fi.searchsorted(row["entry_time"], side="right")
        hi = fi.searchsorted(row["exit_time"], side="right")
        rate_sum = float(fs.iloc[lo:hi].sum())
        sign = -1.0 if str(row["side"]).lower().endswith("long") else 1.0
        notional = float(row["size"]) * float(row["entry_price"])
        return sign * rate_sum * notional

    return df.apply(_one, axis=1)


def _occupancy(df: pd.DataFrame) -> dict:
    """트랜치 동시보유 계측 — 평균/최대 동시 트랜치·time-in-market(%).

    N-트랜치 결과 해석에 필수: 포착률이 왜 그만큼 올랐는지(포화 여부)를 본다.
    시간가중 평균 = Σ(보유시간)/전체구간. 이벤트 스윕으로 최대 동시성 산출."""
    if len(df) == 0:
        return {}
    span_s = (df["exit_time"].max() - df["entry_time"].min()).total_seconds()
    if span_s <= 0:
        return {}
    hold_s = (df["exit_time"] - df["entry_time"]).dt.total_seconds()
    events = ([(t, 1) for t in df["entry_time"]] + [(t, -1) for t in df["exit_time"]])
    # ★I-009★ 동시각(exit_time == entry_time)은 **청산 먼저** 정렬해야 한다 — 엔진이
    # 봉 루프에서 청산(슬롯 반납) 후 진입하므로. 진입 먼저 정렬하면 max_concurrent 가
    # 용량(max_slots)을 넘는 허수(N+1 등)를 만든다(손익 무영향·진단 지표만 오염).
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    open_s = 0.0
    prev_t = None
    for t, delta in events:
        if prev_t is not None and cur > 0:
            open_s += (t - prev_t).total_seconds()
        cur += delta
        peak = max(peak, cur)
        prev_t = t
    return {
        "avg_concurrent": round(float(hold_s.sum()) / span_s, 2),
        "max_concurrent": int(peak),
        "time_in_market_pct": round(open_s / span_s * 100, 1),
        "avg_hold_hours": round(float(hold_s.mean()) / 3600, 1),
    }


def _max_consecutive_losses(pnl: pd.Series) -> int:
    """최장 연속 손실 거래 수 (라이브 인내구간 — 6.4 판단 재료)."""
    best = cur = 0
    for v in pnl:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return int(best)


def _winrate_breakeven(pnl: pd.Series) -> tuple[float, float]:
    """실현 승률 + breakeven 승률(=평균손실/(평균이익+평균손실), 실현 payoff 기준)."""
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n = len(pnl)
    win_rate = round(len(wins) / n * 100, 2) if n else float("nan")
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = -float(losses.mean()) if len(losses) else 0.0  # 양수화
    be = round(avg_l / (avg_w + avg_l) * 100, 2) if (avg_w + avg_l) > 0 else float("nan")
    return win_rate, be


def _analyze(policy_spec: dict, trades: list[dict], funding_csv: str,
             init: float) -> dict:
    """trades → 계측 dict (gross/fee/net·펀딩후·승률vsBE·연도별·청산사유·capture)."""
    tag = policy_spec.get("artifact_path", "?")
    theta = float(policy_spec.get("theta", 0.40))
    src = policy_spec.get("pred_source", "ens")
    out: dict = {"decision_tf": policy_spec.get("decision_tf"),
                 "horizon_bars": policy_spec.get("horizon_bars"),
                 "artifact": tag}
    df = pd.DataFrame(trades)
    if len(df) == 0:
        out["n_trades"] = 0
        return out
    df["gross"] = df["pnl"] + df["trading_fee"] + df["funding_fee"]
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df)
    gross, fee, net = float(df["gross"].sum()), float(df["trading_fee"].sum()), float(df["pnl"].sum())

    df["funding_pnl"] = _apply_funding(df, funding_csv)
    fund_total = float(df["funding_pnl"].sum())
    net_af = net + fund_total
    win_rate, be = _winrate_breakeven(df["pnl"])

    # ★I-010★ bp 정규화 분모 = **실현 노셔널**(size × entry_price)의 합. 선언값
    # (policy_spec["notional"])을 쓰면 conviction 사이징에서 실노셔널이 최대 max_mult 배
    # 커지므로 bp 지표가 그만큼 과대계상된다. 고정사이징에서는 양자가 정확히 일치.
    notl_series = df["size"] * df["entry_price"]
    total_notional = float(notl_series.sum())
    notional = total_notional / n                       # 거래당 평균 실현 노셔널
    out.update(
        n_trades=n, gross=round(gross, 2), fee=round(fee, 2), net=round(net, 2),
        net_pct=round(net / init * 100, 2),
        cost_over_absgross=round(fee / abs(gross), 3) if gross != 0 else None,
        per_trade_gross_pct_notional=round(gross / n / notional * 100, 4),
        win_rate=win_rate, breakeven_wr=be, win_minus_be=round(win_rate - be, 2),
        funding_total=round(fund_total, 1),
        net_after_funding=round(net_af, 1), net_af_pct=round(net_af / init * 100, 2),
        # ---- N 불변 지표 (트랜치 수가 달라도 직접 비교 가능·실현 노셔널 가중) ----
        notional_per_trade=round(notional, 2),
        total_notional=round(total_notional, 2),
        gross_bp_per_trade=round(gross / total_notional * 1e4, 2),
        net_bp_per_trade=round(net / total_notional * 1e4, 2),
        net_af_bp_per_trade=round(net_af / total_notional * 1e4, 2),
        # ---- 라이브 판단용(6.4) ----
        # 실집행 안전마진: 체결 한쪽당 몇 bp 슬리피지까지 net_af 가 0 이 되지 않나
        # (거래당 체결 2회). 현 백테는 슬리피지 0·완벽체결 가정이라 이 값이 실질 여유폭.
        slippage_tolerance_bp_per_fill=round(net_af / total_notional * 1e4 / 2, 2),
        max_consecutive_losses=_max_consecutive_losses(df["pnl"]),
        **_occupancy(df),
    )
    er = df.groupby("exit_reason").agg(count=("pnl", "size"), net=("pnl", "sum"))
    out["exit_reasons"] = {r: dict(count=int(v["count"]), pct=round(v["count"] / n * 100, 1),
                                   net=round(float(v["net"]), 1)) for r, v in er.iterrows()}

    # 연도별 (policy-WF fixed: 각 연도 = OOS 슬라이스)
    df["year"] = df["entry_time"].dt.year
    df["_notional"] = notl_series
    by_year = {}
    for y, g in df.groupby("year"):
        wr, bey = _winrate_breakeven(g["pnl"])
        net_af_y = float(g["pnl"].sum() + g["funding_pnl"].sum())
        by_year[str(int(y))] = dict(
            n=int(len(g)), net=round(float(g["pnl"].sum()), 1),
            net_af=round(net_af_y, 1),
            net_sign="+" if g["pnl"].sum() > 0 else "-",
            # N 불변 — 트랜치 수가 달라도 연도별 품질을 직접 비교(실현 노셔널 가중)
            net_af_bp=round(net_af_y / float(g["_notional"].sum()) * 1e4, 2),
            win_rate=wr, breakeven_wr=bey,
        )
    out["by_year"] = by_year

    # 신호 포착률 (θ 넘는 actionable 대비)
    try:
        art = pd.read_parquet(tag, columns=[f"{src}_up", f"{src}_down"])
        n_act = int((art[[f"{src}_up", f"{src}_down"]].max(axis=1) >= theta).sum())
        out["n_actionable"] = n_act
        out["signal_capture_pct"] = round(n / n_act * 100, 1) if n_act else None
    except Exception:  # noqa: BLE001 — capture 는 진단 보조, 실패해도 채점 유지
        pass
    return out


def evaluate_policy(policy_spec: dict, *, start: str, end: str,
                    funding_csv: str = DEFAULT_FUNDING, init: float = DEFAULT_INIT,
                    scoring: str = "fixed", strategy: str = "dumb_l3",
                    max_slots: int = 1, trades_out: str | None = None,
                    ensemble: list[tuple[str, dict]] | None = None) -> dict:
    """strategy + policy_spec 을 엔진(1m fill)+실펀딩으로 채점. policy-WF 연도별 OOS.

    policy_spec: config dict — decision_tf·horizon_bars·artifact_path·fill_tf·theta·
        레버(trend_gate·avoid_vol·signal_decay·sl_mult·breakeven·ev_k…). 반환 = _analyze dict.
    strategy: 플러그인명(dumb_l3 | adaptive_l3 | econ_l3). scoring: 'fixed' / 'wf'(미구현).
    max_slots: 엔진 용량(N-트랜치). **N 을 바꾸면 net_pct 는 직접 비교 불가** — 총 거래
        노셔널이 포화 정도에 따라 달라지므로 `*_bp_per_trade`(N 불변)로 비교할 것."""
    if scoring == "wf":
        raise NotImplementedError(
            "policy-WF fit 모드는 fittable 정책(conviction 곡선 등)에서 구현. "
            "원리고정 정책은 fit無라 scoring='fixed'(전 연도 OOS)로 충분."
        )
    if scoring != "fixed":
        raise ValueError(f"미지원 scoring: {scoring} (fixed | wf)")
    trades, diag = _run_engine(policy_spec, start, end, init, strategy, max_slots,
                               ensemble)
    if trades_out:
        # 거래 원장 보존 — N 간 **교집합 교차검증**(동일 진입시각·방향 거래의 per-trade
        # gross 가 N 에 무관하게 동일한가)과 사후 진단에 필요. fresh-eyes 권고.
        pd.DataFrame(trades).to_csv(trades_out, index=False)
    out = _analyze(policy_spec, trades, funding_csv, init)
    out["max_slots"] = int(max_slots)
    out.update(diag)
    return out
