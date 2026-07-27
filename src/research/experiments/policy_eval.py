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

def _run_engine(policy_spec: dict, start: str, end: str, init: float,
                strategy: str = "dumb_l3") -> list[dict]:
    """strategy + policy_spec 으로 엔진 실행 → trades. policy_spec 는 fill_tf='1m' 포함 권장.

    strategy: 활성화할 플러그인명(dumb_l3 | adaptive_l3 …). config 섹션이 없으면 생성."""
    cfg = load_config("config/default.yaml")
    cfg["strategies"]["active"] = [strategy]
    spec = {"notional": init, **policy_spec}
    cfg[strategy] = {**cfg.get(strategy, {}), **spec}   # 섹션 생성/병합(엔진 강제키 없음)
    eng = BacktestEngine(cfg, start=start, end=end)
    eng.initialize()
    eng.run()
    return eng.trades


def _apply_funding(df: pd.DataFrame, funding_csv: str, init: float) -> pd.Series:
    """실펀딩(Binance proxy) 거래별 손익. 롱=양의펀딩 지불(-), 숏=수취(+). notional 고정.

    frontier_gate2 와 동일 규약(등가성). 파일 부재 시 0 시리즈."""
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
        return sign * rate_sum * init

    return df.apply(_one, axis=1)


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

    df["funding_pnl"] = _apply_funding(df, funding_csv, init)
    fund_total = float(df["funding_pnl"].sum())
    net_af = net + fund_total
    win_rate, be = _winrate_breakeven(df["pnl"])

    out.update(
        n_trades=n, gross=round(gross, 2), fee=round(fee, 2), net=round(net, 2),
        net_pct=round(net / init * 100, 2),
        cost_over_absgross=round(fee / abs(gross), 3) if gross != 0 else None,
        per_trade_gross_pct_notional=round(gross / n / init * 100, 4),
        win_rate=win_rate, breakeven_wr=be, win_minus_be=round(win_rate - be, 2),
        funding_total=round(fund_total, 1),
        net_after_funding=round(net_af, 1), net_af_pct=round(net_af / init * 100, 2),
    )
    er = df.groupby("exit_reason").agg(count=("pnl", "size"), net=("pnl", "sum"))
    out["exit_reasons"] = {r: dict(count=int(v["count"]), pct=round(v["count"] / n * 100, 1),
                                   net=round(float(v["net"]), 1)) for r, v in er.iterrows()}

    # 연도별 (policy-WF fixed: 각 연도 = OOS 슬라이스)
    df["year"] = df["entry_time"].dt.year
    by_year = {}
    for y, g in df.groupby("year"):
        wr, bey = _winrate_breakeven(g["pnl"])
        by_year[str(int(y))] = dict(
            n=int(len(g)), net=round(float(g["pnl"].sum()), 1),
            net_af=round(float(g["pnl"].sum() + g["funding_pnl"].sum()), 1),
            net_sign="+" if g["pnl"].sum() > 0 else "-",
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
                    scoring: str = "fixed", strategy: str = "dumb_l3") -> dict:
    """strategy + policy_spec 을 엔진(1m fill)+실펀딩으로 채점. policy-WF 연도별 OOS.

    policy_spec: config dict — decision_tf·horizon_bars·artifact_path·fill_tf·theta·
        레버(trend_gate·avoid_vol·signal_decay·sl_mult·breakeven…). 반환 = _analyze dict.
    strategy: 플러그인명(dumb_l3 | adaptive_l3). scoring: 'fixed'(전체OOS→연도별) / 'wf'(6.2+)."""
    if scoring == "wf":
        raise NotImplementedError(
            "policy-WF fit 모드는 fittable 정책(conviction 곡선 등)에서 구현. "
            "원리고정 정책은 fit無라 scoring='fixed'(전 연도 OOS)로 충분."
        )
    if scoring != "fixed":
        raise ValueError(f"미지원 scoring: {scoring} (fixed | wf)")
    trades = _run_engine(policy_spec, start, end, init, strategy)
    return _analyze(policy_spec, trades, funding_csv, init)
