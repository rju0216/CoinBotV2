"""백테스트 엔진.

DataStore에 의존하지 않고 메모리에서 trades·equity_curve를 누적한 뒤
종료 시 `data/backtest_reports/00_Working/` 하위로 결과 파일을 출력.

리포트 디렉토리 구조 (merge_yearly_reports.py 호환):
  data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{config_name}/{config_name}/
    ├── trades.csv
    ├── equity_curve.csv
    ├── metrics.json
    ├── config_snapshot.yaml
    └── equity_curve.png

마스터 TF(가장 작은 활성 TF) 캔들을 순회하며:
  1) SL/TP 캔들 체결 검사 (정책 (a) SL 우선)
  2) update_stop_loss / should_force_exit 훅 호출
  3) 봉 경계 TF별 evaluate_strategies_on_bar dispatch
종료 시 잔여 포지션은 ENGINE_SHUTDOWN 사유로 강제 청산.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.core.engine_base import AbstractEngine
from src.core.enums import ExitReason, PositionSide
from src.core.types import Position
from src.data.historical import HistoricalDataLoader
from src.strategy.features import compute_multi_tf_features

logger = logging.getLogger(__name__)


REPORT_BASE = Path("data/backtest_reports/00_Working")


@dataclass
class BacktestResult:
    initial_balance: float
    final_balance: float
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return self.total_pnl / self.initial_balance * 100.0

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def num_winners(self) -> int:
        return sum(1 for t in self.trades if (t.get("pnl") or 0) > 0)

    @property
    def num_losers(self) -> int:
        return sum(1 for t in self.trades if (t.get("pnl") or 0) < 0)

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return self.num_winners / self.num_trades * 100.0

    @property
    def max_drawdown_pct(self) -> float:
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, eq in self.equity_curve:
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def summary(self) -> dict[str, Any]:
        return {
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.final_balance, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "num_trades": self.num_trades,
            "num_winners": self.num_winners,
            "num_losers": self.num_losers,
            "win_rate": round(self.win_rate, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
        }


class BacktestEngine(AbstractEngine):
    def __init__(
        self,
        config: dict[str, Any],
        start: str | datetime,
        end: str | datetime,
    ) -> None:
        super().__init__(config, mode="backtest")
        self.start_dt = self._parse_dt(start)
        self.end_dt = self._parse_dt(end)
        self.candles_per_tf: dict[str, pd.DataFrame] = {}
        self.equity_curve: list[tuple[datetime, float]] = []
        # 메모리 trades 관리 (I-009 (나) — DataStore 미사용)
        self._next_trade_id = 0
        self._open_trades: dict[int, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []

    @staticmethod
    def _parse_dt(value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        s = str(value)
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = datetime.strptime(s, "%Y-%m-%d")
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    # ---- 추상 구현 ----

    async def initialize(self) -> None:
        await self.broker.initialize()
        balance = await self.broker.get_balance()
        self.risk_manager.set_initial_balance(balance)
        await self._load_candles()
        self._build_features_cache()

    def _build_features_cache(self) -> None:
        """활성 strategies의 entry_timeframe별로 OOS 전체 features 사전계산.

        Phase E-2-2-OPT Step 1 — 매 봉 generate_signal에서 compute_multi_tf_features를
        처음부터 재계산하던 것을 1회로 축소. plugin은 ctx.precomputed_features를
        slice해서 사용 (lookahead는 features.get_features_for_ctx에서 ts < now로 차단).
        """
        entry_tfs = {s.entry_timeframe for s in self.strategies}
        for tf in entry_tfs:
            if tf in self.candles_per_tf and not self.candles_per_tf[tf].empty:
                self._features_cache[tf] = compute_multi_tf_features(
                    self.candles_per_tf, tf
                )
                logger.info(
                    "Features cache built: entry_tf=%s, rows=%d",
                    tf, len(self._features_cache[tf]),
                )

    async def shutdown(self) -> None:
        await self.broker.close()

    async def run(self) -> None:
        if self.master_timeframe is None or not self.candles_per_tf:
            logger.error("Cannot run: no master timeframe or candles loaded")
            return
        master_df = self.candles_per_tf.get(self.master_timeframe)
        if master_df is None or master_df.empty:
            logger.warning("Master candles empty for %s", self.master_timeframe)
            return

        start, end = self.start_dt, self.end_dt
        master_df = master_df.loc[
            (master_df.index >= pd.Timestamp(start))
            & (master_df.index <= pd.Timestamp(end))
        ]
        if master_df.empty:
            logger.warning("No master candles in range %s ~ %s", start, end)
            return
        if not self.strategies:
            logger.warning(
                "Backtest with 0 active strategies — no trades will occur"
            )
        logger.info(
            "Backtest run: %d %s candles, strategies=%s",
            len(master_df),
            self.master_timeframe,
            [s.name for s in self.strategies],
        )

        for ts, candle in master_df.iterrows():
            high = float(candle["high"])
            low = float(candle["low"])
            open_ = float(candle["open"])
            now = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

            # SL/TP 안전장치 — 봉 안에서 hit 여부는 high/low로 판정 (라이브와 일치)
            if self._position is not None:
                fill = self.check_candle_sl_tp(self._position, high, low)
                if fill is not None:
                    exit_price, reason = fill
                    await self.close_position(exit_price, reason, now=now)

            # I-B007 수정: ts 시점에는 직전 봉까지의 데이터로 평가 + open 가격으로 진입
            # _slice_candles는 ts 미만 슬라이스 (lookahead 제거)
            candles_slice = self._slice_candles(ts)
            balance = await self.broker.get_balance()

            if self._position is not None:
                exit_decision = self.check_strategy_exits(
                    candles_slice, open_, balance, now
                )
                if exit_decision is not None:
                    await self.close_position(open_, exit_decision.reason, now=now)
                    balance = await self.broker.get_balance()

            for tf in self.timeframes:
                if self._is_tf_boundary(now, tf):
                    await self.evaluate_strategies_on_bar(
                        tf, candles_slice, open_, balance, now
                    )
                    balance = await self.broker.get_balance()

            self.equity_curve.append((now, balance))

        if self._position is not None:
            last_ts = master_df.index[-1]
            last_close = float(master_df["close"].iloc[-1])
            last_now = (
                last_ts.to_pydatetime()
                if hasattr(last_ts, "to_pydatetime")
                else last_ts
            )
            await self.close_position(
                last_close, ExitReason.ENGINE_SHUTDOWN, now=last_now
            )
            balance = await self.broker.get_balance()
            self.equity_curve.append((last_now, balance))

        logger.info(
            "Backtest complete: equity=%.2f, trades=%d",
            balance if self.equity_curve else self.risk_manager.initial_balance,
            len(self.trades),
        )

    # ---- 거래 기록 (메모리) ----

    async def _record_trade_open(
        self,
        strategy_name: str,
        side: PositionSide,
        size: float,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        now: datetime,
    ) -> int:
        self._next_trade_id += 1
        tid = self._next_trade_id
        self._open_trades[tid] = {
            "id": tid,
            "strategy_name": strategy_name,
            "side": side.value,
            "size": size,
            "entry_price": entry_price,
            "entry_time": now,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
        }
        return tid

    async def _record_trade_close(
        self,
        trade_id: int,
        position: Position,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        trading_fee: float,
        funding_fee: float,
        exit_reason: str,
        now: datetime,
    ) -> None:
        rec = self._open_trades.pop(trade_id, None)
        if rec is None:
            rec = {
                "id": trade_id,
                "strategy_name": position.strategy_name,
                "side": position.side.value,
                "size": position.size,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
            }
        rec.update(
            {
                "exit_time": now,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "trading_fee": trading_fee,
                "funding_fee": funding_fee,
                "exit_reason": exit_reason,
                "status": "closed",
            }
        )
        self.trades.append(rec)

    # ---- 캔들 로딩 ----

    async def _load_candles(self) -> None:
        loader = HistoricalDataLoader(self.config)
        try:
            start_ms = int(self.start_dt.timestamp() * 1000)
            end_ms = int(self.end_dt.timestamp() * 1000)
            for tf in self.timeframes:
                df = await loader.download_range_merged(tf, start_ms, end_ms)
                self.candles_per_tf[tf] = df
                logger.info("Loaded %d %s candles", len(df), tf)
        finally:
            await loader.close()

    def _is_tf_boundary(self, ts: datetime, tf: str) -> bool:
        if tf == "1m":
            return ts.second == 0
        if tf == "5m":
            return ts.minute % 5 == 0 and ts.second == 0
        if tf == "15m":
            return ts.minute % 15 == 0 and ts.second == 0
        if tf == "1h":
            return ts.minute == 0 and ts.second == 0
        if tf == "4h":
            return ts.hour % 4 == 0 and ts.minute == 0 and ts.second == 0
        if tf == "1d":
            return ts.hour == 0 and ts.minute == 0 and ts.second == 0
        return False

    def _slice_candles(self, ts) -> dict[str, pd.DataFrame]:
        """ts 시점 직전까지의 캔들 반환 (I-B007 lookahead 제거).

        과거에는 `df.loc[:ts]`로 ts 봉을 포함시켰으나,
        이는 봉 시작 시점에 그 봉의 close 정보가 피처에 들어가는 lookahead bias.
        라이브 환경에서는 ts 시점에 그 봉이 시작도 안 한 상태이므로,
        직전 봉까지의 데이터로만 의사결정해야 함.
        """
        result: dict[str, pd.DataFrame] = {}
        for tf, df in self.candles_per_tf.items():
            if df.empty:
                result[tf] = df
            else:
                result[tf] = df[df.index < ts]
        return result

    # ---- 결과 집계 ----

    async def get_result(self) -> BacktestResult:
        # I-010: equity_curve 비어있으면 initial_balance fallback
        initial = self.risk_manager.initial_balance
        if self.equity_curve:
            final = self.equity_curve[-1][1]
        else:
            final = initial
        return BacktestResult(
            initial_balance=initial,
            final_balance=final,
            equity_curve=list(self.equity_curve),
            trades=list(self.trades),
        )

    # ---- 결과 파일 출력 (I-011) ----

    def write_reports(
        self,
        config_path: str | Path | None = None,
        out_root: str | Path | None = None,
    ) -> Path:
        """리포트 5종(trades / equity_curve / metrics / config_snapshot / png)을
        `data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{name}/{name}/`
        하위에 저장하고 디렉토리 경로를 반환.

        out_root가 지정되면 기본 경로 대신 그 디렉토리를 사용 (Phase E-2 evaluate_models.py).
        """
        config_name = "default"
        if config_path:
            config_name = Path(str(config_path)).stem
        if out_root is None:
            today = datetime.now().strftime("%y%m%d")
            start_str = self.start_dt.strftime("%Y-%m-%d")
            end_str = self.end_dt.strftime("%Y-%m-%d")
            out_root_path = REPORT_BASE / (
                f"{today}_backtest_{start_str}_{end_str}_{config_name}"
            )
        else:
            out_root_path = Path(out_root)
        out = out_root_path / config_name
        out.mkdir(parents=True, exist_ok=True)

        # trades.csv
        if self.trades:
            df = pd.DataFrame(self.trades)
            cols_order = [
                "id", "strategy_name", "side", "size",
                "entry_time", "entry_price", "exit_time", "exit_price",
                "stop_loss", "take_profit",
                "pnl", "pnl_pct", "trading_fee", "funding_fee",
                "exit_reason", "status",
            ]
            ordered = [c for c in cols_order if c in df.columns] + [
                c for c in df.columns if c not in cols_order
            ]
            df = df[ordered]
            df.to_csv(out / "trades.csv", index=False)
        else:
            (out / "trades.csv").write_text(
                "id,strategy_name,side,size,entry_time,entry_price,"
                "exit_time,exit_price,stop_loss,take_profit,pnl,pnl_pct,"
                "trading_fee,funding_fee,exit_reason,status\n",
                encoding="utf-8",
            )

        # equity_curve.csv
        if self.equity_curve:
            ec = pd.DataFrame(self.equity_curve, columns=["timestamp", "balance"])
            ec["equity"] = ec["balance"]  # 현재 unrealized 미추적, balance==equity
            ec.set_index("timestamp", inplace=True)
            ec.to_csv(out / "equity_curve.csv")
        else:
            (out / "equity_curve.csv").write_text(
                "timestamp,balance,equity\n", encoding="utf-8"
            )

        # metrics.json
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self._build_metrics(), f, indent=2, default=str, ensure_ascii=False)

        # config_snapshot.yaml — 자격증명 제거
        snapshot = {k: v for k, v in self.config.items()}
        if "exchange" in snapshot:
            ex = dict(snapshot["exchange"])
            for secret_key in ("api_key", "secret", "passphrase"):
                ex.pop(secret_key, None)
            snapshot["exchange"] = ex
        with open(out / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            yaml.dump(
                snapshot,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        # equity_curve.png
        try:
            self._plot_equity(out / "equity_curve.png")
        except Exception as e:
            logger.warning("Failed to render equity_curve.png: %s", e)

        return out

    def _build_metrics(self) -> dict[str, Any]:
        initial = self.risk_manager.initial_balance
        final = self.equity_curve[-1][1] if self.equity_curve else initial
        total_pnl = final - initial
        total_pct = (total_pnl / initial * 100.0) if initial > 0 else 0.0

        n_total = len(self.trades)
        winning = [t for t in self.trades if (t.get("pnl") or 0) > 0]
        losing = [t for t in self.trades if (t.get("pnl") or 0) < 0]
        n_winners = len(winning)
        n_losers = len(losing)
        win_rate = (n_winners / n_total * 100.0) if n_total > 0 else 0.0

        gp = sum(t["pnl"] for t in winning) if winning else 0.0
        gl = abs(sum(t["pnl"] for t in losing)) if losing else 0.0
        pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

        max_dd = 0.0
        if self.equity_curve:
            peak = self.equity_curve[0][1]
            for _, eq in self.equity_curve:
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = (peak - eq) / peak * 100.0
                    if dd > max_dd:
                        max_dd = dd

        avg_win = (gp / n_winners) if n_winners > 0 else 0.0
        avg_loss = (gl / n_losers) if n_losers > 0 else 0.0

        return {
            "integrated": {
                "initial_balance": round(initial, 2),
                "final_balance": round(final, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_pct, 2),
                "total_trades": n_total,
                "winning_trades": n_winners,
                "losing_trades": n_losers,
                "win_rate_pct": round(win_rate, 2),
                "gross_profit": round(gp, 2),
                "gross_loss": round(gl, 2),
                "profit_factor": (
                    round(pf, 2) if pf != float("inf") else "inf"
                ),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "max_drawdown_pct": round(max_dd, 2),
            },
            "by_strategy_name": self._split_by(
                lambda t: t.get("strategy_name", "_unknown")
            ),
            "by_exit_reason": self._split_by(
                lambda t: t.get("exit_reason", "_unknown")
            ),
            "by_direction": self._split_by(lambda t: t.get("side", "_unknown")),
        }

    def _split_by(self, key_fn) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for t in self.trades:
            k = key_fn(t)
            groups.setdefault(k, []).append(t)
        out: dict[str, dict[str, Any]] = {}
        for k, trs in groups.items():
            wins = [t for t in trs if (t.get("pnl") or 0) > 0]
            losses = [t for t in trs if (t.get("pnl") or 0) < 0]
            gp = sum(t["pnl"] for t in wins) if wins else 0.0
            gl = abs(sum(t["pnl"] for t in losses)) if losses else 0.0
            pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
            out[k] = {
                "trades": len(trs),
                "winning_trades": len(wins),
                "losing_trades": len(losses),
                "win_rate_pct": (
                    round(len(wins) / len(trs) * 100.0, 2) if trs else 0.0
                ),
                "pnl": round(sum(t["pnl"] for t in trs), 2),
                "profit_factor": (
                    round(pf, 2) if pf != float("inf") else "inf"
                ),
            }
        return out

    def _plot_equity(self, path: Path) -> None:
        if not self.equity_curve:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        timestamps = [t for t, _ in self.equity_curve]
        balances = [b for _, b in self.equity_curve]
        initial = self.risk_manager.initial_balance

        fig, axes = plt.subplots(
            2, 1, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]}
        )
        axes[0].plot(timestamps, balances, color="blue", linewidth=1, label="Equity")
        axes[0].axhline(
            y=initial, color="gray", linestyle="--", alpha=0.5,
            label=f"Initial ${initial:,.0f}",
        )
        axes[0].set_title(
            f"Backtest Equity Curve "
            f"({self.start_dt.date()} ~ {self.end_dt.date()})"
        )
        axes[0].set_ylabel("Equity ($)", color="blue")
        axes[0].tick_params(axis="y", labelcolor="blue")
        axes[0].grid(True, alpha=0.3)

        # BTC overlay
        try:
            btc_csv = Path("data/candles/BTC_USDT_USDT_1d.csv")
            if btc_csv.exists():
                btc = pd.read_csv(
                    btc_csv, parse_dates=["timestamp"], index_col="timestamp"
                )
                btc.index = pd.to_datetime(btc.index, utc=True)
                btc = btc.loc[
                    (btc.index >= pd.Timestamp(self.start_dt))
                    & (btc.index <= pd.Timestamp(self.end_dt))
                ]
                if not btc.empty:
                    ax2 = axes[0].twinx()
                    ax2.plot(
                        btc.index, btc["close"],
                        color="orange", linewidth=1, alpha=0.7, label="BTC",
                    )
                    ax2.set_ylabel("BTC Price ($)", color="orange")
                    ax2.tick_params(axis="y", labelcolor="orange")
        except Exception:
            pass
        axes[0].legend(loc="upper left")

        eq_series = pd.Series(balances, index=pd.DatetimeIndex(timestamps))
        peak = eq_series.expanding().max()
        dd = (peak - eq_series) / peak * 100.0
        axes[1].fill_between(dd.index, 0, dd, color="red", alpha=0.3)
        axes[1].set_title("Drawdown (%)")
        axes[1].set_ylabel("Drawdown %")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    # ---- 테스트용: 외부 캔들 주입 ----

    def inject_candles(self, candles_per_tf: dict[str, pd.DataFrame]) -> None:
        self.candles_per_tf = {
            tf: df.copy() for tf, df in candles_per_tf.items()
        }
