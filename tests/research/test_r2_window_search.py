"""R2 창 순차탐색 러너 회귀 — coordinate descent 메커니즘(합성 데이터·소형 split).

실 데이터 전체 탐색(~6h)은 별도 실행. 여기선 러너 배선이 올바른지 확인: config 캐시·
선택(MLP ll 최소)·트리 참고 기록·순위상관·예산 계수·ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data.loader import csv_filename
from src.research.experiments.r2_window_search import run_r2
from src.research.validation.ledger import RunLedger


def _write_synth_1h(candle_dir, n=1400, seed=0):
    """신호 있는 합성 1h OHLCV CSV (배리어가 터지도록 변동성 확보)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    close = 10000 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.abs(rng.normal(1000, 200, n))},
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    df.to_csv(candle_dir / csv_filename("BTC/USDT:USDT", "1h"))


def test_r2_coordinate_descent_mechanics(tmp_path):
    _write_synth_1h(tmp_path)
    ledger = str(tmp_path / "r2_ledger.jsonl")
    res = run_r2(
        candle_dir=str(tmp_path), ledger_path=ledger,
        cycles=1, mlp_seeds=(0,), max_comparisons=20,
        dims=[("er", "window", [24, 48])],
        split={"train_min": 300, "val_size": 150},
    )
    assert res is not None
    # 선택된 er 창은 후보 중 하나
    assert res.selected["er"]["window"] in (24, 48)
    # 스윕 1개(er.window), 곡선에 2 후보
    assert len(res.history) == 1
    assert len(res.history[0]["curve"]) == 2
    # 선택 = MLP ll 최소 후보와 일치
    curve = {c: mll for c, mll, tll in res.history[0]["curve"]}
    assert res.selected["er"]["window"] == min(curve, key=curve.get)
    # 순위상관(트리 vs MLP) 산출
    assert len(res.rank_agreement) == 1
    # 예산 계수: MLP 비교만 카운트(트리 관찰용 제외). default+er24 = 2 unique
    led = RunLedger(ledger)
    assert led.comparison_count("R2_window_search") == res.n_comparisons
    assert res.n_comparisons <= 2      # default(er48) + er24 (er48==default 캐시)


def test_r2_skip_guard_no_data(tmp_path):
    # 1h 부재 → None (skip-guard)
    assert run_r2(candle_dir=str(tmp_path)) is None
