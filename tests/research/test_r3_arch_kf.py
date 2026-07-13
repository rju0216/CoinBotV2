"""R3.2 구조·KF 러너 회귀 — build 캐시(KF 재빌드)·모델 파라미터·평가 배선(합성·소형)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data.loader import csv_filename
from src.research.experiments.r3_arch_kf import run_r32
from src.research.validation.ledger import RunLedger


def _write_synth_1h(candle_dir, n=1400, seed=0):
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


def test_r32_mechanics(tmp_path):
    _write_synth_1h(tmp_path)
    ledger = str(tmp_path / "r32_ledger.jsonl")
    res = run_r32(
        candle_dir=str(tmp_path), ledger_path=ledger, mlp_seeds=(0,),
        configs=[("base d2w32", None, None),
                 ("d3w32", None, {"trunk_depth": 3}),
                 ("KF dof2", {"dof": 2.0}, None)],   # KF override → X 재빌드 경로
        split={"train_min": 300, "val_size": 150},
    )
    assert res is not None
    assert len(res.evals) == 3
    assert res.baseline_ll > 0
    assert res.best in [e[0] for e in res.evals]
    led = RunLedger(ledger)
    assert led.comparison_count("R3_arch_kf") == 3


def test_r32_skip_guard(tmp_path):
    assert run_r32(candle_dir=str(tmp_path)) is None
