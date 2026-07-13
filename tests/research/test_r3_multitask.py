"""R3 멀티태스크 러너 회귀 — reach 준비·멀티태스크 배선·평가(합성 데이터·소형 split).

실 데이터 전체(~45분)는 별도. 여기선 러너 배선: reach 준비(expire 마스킹)·단일/멀티 config
평가·ledger. 멀티태스크 모델 자체는 test_models.py 가 검증.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.data.loader import csv_filename
from src.research.experiments.r3_multitask import run_r3
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


def test_r3_runner_mechanics(tmp_path):
    _write_synth_1h(tmp_path)
    ledger = str(tmp_path / "r3_ledger.jsonl")
    res = run_r3(
        candle_dir=str(tmp_path), ledger_path=ledger, mlp_seeds=(0,),
        configs=[("single-task/z", False, 0.0, "z"),
                 ("multitask λ0.1/z", True, 0.1, "z")],
        split={"train_min": 300, "val_size": 150},
    )
    assert res is not None
    assert len(res.evals) == 2
    names = [e.name for e in res.evals]
    assert "single-task/z" in names
    # 기준선·최소 config 산출
    assert res.baseline_ll > 0
    assert res.best in names
    # 예산 계수 = 평가 config 수
    led = RunLedger(ledger)
    assert led.comparison_count("R3_multitask") == 2


def test_r3_skip_guard(tmp_path):
    assert run_r3(candle_dir=str(tmp_path)) is None
