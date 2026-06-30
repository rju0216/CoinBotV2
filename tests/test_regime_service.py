"""RegimeService 테스트 (D4-2d).

메커니즘(수동 모델, 빠름): warmup None·캐시·모델 선택·sliding-window 일관(H4).
E2E 통합(build_model, 1건): feature→zscore→filter→매핑→Contract 전 파이프라인.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.regime.artifact import RegimeModel, load_model, save_model
from src.strategy.regime.contract import Contract, RegimeDirection, RegimeType
from src.strategy.regime.features import FeatureConfig, ZScoreParams
from src.strategy.regime.hmm import GaussianEmission, HMM
from src.strategy.regime.mapping import StateMapping, StateStats
from src.strategy.regime.service import RegimeService
from src.strategy.regime.training import build_model


def _make_candles(n, start="2022-01-01", seed=0):
    rng = np.random.default_rng(seed)
    prices = 50000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": 1.0,
        },
        index=idx,
    )


def _manual_model(types, directions, valid_period):
    em = GaussianEmission(2, 3)
    em.means = np.array([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]])
    em.covs = np.array([np.eye(3), np.eye(3)])
    hmm = HMM(2, em)
    hmm.log_pi = np.log([0.5, 0.5])
    hmm.log_A = np.log([[0.9, 0.1], [0.1, 0.9]])
    mapping = StateMapping(
        types=types,
        directions=directions,
        tau=1.0,
        stats=(StateStats(0.0, 1.0), StateStats(0.0, 1.0)),
    )
    zscore = ZScoreParams(
        mean={"log_return": 0.0, "realized_vol": 0.0, "efficiency_ratio": 0.0},
        std={"log_return": 1.0, "realized_vol": 1.0, "efficiency_ratio": 1.0},
    )
    return RegimeModel(hmm, mapping, zscore, FeatureConfig(), valid_period, {})


_TREND_MODEL = _manual_model(
    (RegimeType.TREND, RegimeType.TREND),
    (RegimeDirection.LONG, RegimeDirection.LONG),
    ("2022-01-01", "2030-01-01"),
)


# ---- 메커니즘 ----

def test_warmup_returns_none():
    svc = RegimeService([_TREND_MODEL], window=30)
    assert svc.get_contract(_make_candles(40)) is None  # < warmup(48)


def test_returns_contract_with_enough_candles():
    svc = RegimeService([_TREND_MODEL], window=30)
    c = svc.get_contract(_make_candles(120))
    assert isinstance(c, Contract)
    assert c.type == RegimeType.TREND  # 매핑이 둘 다 trend
    assert 0.0 <= c.confidence <= 1.0 + 1e-9
    assert c.volatility > 0  # ATR


def test_same_ts_cached():
    svc = RegimeService([_TREND_MODEL], window=30)
    candles = _make_candles(120)
    c1 = svc.get_contract(candles)
    c2 = svc.get_contract(candles)  # 같은 ts → 캐시
    assert c1 is c2


def test_empty_candles_none():
    svc = RegimeService([_TREND_MODEL], window=30)
    assert svc.get_contract(_make_candles(0)) is None


def test_no_model_for_ts_returns_none():
    # 모델 valid_period 가 미래 → 현재 ts 커버 안 함
    future_model = _manual_model(
        (RegimeType.TREND, RegimeType.TREND),
        (RegimeDirection.LONG, RegimeDirection.LONG),
        ("2025-01-01", "2030-01-01"),
    )
    svc = RegimeService([future_model], window=30)
    assert svc.get_contract(_make_candles(120, start="2022-01-01")) is None


def test_model_selected_by_ts():
    a = _manual_model(
        (RegimeType.TREND, RegimeType.TREND),
        (RegimeDirection.LONG, RegimeDirection.LONG),
        ("2022-01-01", "2023-01-01"),
    )
    b = _manual_model(
        (RegimeType.RANGE, RegimeType.RANGE),
        (RegimeDirection.NONE, RegimeDirection.NONE),
        ("2023-01-01", "2024-01-01"),
    )
    svc = RegimeService([a, b], window=30)
    candles = _make_candles(400, start="2022-12-20")  # 2022-12-20 ~ 2023-01-05
    # 2022 봉 (warmup 후) → A(TREND)
    c_2022 = svc.get_contract(candles.iloc[:150])  # ts ≈ 2022-12-26
    assert candles.index[149] < pd.Timestamp("2023-01-01", tz="UTC")
    assert c_2022.type == RegimeType.TREND
    svc.reset()
    # 2023 봉 → B(RANGE)
    c_2023 = svc.get_contract(candles.iloc[:350])  # ts ≈ 2023-01-03
    assert candles.index[349] >= pd.Timestamp("2023-01-01", tz="UTC")
    assert c_2023.type == RegimeType.RANGE


def test_sliding_window_independent_of_older_history():
    # 같은 ts 의 contract 는 직전 window+warmup 봉에만 의존 (H4: 백테=라이브 토대).
    svc = RegimeService([_TREND_MODEL], window=30)
    candles = _make_candles(300)
    T = 250
    full = svc.get_contract(candles.iloc[: T + 1])
    svc.reset()
    short = svc.get_contract(candles.iloc[T - 100 : T + 1])  # 마지막 101봉만
    assert full == short  # window30+warmup48=78 ≪ 100 → 더 오래된 과거 무관


# ---- E2E 통합 ----

def test_e2e_build_save_load_serve(tmp_path):
    candles = _make_candles(800, seed=1)
    train = candles.iloc[:600]
    model = build_model(
        train, k=2, tau=1.0,
        valid_period=("2022-01-01", "2030-01-01"),
        n_init=2, seed=0,
    )
    # artifact 왕복
    path = tmp_path / "m.json"
    save_model(model, path)
    loaded = load_model(path)

    svc = RegimeService([loaded], window=50)
    got = 0
    for i in (650, 700, 750, 799):
        c = svc.get_contract(candles.iloc[: i + 1])
        svc.reset()
        assert isinstance(c, Contract)
        assert c.type in (RegimeType.TREND, RegimeType.RANGE)
        assert c.direction in (
            RegimeDirection.LONG, RegimeDirection.SHORT, RegimeDirection.NONE,
        )
        assert c.volatility > 0
        got += 1
    assert got == 4


def test_inference_does_not_use_smoothed(monkeypatch):
    """추론(get_contract)은 forward-only filter 만 — smoothed_posterior(미래 사용)를
    호출하면 안 됨. smoothed 를 폭탄으로 바꿔 호출 시 실패하도록 가드(회귀 고정).
    """
    model = _manual_model(
        (RegimeType.TREND, RegimeType.RANGE),
        (RegimeDirection.LONG, RegimeDirection.NONE),
        ("2022-01-01", "2030-01-01"),
    )

    def _boom(*a, **k):
        raise AssertionError("추론이 smoothed_posterior(미래 사용)를 호출함")

    monkeypatch.setattr(model.hmm, "smoothed_posterior", _boom)
    svc = RegimeService([model], window=30)
    c = svc.get_contract(_make_candles(120))  # smoothed 안 부르면 정상
    assert isinstance(c, Contract)
