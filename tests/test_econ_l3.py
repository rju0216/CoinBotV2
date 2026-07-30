"""EconL3(경제성·로버스트 진입 3층) 단위 테스트 (Phase 6 Step 6.3).

검증 축:
- 퇴화 등가: 레버 전부 off → DumbL3 와 신호 시퀀스 완전 일치 (회귀 가드)
- 축 A(EV 게이트): 실비용률 공식 · 문턱 경계 · **θ 통과/EV 차단**(설계의 핵심 주장)
- 비용상수 ↔ config/default.yaml `fees` 드리프트 박제 (규칙 13)
- 축 B(로버스트): 시드 만장일치 · 지속성 K · **폴드 경계 갭 가드** · 미래행 불참조(인과)
합성 아티팩트(4h) — 실데이터 비의존.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.enums import SignalSide
from src.core.types import StrategyContext
from src.strategy.plugins.dumb_l3 import DumbL3
from src.strategy.plugins.econ_l3 import (
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_TAKER_FEE_PCT,
    EconL3,
)
from src.utils.config_loader import load_config

# 합성 아티팩트 봉 구성 (4h) — b4 와 b5 사이 20:00 봉을 **비워** 갭 가드를 검증한다.
#  b0 LONG강 / b1 LONG강 / b2 LONG강(시드 4:1 분열) / b3 SHORT 저마진(θ통과·EV탈락)
#  b4 SHORT강 / [갭] / b5 SHORT강(갭 직후) / b6 SHORT강
_ENS_UP = [0.60, 0.60, 0.60, 0.44, 0.20, 0.20, 0.20]
_ENS_DOWN = [0.20, 0.20, 0.20, 0.45, 0.60, 0.60, 0.60]
_ENS_EXPIRE = [0.20, 0.20, 0.20, 0.11, 0.20, 0.20, 0.20]
_W = 0.02
_HORIZON = 6                       # 6봉 × 4h = 24h 보유 → 펀딩 3구간
# 실비용률 = 2×(0.0005+0) + (24/8)×0.0001 = 0.0013 (13bp)
_COST = 0.0013


def _make_artifact(tmp_path, n=7, seed_split_at=2, name="art.parquet"):
    """n봉 합성 아티팩트. seed_split_at 봉에서 s4 만 반대 방향(4:1 분열)."""
    base = pd.date_range("2021-01-01 00:00", periods=6, freq="4h", tz="UTC")
    idx = pd.DatetimeIndex(
        list(base[:5]) + [base[5] + pd.Timedelta(hours=4), base[5] + pd.Timedelta(hours=8)]
    )[:n]                                        # base[5]=20:00 을 건너뛴 = 갭
    data = {
        "ens_up": _ENS_UP[:n], "ens_down": _ENS_DOWN[:n], "ens_expire": _ENS_EXPIRE[:n],
        "barrier_frac": [_W] * n,
    }
    for s in range(5):
        flip = [s == 4 and i == seed_split_at for i in range(n)]
        data[f"s{s}_up"] = [d if f else u for u, d, f in zip(_ENS_UP[:n], _ENS_DOWN[:n], flip)]
        data[f"s{s}_down"] = [u if f else d for u, d, f in zip(_ENS_UP[:n], _ENS_DOWN[:n], flip)]
        data[f"s{s}_expire"] = _ENS_EXPIRE[:n]
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / name
    df.to_parquet(p)
    return str(p), idx


@pytest.fixture
def art(tmp_path):
    return _make_artifact(tmp_path)


def _params(path, **over):
    base = {"artifact_path": path, "decision_tf": "4h", "theta": 0.40,
            "horizon_bars": _HORIZON, "notional": 10_000.0}
    base.update(over)
    return base


def _ctx(idx, i, price=100_000.0):
    """엔진과 동일하게 **직전 완성봉까지의** df 를 넘긴다(마지막 행 = 평가 대상 봉)."""
    sl = idx[: i + 1]
    df = pd.DataFrame({"open": [price] * len(sl), "high": [price] * len(sl),
                       "low": [price] * len(sl), "close": [price] * len(sl)}, index=sl)
    return StrategyContext(candles={"4h": df}, current_price=price, balance=10_000.0,
                           position=None, is_slot_occupied=False, params={}, now=sl[-1])


# --- 퇴화 등가 -------------------------------------------------------------

def test_degenerate_equals_dumb(art):
    """레버 전부 off(ev_k=0·만장일치 off·K=0) → DumbL3 와 봉별 신호 완전 일치."""
    path, idx = art
    econ, dumb = EconL3(_params(path)), DumbL3(_params(path))
    for i in range(len(idx)):
        ctx = _ctx(idx, i)
        e, d = econ.generate_signal(ctx), dumb.generate_signal(ctx)
        assert e.side == d.side, f"bar{i} side 불일치"
        assert e.confidence == pytest.approx(d.confidence)
        assert e.meta.get("barrier_frac") == d.meta.get("barrier_frac")


# --- 축 A: EV 게이트 -------------------------------------------------------

def test_cost_frac_formula(art):
    """실비용률 = 2×(taker+slippage) + (보유시간/8)×펀딩상수."""
    path, _ = art
    p = EconL3(_params(path, ev_k=1.0))
    assert p.cost_frac == pytest.approx(_COST)


def test_cost_frac_scales_with_horizon(art):
    """지평이 길수록 펀딩분만큼 문턱이 올라간다(무튜닝 적응)."""
    path, _ = art
    short_h = EconL3(_params(path, ev_k=1.0, horizon_bars=6)).cost_frac
    long_h = EconL3(_params(path, ev_k=1.0, horizon_bars=30)).cost_frac   # 120h → 15구간
    assert long_h > short_h
    assert long_h == pytest.approx(0.001 + (30 * 4 / 8) * 0.0001)


def test_ev_gate_boundary(art):
    """b0: ev = w×(0.60−0.20) = 0.008. 문턱 = k×0.0013 을 사이에 두고 통과/차단."""
    path, idx = art
    k_edge = 0.008 / _COST                        # ≈ 6.1538
    assert EconL3(_params(path, ev_k=k_edge * 0.99)).generate_signal(
        _ctx(idx, 0)).side == SignalSide.LONG
    assert EconL3(_params(path, ev_k=k_edge * 1.01)).generate_signal(
        _ctx(idx, 0)).side == SignalSide.HOLD


def test_ev_gate_blocks_low_margin_that_theta_passes(art):
    """★설계 핵심★ b3(p_dir 0.45 ≥ θ0.40 이지만 마진 0.01) — θ는 통과시키고 EV는 차단."""
    path, idx = art
    ctx = _ctx(idx, 3)
    assert DumbL3(_params(path)).generate_signal(ctx).side == SignalSide.SHORT
    assert EconL3(_params(path, ev_k=1.0)).generate_signal(ctx).side == SignalSide.HOLD


def test_ev_gate_symmetric_for_short(art):
    """b4(SHORT 강, 마진 0.40) 는 같은 ev_k 에서 통과 — 방향 편향 없음."""
    path, idx = art
    sig = EconL3(_params(path, ev_k=1.0)).generate_signal(_ctx(idx, 4))
    assert sig.side == SignalSide.SHORT
    assert sig.meta["ev_frac"] == pytest.approx(_W * 0.40)
    assert sig.meta["cost_frac"] == pytest.approx(_COST)


def test_cost_constants_match_config():
    """규칙 13 — 플러그인 기본 비용상수가 config/default.yaml `accounting` 과 어긋나지 않는다."""
    fees = load_config("config/default.yaml")["accounting"]
    assert DEFAULT_TAKER_FEE_PCT == pytest.approx(float(fees["taker_fee_pct"]))
    assert DEFAULT_SLIPPAGE_PCT == pytest.approx(float(fees["slippage_pct"]))


# --- 축 B: 로버스트 선별 ---------------------------------------------------

def test_seed_unanimity_blocks_split_bar(art):
    """b0(5:0 합의) 통과 · b2(4:1 분열) 차단. 앙상블 신호 자체는 둘 다 LONG."""
    path, idx = art
    p = EconL3(_params(path, seed_unanimity=True))
    assert p.generate_signal(_ctx(idx, 0)).side == SignalSide.LONG
    assert p.generate_signal(_ctx(idx, 2)).side == SignalSide.HOLD
    assert DumbL3(_params(path)).generate_signal(_ctx(idx, 2)).side == SignalSide.LONG


def test_seed_unanimity_requires_columns(tmp_path):
    """시드 열 없는 아티팩트 + seed_unanimity=True → 하드페일(조용한 통과 금지)."""
    idx = pd.date_range("2021-01-01", periods=2, freq="4h", tz="UTC")
    df = pd.DataFrame({"ens_up": [0.6, 0.6], "ens_down": [0.2, 0.2],
                       "ens_expire": [0.2, 0.2], "barrier_frac": [0.02, 0.02]}, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / "noseed.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="시드 열 부재"):
        EconL3(_params(str(p), seed_unanimity=True))


def test_persistence_k2_and_gap_guard(art):
    """K=2: 직전 봉과 방향 연속이어야 통과. **갭 직후 봉(b5)은 연속 아님 → 차단.**

    방향 = [L,L,L,S,S,S,S] (b3 에서 전환). b4→b5 사이 20:00 봉이 비어 있음."""
    path, idx = art
    p = EconL3(_params(path, persistence_bars=2))
    sides = [p.generate_signal(_ctx(idx, i)).side for i in range(len(idx))]
    assert sides[0] == SignalSide.HOLD          # 최초 봉 = 직전 없음
    assert sides[1] == SignalSide.LONG          # b0→b1 연속 LONG
    assert sides[2] == SignalSide.LONG
    assert sides[3] == SignalSide.HOLD          # 방향 전환(L→S)
    assert sides[4] == SignalSide.SHORT         # b3→b4 연속 SHORT
    assert sides[5] == SignalSide.HOLD          # ★갭 가드★ 온그리드 아님
    assert sides[6] == SignalSide.SHORT         # b5→b6 연속 복귀


def test_persistence_k3_stricter(art):
    """K=3 은 K=2 보다 좁다(3봉 연속 요구) — b1 은 탈락, b2 는 통과."""
    path, idx = art
    p = EconL3(_params(path, persistence_bars=3))
    assert p.generate_signal(_ctx(idx, 1)).side == SignalSide.HOLD
    assert p.generate_signal(_ctx(idx, 2)).side == SignalSide.LONG


def test_persistence_flags_are_causal(tmp_path):
    """★인과 박제★ 지속성 플래그는 과거 행만 쓴다 — 미래 행을 잘라내도 공통 구간 불변."""
    full_path, _ = _make_artifact(tmp_path, n=7, name="full.parquet")
    trunc_path, _ = _make_artifact(tmp_path, n=4, name="trunc.parquet")
    f = EconL3(_params(full_path, persistence_bars=2))._pred["_econ_persist_ok"]
    t = EconL3(_params(trunc_path, persistence_bars=2))._pred["_econ_persist_ok"]
    assert list(f.iloc[: len(t)]) == list(t)


def test_combined_gates_intersect(art):
    """두 축은 독립 AND — b2 는 EV 통과하지만 시드 분열로 차단된다."""
    path, idx = art
    ev_only = EconL3(_params(path, ev_k=1.0))
    both = EconL3(_params(path, ev_k=1.0, seed_unanimity=True))
    assert ev_only.generate_signal(_ctx(idx, 2)).side == SignalSide.LONG
    assert both.generate_signal(_ctx(idx, 2)).side == SignalSide.HOLD
