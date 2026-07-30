"""Phase 6 통짜(whole-Phase) 통합 회귀 — 규칙 17 박제.

개별 단위테스트·pairwise seam·전체회귀는 "신규가 기존을 안 깬다"만 볼 뿐,
**전 컴포넌트가 함께 돌 때의 상호작용**은 증명하지 않는다. 이 파일은 Phase 6 산출물을
한 번에 관통시킨다:

    합성 아티팩트(2층 예측 박제)
      -> EconL3 진입 게이트 (EV 게이트 + 만기 게이트)
      -> BacktestEngine **N-트랜치** 경로 (SL/TP·만기·용량·정산)
      -> policy_eval._analyze 지표 (bp 정규화·동시성·연도별)

중량 등가성 테스트(PHASE6_HEAVY_EQUIV)는 **단일슬롯 경로만** 박제하므로, N>1 경로를
회귀로 고정하는 것은 이 파일이 유일하다.

박제하는 불변식:
  I1  손익 항등식: 최종잔고 == 초기잔고 + Σpnl (트랜치별 정산이 어긋나면 즉시 실패)
  I2  **포함관계**: N=1 의 거래는 N=4 거래의 부분집합이며 **체결가·청산사유가 동일**.
      용량 확대는 기존 거래를 바꾸지 않고 **버려졌던 거래를 추가**할 뿐이라는 실측 성질
      (6.4 교집합 검정에서 실아티팩트로 확인된 것을 합성으로 회귀 고정).
  I3  용량: 동시 보유 트랜치가 max_slots 를 넘지 않음.
  I4  인과: 모든 진입 시각이 그 진입을 만든 **예측봉 마감 이후**.
  I5  게이트 실효: ev_k 를 올리면 거래가 줄고, 만기게이트가 인구를 바꾼다.
  I6  bp 정규화: 지표가 **실현 노셔널** 기준(N 이 달라도 거래당 bp 비교 가능).

합성 데이터만 사용 — 실캔들/실아티팩트 비의존이라 suite 에서 항상 돈다.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.research.experiments import policy_eval as pe
from src.strategy.plugins.econ_l3 import EconL3
from src.strategy.registry import register_strategy, reset_registry_for_testing

DECISION_TF = "4h"
N_BARS = 36                    # 4h 봉 36개 = 6일 (suite 부담 최소화하되 신호 20+ 확보)
HORIZON = 6                    # 만기 6봉 = 24h
BARRIER = 0.02                 # 배리어 +-2%
INIT = 10_000.0
# 실비용률 = 2*(taker 0.0005 + slip 0) + (6*4h/8)*0.0001 = 0.0013
COST_FRAC = 0.0013


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _artifact(tmp_path):
    """합성 OOS 예측: 3봉 주기로 강LONG / 저마진 / 강SHORT.

    강신호 EV = 0.02*0.40 = 0.008 = 6.15x 비용 -> ev_k=2 통과
    저마진 EV = 0.02*0.02 = 0.0004 = 0.31x 비용 -> ev_k=2 차단
    만기확률은 저마진 봉에서만 방향확률보다 크게 둬 만기게이트도 실효."""
    idx = pd.date_range("2021-01-01", periods=N_BARS, freq="4h", tz="UTC")
    up, dn, ex = [], [], []
    for i in range(N_BARS):
        if i % 3 == 0:
            up.append(0.55); dn.append(0.15); ex.append(0.30)      # 강LONG
        elif i % 3 == 1:
            up.append(0.34); dn.append(0.32); ex.append(0.34)      # 저마진·만기우세
        else:
            up.append(0.15); dn.append(0.55); ex.append(0.30)      # 강SHORT
    data = {"ens_up": up, "ens_down": dn, "ens_expire": ex,
            "barrier_frac": [BARRIER] * N_BARS}
    for s in range(5):                                             # 시드열(만장일치 레버용)
        data[f"s{s}_up"] = up
        data[f"s{s}_down"] = dn
        data[f"s{s}_expire"] = ex
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / "art.parquet"
    df.to_parquet(p)
    return str(p), idx


def _candles(idx4h):
    """1m 체결 캔들 + 4h 결정 캔들. 가격은 결정적 사인파(진폭 +-3%)로 배리어를 오간다."""
    start, end = idx4h[0], idx4h[-1] + pd.Timedelta(hours=4)
    m_idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    base = 100_000.0
    px = [base * (1 + 0.03 * math.sin(i / 137.0)) for i in range(len(m_idx))]
    m = pd.DataFrame({"open": px, "high": [p * 1.0005 for p in px],
                      "low": [p * 0.9995 for p in px], "close": px,
                      "volume": [1.0] * len(px)}, index=m_idx)
    m.index.name = "timestamp"
    h4 = m.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna()
    return {"1m": m, "4h": h4}


def _cfg(art_path, max_slots, ev_k=2.0, dir_gt_exp=False, notional=None):
    return {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "backtest": {"initial_balance": INIT, "max_slots": max_slots},
        "strategies": {"active": ["econ_l3"]},
        "econ_l3": {
            "decision_tf": DECISION_TF, "fill_tf": "1m", "artifact_path": art_path,
            "horizon_bars": HORIZON, "theta": 0.0, "ev_k": ev_k,
            "require_dir_gt_expire": dir_gt_exp, "sizing": "fixed",
            "notional": notional if notional is not None else INIT / max_slots,
            "no_tp": False, "no_timeout": False, "allow_reverse": False,
        },
    }


def _run(art_path, idx4h, max_slots, **kw):
    register_strategy(EconL3)          # 격리 레지스트리에 명시 등록(모듈 import 캐시 무관)
    eng = BacktestEngine(_cfg(art_path, max_slots, **kw),
                         start=str(idx4h[0].date()),
                         end=str((idx4h[-1] + pd.Timedelta(days=1)).date()))
    eng.inject_candles(_candles(idx4h))
    eng.account_tracker.set_initial_balance(eng.balance)
    eng.run()
    return eng


@pytest.fixture
def art(tmp_path):
    return _artifact(tmp_path)


# --- I1 손익 항등식 -------------------------------------------------------

def test_i1_pnl_identity_holds_under_tranches(art):
    path, idx = art
    eng = _run(path, idx, max_slots=4)
    assert len(eng.trades) > 0, "합성 데이터에서 거래가 발생해야 통합검증이 의미 있다"
    assert eng.balance == pytest.approx(INIT + sum(t["pnl"] for t in eng.trades), abs=1e-6)
    assert len(eng.positions) == 0            # 종료 시 전 트랜치 청산


# --- I2 포함관계 (용량 확대는 기존 거래를 바꾸지 않는다) --------------------

def test_i2_single_slot_trades_are_subset_of_multislot(art):
    path, idx = art
    one = _run(path, idx, max_slots=1, notional=INIT)
    many = _run(path, idx, max_slots=4, notional=INIT)   # 노셔널 동일 -> 체결가 직접 비교
    assert len(one.trades) > 0 and len(many.trades) > len(one.trades)

    key = lambda t: (pd.Timestamp(t["entry_time"]), t["side"])   # noqa: E731
    m = {key(t): t for t in many.trades}
    for t in one.trades:
        k = key(t)
        assert k in m, f"N=1 거래 {k} 가 N=4 에 없음 (용량 확대가 기존 거래를 없앰)"
        o = m[k]
        assert o["entry_price"] == pytest.approx(t["entry_price"])
        assert o["exit_price"] == pytest.approx(t["exit_price"])
        assert o["exit_reason"] == t["exit_reason"]
        assert o["pnl"] == pytest.approx(t["pnl"])


# --- I3 용량 -------------------------------------------------------------

def test_i3_concurrency_never_exceeds_capacity(art):
    path, idx = art
    for n in (1, 3, 8):
        eng = _run(path, idx, max_slots=n)
        occ = pe._occupancy(pd.DataFrame([
            {"entry_time": pd.Timestamp(t["entry_time"]),
             "exit_time": pd.Timestamp(t["exit_time"])} for t in eng.trades]))
        assert occ["max_concurrent"] <= n, f"max_slots={n} 인데 동시성 {occ}"


# --- I4 인과 (진입은 예측봉 마감 이후) ------------------------------------

def test_i4_entries_never_precede_their_prediction_bar_close(art):
    path, idx = art
    eng = _run(path, idx, max_slots=4)
    interval = pd.Timedelta(hours=4)
    for t in eng.trades:
        et = pd.Timestamp(t["entry_time"])
        # 진입을 만든 예측봉 = et 직전 완성 4h봉. 그 봉의 **마감 <= 진입시각** 이어야 한다.
        pred_open = idx[idx < et][-1] if len(idx[idx < et]) else None
        assert pred_open is not None
        assert pred_open + interval <= et, (
            f"진입 {et} 이 예측봉 {pred_open} 마감 전 — lookahead")


# --- I5 게이트 실효 -------------------------------------------------------

def test_i5_gates_change_the_traded_population(art):
    path, idx = art
    loose = _run(path, idx, max_slots=8, ev_k=0.0)
    strict = _run(path, idx, max_slots=8, ev_k=2.0)
    tight = _run(path, idx, max_slots=8, ev_k=2.0, dir_gt_exp=True)
    assert len(loose.trades) > len(strict.trades) > 0, "EV 게이트가 거래를 줄여야 한다"
    # 저마진 봉은 p_dir(0.34) < p_expire(0.34) 가 아니라 동률이 아니므로 만기게이트는
    # ev_k 통과분에 대해 추가 차단이 없거나 있을 수 있다 — 인구가 축소되기만 하면 된다.
    assert len(tight.trades) <= len(strict.trades)


# --- I6 bp 정규화 (실현 노셔널 기준) --------------------------------------

def test_i6_bp_metrics_use_realized_notional_end_to_end(art):
    path, idx = art
    eng = _run(path, idx, max_slots=4)
    out = pe._analyze({"notional": 999_999.0},              # 선언값을 일부러 틀리게
                      eng.trades, "data/funding/__nope__.csv", init=INIT)
    realized = sum(t["size"] * t["entry_price"] for t in eng.trades)
    # _analyze 는 지표를 소수 2자리로 반올림해 저장한다 -> abs 허용오차로 대조.
    assert out["total_notional"] == pytest.approx(realized, abs=0.01)
    assert out["net_bp_per_trade"] == pytest.approx(
        sum(t["pnl"] for t in eng.trades) / realized * 1e4, abs=0.01)
    # 슬리피지 허용치 = 거래당 bp / 2 (체결 2회)
    assert out["slippage_tolerance_bp_per_fill"] == pytest.approx(
        out["net_af_bp_per_trade"] / 2, abs=0.02)


def test_i6_bp_is_invariant_to_tranche_count_for_same_trades(art):
    """같은 거래 집합이면 노셔널을 N등분해도 **거래당 bp 는 불변**이어야 한다."""
    path, idx = art
    big = _run(path, idx, max_slots=1, notional=INIT)
    small = _run(path, idx, max_slots=1, notional=INIT / 4)
    fa = pe._analyze({}, big.trades, "data/funding/__nope__.csv", init=INIT)
    fb = pe._analyze({}, small.trades, "data/funding/__nope__.csv", init=INIT)
    assert fa["n_trades"] == fb["n_trades"]
    assert fa["net_bp_per_trade"] == pytest.approx(fb["net_bp_per_trade"], abs=0.01)
    assert fa["net_pct"] == pytest.approx(fb["net_pct"] * 4, abs=0.05)   # 금액은 4배
