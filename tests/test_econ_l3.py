"""EconL3(경제성·로버스트 진입 3층) 단위 테스트 (Phase 6 Step 6.3).

검증 축:
- 퇴화 등가: 레버 전부 off → DumbL3 와 신호 시퀀스 완전 일치 (회귀 가드)
- 축 A(EV 게이트): 실비용률 공식 · 문턱 경계 · **θ 통과/EV 차단**(설계의 핵심 주장)
- 비용상수 ↔ config/default.yaml `fees` 드리프트 박제 (규칙 13)
- 축 B(로버스트): 시드 만장일치 · 지속성 K · **폴드 경계 갭 가드** · 미래행 불참조(인과)
합성 아티팩트(4h) — 실데이터 비의존.
"""

from __future__ import annotations

import numpy as np
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


# --- 축 C: 선택률 고정 (G-C, Phase 7 Step 7.5c) -----------------------------
#
# 합성 아티팩트는 **온그리드 연속 4h**이고 점수(=|p_up−p_down|×barrier_frac)를 통제해 만든다.
# 창은 ev_rank_window_days 로 축소(4h → 1일 = 6봉)해 워밍업·상한을 짧은 계열에서 재현한다.
#
# ★점수열은 반드시 비단조★ — 단조 증가열을 쓰면 현재 봉이 항상 창의 최댓값이라 통과율이
# 100% 로 나오고, 인과 테스트도 "전부 True == 전부 True" 인 공허한 green 이 된다.

_RANK_BARS_PER_DAY = 6


def _rank_margins(n, lo=0.02, hi=0.60, mult=37, mod=101):
    """결정적 유사난수 마진(비단조·재현가능). 부호는 방향, 절대값이 점수를 만든다."""
    return [lo + (hi - lo) * (((i * mult) % mod) / (mod - 1)) for i in range(n)]


def _make_rank_artifact(tmp_path, margins, w=_W, expire=None, name="rank.parquet"):
    """margins[i] = p_up − p_down. p_expire 를 주면 만기게이트 적격을 통제한다."""
    n = len(margins)
    idx = pd.date_range("2021-01-01 00:00", periods=n, freq="4h", tz="UTC")
    exp = [0.02] * n if expire is None else list(expire)
    up, dn = [], []
    for m, e in zip(margins, exp):
        rest = 1.0 - e
        assert abs(m) <= rest + 1e-12, f"마진 {m} 이 잔여질량 {rest} 초과 - 확률 음수"
        up.append(rest / 2 + m / 2)
        dn.append(rest / 2 - m / 2)
    data = {"ens_up": up, "ens_down": dn, "ens_expire": exp, "barrier_frac": [w] * n}
    for s in range(5):                       # 시드 열(만장일치 경로 재사용 대비)
        data[f"s{s}_up"], data[f"s{s}_down"], data[f"s{s}_expire"] = up, dn, exp
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    p = tmp_path / name
    df.to_parquet(p)
    return str(p), idx


def _rank_params(path, rate=1 / 3, days=1, **over):
    base = {"theta": 0.0, "ev_rank_rate": rate, "ev_rank_window_days": days}
    base.update(over)                        # over 가 항상 이긴다(중복 kwarg 방지)
    return _params(path, **base)


def _assert_nonvacuous(flags, label):
    """플래그가 전부 True/False 면 그 위의 어떤 대조도 무의미하다(공허한 green 차단)."""
    vals = list(flags)
    assert any(vals) and not all(vals), f"{label}: 플래그가 상수 - 검증이 공허함"


def test_rank_gate_causal_under_truncation(tmp_path):
    """★인과 박제★ 미래 행을 잘라내도 공통 구간의 게이트 플래그가 불변."""
    margins = _rank_margins(40)
    full, _ = _make_rank_artifact(tmp_path, margins, name="rank_full.parquet")
    trunc, _ = _make_rank_artifact(tmp_path, margins[:20], name="rank_trunc.parquet")
    f = EconL3(_rank_params(full))._pred["_econ_rank_ok"]
    t = EconL3(_rank_params(trunc))._pred["_econ_rank_ok"]
    _assert_nonvacuous(t, "truncated")
    assert list(f.iloc[: len(t)]) == list(t)


def test_rank_gate_causal_under_future_perturbation(tmp_path):
    """★인과 박제★ 미래 봉 점수를 최대로 흔들어도 과거 플래그 불변(절단이 아닌 교란)."""
    margins = _rank_margins(40)
    perturbed = list(margins)
    perturbed[25:] = [0.95] * len(perturbed[25:])
    a, _ = _make_rank_artifact(tmp_path, margins, name="rank_a.parquet")
    b, _ = _make_rank_artifact(tmp_path, perturbed, name="rank_b.parquet")
    fa = EconL3(_rank_params(a))._pred["_econ_rank_ok"]
    fb = EconL3(_rank_params(b))._pred["_econ_rank_ok"]
    _assert_nonvacuous(fa.iloc[:25], "pre-perturbation")
    assert list(fa.iloc[:25]) == list(fb.iloc[:25])
    assert list(fa) != list(fb), "교란이 이후 구간조차 안 바꿈 - 교란 주입 실패"


def test_rank_gate_warmup_no_trade(tmp_path):
    """창 미충족 구간은 무거래 - 첫 (창-1)봉 전부 False."""
    path, idx = _make_rank_artifact(tmp_path, _rank_margins(30))
    p = EconL3(_rank_params(path))
    win = p.rank_window_bars
    assert win == _RANK_BARS_PER_DAY                     # 1일 × 6봉/일
    flags = list(p._pred["_econ_rank_ok"])
    _assert_nonvacuous(flags, "warmup case")
    assert not any(flags[: win - 1]), "워밍업 구간에서 진입 발생"
    assert p.rank_warmup_bars == win - 1
    for i in range(win - 1):                             # 엔진 호출 경로에서도 무거래
        assert p.generate_signal(_ctx(idx, i)).side == SignalSide.HOLD


@pytest.mark.parametrize("rate", [0.2, 1 / 3, 0.5])
def test_rank_gate_holds_selection_rate(tmp_path, rate):
    """★핵심★ 점수 분포가 달라도 실현 통과율이 목표에 고정된다."""
    n = 600
    a, _ = _make_rank_artifact(tmp_path, _rank_margins(n), name="rk_a.parquet")
    # 스케일이 10배 작고 시간에 따라 붕괴하는 계열(실 아티팩트의 마진 붕괴 모사)
    b, _ = _make_rank_artifact(
        tmp_path, [m * 0.1 * (0.995 ** i) for i, m in enumerate(_rank_margins(n, mult=53))],
        name="rk_b.parquet")
    for path in (a, b):
        p = EconL3(_rank_params(path, rate, days=5))     # 창 30봉
        _assert_nonvacuous(p._pred["_econ_rank_ok"], path)
        assert p.rank_realized_rate_post_warmup == pytest.approx(rate, abs=0.03), (
            f"{path}: 실현 {p.rank_realized_rate_post_warmup:.3f} vs 목표 {rate:.3f}")
        assert p.rank_capped_bars == 0


def test_rank_gate_rate_is_config_invariant(tmp_path):
    """★arm 의 존재 이유★ 절대문턱(EV)은 통과율이 갈리지만 백분위는 안 갈린다."""
    n, rate = 600, 0.2
    small, idx = _make_rank_artifact(
        tmp_path, _rank_margins(n, lo=0.005, hi=0.03), name="rk_s.parquet")
    big, _ = _make_rank_artifact(
        tmp_path, _rank_margins(n, lo=0.30, hi=0.60), name="rk_g.parquet")

    def _ev_pass_rate(path):
        m = EconL3(_params(path, theta=0.0, ev_k=2.0))
        return sum(m.generate_signal(_ctx(idx, i)).is_actionable
                   for i in range(n)) / n

    lo_rate, hi_rate = _ev_pass_rate(small), _ev_pass_rate(big)
    assert lo_rate < 0.05 and hi_rate > 0.95, (
        f"대비군 무효: EV 통과율 {lo_rate:.2f} vs {hi_rate:.2f}")
    for path in (small, big):                             # 같은 두 config 를 백분위로
        p = EconL3(_rank_params(path, rate, days=5))
        assert p.rank_realized_rate_post_warmup == pytest.approx(rate, abs=0.03)


def test_rank_gate_capped_when_eligible_below_target(tmp_path):
    """적격(만기게이트 통과) 봉이 목표율보다 적으면 적격 전부 통과 + 상한 플래그."""
    n, rate = 300, 0.5
    expire = [0.02 if i % 6 == 0 else 0.90 for i in range(n)]   # 적격률 1/6 < 목표 1/2
    path, _ = _make_rank_artifact(
        tmp_path, _rank_margins(n, lo=0.01, hi=0.09), expire=expire, name="rk_cap.parquet")
    p = EconL3(_rank_params(path, rate, days=5, require_dir_gt_expire=True))
    ok = p._pred["_econ_rank_ok"].to_numpy()
    warm = p.rank_window_bars - 1
    eligible = p._pred["ens_expire"].to_numpy() < 0.5
    assert p.rank_capped_bars > 0, "상한이 걸렸는데 플래그가 0"
    assert p.rank_realized_rate_post_warmup == pytest.approx(1 / 6, abs=0.02)
    assert ok[warm:][eligible[warm:]].all(), "상한 시 적격 봉은 전부 통과해야 함"
    assert not ok[~eligible].any(), "부적격 봉이 통과"


def test_rank_gate_off_is_byte_identical(tmp_path):
    """비파괴 - ev_rank_rate=0 이면 기존 동작과 신호 시퀀스 완전 일치, 열도 안 만든다."""
    path, idx = _make_rank_artifact(tmp_path, _rank_margins(30))
    off, dumb = EconL3(_params(path, theta=0.0)), DumbL3(_params(path, theta=0.0))
    assert "_econ_rank_ok" not in off._pred.columns
    for i in range(len(idx)):
        e, d = off.generate_signal(_ctx(idx, i)), dumb.generate_signal(_ctx(idx, i))
        assert e.side == d.side and e.confidence == pytest.approx(d.confidence)


@pytest.mark.parametrize("bad", [
    {"ev_k": 2.0},                        # 절대문턱과 동시 = G-A 도 G-C 도 아님
    {"mtf_gate": True},                   # mask 에 반영 안 되는 게이트 → 조용한 이탈
    {"trend_gate": True},
    {"avoid_vol": "high"},
    {"ev_rank_window_days": 0},
    {"ev_rank_rate": 1.0},                # (0,1) 밖
    {"ev_rank_rate": -0.1},
])
def test_rank_gate_hardfails_on_conflicting_config(tmp_path, bad):
    """조용한 선택률 이탈을 만드는 조합은 전부 구성 단계에서 예외."""
    path, _ = _make_rank_artifact(tmp_path, _rank_margins(30))
    with pytest.raises(ValueError):
        EconL3(_rank_params(path, **bad))


def test_rank_gate_hardfails_on_nan_barrier(tmp_path):
    """barrier_frac NaN 은 비교가 전부 False 라 조용히 표본에서 빠진다 → 하드페일."""
    path, _ = _make_rank_artifact(tmp_path, _rank_margins(30), name="rk_nan.parquet")
    df = pd.read_parquet(path)
    df.loc[df.index[5], "barrier_frac"] = float("nan")
    df.to_parquet(path)
    with pytest.raises(ValueError, match="barrier_frac"):
        EconL3(_rank_params(path))


# --- fresh-eyes 회귀 공백 보강 (L-1 / L-8) -------------------------------

def test_rank_gate_hardfails_on_unsorted_or_duplicate_index(tmp_path):
    """★L-1★ 비정렬 인덱스는 rolling 창을 **비인과**로 만든다 - 하드페일.

    가드가 없으면 '과거 창'이 조용히 미래를 포함하고, 중복은 같은 봉을 두 번 가중한다.
    """
    path, _ = _make_rank_artifact(tmp_path, _rank_margins(30), name="rk_ord.parquet")
    df = pd.read_parquet(path)

    rev = path.replace(".parquet", "_rev.parquet")
    df.iloc[::-1].to_parquet(rev)
    with pytest.raises(ValueError, match="시간순"):
        EconL3(_rank_params(rev))

    dup = path.replace(".parquet", "_dup.parquet")
    pd.concat([df, df.iloc[[5]]]).sort_index().to_parquet(dup)
    with pytest.raises(ValueError, match="중복"):
        EconL3(_rank_params(dup))


def test_rank_gate_mask_honors_theta_tie_convention(tmp_path):
    """★L-8★ θ>0 × rank 조합 - mask 의 θ 규약이 엔진(`>=`)과 같아야 한다.

    fresh-eyes 가 수동으로 확인한 동작을 회귀로 박제한다(그 전엔 축 C 테스트가 전부
    theta=0 이라 미검증이었다). θ 와 **정확히 같은** p_dir 봉은 통과해야 한다.
    """
    n = 300
    margins = _rank_margins(n)
    path, idx = _make_rank_artifact(tmp_path, margins, name="rk_theta.parquet")
    df = pd.read_parquet(path)
    p_dir = np.maximum(df.ens_up.to_numpy(), df.ens_down.to_numpy())
    theta = float(p_dir[7])                                    # 7번 봉이 정확히 경계

    m = EconL3(_rank_params(path, 0.5, days=5, theta=theta))
    mask_n = int((p_dir >= theta).sum())
    assert 0 < mask_n < n, "경계 θ 가 인구를 나누지 못함 - 검증이 공허함"
    # 게이트 통과봉은 전부 θ 이상이어야 한다(θ 미만이 새면 mask 불일치).
    ok = m._pred["_econ_rank_ok"].to_numpy()
    assert (p_dir[ok] >= theta).all(), "θ 미만 봉이 통과 - mask 가 엔진 규약과 불일치"
    # 타이 봉(p_dir == theta)은 `>=` 규약상 **적격**이다(축출되면 안 된다).
    tie = np.isclose(p_dir, theta)
    assert tie.any()
    m0 = EconL3(_rank_params(path, 0.5, days=5, theta=0.0))
    assert int(m0._pred["_econ_rank_ok"].sum()) >= int(ok.sum()), "θ 상향이 인구를 늘림"


def test_rank_gate_mask_honors_expire_tie_convention(tmp_path):
    """★L-8★ 만기게이트 타이(p_dir == p_expire)는 `>` 규약상 **부적격**이어야 한다."""
    n = 120
    margins = _rank_margins(n, lo=0.02, hi=0.30)
    # 절반은 p_dir == p_expire 로 정확히 동률이 되게 만든다.
    idx = pd.date_range("2021-01-01 00:00", periods=n, freq="4h", tz="UTC")
    up, dn, ex = [], [], []
    for i, mrg in enumerate(margins):
        if i % 2 == 0:                       # 동률 봉: p_dir == p_expire
            e = (1.0 - mrg) / 2.0            # up = e + m, dn = e - ... -> p_dir == e
            up.append(e + mrg); dn.append(e - mrg + mrg); ex.append(e + mrg)
            s = up[-1] + dn[-1] + ex[-1]
            up[-1], dn[-1], ex[-1] = up[-1] / s, dn[-1] / s, ex[-1] / s
        else:
            up.append(0.60); dn.append(0.20); ex.append(0.20)
    data = {"ens_up": up, "ens_down": dn, "ens_expire": ex, "barrier_frac": [_W] * n}
    for s_i in range(5):
        data[f"s{s_i}_up"], data[f"s{s_i}_down"], data[f"s{s_i}_expire"] = up, dn, ex
    p = tmp_path / "rk_tie.parquet"
    d = pd.DataFrame(data, index=idx)
    d.index.name = "timestamp"
    d.to_parquet(p)

    m = EconL3(_rank_params(str(p), 0.5, days=5, require_dir_gt_expire=True))
    pd_arr = np.maximum(np.array(up), np.array(dn))
    tie = np.isclose(pd_arr, np.array(ex))
    assert tie.any(), "동률 봉이 없어 검증이 공허함"
    ok = m._pred["_econ_rank_ok"].to_numpy()
    assert not ok[tie].any(), "p_dir == p_expire 동률 봉이 통과 - `>` 규약 위반"
