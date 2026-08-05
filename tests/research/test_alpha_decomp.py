"""알파/베타 모집단·배리어 정합 분해 회귀 테스트 (Phase 7, I-014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.experiments.alpha_decomp import (
    assert_on_grid,
    barrier_matched_long_returns,
    decompose,
    ev_expire_gate,
)

N = 2


def _fixture(y_true, preds, w=0.05, n=60):
    """artifact + candles. preds = [(up,down,expire)] 반복 패턴."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    rows = [preds[i % len(preds)] for i in range(n)]
    art = pd.DataFrame(rows, columns=["ens_up", "ens_down", "ens_expire"], index=idx)
    art["y_true"] = [y_true[i % len(y_true)] for i in range(n)]
    art["barrier_frac"] = w
    candles = pd.DataFrame({"close": np.full(n + N, 100.0)},
                           index=pd.date_range("2024-01-01", periods=n + N, freq="4h", tz="UTC"))
    return art, candles


def test_barrier_matched_returns_use_label_barrier():
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)], w=0.05)
    r = barrier_matched_long_returns(art, c, N)
    assert r.iloc[0] == pytest.approx(0.05)    # up  → +w
    assert r.iloc[1] == pytest.approx(-0.05)   # down → −w
    assert r.iloc[2] == pytest.approx(0.0)     # expire → 평평한 캔들이라 0
    # 픽스처 캔들이 n+N 이라 전 봉 매칭 → 꼬리 드롭 0 (과거 테스트는 `or True` 로 무의미)
    assert r.notna().all()
    assert r.attrs["n_tail_dropped"] == 0 and r.attrs["n_unmatched"] == 0


def test_tail_and_unmatched_coverage_is_reported():
    """★fresh-eyes H-4★ 캔들이 모자라면 조용히 줄지 말고 커버리지로 드러나야 한다."""
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)], n=40)
    r = barrier_matched_long_returns(art, c.iloc[:30], N)   # 캔들을 30개로 자름
    assert r.attrs["n_unmatched"] == 10                      # 매칭 실패 10봉
    assert r.attrs["n_tail_dropped"] > 0                     # 꼬리도 드롭
    assert r.isna().sum() >= 10


def test_all_long_strategy_has_zero_direction_alpha():
    """★I-014 핵심★ 전부 롱이면 무조건롱 벤치와 행동이 같으므로 방향 알파는 정의상 0."""
    art, c = _fixture(["up", "down"], [(0.6, 0.2, 0.2)])   # 항상 롱
    d = decompose(art, c, N)
    assert d.loc["전체", "long_share_pct"] == 100.0
    assert d.loc["전체", "direction_alpha_bp"] == pytest.approx(0.0, abs=1e-6)
    assert d.loc["전체", "strategy_bp"] == pytest.approx(d.loc["전체", "benchmark_bp(베타)"])


def test_perfect_shorts_generate_direction_alpha():
    """하락 봉에서만 숏 → 방향 알파 = 2×|벤치| (부호를 뒤집었으므로)."""
    art, c = _fixture(["down"], [(0.2, 0.6, 0.2)], w=0.05)   # 항상 숏, 라벨 전부 down
    d = decompose(art, c, N)
    assert d.loc["전체", "long_share_pct"] == 0.0
    assert d.loc["전체", "benchmark_bp(베타)"] == pytest.approx(-500.0)   # −w = −5%
    assert d.loc["전체", "strategy_bp"] == pytest.approx(500.0)
    assert d.loc["전체", "direction_alpha_bp"] == pytest.approx(1000.0)
    assert d.loc["전체", "short_alpha_bp"] == pytest.approx(1000.0)


def test_selection_alpha_isolates_which_bars_were_picked():
    """게이트가 좋은 봉만 고르면 방향 알파 0 이어도 선택 알파가 양수로 잡힌다."""
    # 짝수봉 = up(+w), 홀수봉 = down(−w). 전부 롱 예측.
    art, c = _fixture(["up", "down"], [(0.6, 0.2, 0.2)], w=0.05)
    gate = np.array([i % 2 == 0 for i in range(len(art))])   # up 봉만 통과
    d = decompose(art, c, N, gate=gate)
    assert d.loc["전체", "direction_alpha_bp"] == pytest.approx(0.0, abs=1e-6)
    assert d.loc["전체", "benchmark_bp(베타)"] == pytest.approx(500.0)     # 고른 봉은 전부 +w
    assert d.loc["전체", "selection_alpha_bp"] > 400                        # 전체봉 평균(≈0) 대비
    assert d.loc["전체", "n"] == int(gate.sum())


def test_nan_label_hard_fails_instead_of_silent_expire():
    """★fresh-eyes M-1★ y_true NaN(동시터치)이 조용히 expire 로 계상되면 안 된다."""
    art, c = _fixture(["up", "down"], [(0.5, 0.3, 0.2)])
    art.loc[art.index[0], "y_true"] = None
    with pytest.raises(ValueError, match="y_true"):
        barrier_matched_long_returns(art, c, N)


def test_gate_series_with_wrong_index_is_rejected():
    """★fresh-eyes M-2★ Series gate 는 인덱스가 어긋나면 조용히 틀리므로 하드페일."""
    art, c = _fixture(["up", "down"], [(0.5, 0.3, 0.2)])
    bad = pd.Series([True] * len(art), index=art.index[::-1])
    with pytest.raises(ValueError, match="인덱스"):
        decompose(art, c, N, gate=bad)
    ok = pd.Series([True] * len(art), index=art.index)
    assert len(decompose(art, c, N, gate=ok)) > 0        # 정렬되면 통과


def test_dropped_years_are_recorded_not_silent():
    """★fresh-eyes H-4★ 20봉 미만 연도는 제외되지만 **기록**돼야 한다."""
    art, _ = _fixture(["up", "down"], [(0.5, 0.3, 0.2)], n=60)
    # 앞 55봉 = 2024, 뒤 5봉 = 2025 (20봉 미만이라 연도 행에서 제외돼야 함)
    idx = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=55, freq="4h", tz="UTC"))
        + list(pd.date_range("2025-01-01", periods=5, freq="4h", tz="UTC")))
    art.index = idx
    cand = pd.DataFrame({"close": 100.0}, index=pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC"))
        + list(pd.date_range("2025-01-01", periods=200, freq="4h", tz="UTC"))))
    d = decompose(art, cand, N)
    assert "2025" not in d.index                                   # 20봉 미만이라 제외
    assert ("2025", 5) in d.attrs["dropped_years"]                 # 그러나 기록됨


def test_ev_expire_gate_matches_econ_l3_formula():
    idx = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    art = pd.DataFrame({
        "ens_up": [0.60, 0.40, 0.34], "ens_down": [0.20, 0.38, 0.33],
        "ens_expire": [0.20, 0.22, 0.33], "barrier_frac": [0.05, 0.05, 0.05],
    }, index=idx)
    g = ev_expire_gate(art, horizon_bars=24, tf_hours=4.0, ev_k=2.0)
    cost = 2 * 0.0005 + (24 * 4 / 8) * 0.0001            # 22bp
    # 1봉: |0.4|*0.05=200bp >= 2*22=44bp, p_dir 0.6>0.2 → 통과
    assert bool(g[0]) is True
    # 2봉: |0.02|*0.05=10bp < 44bp → EV 미달
    assert bool(g[1]) is False
    # 3봉: |0.01|*0.05=5bp < 44 → 미달 (만기게이트도 0.34>0.33 로는 통과)
    assert bool(g[2]) is False
    assert cost == pytest.approx(0.0022)


# --- F-20: 위치 오프셋의 전제(온그리드 무결손) 강제 ------------------------

def test_on_grid_guard_accepts_clean_candles():
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)])
    assert_on_grid(c, "4h")                                  # 예외 없음
    r = barrier_matched_long_returns(art, c, N, timeframe="4h")
    assert r.notna().all()


def test_on_grid_guard_rejects_missing_bar():
    """★F-20 핵심★ 캔들 결손이 있으면 하드페일 - pos+N 이 시간상 N봉 뒤가 아니게 된다."""
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)])
    holed = c.drop(c.index[10])                              # 한 봉 제거
    with pytest.raises(ValueError, match="온그리드"):
        assert_on_grid(holed, "4h")
    with pytest.raises(ValueError, match="온그리드"):
        barrier_matched_long_returns(art, holed, N, timeframe="4h")


def test_on_grid_guard_is_nonvacuous_the_hole_really_changes_results():
    """가드가 막는 오염이 실재함을 보인다 - 결손 시 값이 실제로 달라진다.

    (가드가 '있으나 마나'면 회귀가 공허하다. expire 봉의 표류 수익이 바뀌는지 본다.)
    """
    idx = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    art = pd.DataFrame({"ens_up": 0.3, "ens_down": 0.2, "ens_expire": 0.5,
                        "y_true": "expire", "barrier_frac": 0.05}, index=idx)
    px = pd.DataFrame({"close": np.arange(100.0, 100.0 + 45)},  # 단조 증가 -> 위치가 값을 바꿈
                      index=pd.date_range("2024-01-01", periods=45, freq="4h", tz="UTC"))
    clean = barrier_matched_long_returns(art, px, N).to_numpy()
    holed_px = px.drop(px.index[5])
    holed = barrier_matched_long_returns(art, holed_px, N).to_numpy()   # 가드 off(구 경로)
    assert not np.allclose(clean[:20], holed[:20], equal_nan=True), (
        "결손이 결과를 안 바꾸면 이 가드의 회귀는 공허하다")
    with pytest.raises(ValueError):                                     # 가드 on 이면 차단
        barrier_matched_long_returns(art, holed_px, N, timeframe="4h")


def test_on_grid_guard_rejects_duplicate_and_unsorted():
    _, c = _fixture(["up"], [(0.5, 0.3, 0.2)])
    dup = pd.concat([c, c.iloc[[3]]]).sort_index()
    with pytest.raises(ValueError, match="중복"):
        assert_on_grid(dup, "4h")
    rev = c.iloc[::-1]
    with pytest.raises(ValueError, match="시간순"):
        assert_on_grid(rev, "4h")


def test_on_grid_guard_only_checks_used_span():
    """검사 범위는 '아티팩트 구간 + 지평 꼬리' - 그 밖의 결손은 무해하므로 통과."""
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)], n=20)
    tail = pd.DataFrame({"close": [100.0] * 5}, index=pd.date_range(
        c.index[-1] + pd.Timedelta(hours=12), periods=5, freq="4h", tz="UTC"))  # 뒤쪽 갭
    with_far_gap = pd.concat([c, tail])
    barrier_matched_long_returns(art, with_far_gap, N, timeframe="4h")   # 예외 없어야 함


def test_timeframe_none_preserves_legacy_behavior():
    """비파괴 - timeframe 미지정이면 검증 없이 기존과 동일한 값."""
    art, c = _fixture(["up", "down", "expire"], [(0.5, 0.3, 0.2)])
    a = barrier_matched_long_returns(art, c, N).to_numpy()
    b = barrier_matched_long_returns(art, c, N, timeframe="4h").to_numpy()
    assert np.allclose(a, b, equal_nan=True)


# --- F-21(2): ev_expire_gate 의 theta 인자 --------------------------------

def test_ev_expire_gate_theta_default_is_backward_compatible():
    """theta 기본 0.0 은 확률이 항상 >=0 이라 무조건 통과 - 기존 호출 불변."""
    art, _ = _fixture(["up"], [(0.55, 0.15, 0.30), (0.34, 0.32, 0.34)], w=0.02)
    assert np.array_equal(ev_expire_gate(art, 6, 4.0), ev_expire_gate(art, 6, 4.0, theta=0.0))


def test_ev_expire_gate_theta_narrows_population():
    """★F-21(2)★ theta 를 올리면 모집단이 좁아진다 - 과거엔 인자가 없어 조용히 넓었다."""
    art, _ = _fixture(["up"], [(0.55, 0.15, 0.30), (0.44, 0.26, 0.30)], w=0.02)
    base = ev_expire_gate(art, 6, 4.0, theta=0.0)
    tight = ev_expire_gate(art, 6, 4.0, theta=0.50)
    assert base.sum() > tight.sum() > 0, "theta 가 인구를 좁히지 못함"
    assert (tight & ~base).sum() == 0, "theta 는 좁히기만 해야 한다"
