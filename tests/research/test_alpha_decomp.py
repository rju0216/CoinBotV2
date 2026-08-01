"""알파/베타 모집단·배리어 정합 분해 회귀 테스트 (Phase 7, I-014)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.experiments.alpha_decomp import (
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
