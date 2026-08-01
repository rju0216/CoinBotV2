"""알파/베타 분해 — **모집단·배리어 정합** (Phase 7, I-014 해소용).

**왜 새로 만드나 (I-014)**: §12-9 의 분해는 두 군데가 어긋나 있었다.
  ① **벤치가 배리어를 안 씀** — 전략은 ±w 배리어로 손절하는데 벤치는 무배리어 N봉 보유라,
     폭락장에서 *손절이 좌측 꼬리를 자른 것*이 "타이밍 실력(알파)"으로 계상된다.
  ② **모집단이 다름** — 전략은 **게이트 통과 거래**인데 벤치는 **전체 봉** 평균이라,
     게이트의 *진입 선택 효과*까지 방향 알파에 섞인다.

**여기서의 정의** (같은 봉·같은 배리어폭·같은 만기에서만 비교)::

    벤치(베타)   = 그 봉에서 **무조건 롱** 의 배리어 정합 실현
    전략         = 그 봉에서 **모델 방향**으로 진입한 배리어 정합 실현
    방향 알파    = 전략 − 벤치          (게이트 통과봉에서)
    선택 알파    = 게이트봉 벤치 − 전체봉 벤치   (어느 봉을 골랐나의 기여)

방향 알파는 **롱을 잡은 봉에서 정의상 0** 이다(무조건 롱과 행동이 같으므로). 따라서 방향
알파는 전적으로 **숏 거래에서만** 나온다 — "숏을 얼마나 잡았고 그게 맞았나"의 직접 측정이다.

**한계(명시)**: 봉 수준 프록시다. 라벨의 배리어 결과 = 거래 결과로 가정하며(Phase 4 실측
라벨매칭 97~99%), 엔진의 1m 체결·용량(N-트랜치)·수수료·펀딩은 반영하지 않는다. 손익 판정이
아니라 **분해 진단** 용도다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UP, DOWN = "up", "down"


def barrier_matched_long_returns(artifact: pd.DataFrame, candles: pd.DataFrame,
                                 horizon_bars: int) -> pd.Series:
    """그 봉에서 **무조건 롱** 진입 시의 배리어 정합 실현 수익률(분수).

    up → +w, down → −w (w=barrier_frac, 라벨과 동일 폭), expire → ``close_{i+N}/close_i − 1``.
    N봉 뒤 캔들이 없는 꼬리는 NaN.
    """
    c = candles[~candles.index.duplicated()].sort_index()
    if c.index.tz is None:
        c = c.tz_localize("UTC") if hasattr(c, "tz_localize") else c
    pos = c.index.get_indexer(artifact.index)
    cl = c["close"].to_numpy()
    ok = (pos >= 0) & (pos + horizon_bars < len(cl))
    exp_ret = np.full(len(artifact), np.nan)
    exp_ret[ok] = cl[pos[ok] + horizon_bars] / cl[pos[ok]] - 1.0
    w = artifact["barrier_frac"].to_numpy(dtype="float64")
    yt = artifact["y_true"]
    # y_true NaN(삼중배리어 동시터치 = 가장 모호한 봉)은 `== UP/DOWN` 이 모두 False 라
    # **조용히 expire(표류) 분기로 낙하**한다. 하드페일로 막는다(barrier_frac 에는 이미
    # 같은 성격의 가드가 있다 - oos_export).
    n_nan = int(yt.isna().sum())
    if n_nan:
        raise ValueError(
            f"y_true 에 NaN {n_nan}개 - 동시터치 봉이 expire 로 오계상된다. "
            f"아티팩트 생성 경로(harness NaN 드롭)를 확인할 것.")
    ytv = yt.to_numpy()
    out = np.where(ytv == UP, w, np.where(ytv == DOWN, -w, exp_ret))
    out = np.where(ok, out, np.nan)
    s = pd.Series(out, index=artifact.index, name="long_ret")
    # 캔들 매칭 커버리지 노출 (조용한 표본 축소 방지, F-10 정신)
    s.attrs["n_unmatched"] = int((pos < 0).sum())
    s.attrs["n_tail_dropped"] = int(((pos >= 0) & (pos + horizon_bars >= len(cl))).sum())
    return s


def decompose(artifact: pd.DataFrame, candles: pd.DataFrame, horizon_bars: int,
              gate: np.ndarray | pd.Series | None = None,
              pred_source: str = "ens") -> pd.DataFrame:
    """연도별 + 전체 알파/베타 분해 표 (bp 단위). index=연도('전체' 포함).

    gate=None 이면 전 봉이 대상. gate 지정 시 **방향 알파는 게이트 통과봉에서**,
    **선택 알파는 게이트봉 벤치 vs 전체봉 벤치**로 계산한다.
    """
    long_ret = barrier_matched_long_returns(artifact, candles, horizon_bars)
    up = artifact[f"{pred_source}_up"].to_numpy(dtype="float64")
    dn = artifact[f"{pred_source}_down"].to_numpy(dtype="float64")
    is_long = up >= dn
    strat = np.where(is_long, long_ret.to_numpy(), -long_ret.to_numpy())
    valid = ~np.isnan(long_ret.to_numpy())
    # Series 로 받으면 인덱스를 버리고 위치로만 쓰므로, 길이가 같고 순서만 다르면
    # **조용히 틀린다**. 인덱스가 있으면 아티팩트와 동일한지 강제한다.
    if gate is None:
        g = np.ones(len(artifact), bool)
    else:
        if isinstance(gate, pd.Series):
            if not gate.index.equals(artifact.index):
                raise ValueError("gate 인덱스가 artifact 와 불일치 - 위치 정렬 오염 방지")
            gate = gate.to_numpy()
        g = np.asarray(gate, dtype=bool)
        if len(g) != len(artifact):
            raise ValueError(f"gate 길이 {len(g)} != artifact {len(artifact)}")
    g = g & valid
    yr = artifact.index.year

    rows, dropped = {}, []
    for key in list(sorted(set(yr))) + ["전체"]:
        ym = np.ones(len(artifact), bool) if key == "전체" else (yr == key)
        m, mall = g & ym, valid & ym
        if m.sum() < 20:
            # 과거 구현은 여기서 **경고 없이** 연도 행을 버렸다. D-046 진전 바가
            # "연도별 선택알파 **전부** 양수"라 연도 행 누락은 판정을 바꾼다.
            if key != "전체" and m.sum() > 0:
                dropped.append((str(key), int(m.sum())))
            continue
        s_bp = float(np.nanmean(strat[m])) * 1e4
        b_bp = float(np.nanmean(long_ret.to_numpy()[m])) * 1e4
        b_all = float(np.nanmean(long_ret.to_numpy()[mall])) * 1e4
        short_m = m & ~is_long
        rows[str(key)] = {
            "n": int(m.sum()),
            "long_share_pct": round(float(is_long[m].mean()) * 100, 1),
            "strategy_bp": round(s_bp, 1),
            "benchmark_bp(베타)": round(b_bp, 1),
            "direction_alpha_bp": round(s_bp - b_bp, 1),
            "selection_alpha_bp": round(b_bp - b_all, 1),
            "n_short": int(short_m.sum()),
            "short_alpha_bp": (round(float(np.nanmean(
                (strat - long_ret.to_numpy())[short_m])) * 1e4, 1)
                if short_m.sum() >= 20 else np.nan),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "year"
    out.attrs["dropped_years"] = dropped        # [(연도, 표본수)] - 20봉 미만이라 제외
    out.attrs["n_unmatched"] = long_ret.attrs.get("n_unmatched", 0)
    out.attrs["n_tail_dropped"] = long_ret.attrs.get("n_tail_dropped", 0)
    return out


def ev_expire_gate(artifact: pd.DataFrame, horizon_bars: int, tf_hours: float,
                   ev_k: float = 2.0, taker: float = 0.0005,
                   funding_per_8h: float = 0.0001,
                   pred_source: str = "ens") -> np.ndarray:
    """D2 게이트 재현 (EV k=2 + 만기게이트) — 봉 수준. `econ_l3` 규약과 동일 식."""
    up = artifact[f"{pred_source}_up"].to_numpy(dtype="float64")
    dn = artifact[f"{pred_source}_down"].to_numpy(dtype="float64")
    ex = artifact[f"{pred_source}_expire"].to_numpy(dtype="float64")
    w = artifact["barrier_frac"].to_numpy(dtype="float64")
    cost = 2.0 * taker + (horizon_bars * tf_hours / 8.0) * funding_per_8h
    return (np.abs(up - dn) * w >= ev_k * cost) & (np.maximum(up, dn) > ex)
