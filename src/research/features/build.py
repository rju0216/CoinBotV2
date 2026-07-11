"""피처 조립기 — OHLCV → X DataFrame (Phase 2, MasterPlan §2 계약).

각 피처 함수를 **기본창**으로 호출해 열로 이어붙인다(설계 §12 흐름). Phase 2 는
기본창만 — 창 순차탐색(coordinate descent)은 Phase 3 R2(``DEFAULT_PARAMS`` 를 폴드별로
바꿔 재호출). 정규화는 여기서 안 한다(harness 소유, F-6).

**확장 규율**: 새 축은 ``_FEATURE_COLUMNS`` 에 (함수, 파라미터키) 로 등록하면 조립기
구조 변경 없이 붙는다 — 계열2(ER/Hurst, Step 2.2)·계열1(KF, Step 2.3, 다중열) 추가 예정.
각 피처는 Series(단일 열) 또는 DataFrame(다중 열, 예: KF slope+불확실성)을 반환할 수 있다.
"""

from __future__ import annotations

import pandas as pd

from src.research.features import kalman, simple, trend_strength

# 기본창/기본파라미터 (Phase 2 placeholder — Phase 3 R2 coordinate descent 가 탐색).
# 1h 기준 대략 2일(48봉) 규모. Hurst 는 R/S 하위창 분할 위해 더 김(128). KF 는 창이
# 아니라 Q/R/dof 하이퍼(설계 §7 예외). 값 자체는 탐색축이지 확정 아님(기둥 5·설계 §13).
DEFAULT_PARAMS: dict[str, dict] = {
    "vol_change": {"estimator": "atr", "window": 48, "lag": 24},
    "relative_volume": {"window": 48},
    "semi_dev": {"window": 48},
    "er": {"window": 48},
    "hurst": {"window": 128},
    "kf": {"q_level": 1e-5, "q_slope": 1e-6, "r": 1e-4, "dof": 4.0, "warmup": 20},
    # close_position: 창 없음
}


def build_features(
    df: pd.DataFrame,
    params: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """감사 통과 OHLCV → 피처 X DataFrame (index = df.index).

    ``params`` 로 기본창을 덮어쓸 수 있다(Phase 3 창 탐색). 반환 X 는 자기정규화 전
    원시 피처 — harness 가 폴드 train-only 로 정규화한다(계약 §2). 라벨(BarrierLabels)의
    NaN 행과의 정렬·purge 는 splitter/harness 가 처리한다(피처는 index 만 맞춘다).
    """
    p = {k: {**v} for k, v in DEFAULT_PARAMS.items()}
    for k, v in (params or {}).items():
        p.setdefault(k, {}).update(v)

    cols: list[pd.Series | pd.DataFrame] = [
        simple.vol_change(df, **p["vol_change"]),
        simple.relative_volume(df, **p["relative_volume"]),
        simple.rolling_semi_deviation(df, **p["semi_dev"]),
        simple.close_position(df),
        trend_strength.efficiency_ratio(df, **p["er"]),
        trend_strength.hurst(df, **p["hurst"]),
        kalman.kf_trend(df, **p["kf"]),          # DataFrame(kf_slope, kf_uncertainty)
    ]
    X = pd.concat(cols, axis=1)
    return X
