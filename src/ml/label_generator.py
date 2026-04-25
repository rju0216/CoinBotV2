"""레이블 생성기. 미래 수익률 기반 3-class 방향 분류.

레이블 매핑:
    0 = SHORT  (미래 수익률 < -threshold)
    1 = HOLD   (그 사이)
    2 = LONG   (미래 수익률 > +threshold)
    -1 = 미래 데이터 없음 (학습 시 제거)
"""

from __future__ import annotations

import pandas as pd


def generate_direction_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold_pct: float = 0.3,
) -> pd.Series:
    """3-class 방향 레이블 생성.

    Args:
        df: OHLCV DataFrame (close 컬럼 필수)
        horizon: 미래 N봉 후 수익률 기준
        threshold_pct: ±threshold% 이상이면 LONG/SHORT (0.3 = 0.3%)

    Returns:
        pd.Series: 0=SHORT, 1=HOLD, 2=LONG, -1=미래없음
    """
    future_return = df["close"].pct_change(horizon).shift(-horizon) * 100
    labels = pd.Series(1, index=df.index, dtype=int, name="label")
    labels[future_return > threshold_pct] = 2
    labels[future_return < -threshold_pct] = 0
    labels[future_return.isna()] = -1
    return labels
