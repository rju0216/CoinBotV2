"""합성 OHLCV 생성기 — Phase 2 피처 테스트 공유 픽스처 (DRY, 규칙 8).

랜덤워크 종가 + 유효 OHLC(high≥max(o,c), low≤min(o,c)) + 양수 거래량. 인과성·정상성
회귀에서 실데이터 없이 결정론적으로 재현한다(시드 고정). test_volatility 의 로컬
``_ohlc`` 와 동형 — Phase 2 신규 test 는 여기서 import 해 중복을 없앤다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    ret = rng.normal(0, 0.01, n)
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    hi_extra = np.abs(rng.normal(0, 0.005, n))
    lo_extra = np.abs(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + hi_extra)
    low = np.minimum(open_, close) * (1 - lo_extra)
    vol = rng.uniform(1, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
