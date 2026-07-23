"""oos_export 일반화(P1, Phase5 지평 프론티어) seam 테스트.

핵심 = **등가성 가드**: 일반화한 build_frontier_xy 가 채택 config(15m+1h/6h)에서
기존 build_adopted_xy 와 **정확히 동일한 X·y·barrier_frac** 를 낸다 → 파라미터화가
채택 산출을 바꾸지 않음(오염 0). 실데이터 의존 → 캔들 부재 시 skip(규칙 16).
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

_CANDLE_DIR = "data/candles"
_SYM = "BTC_USDT_USDT"


def _have(tf: str) -> bool:
    return os.path.exists(os.path.join(_CANDLE_DIR, f"{_SYM}_{tf}.csv"))


pytestmark = pytest.mark.skipif(
    not (_have("15m") and _have("1h") and _have("4h")),
    reason="프론티어 seam 테스트: 15m/1h/4h 캔들 캐시 필요",
)


def test_frontier_xy_equals_adopted():
    """build_frontier_xy(15m, PRIMARY_LABEL, higher=('1h',)) == build_adopted_xy (등가성 가드)."""
    from src.research.experiments.oos_export import build_adopted_xy, build_frontier_xy
    from src.research.experiments.tf_mtf import DECISION_TF, PRIMARY_LABEL

    _, X_a, y_a, sp_a, tags_a, bf_a = build_adopted_xy(_CANDLE_DIR)
    _, X_f, y_f, sp_f, tags_f, bf_f = build_frontier_xy(
        DECISION_TF, PRIMARY_LABEL, higher_tfs=("1h",), candle_dir=_CANDLE_DIR
    )
    pd.testing.assert_frame_equal(X_f, X_a)
    pd.testing.assert_series_equal(y_f, y_a)
    pd.testing.assert_series_equal(bf_f, bf_a)
    pd.testing.assert_series_equal(tags_f, tags_a)
    # splitter 동일 파라미터
    assert sp_f.label_horizon == sp_a.label_horizon
    assert sp_f.train_min == sp_a.train_min and sp_f.val_size == sp_a.val_size


def test_frontier_single_tf_uses_build_features():
    """단독TF(higher_tfs=()) 는 build_features(8열) — MTF 접두어 열 없음."""
    from src.research.experiments.oos_export import build_frontier_xy
    from src.research.features.build import build_features
    from src.research.experiments.tf_expansion import load_tf
    from src.research.labeling.triple_barrier import LabelParams

    label = LabelParams("atr", 96, 3.0, 24)          # 1h/1일
    _, X, y, sp, tags, bf = build_frontier_xy("1h", label, higher_tfs=(), candle_dir=_CANDLE_DIR)
    # 단독TF X = build_features 와 동일 열
    X_direct = build_features(load_tf("1h", _CANDLE_DIR))
    assert list(X.columns) == list(X_direct.columns)
    assert not any(c.startswith("mtf") for c in X.columns)   # MTF 열 없음
    # barrier_frac: 변동성창 워밍업 구간은 NaN 정상 → 정의된 값은 모두 양수
    assert bf.notna().sum() > 0 and bf.dropna().gt(0).all()   # 배리어폭 유효(워밍업 제외)
