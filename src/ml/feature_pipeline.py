"""오프라인 피처 생성 + parquet 캐싱.

학습 스크립트가 사용. 캔들 데이터를 로드하고 피처를 계산한 뒤 parquet으로 캐싱한다.
캐시가 존재하면 바로 로드, --force-features 옵션으로 재생성 가능.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.historical import HistoricalDataLoader
from src.strategy.features import compute_features, compute_multi_tf_features

FEATURE_CACHE_DIR = Path("data/features")


async def build_features(
    config: dict,
    timeframes: list[str],
    entry_tf: str,
    start_ms: int,
    end_ms: int,
    force: bool = False,
) -> pd.DataFrame:
    """피처 DataFrame 생성 또는 캐시 로드.

    Args:
        config: YAML 기반 config dict
        timeframes: 사용할 타임프레임 목록 (entry_tf 포함)
        entry_tf: 진입 타임프레임 (피처 기준 인덱스)
        start_ms, end_ms: 데이터 범위 (밀리초)
        force: True면 캐시 무시하고 재생성

    Returns:
        피처 DataFrame (NaN 포함 — 호출자가 dropna 처리)
    """
    cache_key = (
        f"features_{'_'.join(sorted(timeframes))}"
        f"_{entry_tf}_{start_ms}_{end_ms}"
    )
    cache_path = FEATURE_CACHE_DIR / f"{cache_key}.parquet"

    if not force and cache_path.exists():
        return pd.read_parquet(cache_path)

    # 캔들 로드
    loader = HistoricalDataLoader(config)
    candles: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        candles[tf] = await loader.download_range_merged(tf, start_ms, end_ms)

    # 피처 계산
    if len(timeframes) == 1:
        features = compute_features(candles[entry_tf])
    else:
        features = compute_multi_tf_features(candles, entry_tf)

    # 캐시 저장
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache_path)

    return features
