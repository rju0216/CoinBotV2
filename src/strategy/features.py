"""공통 피처 엔지니어링. 학습과 추론 양쪽에서 동일하게 사용.

indicators.py의 기존 함수를 조합하여 단일/멀티 타임프레임 피처 벡터를 생성한다.
단일 TF 기준 27개 피처. 멀티TF 시 상위 TF 피처를 forward-fill merge.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.core.types import StrategyContext
from src.data.historical import TF_MS
from src.strategy.indicators import (
    compute_adx,
    compute_atr,
    compute_bbands,
    compute_bb_width,
    compute_choppiness,
    compute_efficiency_ratio,
    compute_ema,
    compute_macd,
    compute_rsi,
)


# 단일 TF 피처명 목록 (27개)
BASE_FEATURE_NAMES: list[str] = [
    # 추세 (4)
    "price_ema10_ratio",
    "price_ema50_ratio",
    "ema10_ema50_ratio",
    "ema20_ema200_ratio",
    # 모멘텀 (5)
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "rsi_7",
    # 변동성 (3)
    "atr_pct",
    "bb_width",
    "bb_position",
    # 추세 강도 (6)
    "adx",
    "plus_di",
    "minus_di",
    "di_diff",
    "choppiness",
    "efficiency_ratio",
    # 거래량 (1)
    "volume_ratio",
    # 수익률 (5)
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "volatility_20",
    # 캔들 구조 (3)
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """단일 타임프레임 OHLCV → 피처 DataFrame (27개 컬럼).

    반환 DataFrame은 df와 같은 인덱스. 초기 행은 NaN — 호출자가 dropna 처리.
    """
    feat = pd.DataFrame(index=df.index)

    # ── 추세 지표 ──
    ema10 = compute_ema(df, 10)
    ema20 = compute_ema(df, 20)
    ema50 = compute_ema(df, 50)
    ema200 = compute_ema(df, 200)

    # pandas_ta가 데이터 부족 시 None을 반환할 수 있으므로 NaN Series로 대체
    _nan = pd.Series(np.nan, index=df.index)
    if ema10 is None:
        ema10 = _nan
    if ema20 is None:
        ema20 = _nan
    if ema50 is None:
        ema50 = _nan
    if ema200 is None:
        ema200 = _nan

    feat["price_ema10_ratio"] = df["close"] / ema10 - 1
    feat["price_ema50_ratio"] = df["close"] / ema50 - 1
    feat["ema10_ema50_ratio"] = ema10 / ema50 - 1
    feat["ema20_ema200_ratio"] = ema20 / ema200 - 1

    # ── 모멘텀 ──
    # compute_macd는 내부에서 pandas_ta를 호출하는데, 데이터 부족 시
    # pandas_ta가 None을 반환하여 columns 할당에서 crash할 수 있으므로
    # 충분한 행이 있을 때만 호출한다.
    macd = None
    if len(df) >= 26:  # MACD slow period
        try:
            macd = compute_macd(df)
        except (AttributeError, TypeError):
            macd = None
    if macd is not None and not macd.empty:
        feat["macd"] = macd["macd"]
        feat["macd_signal"] = macd["signal"]
        feat["macd_hist"] = macd["histogram"]
    else:
        feat["macd"] = _nan
        feat["macd_signal"] = _nan
        feat["macd_hist"] = _nan

    rsi_14 = compute_rsi(df, 14)
    rsi_7 = compute_rsi(df, 7)
    feat["rsi_14"] = rsi_14 if rsi_14 is not None else _nan
    feat["rsi_7"] = rsi_7 if rsi_7 is not None else _nan

    # ── 변동성 ──
    atr14 = compute_atr(df, 14)
    if atr14 is not None:
        feat["atr_pct"] = atr14 / df["close"]
    else:
        feat["atr_pct"] = _nan

    try:
        bb_w = compute_bb_width(df, 20, 2.0)
        feat["bb_width"] = bb_w if bb_w is not None else _nan
    except (AttributeError, TypeError):
        feat["bb_width"] = _nan

    try:
        bb = compute_bbands(df, 20, 2.0)
        if bb is not None and not bb.empty:
            bb_range = (bb["upper"] - bb["lower"]).replace(0, np.nan)
            feat["bb_position"] = (df["close"] - bb["lower"]) / bb_range
        else:
            feat["bb_position"] = _nan
    except (AttributeError, TypeError):
        feat["bb_position"] = _nan

    # ── 추세 강도 ──
    try:
        adx = compute_adx(df, 14)
    except (AttributeError, TypeError):
        adx = None
    if adx is not None and not adx.empty:
        feat["adx"] = adx["adx"]
        feat["plus_di"] = adx["plus_di"]
        feat["minus_di"] = adx["minus_di"]
        feat["di_diff"] = adx["plus_di"] - adx["minus_di"]
    else:
        feat["adx"] = _nan
        feat["plus_di"] = _nan
        feat["minus_di"] = _nan
        feat["di_diff"] = _nan

    # I-B011: 빈 df 입력 시 ta.atr가 None 반환 → .rolling() AttributeError
    # I-B001 해결 시 다른 indicator는 try-except 감쌌으나 choppiness/efficiency_ratio 누락
    try:
        chop = compute_choppiness(df, 14)
        feat["choppiness"] = chop if chop is not None else _nan
    except (AttributeError, TypeError):
        feat["choppiness"] = _nan
    try:
        er = compute_efficiency_ratio(df, 10)
        feat["efficiency_ratio"] = er if er is not None else _nan
    except (AttributeError, TypeError):
        feat["efficiency_ratio"] = _nan

    # ── 거래량 ──
    vol_sma20 = df["volume"].rolling(20).mean()
    feat["volume_ratio"] = df["volume"] / vol_sma20.replace(0, np.nan)

    # ── 수익률 파생 ──
    feat["return_1"] = df["close"].pct_change(1)
    feat["return_5"] = df["close"].pct_change(5)
    feat["return_10"] = df["close"].pct_change(10)
    feat["return_20"] = df["close"].pct_change(20)
    feat["volatility_20"] = df["close"].pct_change().rolling(20).std()

    # ── 캔들 구조 ──
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    feat["body_ratio"] = (df["close"] - df["open"]) / hl_range
    feat["upper_shadow"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / hl_range
    feat["lower_shadow"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / hl_range

    return feat


def _is_in_progress_bar(
    sub_last_ts: pd.Timestamp,
    sub_tf_ms: int,
    now_ms: int,
) -> bool:
    """sub_tf의 마지막 봉이 now 시점에 진행 중인지 판별.

    봉 timestamp는 봉 시작 시각 → 마감 시각 = sub_last_ts + sub_tf_ms.
    now_ms < 마감 시각이면 진행 중. 학습 시엔 historical data라 now_ms가
    봉 마감 시각보다 항상 미래 → 항상 False (학습-추론 일관성 유지).

    I-BL007 Phase 3 fix.
    """
    sub_close_ms = sub_last_ts.value // 10**6 + sub_tf_ms
    return sub_close_ms > now_ms


def compute_multi_tf_features(
    candles: dict[str, pd.DataFrame],
    entry_tf: str,
) -> pd.DataFrame:
    """멀티 타임프레임 피처 병합.

    entry_tf 피처를 기준으로, 상위 TF 피처를 forward-fill merge.
    상위 TF 피처에는 '{tf}_' 접두사를 붙여 컬럼명 충돌을 방지한다.

    I-BL007 Phase 3: sub_tf 마지막 봉이 진행 중이면 그 봉을 제외하고 features
    계산 — 진행 중 봉의 부분 OHLC가 indicator NaN을 만드는 trigger 차단.
    학습 시(historical data)에는 now_ms가 항상 마감 시각보다 미래라 진행 중
    판정이 발동하지 않아 동일 동작 보장.
    """
    base = compute_features(candles[entry_tf])

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for tf, df in candles.items():
        if tf == entry_tf or df.empty:
            continue
        df_use = df
        if tf in TF_MS and _is_in_progress_bar(df.index[-1], TF_MS[tf], now_ms):
            df_use = df.iloc[:-1]
        tf_feat = compute_features(df_use).add_prefix(f"{tf}_")
        base = base.join(
            tf_feat.reindex(base.index).ffill(),
            how="left",
        )

    return base


def get_feature_names(
    entry_tf: str,
    extra_timeframes: list[str] | None = None,
) -> list[str]:
    """피처 컬럼명 목록 반환. 모델 저장/로드 시 사용."""
    all_names = list(BASE_FEATURE_NAMES)
    for tf in extra_timeframes or []:
        if tf != entry_tf:
            all_names.extend(f"{tf}_{n}" for n in BASE_FEATURE_NAMES)
    return all_names


# ─── I-BL007 Phase 3-C: dropna 헬퍼 + 진단 정보 ───


_KNOWN_TF_PREFIXES = ("1m", "5m", "15m", "1h", "4h", "1d")


def _group_nan_by_tf(columns: list[str]) -> dict[str, list[str]]:
    """NaN 컬럼명 리스트를 timeframe prefix별로 그룹화.

    예: ['1h_body_ratio', '4h_atr_pct', 'rsi_14'] →
        {'1h': ['body_ratio'], '4h': ['atr_pct'], 'entry_tf': ['rsi_14']}
    """
    grouped: dict[str, list[str]] = {}
    for col in columns:
        matched = None
        for prefix in _KNOWN_TF_PREFIXES:
            if col.startswith(f"{prefix}_"):
                matched = prefix
                break
        if matched:
            grouped.setdefault(matched, []).append(col[len(matched) + 1:])
        else:
            grouped.setdefault("entry_tf", []).append(col)
    return grouped


def get_clean_last_row(
    features: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray | None, dict]:
    """dropna로 NaN row 제외 후 마지막 row + 진단 정보 반환.

    학습 코드는 features.dropna() 후 학습 (train_*.py). 추론 시 동일 패턴 적용
    하여 학습-추론 일관성 보장. 진행 중 봉이 NaN row를 만들면 dropna로 제외되어
    직전 마감 봉의 features가 사용됨.

    diag["gap_to_latest"]: 가장 최근 봉 대비 사용된 row까지의 봉 수.
      - 0: 가장 최근 봉 사용 (이상적)
      - 1: 1봉 전 사용 (진행 중 봉 제외 발생)
      - N+: N봉 전 사용 (진행 중 봉 영향 잔존)
    long indicator(200 EMA 등) 자연 NaN은 used_ts 이전 row만 영향이라 카운트 X.

    Returns:
        (row_array, diagnostic_meta)
        - row_array: (F,) ndarray, None이면 가용 row 0개
        - diag: gap_to_latest + used_row_ts (정상) 또는 fail_reason + nan_by_tf (실패)
    """
    subset = features[feature_names]
    clean = subset.dropna()

    if len(clean) < 1:
        last_row_nan = subset.iloc[-1].isna() if len(subset) > 0 else None
        nan_cols = (
            [c for c in feature_names if last_row_nan is not None and last_row_nan[c]]
            if last_row_nan is not None
            else feature_names
        )
        return None, {
            "fail_reason": "all_features_nan",
            "nan_by_tf": _group_nan_by_tf(nan_cols),
        }

    used_ts = clean.index[-1]
    gap_to_latest = int((subset.index > used_ts).sum())
    return clean.iloc[-1].values, {
        "gap_to_latest": gap_to_latest,
        "used_row_ts": used_ts,
    }


def get_clean_features_for_sequence(
    features: pd.DataFrame,
    feature_names: list[str],
    lookback: int,
) -> tuple[pd.DataFrame | None, dict]:
    """DL 모델용: dropna로 NaN row 제외 후 lookback 충족 검증.

    Returns:
        (clean_subset, diag) — clean_subset이 None이면 lookback 미달
    """
    subset = features[feature_names]
    clean = subset.dropna()

    if len(clean) < lookback:
        return None, {
            "fail_reason": "dropna_lt_lookback",
            "available_rows": len(clean),
            "required_lookback": lookback,
        }

    used_ts = clean.index[-1]
    gap_to_latest = int((subset.index > used_ts).sum())
    return clean, {
        "gap_to_latest": gap_to_latest,
        "used_row_ts": used_ts,
    }


def get_features_for_ctx(
    ctx: StrategyContext,
    entry_tf: str,
) -> pd.DataFrame:
    """ctx 시점(ctx.now) 미만의 멀티TF 피처 반환.

    백테 모드: BacktestEngine이 OOS 전체 features를 ctx.precomputed_features에
              주입한 상태 → 여기서 ctx.now 미만으로 cutoff (lookahead 방어).
    라이브 모드: ctx.precomputed_features=None → 매 호출마다 즉시 계산. 단 동일하게
              ts < now 필터링 적용 (I-BL007 Phase 3-D) — 봉 t+1 시작 시점의 진행 중
              entry_tf 봉을 features에서 명시적 제외하여 학습/backtest와 동일 cycle
              보장. 신호: 봉 t의 features → 봉 t+1 시작가에 진입 (학습-추론 일관).
              이전엔 진행 중 봉의 NaN trigger를 dropna로 우연히 제외했으나,
              본 fix로 메커니즘 명시화 → gap=0/1 변동성 제거.
    """
    cache = ctx.precomputed_features
    if cache is not None:
        return cache.loc[cache.index < ctx.now]
    full = compute_multi_tf_features(ctx.candles, entry_tf)
    if full.empty:
        return full
    return full.loc[full.index < ctx.now]
