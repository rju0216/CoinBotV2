"""오프라인 캔들 로더 — 순수 CSV(네트워크 없음), 심볼 파라미터화.

백테 인프라의 ``HistoricalDataLoader.load_from_csv``(CSV 파싱 관례)와
``TF_MS``(봉 간격 dict)를 **재사용**하고(MasterPlan Step 0.1 DRY), 연구용으로
UTC 정규화·컬럼/타입 검증을 얹는다.

주의: 이 로더는 **정렬·중복 제거·갭 처리를 하지 않는다.** 그 판정은
``audit.py`` 의 관심사다 (감사·리포트 + 손상만 하드페일 — 무단 정정/채움 금지).
따라서 downstream 은 ``load_ohlcv`` → ``audit_continuity`` 순으로 쓰거나
convenience ``load_audited`` 를 쓴다.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

# CSV 파싱 관례(load_from_csv)와 봉 간격(TF_MS)은 백테 로더와 단일 출처를 공유한다.
# (import 시 ccxt 모듈만 로드될 뿐 네트워크·거래소 인스턴스는 생성되지 않는다.)
from src.data.historical import TF_MS, HistoricalDataLoader
from src.research.data.audit import audit_continuity

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def csv_filename(symbol: str, timeframe: str) -> str:
    """심볼·TF → CSV 파일명. HistoricalDataLoader._csv_path 와 동일 관례.

    예: ("BTC/USDT:USDT", "1h") -> "BTC_USDT_USDT_1h.csv"
    """
    safe = symbol.replace("/", "_").replace(":", "_")
    return f"{safe}_{timeframe}.csv"


def load_ohlcv(
    symbol: str, timeframe: str, candle_dir: str = "data/candles"
) -> pd.DataFrame:
    """(symbol, timeframe) OHLCV CSV 를 로드해 UTC 정규화·검증만 하고 반환.

    Raises:
        FileNotFoundError: CSV 부재.
        ValueError: OHLCV 컬럼 누락 또는 timeframe 미지원.
    """
    if timeframe not in TF_MS:
        raise ValueError(f"미지원 timeframe: {timeframe} (지원: {list(TF_MS)})")

    path = os.path.join(candle_dir, csv_filename(symbol, timeframe))
    if not os.path.exists(path):
        raise FileNotFoundError(f"캔들 CSV 없음: {path}")

    df = HistoricalDataLoader.load_from_csv(path)  # 파싱 관례 재사용

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV 컬럼 누락 {missing}: {path}")

    # UTC tz-aware 정규화 (정렬/중복제거는 audit 의 몫 — 여기서 손대지 않음)
    df = df[OHLCV_COLUMNS].astype(float).copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    return df


def load_audited(
    symbol: str, timeframe: str, candle_dir: str = "data/candles"
) -> pd.DataFrame:
    """표준 진입점: 로드 + 감사 + 손상 시 하드페일. 감사 통과 df 반환.

    구조적 손상(중복·비단조·NaN·OHLC위반·비양수·오프그리드)이면 AuditError.
    갭은 정책상 하드페일이 아니므로 로깅만 한다(리포트는 ``audit_continuity`` 로
    별도 취득). downstream(splitter·regime 등)은 이 함수를 통해 "감사 통과 데이터만
    하류로" 계약을 **구성으로 강제**한다.

    Raises:
        AuditError: 구조적 손상.
        FileNotFoundError / ValueError: load_ohlcv 참조.
    """
    df = load_ohlcv(symbol, timeframe, candle_dir)
    report = audit_continuity(df, timeframe)
    if report.gaps:
        logger.warning(
            "%s %s: 갭 %d개 (missing_bars=%d, coverage=%.6f)",
            symbol, timeframe, len(report.gaps), report.n_missing_bars, report.coverage,
        )
    report.raise_if_corrupt()
    return df


# 재표본용 pandas offset alias (TF_MS 키 ↔ freq 문자열 단일 매핑).
_TF_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def resample_ohlcv(
    df: pd.DataFrame, source_timeframe: str, target_timeframe: str
) -> pd.DataFrame:
    """하위 TF OHLCV → 상위 TF 재표본 (**완전한 버킷만**). UTC 그리드 정렬.

    **왜 (I-001)**: 캐시 상위 TF(1d) 는 off-grid·부분봉 corruption 을 가질 수 있고 그중
    on-grid 값-손상은 ``audit`` 이 검출하지 못한다. 반면 하위 TF(1h) 는 감사-clean 이므로
    거기서 상위를 파생하면 상위도 clean 이 된다(clean 구간에서 공식 상위봉과 정확 일치 검증).
    "감사 통과만 하류로" 를 파생으로 강제하는 표준 경로 (regime·MTF 공용).

    - agg: open=first / high=max / low=min / close=last / volume=sum.
    - **완전한 버킷만**: 상위/하위 봉수 비(예: 1h→1d=24)를 채운 버킷만 남기고 부분(시작/끝
      토막·결측 포함 버킷)은 드롭 — 부분 상위봉의 왜곡을 원천 차단.
    """
    for tf in (source_timeframe, target_timeframe):
        if tf not in TF_MS or tf not in _TF_FREQ:
            raise ValueError(f"미지원 timeframe: {tf} (지원: {list(TF_MS)})")
    src_ms, tgt_ms = TF_MS[source_timeframe], TF_MS[target_timeframe]
    if tgt_ms <= src_ms or tgt_ms % src_ms != 0:
        raise ValueError(
            f"target({target_timeframe})은 source({source_timeframe})의 정수배 상위여야 함"
        )
    expected = tgt_ms // src_ms  # 완전한 버킷당 하위봉 수
    g = df.resample(_TF_FREQ[target_timeframe], label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
    full = g.size() == expected            # 완전 버킷만 (부분·결측 드롭)
    out = out[full]
    out.index.name = df.index.name or "timestamp"
    return out


def forward_fill_completed(
    higher: pd.Series | pd.DataFrame,
    lower_index: pd.DatetimeIndex,
    higher_timeframe: str,
) -> pd.Series | pd.DataFrame:
    """상위 TF 값을 **완성된 봉만** 하위 index 에 인과적으로 매핑 (TF정렬 공용 헬퍼).

    상위 봉의 timestamp 는 봉 **시작**시각이므로, 그 봉은 (시작 + interval) 시점에야
    완성된다. 하위 시각 t 에는 완성시각 <= t 인 가장 최근 상위 봉 값만 쓸 수 있다
    (진행 중 봉 배제 — 미래 누수 방지, 설계 §9 / 기둥 2).

    공용 사용처: ① regime(1d→모델TF 국면 ff, 분석축) ② MTF(상위TF 완성봉 피처를
    결정TF X 에 concat, R4.2). regime.py 에서 data 층으로 승격(features→validation 상향의존
    회피, resample_ohlcv 와 같은 TF정렬 집).
    """
    if higher_timeframe not in TF_MS:
        raise ValueError(f"미지원 timeframe: {higher_timeframe}")
    interval = pd.Timedelta(milliseconds=TF_MS[higher_timeframe])

    is_series = isinstance(higher, pd.Series)
    hi = higher.to_frame(name="_v") if is_series else higher.copy()
    hi = hi.sort_index()
    completion = hi.index + interval  # 각 상위 봉의 완성시각

    right = hi.reset_index(drop=True)
    right["_completion"] = completion
    right = right.sort_values("_completion")

    left = pd.DataFrame({"_ts": pd.DatetimeIndex(lower_index)}).sort_values("_ts")
    merged = pd.merge_asof(
        left, right, left_on="_ts", right_on="_completion", direction="backward"
    )
    merged = merged.set_index("_ts")
    merged.index.name = getattr(lower_index, "name", None) or "timestamp"
    value_cols = [c for c in merged.columns if c != "_completion"]
    result = merged[value_cols]
    # 원래 lower_index 순서로 정렬
    result = result.reindex(pd.DatetimeIndex(lower_index))
    if is_series:
        return result["_v"].rename(higher.name)
    return result
