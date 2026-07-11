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
