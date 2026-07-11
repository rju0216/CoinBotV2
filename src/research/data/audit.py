"""캔들 연속성·정합성 감사 (순수 분석 — 정책 없음).

Step 0.1 핵심 산출물. 결측봉이 있으면 "N봉 앞"이 의도보다 긴 캘린더 시간을
덮고 purge/embargo·창 계산이 봉 기준으로 어긋난다(침묵의 정렬 오류, MasterPlan
F-5). 이 모듈은 **탐지·리포트만** 하고 예외를 던지지 않는다. 하드페일 정책은
호출부(``load_audited`` 또는 ``AuditReport.raise_if_corrupt``)가 적용한다.

정책(MasterPlan Step 0.1 결정): 구조적 손상(중복·비단조·NaN·OHLC 위반·
비양수 가격)은 하드페일, **갭은 리포트만**(무단 채움 금지 — 수익률·변동성 조작
방지). 실질 갭 발견 시 잠재이슈로 등록.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.historical import TF_MS

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


class AuditError(ValueError):
    """구조적 손상이 감지됐을 때 하드페일용 예외."""


@dataclass
class Gap:
    prev: pd.Timestamp   # 갭 직전의 마지막 정상 봉
    next: pd.Timestamp   # 갭 직후의 첫 정상 봉
    missing: int         # 사이에 빠진 봉 수


@dataclass
class AuditReport:
    timeframe: str
    n_rows: int                       # 원본 행수(중복 포함)
    n_unique: int                     # 고유 timestamp 수
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    interval_ms: int
    monotonic: bool                   # 원본 인덱스가 단조증가인가
    n_duplicates: int
    n_nan_rows: int
    n_ohlc_violations: int            # high/low 가 o·c 를 제대로 감싸지 않는 행
    n_nonpositive_price: int          # 가격(o/h/l/c) <= 0
    n_negative_volume: int
    n_irregular: int = 0              # 봉 간격이 interval 의 정수배가 아님(오프-그리드)
    gaps: list[Gap] = field(default_factory=list)
    irregular: list = field(default_factory=list)  # (prev, next, delta) 표본
    expected_bars: int = 0            # [first, last] 를 interval 로 채웠을 때 기대 봉 수
    coverage: float = 0.0             # n_unique / expected_bars

    @property
    def n_missing_bars(self) -> int:
        return sum(g.missing for g in self.gaps)

    @property
    def has_corruption(self) -> bool:
        return (
            not self.monotonic
            or self.n_duplicates > 0
            or self.n_nan_rows > 0
            or self.n_ohlc_violations > 0
            or self.n_nonpositive_price > 0
            or self.n_negative_volume > 0
            or self.n_irregular > 0
        )

    def raise_if_corrupt(self) -> None:
        if self.has_corruption:
            raise AuditError(
                f"구조적 손상 감지 [{self.timeframe}]: "
                f"monotonic={self.monotonic} dup={self.n_duplicates} "
                f"nan={self.n_nan_rows} ohlc_violation={self.n_ohlc_violations} "
                f"nonpositive_price={self.n_nonpositive_price} "
                f"neg_volume={self.n_negative_volume} irregular={self.n_irregular}"
            )

    def summary(self) -> str:
        span = f"{self.first} ~ {self.last}" if self.first is not None else "(빈)"
        return (
            f"[{self.timeframe}] rows={self.n_rows} unique={self.n_unique} {span}\n"
            f"  monotonic={self.monotonic} dup={self.n_duplicates} "
            f"nan={self.n_nan_rows} ohlc_violation={self.n_ohlc_violations} "
            f"nonpositive_price={self.n_nonpositive_price} neg_vol={self.n_negative_volume}\n"
            f"  gaps={len(self.gaps)} missing_bars={self.n_missing_bars} "
            f"irregular={self.n_irregular} "
            f"coverage={self.coverage:.6f} (expected={self.expected_bars})"
        )


def audit_continuity(df: pd.DataFrame, timeframe: str) -> AuditReport:
    """OHLCV df 의 연속성·정합성을 감사해 AuditReport 를 반환(예외 없음)."""
    if timeframe not in TF_MS:
        raise ValueError(f"미지원 timeframe: {timeframe}")
    interval_ms = TF_MS[timeframe]
    interval = pd.Timedelta(milliseconds=interval_ms)

    idx = df.index
    n_rows = len(df)
    monotonic = bool(idx.is_monotonic_increasing)
    n_duplicates = int(idx.duplicated().sum())

    # 손상 판정은 값 기준 — 컬럼이 있어야 검사 가능
    price = df[PRICE_COLUMNS]
    n_nan_rows = int(df[OHLCV_COLUMNS].isna().any(axis=1).sum())
    n_nonpositive_price = int((price <= 0).any(axis=1).sum())
    n_negative_volume = int((df["volume"] < 0).sum())

    # OHLC 정합: high = 최고, low = 최저 여야 함
    hi, lo = df["high"], df["low"]
    o, c = df["open"], df["close"]
    ohlc_ok = (
        (hi >= o) & (hi >= c) & (hi >= lo)
        & (lo <= o) & (lo <= c)
    )
    # NaN 은 위 비교에서 False 가 되므로 NaN 행은 제외하고 위반만 센다
    valid_rows = df[PRICE_COLUMNS].notna().all(axis=1)
    n_ohlc_violations = int((~ohlc_ok & valid_rows).sum())

    # 갭·커버리지는 고유·정렬 인덱스 기준
    uniq = idx[~idx.duplicated()].sort_values()
    n_unique = len(uniq)
    first = uniq[0] if n_unique else None
    last = uniq[-1] if n_unique else None

    gaps: list[Gap] = []
    irregular: list = []
    n_irregular = 0
    expected_bars = 0
    coverage = 0.0
    if n_unique >= 1:
        expected_bars = int(round((last - first) / interval)) + 1
        coverage = n_unique / expected_bars if expected_bars else 0.0
    if n_unique >= 2:
        deltas = uniq[1:] - uniq[:-1]  # TimedeltaIndex (버전 견고)
        step = interval.value          # ns
        for i in range(len(deltas)):
            d = deltas[i]
            if d == interval:
                continue
            if d.value % step == 0 and d > interval:
                # 정수배 간격 = 온-그리드 갭 (봉 결측)
                missing = d.value // step - 1
                if missing > 0:
                    gaps.append(Gap(prev=uniq[i], next=uniq[i + 1], missing=int(missing)))
            else:
                # interval 정수배가 아님 = 오프-그리드 (그리드 파손)
                n_irregular += 1
                if len(irregular) < 50:
                    irregular.append((uniq[i], uniq[i + 1], d))

    return AuditReport(
        timeframe=timeframe,
        n_rows=n_rows,
        n_unique=n_unique,
        first=first,
        last=last,
        interval_ms=interval_ms,
        monotonic=monotonic,
        n_duplicates=n_duplicates,
        n_nan_rows=n_nan_rows,
        n_ohlc_violations=n_ohlc_violations,
        n_nonpositive_price=n_nonpositive_price,
        n_negative_volume=n_negative_volume,
        n_irregular=n_irregular,
        gaps=gaps,
        irregular=irregular,
        expected_bars=expected_bars,
        coverage=coverage,
    )
