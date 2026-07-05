"""Contract — 매핑 레이어 출력 = 매매층이 소비하는 고정 인터페이스 (4필드).

매매 규칙은 오직 이 4필드만 보고 동작한다. 안쪽이 Gaussian-HMM 이든 t-HMM 이든
무엇이든, 이 4필드만 뱉으면 매매 로직은 그대로 굴러간다. (스펙 §2.1)

핵심 분업: confidence → Size, volatility → SL/TP 거리. 절대 섞지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegimeType(Enum):
    """추세장 / 횡보장 / 무매매.

    NONE = 매매하지 않는 레짐(관망). gen1 은 range 를 드롭(O-7 (가))하고 저방향성
    상태를 NONE 으로 매핑해 추세-단독으로 동작한다. RANGE 는 gen2 revival 대비 보존.
    """

    TREND = "trend"
    RANGE = "range"
    NONE = "none"


class RegimeDirection(Enum):
    """방향. range 면 NONE."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass(frozen=True)
class Contract:
    """레짐 판단 결과 (불변 값객체).

    Attributes:
        type: trend / range — |평균수익률|÷변동성 vs τ 로 매핑.
        direction: long / short / none — trend 면 평균수익률 부호, range 면 none.
        confidence: 0~1, filtered posterior γ 의 contract별 집계 ("이 레짐이라는 확신").
            → Size 에 사용.
        volatility: ATR(24), 가격 단위 ("지금 얼마나 출렁이나"). → SL/TP 거리에 사용.
            HMM 입력의 정규화 변동성 feature 와는 별개.
    """

    type: RegimeType
    direction: RegimeDirection
    confidence: float
    volatility: float

    @property
    def is_trend(self) -> bool:
        return self.type == RegimeType.TREND

    @property
    def is_range(self) -> bool:
        return self.type == RegimeType.RANGE

    @property
    def is_none(self) -> bool:
        """무매매 레짐(관망). 매매 로직은 is_trend/is_range 어디에도 안 걸려 no-op."""
        return self.type == RegimeType.NONE
