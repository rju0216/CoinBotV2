"""레짐 발견(HMM)·매핑 레이어.

발견(자유)→매핑(번역)→매매(규칙) 3층 중 앞 2층. 매매층(디스패처 플러그인)은
RegimeService 가 내는 Contract(4필드)만 소비한다 — 안쪽 모델(HMM)을 갈아끼워도
매매 로직 불변.
"""

from src.strategy.regime.contract import (
    Contract,
    RegimeDirection,
    RegimeType,
)

__all__ = ["Contract", "RegimeType", "RegimeDirection"]
