"""Phase 6.4 — 지평 앙상블용 EconL3 별칭 (계획 지렛대 ⑤).

엔진은 전략을 **이름으로** 식별하고 config 섹션도 이름당 하나이므로, 같은 로직을 서로 다른
지평(3·4·5일 아티팩트)으로 **동시에** 태우려면 이름이 다른 인스턴스가 필요하다. 여기 세
서브클래스는 **로직을 전혀 바꾸지 않고 이름만 다르다** — 각자 자기 `artifact_path`·
`horizon_bars` 를 config 로 받는다.

앙상블 = 세 지평이 **공용 트랜치 풀**을 나눠 쓰는 것. 한 봉에서 여러 지평이 동시에 신호를
내면 용량이 남는 한 각각 1 트랜치씩 연다(engine `evaluate_strategies_on_bar`). 지평마다
배리어 폭·만기가 달라 청산 시점이 흩어지므로 **시간 분산**이 생긴다.

주의: 슬롯 경합은 `strategies.active` 순서 = 우선순위다. 용량이 부족하면 앞선 지평이 먼저
채운다(엔진 규약). 앙상블 셀은 용량을 충분히 크게 잡아 이 편향을 없애고 쓴다.
"""

from __future__ import annotations

from src.strategy.plugins.econ_l3 import EconL3
from src.strategy.registry import register_strategy


@register_strategy
class EconL3H3(EconL3):
    """지평 3일 슬리브 (로직 동일, 이름만 분리)."""

    name = "econ_l3_h3"


@register_strategy
class EconL3H4(EconL3):
    """지평 4일 슬리브."""

    name = "econ_l3_h4"


@register_strategy
class EconL3H5(EconL3):
    """지평 5일 슬리브."""

    name = "econ_l3_h5"
