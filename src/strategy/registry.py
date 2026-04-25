"""전략 플러그인 레지스트리.

`@register_strategy` 데코레이터로 클래스를 등록하면 config의
strategies.active 리스트에서 이름으로 활성화 가능. plugins/ 폴더의
모든 모듈을 자동 임포트하여 등록을 트리거한다.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, TypeVar

from src.strategy.base import StrategyModule

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[StrategyModule]] = {}
_PLUGINS_DISCOVERED = False

# 엔진 try_enter가 strategy.params에서 직접 참조하는 필수 키 (§3.2.1).
# 시작 시점에 검증하여 운영 중 첫 진입 시도에서야 발견되는 지연을 방지한다.
REQUIRED_STRATEGY_PARAMS: tuple[str, ...] = (
    "risk_per_trade_pct",
    "max_leverage",
)

T = TypeVar("T", bound=type[StrategyModule])


def register_strategy(cls: T) -> T:
    """전략 클래스 등록 데코레이터.

    필수 클래스 속성: name, entry_timeframe, required_timeframes.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise TypeError(
            f"Strategy class {cls.__name__} must define a non-empty 'name' attribute"
        )
    if not getattr(cls, "entry_timeframe", ""):
        raise TypeError(
            f"Strategy '{name}' must define 'entry_timeframe'"
        )
    if not getattr(cls, "required_timeframes", None):
        raise TypeError(
            f"Strategy '{name}' must define non-empty 'required_timeframes'"
        )
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(
            f"Strategy '{name}' already registered "
            f"({_REGISTRY[name].__module__}). Choose a unique name."
        )
    _REGISTRY[name] = cls
    logger.debug("Registered strategy: %s (%s)", name, cls.__module__)
    return cls


def get_registered_strategies() -> dict[str, type[StrategyModule]]:
    return dict(_REGISTRY)


def get_strategy_class(name: str) -> type[StrategyModule]:
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Strategy '{name}' not found in registry. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def discover_plugins(force: bool = False) -> None:
    """src/strategy/plugins/ 하위 모든 모듈을 임포트하여 @register_strategy 트리거."""
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED and not force:
        return
    try:
        import src.strategy.plugins as plugins_pkg
    except ImportError:
        logger.warning("plugins package not found; no strategies loaded")
        _PLUGINS_DISCOVERED = True
        return
    for _, modname, _ in pkgutil.iter_modules(plugins_pkg.__path__):
        full_name = f"src.strategy.plugins.{modname}"
        try:
            importlib.import_module(full_name)
        except Exception as e:
            logger.error("Failed to import plugin %s: %s", full_name, e)
    _PLUGINS_DISCOVERED = True


def load_active_strategies(config: dict[str, Any]) -> list[StrategyModule]:
    """config['strategies']['active'] 리스트 순서대로 전략 인스턴스 생성.

    순서 = (C) 배타적 경합 정책의 우선순위.
    """
    discover_plugins()
    active_names = (config.get("strategies", {}) or {}).get("active", []) or []
    if not active_names:
        logger.warning(
            "No active strategies declared in config['strategies']['active']"
        )
        return []
    instances: list[StrategyModule] = []
    seen: set[str] = set()
    for name in active_names:
        if name in seen:
            raise ValueError(f"Duplicate strategy in active list: '{name}'")
        seen.add(name)
        cls = get_strategy_class(name)
        params = config.get(name, {}) or {}
        missing = [k for k in REQUIRED_STRATEGY_PARAMS if k not in params]
        if missing:
            raise ValueError(
                f"Strategy '{name}' config is missing required params: "
                f"{missing}. Required keys: {list(REQUIRED_STRATEGY_PARAMS)}"
            )
        instances.append(cls(params))
        logger.info("Loaded strategy: %s", name)
    return instances


def reset_registry_for_testing() -> None:
    """테스트 격리용. 일반 코드에서는 호출하지 말 것."""
    global _PLUGINS_DISCOVERED
    _REGISTRY.clear()
    _PLUGINS_DISCOVERED = False
