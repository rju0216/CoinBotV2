"""전략 플러그인 패키지.

이 폴더 하위에 `@register_strategy` 데코레이터를 단 StrategyModule 서브클래스
파일을 추가하면 registry.discover_plugins()가 자동 임포트하여 등록한다.
신규 전략 추가 = 파일 1개 + config[<strategy.name>] 섹션 1개로 완결.
"""
