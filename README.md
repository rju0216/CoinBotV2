# CoinBot

매매 로직(전략)이 엔진과 분리된 **백테스트 골격**. 전략을 **파일 1개 + config 섹션
1개**로 추가하면 엔진이 봉마감마다 그 전략을 평가·집행한다. 엔진은 전략의 존재를
모른다. 대상 심볼은 OKX 무기한 선물 (`BTC/USDT:USDT`).

> **현재 상태**: 새 퀀트 모델 구현을 위해 **최소 개념 골격만 남긴 상태**.
> `strategies.active: []` 무거래. plugin 폴더는 비어 있으며 아래 3단계로 전략을 추가한다.

---

## Quick Start

```bash
pip install -r requirements.txt

# 모델/리서치(src/research) 개발 시: ML 선택 의존성 추가 설치
# pip install -r requirements-ml.txt

# 1) 백테 캔들 다운로드 (OKX 공개 API — 키 불필요)
python scripts/download_history.py --timeframe 15m --start 2024-01-01 --end 2024-12-31

# 2) 백테스트 실행
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
```

결과는 `data/backtest_reports/backtest_<start>_<end>/` 에 3종 파일
(`trades.csv`, `equity_curve.csv`, `metrics.json`)로 출력된다.
`strategies.active: []` 인 뼈대 상태에서는 무거래로 동작한다.

---

## 전략 추가 (3단계)

### 1. 전략 파일 생성 — `src/strategy/plugins/my_strategy.py`

```python
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.base import StrategyModule
from src.strategy.registry import register_strategy


@register_strategy
class MyStrategy(StrategyModule):
    name = "my_strategy"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]
    sl_tp_fill_priority = "sl_first"   # 동시 도달 처리 (필수)

    # 필수 6 = 거래 정책 전부 모델 소유
    def generate_signal(self, ctx): ...
    def compute_stop_loss(self, ctx, signal): ...       # None = standing SL 없음
    def compute_take_profit(self, ctx, signal, sl): ...
    def compute_position_size(self, ctx, signal, sl): ...
    def should_reverse(self, ctx, position, new_signal): ...
    def allow_entry(self, ctx): ...
```

### 2. config 섹션 — `config/default.yaml`

```yaml
my_strategy:
  risk_per_trade_pct: 0.01   # 전략이 자기 params 로 읽는 정책 임계값 (엔진 강제 없음)
```

### 3. 활성화

```yaml
strategies:
  active: ["my_strategy"]   # 빈 리스트 [] 이면 무거래
```

엔진 코드 수정은 0이어야 한다. 상세는 [INFRA_GUIDE](docs/INFRA_GUIDE.md).

---

## 폴더 구조

```
src/
├── core/         # types(Signal/Position/…) + enums (공용 데이터 vocabulary)
├── backtest/     # engine.py — 봉마감 평가·진입·청산·SL/TP 시뮬 + 리포트 (엔진 전체)
├── strategy/
│   ├── base.py       # StrategyModule 추상 (필수 6 + 선택 훅)
│   ├── registry.py   # @register_strategy 자동 등록 + 활성화
│   └── plugins/      # ★ 신규 전략 위치 (현재 비어 있음)
├── accounting/   # fee_model(PnL 단일 공식) + account_tracker(equity 계측)
├── data/         # historical.py — 백테 캔들 로더 (CSV 캐시 + OKX API 병합)
└── utils/        # config_loader / logger
main: src/main.py  # CLI (backtest)

config/default.yaml   # 통합 설정 1개
docs/INFRA_GUIDE.md   # 아키텍처·구조상 주의점
tests/                # pytest
scripts/download_history.py  # 캔들 다운로드
```

---

## 문서

| 문서 | 대상 |
|---|---|
| [INFRA_GUIDE](docs/INFRA_GUIDE.md) | 아키텍처·모듈 구조·신규 전략 추가법·구조상 주의점·명령어 |
