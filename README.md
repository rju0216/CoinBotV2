# CoinBot

매매 로직이 분리된 자동매매 봇 뼈대. 전략을 **파일 1개 + config 섹션 1개**로
추가하면 backtest / paper / live 모드에서 동일하게 작동한다.

OKX 무기한 선물 (`BTC/USDT:USDT`) 기반.

---

## Quick Start

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 (라이브 모드 시)

프로젝트 루트에 `.env`:

```
OKX_API_KEY=your_key
OKX_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase
```

paper / backtest 만 사용 시 생략 가능.

### 3. 백테스트 실행

```bash
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
```

`data/backtest_reports/00_Working/` 에 결과 5종 파일 생성.

### 4. 페이퍼 / 라이브

```bash
python -m src.main paper --config config/default.yaml
python -m src.main live  --config config/default.yaml
```

⚠️ live 는 실거래 모드. 시작 전 [INFRA_GUIDE §4 주의점](docs/INFRA_GUIDE.md) 필독.

> **현재 상태**: 새 퀀트 모델 구현을 위해 인프라 뼈대만 남긴 상태
> (`InitialInfraSetup` 브랜치). `strategies.active: []` 무거래.
> 전략 plugin 은 비어 있으며, 아래 3단계로 추가한다.

---

## 전략 추가 (3단계)

### 1. 전략 파일 생성

`src/strategy/plugins/my_strategy.py`:

```python
from src.core.enums import SignalSide
from src.core.types import Signal
from src.strategy.base import StrategyModule
from src.strategy.registry import register_strategy


@register_strategy
class MyStrategy(StrategyModule):
    name = "my_strategy"
    entry_timeframe = "1h"
    required_timeframes = ["1h"]

    def generate_signal(self, ctx): ...
    def compute_stop_loss(self, ctx, signal): ...
    def compute_take_profit(self, ctx, signal, sl): ...
```

### 2. config 섹션

```yaml
my_strategy:
  risk_per_trade_pct: 0.01
  max_leverage: 5
  # ... 전략별 파라미터
```

### 3. 활성화

```yaml
strategies:
  active: ["my_strategy"]   # 빈 리스트 [] 이면 무거래 (뼈대 상태)
```

상세·인프라 구조상 주의점은 [INFRA_GUIDE](docs/INFRA_GUIDE.md).

---

## 폴더 구조

```
src/
├── core/         # 엔진 공통 (AbstractEngine, types, enums, policies, event_bus)
├── live/         # CoreEngine (paper/live)
├── backtest/     # BacktestEngine + 결과 리포트 5종 출력
├── strategy/
│   ├── base.py       # StrategyModule 추상
│   ├── registry.py   # 자동 등록 + 검색
│   ├── indicators.py # 공통 지표
│   └── plugins/      # ★ 신규 전략 위치
├── execution/    # Broker / OKX 주문 / 시뮬레이션
├── risk/         # 사이징 + DD락 + 일일 한도
├── accounting/   # 수수료 / 슬리피지 / 펀딩비
├── data/         # WebSocket / 캔들 캐시 / DB
├── utils/        # config_loader / logger / notifier / path_utils
└── main.py       # CLI

config/default.yaml   # 통합 설정 1개
docs/INFRA_GUIDE.md   # 인프라 구조·신규 전략 추가·구조상 주의점
tests/                # pytest 단위·통합 테스트
scripts/              # download_history / run_full_backtest / merge_reports
```

---

## 문서

| 문서 | 대상 |
|---|---|
| [INFRA_GUIDE](docs/INFRA_GUIDE.md) | 아키텍처·모듈 구조·신규 전략 추가법·인프라 구조상 주의점·명령어 |

---

## 라이선스

(미정)
