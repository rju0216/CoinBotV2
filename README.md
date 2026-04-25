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

⚠️ live 는 실거래 모드. 시작 전 [USER_GUIDE §6](docs/01_Guides/USER_GUIDE.md) 필독.

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

상세는 [DEVELOPER_GUIDE](docs/01_Guides/DEVELOPER_GUIDE.md).

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
├── utils/        # config_loader / logger
└── main.py       # CLI

config/default.yaml   # 통합 설정 1개
docs/
├── 00_Work_Report/   # 작업 보고서 (시점별)
└── 01_Guides/        # USER_GUIDE / DEVELOPER_GUIDE
tests/                # pytest 단위·통합 테스트
scripts/              # download_history / run_full_backtest / merge_reports
```

---

## 문서

| 문서 | 대상 |
|---|---|
| [USER_GUIDE](docs/01_Guides/USER_GUIDE.md) | 설치·CLI·백테 결과 해석·라이브 운영 |
| [DEVELOPER_GUIDE](docs/01_Guides/DEVELOPER_GUIDE.md) | 전략 작성·플러그인 인터페이스·엔진 hook |
| [PROTOTYPE_DESIGN_260425](docs/00_Work_Report/PROTOTYPE_DESIGN_260425.md) | 설계 사양 (Day 1 시점) |

---

## 라이선스

(미정)
