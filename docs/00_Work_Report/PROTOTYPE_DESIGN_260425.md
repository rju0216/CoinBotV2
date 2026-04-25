# CoinBot Prototype Design (260425)

## 0. 문서 목적

이 문서는 `60_CoinBot` 프로젝트(ProjectPrototype 브랜치)에서 구축하는 **매매 로직이 제거된 프로토타입 뼈대**의 설계 사양과 단계별 작업 계획을 기록한다. 본 브랜치는 `main`에 머지되지 않으며, 실거래는 별도 경로(`50_CoinBot_Live`)에서 운영된다.

- 작성일: 2026-04-25
- 대상 브랜치: `ProjectPrototype`
- 참조 위치: `c:/Claude_Workspace/60_CoinBot/`

---

## 1. 설계 원칙

1. **확장성**: 신규 전략 추가는 파일 1개 + config 섹션 1개로 완결. 엔진/라우터/다른 전략 수정 불필요.
2. **전략 On/Off 용이성**: `strategies.active` 리스트에서 이름 추가/제거만으로 활성 전환.
3. **뼈대-전략 분리**: 엔진은 전략의 존재를 모르고도 동작 가능한 추상화 유지. 전략 특화 분기는 엔진 코드에 없음.
4. **라이브-백테스트 일관성**: 사이징, 수수료·슬리피지·펀딩비, SL/TP 체결 규칙, 리스크 한도 등 모든 공통 로직은 한 곳(`AbstractEngine` + `FeeModel` + `RiskManager`)에서만 정의.
5. **Regime 판단은 전략의 책임**: 엔진/라우터 레벨의 레짐 감지는 존재하지 않는다. 전략이 필요 시 자체 계산.

---

## 2. 최종 디렉토리 구조

```
src/
├── core/
│   ├── enums.py              # Signal, OrderSide, ExitReason, PositionStatus
│   ├── types.py              # Position, Order, Fill, Candle, StrategyContext
│   ├── event_bus.py          # EventBus
│   ├── policies.py           # ReverseSignalPolicy 등 엔진 전역 정책
│   └── engine_base.py        # AbstractEngine (공통 뼈대)
├── live/
│   └── engine.py             # CoreEngine (실거래/페이퍼)
├── backtest/
│   ├── engine.py             # BacktestEngine
│   └── report.py             # 리포트 생성
├── strategy/
│   ├── base.py               # StrategyModule 추상
│   ├── registry.py           # @register_strategy 데코레이터 + 로더
│   ├── indicators.py         # 공통 지표 라이브러리
│   └── plugins/
│       ├── __init__.py       # auto-discovery
│       └── example.py        # MA 크로스 샘플
├── execution/
│   ├── broker.py             # Facade
│   ├── live_executor.py      # OKX 주문 실행
│   └── paper_executor.py     # 시뮬레이션
├── risk/
│   └── manager.py            # 범용 사이징 + DD락 + 일일 한도
├── accounting/
│   └── fee_model.py          # 수수료/슬리피지/펀딩비 모델
├── data/
│   ├── feed.py               # 실시간 WebSocket
│   ├── historical.py         # 과거 데이터 백필
│   └── store.py              # SQLite 기록
├── utils/
│   ├── config_loader.py
│   └── logger.py
└── main.py                   # CLI (paper/live/backtest)
```

### 제거 완료 항목 (단계 1·13에서 처리)

단계 1에서 `_legacy/` 로 이동 → 단계 13 (`dc8a518`)에서 git에서 영구 삭제됨.

- `src/strategy/router.py` (964줄)
- `src/strategy/{trend_following,range_trading,range_sr,range_vwap,range_grid,squeeze_breakout}.py`
- `src/strategy/scalping/` 전체
- `src/core/engine.py` (옛 모놀리식), `src/backtest/engine.py` 옛 버전
- `src/dashboard/` 전체
- `src/notifications/telegram.py`
- `scripts/analyze_*.py`, `scripts/debug_*.py`
- 기존 `data/coinbot_live.db`, `coinbot_paper.db`

git history 내에서는 커밋 해시(`996e232` 직전~)로 복구 가능.

---

## 3. Strategy 플러그인 인터페이스

### 3.1 StrategyModule 추상 클래스

```python
class StrategyModule(ABC):
    name: str                         # config 네임스페이스와 일치 (고유)
    entry_timeframe: str              # "15m", "4h", "1m" 등
    required_timeframes: list[str]    # 지표 계산에 필요한 TF들
    supports_pyramiding: bool = False # 재진입 opt-in

    @abstractmethod
    def generate_signal(self, ctx: StrategyContext) -> Signal: ...

    @abstractmethod
    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float: ...

    @abstractmethod
    def compute_take_profit(self, ctx: StrategyContext, signal: Signal, sl: float) -> float: ...

    # 선택 훅 (기본 no-op)
    def on_bar_close(self, ctx: StrategyContext, tf: str) -> None: ...
    def update_stop_loss(self, ctx: StrategyContext, position: Position) -> float | None: ...
    def should_force_exit(self, ctx: StrategyContext, position: Position) -> ExitDecision | None: ...
    def on_position_opened(self, position: Position) -> None: ...
    def on_position_closed(self, position: Position, pnl: float) -> None: ...
    def generate_pyramid_signal(self, ctx: StrategyContext, position: Position) -> Signal | None: ...
```

### 3.2 StrategyContext

엔진이 전략에 주입하는 컨텍스트 패키지. 전략이 엔진 내부에 접근하지 않도록 격리.

```python
@dataclass
class StrategyContext:
    candles: dict[str, pd.DataFrame]   # {"1d": df, "4h": df, "15m": df}
    current_price: float
    balance: float
    position: Position | None           # 이 전략 소유 포지션 (C 정책상 전역 슬롯)
    is_slot_occupied: bool              # 다른 전략이 슬롯 점유 중인지
    params: dict                        # config[self.name] 내용
    now: datetime
```

### 3.2.1 전략 params 필수 키

엔진(`try_enter`)이 `strategy.params` dict에서 다음 키를 직접 참조한다. 전략
플러그인의 config 섹션에 **반드시** 포함해야 한다. 단계 11에서 I-008 해결로
**`registry.load_active_strategies` 가 startup 시점에 엄격 검증** — 누락 시
즉시 `ValueError` 로 중단되어 운영 중 조용한 skip을 방지한다.

| 키 | 용도 | 사용처 |
|---|---|---|
| `risk_per_trade_pct` | 포지션 사이징 시 per-trade 리스크 비율 | `RiskManager.calculate_position_size` |
| `max_leverage` | 사이징 시 레버리지 클램프 상한 (거래소 단일 `exchange.leverage`와 별개) | 동상 |

전략별 고유 파라미터는 그 외 자유롭게 정의 가능. `entry_timeframe`·
`required_timeframes`·`supports_pyramiding`은 **클래스 속성**으로 선언한다
(params dict 아님).

### 3.3 Registry

```python
# src/strategy/plugins/example.py
from src.strategy.base import StrategyModule
from src.strategy.registry import register_strategy

@register_strategy
class ExampleMACross(StrategyModule):
    name = "example_macross"
    entry_timeframe = "15m"
    required_timeframes = ["15m"]
    ...
```

엔진은 `config["strategies"]["active"]` 리스트를 읽어 registry에서 이름으로 로드. 순서 = 우선순위.

---

## 4. 엔진 아키텍처

### 4.1 계층

```
AbstractEngine (engine_base.py)
├── load_strategies_from_registry(config)
├── _compute_timeframe_union()             # 활성 전략들의 entry+required TF
├── evaluate_strategies_on_bar(tf, ...)
│   ├── on_bar_close 훅 dispatch (모든 전략, 관련 TF 한정)
│   ├── 슬롯 빔 → entry_timeframe == tf 인 전략 우선순위 순회
│   │   └── strategy.generate_signal(ctx) → try_enter()
│   ├── 슬롯 참 + reverse_policy ≠ ignore → reverse 검사 후 청산·재진입
│   └── 슬롯 참 + 보유 전략.supports_pyramiding → pyramid hook
├── try_enter(strategy, signal, ctx, now)
│   ├── risk_manager.validate_order(balance, current_position_count)
│   ├── strategy.compute_stop_loss / compute_take_profit
│   ├── risk_manager.calculate_position_size(... params로 risk_pct/leverage)
│   ├── broker.open_position()
│   ├── self._record_trade_open(...)        # 추상 — Live: DB / 백테: 메모리
│   ├── broker.place_stop_loss / place_take_profit (라이브 거래소 pending)
│   └── strategy.on_position_opened()
├── check_candle_sl_tp(position, high, low) # 캔들 기반 체결 (정책 (a) SL 우선)
├── check_strategy_exits(...)
│   ├── strategy.update_stop_loss()         # 동적 SL 갱신
│   └── strategy.should_force_exit()        # 전략 특화 청산
└── close_position(exit_price, reason, funding_fee, now)
    ├── broker.cancel_all_orders()
    ├── broker.close_position()
    ├── fee_model.estimate_round_trip + calc_pnl → net PnL
    ├── self._record_trade_close(...)       # 추상 — Live: DB / 백테: 메모리
    ├── risk_manager.add_pnl + update_equity
    └── strategy.on_position_closed()
```

**거래 기록 추상화 (단계 14 도입)**: `_record_trade_open` / `_record_trade_close`
는 추상 메서드. CoreEngine은 `DataStore` 에 영속 기록, BacktestEngine은
메모리 dict에 누적 (DB 누적 문제 I-009 회피).

### 4.2 CoreEngine (live/engine.py)

- async 메인 루프, DataFeed WebSocket 구독
- Broker에 실주문 전송, 거래소 SL/TP pending 주문 사용
- DB 상태 복원 (재시작 대응)

### 4.3 BacktestEngine (backtest/engine.py)

- sync 캔들 순회 (마스터 TF = 활성 전략 entry_timeframe 중 최소값)
- SL/TP 체결 시뮬레이션: **한 캔들 내 SL/TP 동시 도달 시 SL 우선** (`AbstractEngine.check_candle_sl_tp`)
- **DataStore 미사용 (단계 14, I-009 (나) 적용)**: 메모리 dict로 trades 누적,
  매 실행 깨끗이 시작
- 결과는 `BacktestResult` dataclass + **`write_reports()` 가 5종 파일을
  `data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{config_name}/{config_name}/`
  에 출력** (`trades.csv`, `equity_curve.csv`, `metrics.json`,
  `config_snapshot.yaml`, `equity_curve.png`)

---

## 5. 엔진 정책 결정 사항

| # | 정책 | 결정 | 비고 |
|---|---|---|---|
| 1 | 역방향 신호 | **Ignore (default)** | `ReverseSignalPolicy` 추상화로 config 교체 가능. `"reverse"`, `"same_strategy_only"` 옵션 확장 가능 |
| 2 | 재진입 신호 | **Ignore (default)** | `supports_pyramiding=True` opt-in 시 `generate_pyramid_signal` 훅 호출 |
| 3 | SL/TP 동시 도달 | **SL 우선** | 백테/페이퍼 한정. `AbstractEngine.check_candle_sl_tp()` 에 규칙 내장 |
| 4 | 타임프레임 | **전략이 entry_timeframe 선언, 엔진이 최소값으로 마스터 루프 산출** | 보조 TF는 `required_timeframes`로 주입만 받음 |
| 5 | DB | **새 DB 시작**, 기존은 `data/_legacy/` | 스키마: `owner` → `strategy_name` |
| 6 | 실행 명령 | **CLI subcommand 필수** | config에서 `mode:` 삭제 |
| 7 | 라이브 재시작 포지션 | **뼈대(active=0): 에러 중단** / **전략≥1 + orphan: SL/TP 유지 + 자동 입양(7-1)** | orphan 상태에서는 전략 특화 훅(`should_force_exit`, `update_stop_loss`) 스킵, 경고 로그. 전략이 active로 돌아오면 자동 매칭 복구 |
| 8 | SL/TP 체결 로직 | **엔진 담당** | 라이브: 거래소 pending 주문. 백테: 캔들 비교. 동적 갱신은 `update_stop_loss` 훅으로 |

---

## 6. Config 스키마

`config/default.yaml` (통합 config 1개). 단계 13a 이후 **뼈대 상태**: `strategies.active: []` 로 무거래.

```yaml
exchange:
  name: "okx"
  symbol: "BTC/USDT:USDT"
  sandbox: false
  leverage: 5                         # 거래소 단일 레버리지 (전략별 max_leverage와 별개)
  # api_key / secret / passphrase 는 .env (OKX_API_KEY 등)로 주입

engine:
  reverse_signal_policy: "ignore"    # "ignore" | "reverse" | "same_strategy_only"

risk:                                 # 엔진 전역 안전장치
  max_daily_loss_pct: 0.05
  max_drawdown_pct: 0.35
  max_position_size_btc: 1.0
  max_concurrent_positions: 1         # (C) 배타적 경합 — 전역 슬롯 1개

accounting:
  taker_fee_pct: 0.0005
  slippage_pct: 0.0
  funding_enabled: true               # 라이브에서만 실제 fetch, 백테는 0

paper:
  initial_balance: 10000.0            # 페이퍼/백테 시작 잔액

data:
  history_bars: 300                   # 기동 시 백필 캔들 개수 (TF별 동일)
  candle_dir: "data/candles"

database:
  path: "data/coinbot.db"             # 모드별 접미사 자동: coinbot_live.db / coinbot_paper.db (백테는 DataStore 미사용)

logging:
  level: "INFO"
  file: "logs/coinbot.log"
  max_size_mb: 50
  backup_count: 5

strategies:
  active: []                          # 뼈대 상태. 예: ["example_macross"] 로 켜면 활성

# ---- 전략 네임스페이스 (참고용 샘플) ----
# strategy/plugins/example.py 의 ExampleMACross 가 등록되어 있다.
# active 리스트에 "example_macross" 를 추가해야 동작.
example_macross:
  risk_per_trade_pct: 0.01            # 필수 (엔진 사이징)
  max_leverage: 5                     # 필수 (엔진 사이징)
  ma_fast: 20
  ma_slow: 50
  atr_period: 14
  atr_sl_mult: 1.5
  reward_risk_ratio: 2.0
```

### 전략별 필수 키

- `risk_per_trade_pct`, `max_leverage` — 누락 시 startup `ValueError` (I-008)

전략 클래스 속성으로 선언 (params 아님):

- `entry_timeframe`, `required_timeframes`, `supports_pyramiding`

그 외 파라미터는 전략 구현체가 자유롭게 정의.

---

## 7. 실행 방식

CLI subcommand 필수. 미지정 시 argparse 에러로 종료.

```
python -m src.main paper    --config config/default.yaml
python -m src.main live     --config config/default.yaml
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
```

- `mode:` 필드를 config에서 제거. 실행 경로는 subcommand로 결정.
- `backtest` 서브커맨드는 `--start`, `--end` 필수.

---

## 8. 뼈대에 보존되는 "항상 적용되는 로직"

- 수수료 (taker_fee_pct, round-trip) — `accounting/fee_model.py`
- 슬리피지 — 동일
- 펀딩비 (라이브 실시간 / 백테 0) — 동일
- 포지션 사이징 (risk_per_trade_pct, max_leverage) — `risk/manager.py`
- 일일 손실 한도 / 최대 드로우다운 락 / 수동 unlock — `risk/manager.py`
- Daily PnL 리셋 — `AbstractEngine`
- DB 기록 (trades, equity, snapshots) — `data/store.py`
- Event Bus 이벤트 (ORDER_FILLED, POSITION_CLOSED, DRAWDOWN_LOCKED, ERROR, EQUITY_UPDATE) — `core/event_bus.py`
- 상태 복원 (재시작) — 라이브 엔진만

### 제거되는 공통 기능

- 웹 대시보드, status JSON 출력
- Telegram 통지
- exit_plan (partial TP + runner)
- macro_env_filter, macro_bias, range_bounded_momentum
- owner 태그 (→ strategy_name으로 대체)

---

## 9. 단계별 작업 계획

각 단계는 독립 commit. 완료 시 사용자 승인 후 커밋, 이후 다음 단계 착수.

| # | 단계 | 주요 산출물 | 규모 |
|---|------|-------------|------|
| 0 | 설계 문서화 | 본 문서 | — |
| 1 | 기존 코드 `_legacy/` 이동 | `src/_legacy/`, `data/_legacy/` | — |
| 2 | core 공통 타입 | `core/enums.py`, `core/types.py`, `core/event_bus.py`, `core/policies.py` | ~250 |
| 3 | FeeModel | `accounting/fee_model.py` | ~100 |
| 4 | RiskManager 축소 | `risk/manager.py` | ~100 |
| 5 | 데이터 레이어 정리 | `data/feed.py`, `data/historical.py`, `data/store.py` | 축소 |
| 6 | Execution 정리 | `execution/broker.py`, `execution/live_executor.py`, `execution/paper_executor.py` | 축소 |
| 7 | Strategy 인터페이스 | `strategy/base.py`, `strategy/registry.py`, `strategy/indicators.py` | ~200 |
| 8 | AbstractEngine | `core/engine_base.py` | ~300 |
| 9 | BacktestEngine | `backtest/engine.py` | ~400 |
| 10 | CoreEngine | `live/engine.py` | ~400 |
| 11 | 샘플 전략 + I-008 | `strategy/plugins/example.py` + registry 검증 | ~150 |
| 12 | CLI + default config + utils + scripts | `main.py`, `config/default.yaml`, `utils/*` | ~250 |
| 13 | `_legacy/` 제거 | — | — |
| 13a | (β) 적용 | `default.yaml`의 `active: []` | — |
| 14 | I-009/010/011 해결 | `BacktestEngine` DataStore 분리 + `write_reports` + 다중 전략 검증 | ~400 |
| 15 | scripts 갱신 | `merge_yearly_reports.py` (`owner` → `strategy_name`) | ~50 |
| 16 | 문서·의존성 정리 | `README.md`, `CLAUDE.md`, `docs/`, `requirements.txt` | — |

**총 예상 규모 ~2,500줄** (단계 13~14에서 추가 산출물 반영)

> 단계 14는 단계 13 직후 점검에서 발견된 I-009/010/011 처리를 위해 14/15/16으로
> 분할되었다. 원래 단계 14("문서/README")는 단계 16으로 이동.

### 9.1 단계 1 후 보존된 재활용 자산

`_legacy/`로 이동되지 않고 신규 구조에서 재활용되는 항목:

| 위치 | 자산 | 용도 |
|---|---|---|
| `data/candles/` | OKX 백필 캔들 데이터 | 신규 백테스트 엔진(단계 9)이 그대로 사용 |
| `data/backtest_reports/` | 과거 리포트 보관 | 신규 리포트 생성과 충돌 없음 |
| `scripts/download_history.py` | 캔들 백필 유틸 | 데이터 수급 |
| `scripts/run_full_backtest.bat` | 풀 백테스트 실행 스크립트 | 단계 12 후 인자 형태 일부 갱신 가능 |
| `scripts/merge_reports.bat` | 리포트 통합 | 동일 |
| `scripts/merge_yearly_reports.py` | 연도별 리포트 통합 | 동일 |

`_legacy/` 안에서 단계별로 끌어와 활용하는 자산:

| 단계 | `_legacy/` 참조 대상 | 활용 방식 |
|---|---|---|
| 단계 2 | `src/_legacy/core/event_bus.py`, `enums.py` | 골격 그대로 이전, 일부 enum 정리 |
| 단계 4 | `src/_legacy/risk/manager.py` | 사이징/DD 락 로직 발췌, owner 분기 제거 |
| 단계 4 후 | `tests/_legacy/test_risk_manager.py` | 사이징 관련 4~5개 테스트만 선별해 신규 `tests/test_risk_manager.py`로 재작성 |
| 단계 5 | `src/_legacy/data/{feed,historical,store}.py` | DB 스키마에서 `owner` → `strategy_name` 컬럼 변경 후 이전 |
| 단계 6 | `src/_legacy/execution/{broker,live_executor,paper_executor}.py` | exit_plan, owner 분기 제거 후 이전 |
| 단계 7 | `src/_legacy/strategy/indicators.py` | 그대로 이전 (공통 지표 라이브러리) |
| 단계 9 | `src/_legacy/backtest/report.py` | `owner` 별 집계 → `strategy_name` 별로 갱신 후 이전 |

---

## 10. 검증 기준

- **단계 9 완료 후**: 샘플 전략으로 1년 백테 수행, 수수료·사이징·PnL 공식 무결성 확인.
- **단계 10 완료 후**: paper 모드로 1~2시간 실행, 이벤트 흐름·DB 기록·재시작 복원 확인.
- **단계 12 완료 후**: 세 subcommand(paper/live/backtest) 정상 실행 확인.
- **단계 14 완료 후**: 백테 종료 시 5종 결과 파일이 디스크에 생성됨 확인. 다중 전략 활성 시 (C) 정책(우선순위·슬롯 점유·평가 스킵) 단위 테스트 통과.
- **단계 15 완료 후**: `merge_yearly_reports.py` 가 `strategy_name` 컬럼으로 정상 집계, 7년치 병렬 백테 후 `merge_reports.bat` end-to-end 동작 확인.
- **단계 16 완료 후**: 신규 README 따라 사용자가 전략 추가 + 백테 실행 가능. `requirements.txt` 가 뼈대 의존성만 포함.

---

## 11. 잠재 이슈 트래커

진행 중 발견된 미해결 이슈를 단계별로 추적. 해결 완료 시 "해결 단계" 컬럼에 단계 번호를 기록하고 "상태" 를 "해결" 로 변경.

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-001 | 6 | LiveExecutor.get_position()이 strategy_name을 알지 못함. 라이브 재시작 시 거래소 포지션 + DB의 open trade 매칭이 필요 (자동 입양 정책 7-1) | live/engine.py 의 상태 복원 로직 | 10 | 해결 (`_restore_state._match_trade_to_exchange` side+size 매칭 + 자동 입양) |
| I-002 | 6 | PaperExecutor.restore_state()가 strategy_name을 보존하지 않음. CoreEngine이 DB open trade의 strategy_name을 Position 객체에 주입하도록 처리해야 함 | live/engine.py | 10 | 해결 (`_restore_state`가 DB open trade에서 strategy_name을 Position에 주입, orphan 판정 포함) |
| I-003 | 6 | `paper.initial_balance` config 키가 신설되었으나 default config 스키마에 미반영 | config/default.yaml | 12 | 해결 (default.yaml의 `paper.initial_balance: 10000.0` 반영) |
| I-004 | 6 | `exchange.leverage` 신설 키가 default config 스키마에 미반영 | config/default.yaml | 12 | 해결 (default.yaml의 `exchange.leverage: 5` 반영) |
| I-005 | 10 | `ccxt.pro watch_ohlcv`가 봉 진행 중 재발행 시 같은 봉에 대해 전략이 여러 번 평가될 위험 | live/engine.py | 10-a | 해결 (`_should_process_bar` 헬퍼로 TF별 마지막 ts 초과 시에만 전략 평가, 최신 가격은 매번 반영) |
| I-006 | 10 | `src/utils/` 폴더 부재 — `logger.py`, `config_loader.py` 미존재. 단계 12에서 main.py와 함께 생성 | src/utils/ | 12 | 해결 (`utils/logger.py` + `utils/config_loader.py` 작성) |
| I-007 | 10 | `scripts/run_full_backtest.bat` 가 기존 `python -m src.main backtest --config ...` 인터페이스를 전제. 단계 12의 CLI subcommand(`backtest --config ... --start ... --end ...`)와 인자 호환 확인 필요 | scripts/run_full_backtest.bat | 12 | 해결 (batch를 `download_history.py` 호출 + 신규 CLI 인터페이스로 수정) |
| I-008 | 10 (점검) | 전략 params 필수 키 누락이 시작 시점이 아닌 첫 진입 시도에서만 발견 (`KeyError`로 조용히 skip됨) | strategy/registry.py | 11 | 해결 (`REQUIRED_STRATEGY_PARAMS` 상수 + `load_active_strategies`가 startup에 엄격 검증하여 ValueError 발생) |
| I-009 | 13 (점검) | 백테 DB가 매 실행마다 누적되어 신규 백테 결과에 이전 trades가 합쳐짐 | backtest/engine.py | 14 | 해결 — (나) 적용: BacktestEngine을 DataStore 의존에서 분리, 메모리에서 trades/equity_curve 누적 |
| I-010 | 13 (점검) | 활성 전략 0개일 때 `equity_curve` 가 비어있어 `final_balance=0.0` 으로 fallback → `total_pnl=-100%` 같은 왜곡된 표시 | backtest/engine.py | 14 | 해결 — `BacktestEngine.get_result()` 가 `equity_curve` 비어있으면 `initial_balance` 로 fallback |
| I-011 | 13 (점검) | `BacktestEngine` 이 결과 파일(`trades.csv`, `equity_curve.csv`, `metrics.json`, `config_snapshot.yaml`, `equity_curve.png`)을 디스크에 저장하지 않아 `merge_yearly_reports.py` / `merge_reports.bat` 동작 불가 | backtest/engine.py + scripts/ | 14 (출력) / 15 (merge 호환) | 해결 — 단계 14에서 `BacktestEngine.write_reports()` 로 5종 파일 출력. 단계 15에서 merge_yearly_reports.py의 `owner` → `strategy_name` 컬럼 갱신 + 신규 metrics 포맷 호환 (단위 테스트 5건 통과, 옛 `owner` 컬럼은 fallback 지원) |

---

## 12. 진행 기록

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| 0 | 설계 문서화 | 완료 | `48e8e95` | 본 문서 작성 |
| 1 | `_legacy/` 이동 | 완료 | `996e232` | 111 rename, 코드 변경 없음. `data/candles`, `backtest_reports` 및 4개 유틸 스크립트는 보존 |
| 2 | core 공통 타입 | 완료 | `827b028` | enums(7종) + types(8종) + event_bus + policies(3종 + builder). 임포트·정책 동작 smoke test 통과 |
| 3 | FeeModel | 완료 | `bd88591` | `accounting/fee_model.py` 신규. taker_fee + slippage = per_side_rate, round-trip 추정 + LONG/SHORT PnL 정산. funding은 라이브 fetch / 백테 0 (확장 여지 보존). LONG 익절·SHORT 손실·NONE side 등 8개 케이스 smoke test 통과 |
| 4 | RiskManager 축소 | 완료 | `5394ba7` | `risk/manager.py` 재작성. owner 분기·SL/TP 산정·동방향 차단 모두 제거. 사이징은 전략이 risk_per_trade_pct/max_leverage를 키워드로 전달하는 방식으로 일원화. validate_order는 DD락·일일손실·DD트리거·동시포지션수만 검사. `tests/test_risk_manager.py` 14건 통과 (사이징 6 / 검증 5 / DD락 2 / PnL 1) |
| 5 | data 레이어 정리 | 완료 | `cbd3972` | `data/store.py`: owner→strategy_name, partial 트레이드·DB 마이그레이션 제거, exit_reason 컬럼 추가. `data/feed.py`: 매크로/트렌드/스캘핑 TF 하드코딩 제거 → 엔진이 활성 TF union을 주입. `data/historical.py`: 거의 그대로 이전. DataStore lifecycle smoke test 통과 (log/close trade, equity, snapshot, initial_balance idempotency) |
| 5a | pytest 보완 | 완료 | `979aad8` | `tests/conftest.py` 신규. `tests/_legacy/**` collection 차단으로 신규 14건 테스트가 회귀 없이 실행되도록 함 |
| 6 | execution 정리 | 완료 | `1b92c01` | broker 단순 facade로 축소, `mode`를 인자로 받음 (config 의존 제거). LiveExecutor·PaperExecutor 모두 `Position` 의존 제거 → dict 반환으로 표준화. `place_partial_tp_limit` 삭제, PaperExecutor의 SL/TP 추적 제거 (엔진 책임). `update_price` 제거, `close_position`은 `fill_price` 인자로 명시. config 키 변경: `exchange.leverage` 신설 (전략별 max_leverage는 사이징 클램프 전용). 시그니처 변경: `PositionSide`/`OrderType` enum 사용. PaperExecutor lifecycle smoke test 통과 (LONG 익절 +100 / SHORT 익절 +50), 기존 14건 테스트 회귀 없음 |
| 7 | Strategy 인터페이스 | 완료 | `0a9f036` | `strategy/base.py`: StrategyModule 추상 (필수 3 + 선택 6 + supports_pyramiding). `strategy/registry.py`: `@register_strategy` 데코레이터 + auto-discovery + load_active_strategies (선언 순서=우선순위). `strategy/indicators.py`: 그대로 이전. `strategy/plugins/` 폴더 마련. `tests/test_strategy_registry.py` 13건 추가 — 전체 27건 통과 |
| 8 | AbstractEngine | 완료 | `b5c9c22` | `core/engine_base.py` 신규. 봉 마감 dispatch → 슬롯 빔 시 우선순위 진입 / 슬롯 참 시 reverse_policy 검사 (ignore이면 스킵) / 보유 전략 supports_pyramiding 시 pyramid hook. try_enter는 검증·사이징·주문·DB·SL/TP pending·Position 등록·on_position_opened 일괄. close_position은 cancel→close→FeeModel.calc_pnl→DB→risk_manager.add_pnl→on_position_closed (orphan은 스킵+경고). check_candle_sl_tp는 SL 우선 정책 (a) 구현. `tests/test_engine_base.py` 15건 추가 — 전체 42건 통과 |
| 9 | BacktestEngine | 완료 | `d0b239f` | `backtest/engine.py` 신규. AbstractEngine 상속, 마스터 TF(최소 활성 TF) 캔들 순회. 각 캔들에서 SL/TP 캔들 체결 → update_stop_loss/should_force_exit → 봉 경계 TF별 evaluate_strategies_on_bar dispatch. 종료 시 잔여 포지션 ENGINE_SHUTDOWN 사유로 강제 청산. `HistoricalDataLoader`로 자동 백필 + `inject_candles` 테스트 우회. `BacktestResult` dataclass에 PnL/승률/MDD 등 메트릭 계산. 합성 캔들 end-to-end 4건 (SL hit 손실 / TP hit 수익 / ENGINE_SHUTDOWN / metrics 계산) + 전체 46건 통과 |
| 10 | CoreEngine | 완료 | `9e692b1` | `live/engine.py` 신규. AbstractEngine 상속 + DataFeed BAR_CLOSED 이벤트 구독 기반 실시간 처리. `_restore_state`가 5개 시나리오(clean/stale DB/뼈대+거래소 포지션=에러/매칭 성공/orphan/unknown)로 잠재 이슈 I-001/I-002 해결. 자동 입양(7-1): strategy_name in active→OPEN, 아니면 ORPHAN. 뼈대+거래소 포지션(7 (a))은 RuntimeError. 라이브 모드는 `_close_with_funding`으로 fetch_funding_history 후 FeeModel.calc_pnl 정산. 단위 테스트 11건 추가 (5개 매칭 시나리오 + 4개 matcher 헬퍼) — 전체 57건 통과 |
| 10-a | I-005 보완 | 완료 | `c9b6d31` | 점검 중 발견: watch_ohlcv가 봉 진행 중 재발행 시 같은 봉에 대해 전략이 중복 평가될 위험. `_should_process_bar(tf, ts_ms)` 헬퍼로 TF별 마지막 ts 초과 시에만 전략 평가. 최신 가격(append_candle)은 매 이벤트마다 반영. 단위 테스트 5건 추가 — 전체 62건 통과. Strategy params 필수 키(risk_per_trade_pct, max_leverage)를 설계 문서 §3.2.1에 명시. I-006(`src/utils/` 부재), I-007(`scripts/run_full_backtest.bat` CLI 호환)을 트래커에 추가 |
| 11 | 샘플 전략 + I-008 | 완료 | `9a7b091` | `strategy/plugins/example.py` 신규 (`ExampleMACross` — 15m EMA 크로스, ATR 기반 SL, R:R 기반 TP). I-008 처리: `strategy/registry.py`에 `REQUIRED_STRATEGY_PARAMS` 상수 + `load_active_strategies`가 startup에 필수 키 검증하여 ValueError (운영 중 조용한 skip 방지). 단위 테스트 8건 추가 (샘플 전략 generate_signal/SL/TP 6건 + registry 필수 키 3건 중 2건은 개별 누락, 1건은 empty + 백테 통합 1건) + 기존 테스트 필수 키 반영 보정. 전체 73건 통과 |
| 12 | CLI + default config + utils | 완료 | `191bbc4` | `src/utils/{logger,config_loader}.py` 신규 (I-006 해결). `src/main.py` 신규 — argparse subcommand(paper/live/backtest) 필수, config에서 mode 제거 (정책 6 (c)). `config/default.yaml` 작성 — paper.initial_balance·exchange.leverage·engine.reverse_signal_policy 등 단계 2~11 누적 키 모두 반영 (I-003/I-004 해결). `scripts/download_history.py`를 다중 TF + 범위 지원으로 재작성, `scripts/run_full_backtest.bat`를 신규 CLI에 호환되게 수정 (I-007 해결). 단위 테스트 12건 추가 (config_loader 4 / CLI 파서 6 / default.yaml 스키마 검증 1 / 통합 1) — 전체 85건 통과 |
| 12-a | 전면 점검 | 완료 | `7a589f8` | 단계 13 진행 전 CLI·백테·플러그인 확장 워크플로·config 스키마·테스트 스위트·커밋 히스토리 6개 축으로 전수 점검. 실 1주일 백테로 end-to-end 동작 확인(12건 체결, daily loss limit 트리거). 임시 플러그인 파일 생성→자동 발견→다중 전략 우선순위→TF union→필수 키 startup 차단 실습. §13에 점검 결과 기록. 잠재 이슈 잔존 0건 |
| 13 | `_legacy/` 삭제 | 완료 | `dc8a518` | 옵션 (a) 전체 삭제. `src/_legacy`, `config/_legacy`, `tests/_legacy`, `scripts/_legacy`, `data/_legacy` 모두 `git rm -r`로 제거 (112개 파일). `tests/conftest.py`의 `collect_ignore_glob = ["_legacy/**"]`는 불필요해져 함께 제거. 삭제 후 전체 85건 테스트 통과 + 실 백테 1~3일 재검증(16건 체결, 승률 31.25%) 정상 동작. git history 내에서는 `996e232`~이전 커밋으로 복구 가능 |
| 13a | (β) 적용 | 완료 | `eab5666` | `config/default.yaml` 의 `strategies.active`를 `[]`로 비워 진정한 "전략 0개 뼈대" 상태로 전환. `example_macross` 섹션과 `example.py` 파일은 새 전략 작성의 참고 예시로 유지. 점검 중 발견한 I-009(백테 DB 누적), I-010(무거래 시 final_balance 왜곡), I-011(리포트 파일 미생성)을 잠재 이슈 트래커에 등록. 단계 14를 14/15/16으로 분할하여 I-009 (나) 방식으로 처리 결정 |
| 14 | I-009/010/011 해결 | 완료 | (본 커밋) | AbstractEngine에 `_record_trade_open` / `_record_trade_close` 추상 메서드 도입, `data_store` 의존 제거. CoreEngine은 자체 DataStore 보유 + 두 메서드 DB 구현. BacktestEngine은 메모리 dict로 trades 관리(I-009 해결). `get_result` 가 `equity_curve` 비어있으면 `initial_balance` fallback(I-010 해결). `write_reports()` 가 5종 파일을 `data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{config_name}/{config_name}/` 출력(I-011 부분 해결, merge 호환은 단계 15). main.py가 출력 디렉토리 메시지 표시. 단위 테스트 6건 추가 — 3건 (무거래 fallback / 5종 파일 / 빈 헤더) + **다중 전략 (C) 정책 검증 3건** (우선순위 진입 / 슬롯 차있을 때 generate_signal 미호출 검증 / 우선순위 뒤집어 lower-rank 전략이 슬롯 독점). 전체 91건 통과 |
| 15 | scripts 갱신 | 완료 | (본 커밋) | `scripts/merge_yearly_reports.py` 의 `compute_merged_metrics` 에서 `owner` 컬럼 분기를 `strategy_name` 으로 변경하되, 컬럼이 없으면 `owner` 로 fallback 하도록 안전장치 추가. 메트릭 dict 키 `by_owner` → `by_strategy_name` 으로 갱신, `print_summary` 헤더도 "By Strategy" 로 변경. `tests/test_merge_yearly_reports.py` 5건 추가 (신규 컬럼 split / 옛 owner fallback / 컬럼 없는 경우 빈 split / exit_reason·side 분할 무회귀 / integrated 필드 호환) — 전체 96건 통과. `merge_reports.bat` wrapper는 인자 그대로 호출하므로 변경 불필요. **end-to-end 시연**: `default.yaml` 임시 `active: ["example_macross"]` → 2024-01·02 백테 2회 → `merge_yearly_reports.py --tag 260425 --config-name default` 호출 → 통합 디렉토리 `260425_backtest_MERGE_2024-01-01_2024-02-29_default/default/` 생성 (5종 파일 모두 정상). 통합 결과: 총 17건 / 승률 17.6% / MDD 7.81% / `by_strategy_name`·`by_exit_reason`·`by_direction`·`yearly_summaries` 모두 정상 출력. 시연 후 `default.yaml` 의 `active: []` 와 시연 산출물 모두 복구 |
| 16 | 문서·의존성 정리 | 완료 | `c0b6806` | `docs/` 재구성: 옛 11개 전략 문서 삭제, `docs/00_Work_Report/` 폴더 신설 + `PROTOTYPE_DESIGN_260425.md` 이동, `docs/01_Guides/` 폴더 신설 + `USER_GUIDE.md`(설치·CLI·백테 결과·라이브 운영) + `DEVELOPER_GUIDE.md`(전략 작성·StrategyModule 인터페이스·신규 모듈 추가·보안) 신규 작성. README.md 간결형 전면 재작성. CLAUDE.md 전면 재작성 — Session Startup 인라인, 응답 규칙 4개 (확인되지 않은 사실 단정 X 신규 추가), 협업 규칙 7개 등재, 50_CoinBot_Live 언급 제거. requirements.txt 정리 (streamlit/plotly/python-telegram-bot 제거). .gitignore 갱신 (`data/coinbot_*.db`, `00_Working/`, `candles/*.csv`, `.claude/settings.local.json`). 회귀 96건 통과 |
| 16-a | 최종 점검 | 완료 | (본 커밋) | 단계 13~16 결산을 §14 로 추가. 전체 96건 회귀 + 20+ 모듈 임포트 + CLI `--help` 재확인 모두 정상. 새 프로젝트로 옮기는 안내(§14.7) 한 줄 추가. 잠재 이슈 트래커 잔존 0건 (I-001~I-011 모두 해결) |

---

## 13. 단계 0~12 전면 점검 결과 (2026-04-25)

단계 13(`_legacy/` 삭제) 진행 전 구현 완결성과 플러그인 확장성을 검증한
결과. 모든 항목 통과.

### 13.1 전체 건강 상태

| 항목 | 값 |
|---|---|
| 신규 소스 파일 수 | 22개 (`src/` 21 + `src/strategy/plugins/example.py` 1) |
| 신규 테스트 파일 수 | 7개 + conftest.py |
| 총 라인 수 (src + tests) | ~3,600줄 |
| 단위 테스트 | 85건 / 모두 통과 |
| 독립 커밋 | 15건 (단계 0~12 + 05a + 10-a) |
| 잠재 이슈 잔존 | 0건 (I-001~I-008 모두 해결) |

### 13.2 CLI 검증

| 케이스 | 결과 |
|---|---|
| `python -m src.main --help` / `{paper,live,backtest} --help` | 정상 |
| subcommand 누락 | `exit 2` 거부 |
| `--config` 누락 | `exit 2` 거부 |
| `backtest` 의 `--start`/`--end` 누락 | `exit 2` 거부 |
| 알 수 없는 subcommand (`wrong`) | `exit 2` 거부 |
| `scripts/download_history.py --help` | 정상 |

### 13.3 실 백테스트 end-to-end

`python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-01-07`

| 관찰 | 결과 |
|---|---|
| CSV 캐시 로드 | 15m 캔들 정상 로드 |
| 전략 로드 | `example_macross` 단일 활성 |
| 체결 건수 | 12건 (승 3 / 패 9) |
| win_rate | 25.0% |
| max_drawdown_pct | 3.06% |
| daily loss limit 트리거 | 확인됨 — 이후 진입 차단 |
| ExitReason DB 기록 | `sl_hit` 등 정상 저장 |
| Summary 표준 출력 | initial/final balance, total_pnl, num_trades 등 포맷 정상 |
| 신규 DB 파일 자동 생성 | `data/coinbot_backtest.db` (신규 스키마) |

### 13.4 플러그인 확장 워크플로 실습

임시 `_smoke_test_plugin.py` 를 `src/strategy/plugins/` 에 생성하여 끝까지 돌린 후 정리:

| 단계 | 결과 |
|---|---|
| 1. 파일 생성만으로 auto-discovery | ✓ `smoke_test` 등록 성공 |
| 2. 두 전략 동시 활성 (`["smoke_test", "example_macross"]`) | ✓ 선언 순서 보존 로드 |
| 3. TF union 자동 산출 (smoke_test: 4h/1d + example_macross: 15m) | ✓ `["15m", "4h", "1d"]`, master=`15m` |
| 4. 필수 키 누락 startup 차단 | ✓ `ValueError: missing required params: ['max_leverage']` |
| 5. 임시 파일 제거 후 레지스트리 복구 | ✓ 정상 |

→ **"파일 1개 + config 섹션 1개"로 전략 확장** 설계 목표 달성 확인.

### 13.5 Config 스키마 완전성

`config/default.yaml` 10개 최상위 섹션 전부 실제 로드 검증:

```
accounting / data / database / engine / example_macross /
exchange / logging / paper / risk / strategies
```

각 섹션 키는 해당 컴포넌트의 `__init__` / `from_config` 가 참조하는 키와 일치.
활성 전략의 필수 키(`risk_per_trade_pct`, `max_leverage`) 존재도 자동 검증.

### 13.6 테스트 스위트 구성 (단계 14 시점 기준 91건)

| 파일 | 건수 | 검증 대상 |
|---|---|---|
| test_risk_manager.py | 14 | 사이징 / validate_order / DD락 / PnL |
| test_strategy_registry.py | 18 | 등록·조회 / load_active_strategies / 필수 키 검증(I-008) |
| test_engine_base.py | 15 | 추상 / TF union / SL 우선 정책 (a) |
| test_backtest_engine.py | 10 | end-to-end SL hit/TP hit/shutdown + I-010 fallback + I-011 5종 파일 + 다중 전략 (C) 정책 3건 |
| test_live_restore_state.py | 16 | I-001/I-002 5개 시나리오 + I-005 봉 중복 차단 |
| test_example_strategy.py | 8 | MA 크로스 신호·SL/TP·백테 통합 |
| test_config_loader.py | 12 | YAML 로드 / .env 주입 / CLI 파서 / default.yaml 스키마 |

> 단계 12 시점 시점 그대로 적용한 §13.1~13.5는 단계 13~14 진행 후 변경 사항을
> §13.9 "후속 처리" 에 반영.

### 13.7 커밋 히스토리

```
48e8e95 [260425_프로토타입_00_설계문서작성]
996e232 [260425_프로토타입_01_legacy폴더이동]
827b028 [260425_프로토타입_02_core공통타입정리]
bd88591 [260425_프로토타입_03_FeeModel작성]
5394ba7 [260425_프로토타입_04_RiskManager범용화]
cbd3972 [260425_프로토타입_05_data레이어정리]
979aad8 [260425_프로토타입_05a_pytest_legacy무시]
1b92c01 [260425_프로토타입_06_execution정리]
0a9f036 [260425_프로토타입_07_strategy인터페이스]
b5c9c22 [260425_프로토타입_08_AbstractEngine]
d0b239f [260425_프로토타입_09_BacktestEngine]
9e692b1 [260425_프로토타입_10_CoreEngine]
c9b6d31 [260425_프로토타입_10a_I005_봉중복차단]
9a7b091 [260425_프로토타입_11_샘플전략_I008검증]
191bbc4 [260425_프로토타입_12_CLI_config_utils]
7a589f8 [260425_프로토타입_12a_전면점검결과문서화]
dc8a518 [260425_프로토타입_13_legacy폴더삭제]
eab5666 [260425_프로토타입_13a_뼈대상태로_active비움]
```

(단계 14 커밋은 본 갱신 후 추가될 예정)

### 13.8 단계 13 진행 결정

`data/_legacy/*.db` 의 과거 라이브 거래 이력은 신규 프로토타입이 참조하지 않음
(프로토타입 DB는 `data/coinbot_{live,paper,backtest}.db` 로 자동 분리). 따라서
**옵션 (a) `_legacy/` 전체 삭제** 로 단계 13 진행. git history에 이력이 유지되므로
필요 시 커밋 해시로 복구 가능.

### 13.9 미해결 / 후속 처리

단계 13~14 진행 중 다음과 같이 추가 처리됨:

- **`_legacy/` 삭제**: 옵션 (a) 적용 — 단계 13 (`dc8a518`) 완료
- **(β) 적용**: `strategies.active: []` 로 뼈대 상태 전환 — 단계 13a (`eab5666`) 완료
- **백테 리포트 5종 파일 출력**: `BacktestEngine.write_reports()` — 단계 14 완료 (I-011 부분 해결)
- **다중 전략 (C) 정책 검증**: 단위 테스트 3건 추가 — 단계 14 완료
- **`merge_yearly_reports.py` `owner` → `strategy_name` 호환**: 단계 15 (`a4f31ba`) 완료 (I-011 완전 해결)
- **`requirements.txt` 정리** (`streamlit`, `plotly`, `python-telegram-bot` 등): 단계 16 (`c0b6806`) 완료
- **`CLAUDE.md` / `README.md` / `docs/` 재구성**: 단계 16 완료

---

## 14. 단계 13~16 최종 점검 결과 (2026-04-25)

뼈대 프로토타입의 마지막 점검. 단계 16-a 시점 기준.

### 14.1 전체 건강 상태

| 항목 | 값 |
|---|---|
| 신규 소스 파일 수 | 22개 (`src/` 21 + `src/strategy/plugins/example.py` 1) |
| 신규 테스트 파일 수 | 8개 (test_merge_yearly_reports.py 추가) |
| 단위 테스트 | **96건 / 모두 통과** |
| 독립 커밋 | **18건** (단계 0~16 + 05a + 10a + 12a + 13a + 16a) |
| 잠재 이슈 잔존 | **0건** (I-001~I-011 전부 해결) |
| 최종 라인 수 (src + tests) | ~4,100줄 |

### 14.2 CLI 검증 (재확인)

| 케이스 | 결과 |
|---|---|
| `python -m src.main --help` / `{paper,live,backtest} --help` | 정상 |
| subcommand·`--config`·`--start`/`--end` 누락 / 잘못된 subcommand | 모두 `exit 2` 거부 |
| `scripts/download_history.py --help` | 정상 |
| 20+ 모듈 import | 정상 |

### 14.3 백테 end-to-end (단계 15 시연)

`default.yaml` 임시 `active: ["example_macross"]` 로 2024-01·02 백테 2회 실행 결과:

| 백테 | 거래 | PnL | MDD |
|---|---|---|---|
| 2024-01 | 12 | -3.06% | 3.06% |
| 2024-02 | 5 | -4.90% | 4.90% |
| 통합 (merge) | 17 | -7.81% | 7.81% |

`merge_yearly_reports.py --tag 260425 --config-name default` 로 통합 5종 파일 생성 + `by_strategy_name`/`by_exit_reason`/`by_direction`/`yearly_summaries` 모두 정상 출력. 시연 후 `default.yaml` 의 `active: []` 와 시연 산출물 모두 복구됨.

### 14.4 다중 전략 (C) 정책 검증 (단계 14)

단위 테스트 3건으로 검증:

| 검증 | 결과 |
|---|---|
| 우선순위 진입 (선언 순서대로 슬롯 점유) | ✓ |
| 슬롯 차있을 때 `generate_signal` 호출 차단 (Ignore 정책) | ✓ |
| 우선순위 뒤집어 lower-rank 전략이 슬롯 독점 | ✓ |

### 14.5 문서 구조 최종

```
docs/
├── 00_Work_Report/
│   └── PROTOTYPE_DESIGN_260425.md   ← 본 문서
└── 01_Guides/
    ├── USER_GUIDE.md                ← 설치·CLI·백테 결과 해석·라이브 운영
    └── DEVELOPER_GUIDE.md           ← 전략 작성·StrategyModule·hook·보안
```

`README.md` 는 Quick Start + 전략 추가 3단계 + 폴더 구조 + 문서 링크의 간결형.
`CLAUDE.md` 는 Session Startup 인라인 + 응답 규칙 4개 + 협업 규칙 7개 (단계 0~15 작업 패턴 표준화).

### 14.6 잠재 이슈 트래커 최종

| ID | 발견 단계 | 해결 단계 | 상태 |
|---|---|---|---|
| I-001 | 6 (점검) | 10 | ✅ 해결 |
| I-002 | 6 (점검) | 10 | ✅ 해결 |
| I-003 | 6 (점검) | 12 | ✅ 해결 |
| I-004 | 6 (점검) | 12 | ✅ 해결 |
| I-005 | 10 (점검) | 10-a | ✅ 해결 |
| I-006 | 10 (점검) | 12 | ✅ 해결 |
| I-007 | 10 (점검) | 12 | ✅ 해결 |
| I-008 | 10 (점검) | 11 | ✅ 해결 |
| I-009 | 13 (점검) | 14 | ✅ 해결 |
| I-010 | 13 (점검) | 14 | ✅ 해결 |
| I-011 | 13 (점검) | 14·15 | ✅ 해결 |

### 14.7 새 프로젝트로 옮기는 가이드

본 브랜치(`ProjectPrototype`)는 새 프로젝트 main 브랜치 시작점으로 사용된다.
사용자가 직접 다음 작업을 수행:

1. 60_CoinBot 폴더를 새 위치로 복사 (또는 `git clone`)
2. 새 위치에서 `git init` 후 첫 커밋 또는 기존 git history 보존(branch rename)
3. 새 git remote 연결 (`git remote add origin ...`)
4. 첫 push 전 `.env` 작성, `data/candles/` 비어있으면 `scripts/download_history.py` 로 백필

CLAUDE.md 의 Session Startup 체크리스트가 새 세션에서도 동일하게 작동.

### 14.8 커밋 히스토리 (전체)

```
48e8e95 [260425_프로토타입_00_설계문서작성]
996e232 [260425_프로토타입_01_legacy폴더이동]
827b028 [260425_프로토타입_02_core공통타입정리]
bd88591 [260425_프로토타입_03_FeeModel작성]
5394ba7 [260425_프로토타입_04_RiskManager범용화]
cbd3972 [260425_프로토타입_05_data레이어정리]
979aad8 [260425_프로토타입_05a_pytest_legacy무시]
1b92c01 [260425_프로토타입_06_execution정리]
0a9f036 [260425_프로토타입_07_strategy인터페이스]
b5c9c22 [260425_프로토타입_08_AbstractEngine]
d0b239f [260425_프로토타입_09_BacktestEngine]
9e692b1 [260425_프로토타입_10_CoreEngine]
c9b6d31 [260425_프로토타입_10a_I005_봉중복차단]
9a7b091 [260425_프로토타입_11_샘플전략_I008검증]
191bbc4 [260425_프로토타입_12_CLI_config_utils]
7a589f8 [260425_프로토타입_12a_전면점검결과문서화]
dc8a518 [260425_프로토타입_13_legacy폴더삭제]
eab5666 [260425_프로토타입_13a_뼈대상태로_active비움]
471745f [260425_프로토타입_14_백테리포트_파일출력_다중전략검증]
a4f31ba [260425_프로토타입_15_merge_yearly_reports_갱신]
c0b6806 [260425_프로토타입_16_문서및의존성정리]
(본 커밋) [260425_프로토타입_16a_최종점검]
```
