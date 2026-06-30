# INFRA_GUIDE — CoinBot 인프라 뼈대

이 문서는 **새 퀀트 모델을 바로 구현할 수 있도록 인프라만 남긴 상태**의
단일 안내서다. 과거 프로젝트에서 구현했던 특정 모델(룰베이스 추세추종 ·
ML/DL/RL · 앙상블)과 그 학습/평가 파이프라인, 모델 특화 인사이트는
의도적으로 제거했다. 여기 담긴 것은 **어떤 모델에도 공통인 매매 실행
인프라**와 그 구조상 주의점뿐이다.

---

## 1. 아키텍처 한눈에

- **플러그인 + 엔진 분리**: 엔진은 전략의 존재를 모른다. 전략은
  `src/strategy/plugins/<name>.py` 파일 1개 + `config/default.yaml` 섹션 1개로
  추가된다. **엔진 코드 수정 0** 이 원칙이다 — 수정이 필요하면 추상화가
  잘못된 것.
- **3 모드 동일 코어**: `backtest` / `paper` / `live` 가 같은 `FeeModel`(PnL) ·
  `AccountTracker`(equity 계측) · 거래 기록 스키마 + **동일 전략(사이징·SL/TP·
  진입게이트 포함)** 을 공유한다 (라이브-백테 일관성). 거래 정책은 엔진이 아닌
  전략(모델)이 소유한다.
- **실행 모드는 CLI subcommand 로 결정** (`config` 에 mode 필드 없음).
- 대상 거래소: **OKX 무기한 선물** (`BTC/USDT:USDT`).

```
DataFeed/Historical(캔들) → 봉 마감 → AbstractEngine.evaluate_strategies_on_bar
   → StrategyModule.generate_signal → (진입) allow_entry·compute_stop_loss/
     take_profit·compute_position_size(전부 모델) → Broker 주문 → Position 등록
   → 봉마다 SL/TP·strategy exit 검사 → 청산 시 FeeModel 정산 → 기록
```

## 2. 남은 모듈 구조

```
src/
├── core/            # AbstractEngine + types/enums/event_bus  ★엔진 코어(메커니즘)
│   ├── engine_base.py   # AbstractEngine: 봉마감 평가·진입·청산 집행(정책은 전략 호출)
│   ├── types.py         # Signal / StrategyContext / AccountState / Position / Fill
│   ├── enums.py         # SignalSide / PositionSide / ExitReason / EventType ...
│   └── event_bus.py     # 비동기 pub/sub
├── live/engine.py    # CoreEngine (paper/live): DataFeed 구동, 상태 복원, 모니터링 로그
├── live/trade_sync.py# TradeSyncer: 라이브 종료 후 OKX 실값으로 DB 거래 정정
├── backtest/engine.py# BacktestEngine: 캔들 순회 + 5종 리포트 출력
├── strategy/
│   ├── base.py          # StrategyModule 추상 (필수 6 = 거래 정책 + 선택 훅)
│   ├── registry.py      # @register_strategy + auto-discovery + 활성화
│   ├── indicators.py    # 공통 TA 라이브러리 (pandas_ta_classic 래핑)
│   ├── helpers/         # opt-in 정책 공식(sizing/risk_gates) — 엔진 미호출
│   └── plugins/         # ★ 신규 전략 파일을 여기에 추가 (현재 비어 있음)
├── execution/        # Broker(facade) + LiveExecutor(OKX) + PaperExecutor(시뮬)
├── accounting/       # fee_model(수수료·PnL 단일 공식) + account_tracker(equity 계측)
├── data/             # feed / historical / store(SQLite) / orderbook
└── utils/            # config_loader / logger / notifier / path_utils

config/default.yaml   # 인프라 공통 설정 + (비어 있는) 전략 섹션 자리 (정책값 없음)
scripts/              # download_history / run_full_backtest / merge_(yearly_)reports
tests/                # 인프라 회귀 테스트 (전략 픽스처는 tests/strategy_stub.py)
```

### 모듈 레이어 (언제 실행되나)

코드의 약 절반은 **라이브/페이퍼 실거래 운영** 부분이라, 백테로 모델을
개발·검증하는 단계에서는 실행되지 않는다. "왜 이렇게 많은가"의 답.

- **Layer A — 모든 모드 공통 (백테 단계부터 필수)**:
  `core/*`, `backtest/engine.py`, `strategy/{base,registry,indicators,helpers}`,
  `accounting/{fee_model,account_tracker}`, `data/historical`,
  `execution/{broker,paper_executor}` (백테 체결도 사용), `utils/*`, `main.py`.
- **Layer B — 라이브/페이퍼 실거래 운영 (라이브 갈 때만 · OKX 종속, ~2,700줄)**:
  `live/engine.py`(CoreEngine), `execution/live_executor.py`(OKX 주문+circuit
  breaker), `live/trade_sync.py`(OKX 실값 DB 정정), `data/feed.py`(websocket),
  `data/store.py`(SQLite, 백테는 메모리라 미사용), `data/orderbook.py`(페이퍼
  슬리피지), `utils/notifier.py`(알림). 거래소 교체 시 주 리팩토링 대상.

> 커플링 메모: `broker.py` 가 `live_executor` 를 최상단 import, `paper_executor`
> 가 `orderbook.compute_market_impact` 를 import 한다. 향후 Layer B 를 떼어내
> 연구 전용 뼈대로 줄이려면 이 두 곳을 lazy/옵션 import 로 디커플링해야 한다.

## 3. 새 모델(전략) 추가 방법

1. `src/strategy/plugins/my_strategy.py` 작성:
   ```python
   from src.core.enums import SignalSide
   from src.core.types import Signal, StrategyContext
   from src.strategy.base import StrategyModule
   from src.strategy.registry import register_strategy
   from src.strategy.helpers.sizing import risk_based_size  # opt-in (원하면)

   @register_strategy
   class MyStrategy(StrategyModule):
       name = "my_strategy"
       entry_timeframe = "15m"
       required_timeframes = ["15m"]
       sl_tp_fill_priority = "sl_first"   # 동시도달 처리 (필수 선언)

       # 필수 5 = 거래 정책 전부 모델 소유
       def generate_signal(self, ctx) -> Signal: ...
       def compute_stop_loss(self, ctx, signal) -> float | None: ...   # None=standing SL 없음
       def compute_take_profit(self, ctx, signal, sl) -> float | None: ...
       def compute_position_size(self, ctx, signal, sl) -> float: ...  # 사이징 공식
       def allow_entry(self, ctx) -> bool: ...                          # 진입 게이트(리스크)
   ```
2. `config/default.yaml` 에 `my_strategy:` 섹션 추가 — **엔진이 강제하는 필수 키는
   없다.** 정책 임계값(예: `risk_per_trade_pct`, `max_drawdown_pct`)을 *모델이
   자기 params 로* 정의하고 위 메서드에서 읽는다.
3. `strategies.active` 리스트에 `"my_strategy"` 추가 (리스트 순서 = 우선순위).

### StrategyModule 인터페이스 — 거래 정책은 전부 모델 소유
- **필수 5 (abstract — 엔진에 기본값 없음)**: `generate_signal`(진입 신호) /
  `compute_stop_loss`(None=standing SL 없음) / `compute_take_profit` /
  `compute_position_size`(사이징 공식·레버리지·변동성) /
  `allow_entry`(DD/일일손실 등 진입 게이트) + 클래스 속성 `sl_tp_fill_priority`.
  (레짐 전환 청산은 `should_force_exit` 로 — reverse flow 는 제거됨.)
- **선택 훅 (기본 no-op)**: `on_bar_close` · `update_stop_loss`(trailing) ·
  `update_take_profit`(동적 TP, 이동 중심선 등) · `should_force_exit`(regime/timeout 청산) ·
  `on_position_opened` · `on_position_closed` · `generate_pyramid_signal`(클래스속성
  `supports_pyramiding=True` 일 때만 호출 — 현재 엔진 dispatch 는 stub 으로 실 추가진입 미구현).
- **opt-in 헬퍼**: 흔한 공식은 `src/strategy/helpers/`(sizing·risk_gates)에서
  골라 import. **엔진은 이 헬퍼를 호출하지 않는다** — 모델이 원할 때만 쓴다.
- 모델 학습/추론·피처 엔지니어링도 **전략이 자체 책임**. 인프라는 피처 파이프라인을
  제공하지 않는다 (§4 주의점 참조).

## 4. 인프라 구조상 주의점 (구현 전 반드시 숙지)

0. **엔진=메커니즘 / 모델=정책 (핵심 원칙)**: 엔진은 거래 *정책*을 결정하지
   않는다. 진입 게이트·사이징·SL/TP·동시도달 우선순위는 전부
   `StrategyModule` 의 추상 메서드로 모델이 소유한다. 엔진의 `try_enter` 등은
   그 메서드를 호출하고 결과를 집행할 뿐, 정책 기본값을 갖지 않는다. 남는
   조건문은 슬롯 상태·actionable 여부·가격 교차 감지 같은 *메커니즘*뿐이다.

1. **단일 전역 슬롯 (엔진 메커니즘)**: 동시에 1개 포지션만 보유(`_position` 단일
   슬롯). 여러 전략이 active 면 리스트 순서가 우선순위. 멀티 포지션이 필요하면
   엔진 슬롯 로직(`engine_base.py`)을 확장해야 한다 — 현재 가정에 잠겨 있음.

2. **봉 마감 기반 평가**: 진입 신호는 `entry_timeframe` **봉 마감 시점**에만
   생성된다. 틱 단위 의사결정 모델은 이 골격과 맞지 않는다.

3. **Lookahead 차단 위치 (직접 손대지 말 것)**:
   - 백테: `BacktestEngine._slice_candles` 가 `ts` **미만** 캔들만 전달.
   - 라이브: `AbstractEngine.LAST_CLOSED_BAR_IDX` 로 진행 중 봉을 절단
     (`CoreEngine` 는 `-2`, 백테 `-1`). 그래서 `ctx.candles[tf].iloc[-1]` 은
     항상 **직전 마감 봉**이다 (라이브에서 ccxt watch_ohlcv 가 발행하는
     진행 중 봉이 들어와도 plugin 이 보지 못함).
   - `indicators.py` 의 Donchian 은 `shift(1)` 로 현재 봉 제외 (causal).

4. **SL/TP 는 모델이 거는 선언형 청산 주문**:
   - 값은 모델 소유(`compute_stop_loss`/`compute_take_profit`/`update_stop_loss`).
     `compute_stop_loss` 가 `None` 이면 standing SL 없음 → 청산을 `should_force_exit`
     (명령형)로 전부 가져갈 수 있다.
   - 라이브: 거래소 conditional order. 백테/페이퍼: `check_candle_sl_tp` 가 캔들
     high/low 로 *거래소 체결을 시뮬* (가격 교차 감지만; 동시 도달 시 모델의
     `sl_tp_fill_priority` 로 결정). 이 시뮬을 모델로 옮기면 백테≠라이브 괴리 위험.

5. **PnL·수수료는 `FeeModel.calc_pnl` 단일 공식** (3 모드 공유). 백테 결과
   의심 시 데이터 단위 정합성부터 검증: `trades pnl 합` ↔ `metrics total_pnl`
   ↔ `equity_curve 변화량`.

6. **사이징·리스크 게이트는 모델 메서드**: `compute_position_size`(공식·레버리지·
   변동성) / `allow_entry`(DD·일일손실 등 진입 차단). 판단 재료는 `ctx.account`
   (`AccountState`: balance·equity·peak·daily_pnl·drawdown_pct). 흔한 공식은
   `strategy/helpers/`(`risk_based_size`, `drawdown_exceeded`, `daily_loss_exceeded`)에
   opt-in. `accounting/account_tracker.py` 는 equity/peak/daily_pnl 을 *계측*만 하고
   정책 enforcement 는 하지 않는다.

7. **거래소 종속 영역 (OKX 한정)**: `execution/live_executor.py` (conditional
   order 문법, contract 단위 변환), `live/trade_sync.py` (positions-history,
   fetch_my_trades) 는 OKX API 에 강결합. **다른 거래소로 바꾸면 이 두 파일이
   주 리팩토링 대상.** 백테/페이퍼/회계는 거래소 무관.

8. **엔진 전역 안전장치는 circuit breaker 뿐**: API 연속 실패 시 신규 진입 차단
   (`config.risk.circuit_breaker`, 거래소 연결 안전 — 거래 정책 아님). DD락·일일
   손실 한도 같은 *거래 리스크 정책*은 모델의 `allow_entry` 로 이관됐다(엔진에 없음).

9. **재시작 상태 복원 (라이브)**: `CoreEngine._restore_state` 가 거래소 포지션
   ↔ DB open trade 를 매칭해 Position 을 자동 복원/입양. 전략 0개(뼈대)인데
   거래소에 포지션이 있으면 에러로 중단.

10. **DB 는 라이브/페이퍼만**: `data/coinbot_{live,paper}.db` (모드별 자동 접미사).
    백테는 `DataStore` 미사용 — 메모리에 trades/equity 누적 후 종료 시 리포트 파일로만 출력.

11. **모니터링 로그는 generic**: `CoreEngine._log_signal_status` 는 side ·
    confidence · bar 컨텍스트 + 선택적 `signal.meta["note"]` 문자열만 출력.
    모델별 진단(확률 분포 등)이 필요하면 전략이 `meta["note"]` 로 요약해 넘기거나
    자체 로깅한다.

12. **피처/학습 인프라 없음 (의도적)**: 과거의 `src/strategy/features.py`,
    `src/ml/*` (피처 파이프라인 · 라벨 생성 · walk-forward · 모델 클래스 ·
    calibration · RL 환경) 는 전부 제거됨. 새 모델이 ML/DL 라면 피처 생성 ·
    학습 스크립트 · 직렬화 로드를 **새로 설계**한다. 인프라가 강요하는 형식은
    없다 (`generate_signal` 이 `Signal` 만 반환하면 됨).

## 5. 명령어

```bash
# 백테스트 (사용자가 직접 수행)
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
#   결과: data/backtest_reports/00_Working/.../ 5종 (trades.csv, equity_curve.csv,
#         metrics.json, config_snapshot.yaml, equity_curve.png)

# 페이퍼 / 라이브
python -m src.main paper --config config/default.yaml
python -m src.main live  --config config/default.yaml

# 캔들 다운로드
python scripts/download_history.py --config config/default.yaml --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-01-01

# 다중 연도 병렬 백테 + 통합 (Windows)
scripts\run_full_backtest.bat config/default.yaml
scripts\merge_reports.bat <tag> default

# 테스트
python -m pytest tests/ -q
```

`strategies.active: []` 인 현재 상태에서는 어떤 모드든 **무거래**로 동작한다
(엔진 골격만 도는 안전 상태). 전략을 추가해야 매매가 발생한다.
