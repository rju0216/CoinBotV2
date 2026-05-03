# Developer Guide

뼈대 프로토타입 위에 매매 로직을 추가하는 개발자를 위한 가이드.
설계 사양 전체는 `docs/00_Work_Report/PROTOTYPE_DESIGN_260425.md` 참조.

---

## 1. 핵심 설계 원칙

1. **확장성**: 신규 전략 추가 = 파일 1개 + config 섹션 1개. 엔진/라우터/타 전략 수정 X
2. **전략 On/Off**: `strategies.active` 리스트로만 제어
3. **뼈대-전략 분리**: 엔진은 전략의 존재를 모르고도 동작 가능
4. **라이브-백테스트 일관성**: 사이징·수수료·SL/TP 체결 규칙은 단일 정의 (`AbstractEngine` + `FeeModel` + `RiskManager`)
5. **Regime 판단은 전략 책임**: 엔진 레벨의 레짐 감지는 없음

---

## 2. 디렉토리 구조

```
src/
├── core/
│   ├── enums.py              # SignalSide, OrderSide, ExitReason, PositionStatus, EventType
│   ├── types.py              # Signal, Position, Order, Fill, Candle, ExitDecision, StrategyContext
│   ├── event_bus.py          # EventBus (subscribe/publish)
│   ├── policies.py           # ReverseSignalPolicy 추상 + 3종
│   └── engine_base.py        # AbstractEngine (공통 뼈대)
├── live/
│   └── engine.py             # CoreEngine (paper/live)
├── backtest/
│   └── engine.py             # BacktestEngine + BacktestResult + write_reports
├── strategy/
│   ├── base.py               # StrategyModule 추상
│   ├── registry.py           # @register_strategy + auto-discovery
│   ├── indicators.py         # 공통 지표 (EMA, ADX, BB, ATR, RSI 등)
│   └── plugins/              # ★ 신규 전략 위치
│       └── example.py        # 샘플 (참고용)
├── execution/
│   ├── broker.py             # Live/Paper executor facade
│   ├── live_executor.py      # OKX 주문 실행
│   └── paper_executor.py     # 시뮬레이션
├── risk/
│   └── manager.py            # 사이징 + DD락 + 일일 한도
├── accounting/
│   └── fee_model.py          # 수수료/슬리피지/펀딩비
├── data/
│   ├── feed.py               # WebSocket 캔들 구독
│   ├── historical.py         # 캐시·API 병합 백필
│   └── store.py              # SQLite (라이브/페이퍼만)
├── utils/
│   ├── config_loader.py      # YAML + .env
│   └── logger.py             # rich + 회전 파일
└── main.py                   # CLI 엔트리포인트
```

---

## 3. 신규 전략 작성 (3단계)

### 3.1 전략 파일 생성

`src/strategy/plugins/my_strategy.py`:

```python
from __future__ import annotations

import pandas as pd

from src.core.enums import SignalSide
from src.core.types import Signal, StrategyContext
from src.strategy.base import StrategyModule
from src.strategy.indicators import compute_ema, compute_atr
from src.strategy.registry import register_strategy


@register_strategy
class MyStrategy(StrategyModule):
    name = "my_strategy"               # config 네임스페이스와 일치
    entry_timeframe = "1h"             # 신호 생성 TF
    required_timeframes = ["1h", "4h"] # 지표 계산용 TF 합집합

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        df = ctx.candles[self.entry_timeframe]
        if len(df) < 50:
            return Signal(side=SignalSide.HOLD)
        ema_fast = compute_ema(df, self.params.get("ema_fast", 20)).iloc[-1]
        ema_slow = compute_ema(df, self.params.get("ema_slow", 50)).iloc[-1]
        if ema_fast > ema_slow * 1.001:
            return Signal(side=SignalSide.LONG)
        if ema_fast < ema_slow * 0.999:
            return Signal(side=SignalSide.SHORT)
        return Signal(side=SignalSide.HOLD)

    def compute_stop_loss(self, ctx: StrategyContext, signal: Signal) -> float:
        atr = compute_atr(
            ctx.candles[self.entry_timeframe], self.params.get("atr_period", 14)
        ).iloc[-1]
        mult = self.params.get("atr_sl_mult", 1.5)
        if signal.side == SignalSide.LONG:
            return ctx.current_price - atr * mult
        return ctx.current_price + atr * mult

    def compute_take_profit(
        self, ctx: StrategyContext, signal: Signal, stop_loss: float
    ) -> float:
        rr = self.params.get("reward_risk_ratio", 2.0)
        risk = abs(ctx.current_price - stop_loss)
        if signal.side == SignalSide.LONG:
            return ctx.current_price + risk * rr
        return ctx.current_price - risk * rr
```

### 3.2 config 섹션 추가

`config/default.yaml`:

```yaml
my_strategy:
  risk_per_trade_pct: 0.01         # 필수 (엔진 사이징)
  max_leverage: 5                  # 필수 (엔진 사이징)
  ema_fast: 20
  ema_slow: 50
  atr_period: 14
  atr_sl_mult: 1.5
  reward_risk_ratio: 2.0
```

### 3.3 활성화

```yaml
strategies:
  active: ["my_strategy"]
```

저장 후 즉시 다음 백테/페이퍼/라이브 실행 시 자동 적용. **엔진 코드 수정 0**.

---

## 4. StrategyModule 인터페이스

### 4.1 클래스 속성 (필수 선언)

| 속성 | 타입 | 설명 |
|---|---|---|
| `name` | `str` | 고유 식별자. config 네임스페이스 키와 일치해야 함 |
| `entry_timeframe` | `str` | `"1m"` / `"5m"` / `"15m"` / `"1h"` / `"4h"` / `"1d"` 중 하나 |
| `required_timeframes` | `list[str]` | 지표 계산에 필요한 TF 합집합 (entry_timeframe 포함) |
| `supports_pyramiding` | `bool` | 기본 `False`. `True` 면 `generate_pyramid_signal` 훅 활성 |

### 4.2 필수 메서드

| 메서드 | 호출 시점 | 반환 |
|---|---|---|
| `generate_signal(ctx)` | entry_tf 봉 마감 + 슬롯 빔 | `Signal(side=LONG/SHORT/HOLD)` |
| `compute_stop_loss(ctx, signal)` | 진입 직전 | `float` (가격) |
| `compute_take_profit(ctx, signal, sl)` | 진입 직전 | `float` (가격) |

### 4.3 선택 훅 (기본 no-op)

| 훅 | 호출 시점 | 용도 |
|---|---|---|
| `on_bar_close(ctx, tf)` | 모든 TF 봉 마감 | 보조 TF 상태 갱신 |
| `update_stop_loss(ctx, position)` | 봉 마감 + 보유 중 | 동적 SL (trailing 등). `None` = 변경 없음 |
| `should_force_exit(ctx, position)` | 봉 마감 + 보유 중 | 전략 특화 청산 (시간 timeout, regime 변화 등). `ExitDecision` 반환 시 청산 |
| `on_position_opened(position)` | 진입 직후 | 내부 상태 초기화 |
| `on_position_closed(position, pnl)` | 청산 직후 | 결과 학습/통계 |
| `generate_pyramid_signal(ctx, position)` | 보유 중 + supports_pyramiding | 추가 진입 신호 (interface만, 실 처리는 향후 확장) |

### 4.4 StrategyContext

```python
@dataclass
class StrategyContext:
    candles: dict[str, pd.DataFrame]   # {"1d": df, "4h": df, "15m": df}
    current_price: float
    balance: float
    position: Position | None           # 이 전략 소유 포지션 (없으면 None)
    is_slot_occupied: bool              # 다른 전략이 슬롯 점유 중인지
    params: dict                        # config[self.name] 내용
    now: datetime
```

`ctx.candles[tf]` 는 그 시점까지의 캔들 DataFrame. 마지막 행이 방금 마감된 봉.

### 4.5 Signal

```python
@dataclass
class Signal:
    side: SignalSide                    # LONG / SHORT / HOLD
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)  # 디버그 정보
```

---

## 5. 엔진 동작 흐름 (이해용)

### 5.1 봉 마감 시 (백테/라이브 동일)

```
BAR_CLOSED 이벤트
  ↓
1) 보유 중이면 → check_candle_sl_tp(high, low) → 도달 시 close_position
2) 보유 중이면 → check_strategy_exits → update_stop_loss + should_force_exit
3) evaluate_strategies_on_bar:
   3-a) 모든 전략에 on_bar_close 훅 dispatch
   3-b) 슬롯 빔 → entry_tf 일치 전략 우선순위 순회 → 첫 actionable 신호로 진입
   3-c) 슬롯 참 + reverse_policy != ignore → reverse 검사 후 청산·재진입
   3-d) 슬롯 참 + 보유 전략.supports_pyramiding → pyramid hook
4) equity 로깅
```

### 5.2 진입 (try_enter)

1. `risk_manager.validate_order(balance, current_position_count)` — DD락·일일손실·동시포지션 검사
2. `strategy.compute_stop_loss / compute_take_profit` 호출
3. `risk_manager.calculate_position_size(price, sl, balance, risk_per_trade_pct=..., max_leverage=...)` 사이징
4. `broker.open_position(side, size, fill_price)` — Live: OKX 주문 / Paper: 즉시 진입
5. `_record_trade_open` — Live: DB / 백테: 메모리
6. `broker.place_stop_loss / place_take_profit` — 라이브에서만 거래소 pending 주문 (안전장치)
7. `Position` 객체 생성 + `strategy.on_position_opened()`

### 5.3 청산 (close_position)

1. `broker.cancel_all_orders()` — pending SL/TP 취소
2. `broker.close_position(side, size, fill_price)` — 시장가 청산
3. `fee_model.estimate_round_trip` + `calc_pnl` — 수수료 + LONG/SHORT PnL
4. `_record_trade_close` — Live: DB / 백테: 메모리 (메모리는 종료 시 trades.csv로 출력)
5. `risk_manager.add_pnl + update_equity`
6. `strategy.on_position_closed(position, net_pnl)`

---

## 6. (C) 배타 경합 정책 — 다중 전략

`risk.max_concurrent_positions: 1` 정책상 **전역 슬롯 1개**. 여러 전략이
활성이어도 한 번에 한 전략의 포지션만 보유.

```
슬롯 비어있을 때 봉 마감 →
  for strategy in strategies (config.active 순서 = 우선순위):
      if strategy.entry_timeframe == 현재 봉 TF:
          signal = strategy.generate_signal(ctx)
          if signal is actionable:
              try_enter(strategy, signal) → 슬롯 차임 → break
```

→ **선언 순서가 우선순위**. config의 `active` 리스트 앞에 둔 전략이 먼저 평가됨.

### reverse_signal_policy

`engine.reverse_signal_policy` config 값:

| 값 | 동작 |
|---|---|
| `"ignore"` (default) | 슬롯 차있을 때 다른 전략의 `generate_signal` 호출 X |
| `"reverse"` | 보유 포지션과 반대 방향 신호가 들어오면 청산 후 재진입 |
| `"same_strategy_only"` | 같은 전략의 반대 신호일 때만 reverse |

---

## 7. 공통 지표 (`strategy/indicators.py`)

```python
from src.strategy.indicators import (
    compute_ema, compute_sma, compute_ma,
    compute_macd, compute_adx, compute_bbands, compute_atr,
    compute_choppiness, compute_efficiency_ratio,
    compute_rsi, compute_bb_width,
)
```

신규 지표 추가 시 이 모듈에 함수 추가. pandas DataFrame 입력 → Series 또는 DataFrame 반환.

---

## 8. 테스트 작성

### 8.1 위치

- 단위 테스트: `tests/test_<module>.py`
- 신규 전략 테스트: `tests/test_<strategy>.py`

### 8.2 격리: registry reset

각 테스트가 다른 전략 등록을 간섭하지 않게:

```python
import pytest
from src.strategy.registry import reset_registry_for_testing

@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()
```

### 8.3 BacktestEngine end-to-end 테스트

```python
@pytest.mark.asyncio
async def test_my_strategy(tmp_path):
    register_strategy(MyStrategy)
    config = {
        "exchange": {"symbol": "BTC/USDT:USDT"},
        "database": {"path": str(tmp_path / "bt.db")},
        "paper": {"initial_balance": 10000},
        "accounting": {"taker_fee_pct": 0.0005, "slippage_pct": 0.0},
        "risk": {
            "max_daily_loss_pct": 0.5, "max_drawdown_pct": 0.5,
            "max_position_size_btc": 1.0, "max_concurrent_positions": 1,
        },
        "strategies": {"active": ["my_strategy"]},
        "my_strategy": {"risk_per_trade_pct": 0.01, "max_leverage": 5, ...},
    }
    eng = BacktestEngine(config, start="2024-01-01", end="2024-01-31")
    eng.inject_candles({"1h": _make_candles(...)})  # HistoricalDataLoader 우회
    await eng.broker.initialize()
    bal = await eng.broker.get_balance()
    eng.risk_manager.set_initial_balance(bal)
    await eng.run()
    result = await eng.get_result()
    await eng.shutdown()
    assert result.num_trades >= 1
```

### 8.4 fees/slippage 정합성 회귀 테스트 (I-B012 회귀 방지)

paper_executor가 fees를 balance에 정확히 반영하는지 검증. `tests/test_backtest_fees.py` 참조.

3 invariant:
- `initial_balance + sum(trades.csv.pnl)` == `equity_curve.csv` 마지막 row balance
- `sum(trades.csv.pnl)` == `metrics.json["integrated"]["total_pnl"]`
- fees 증가 시 final_balance 단조 감소

paper_executor / FeeModel / BacktestEngine.close_position 흐름 변경 시 반드시 통과해야 함.

---

## 9. 보안 가이드

| 항목 | 정책 |
|---|---|
| OKX API 자격증명 | `.env` 만 사용. config·코드·로그 어디에도 평문 X |
| `config_snapshot.yaml` | `BacktestEngine.write_reports` 가 자격증명 키 자동 제거 후 저장 |
| 로그 | `setup_logger` 가 자격증명을 로그에 출력하지 않음 (직접 print 금지) |
| 거래 데이터 | `data/coinbot_*.db` 는 `.gitignore`. 사적 데이터 노출 방지 |

신규 코드 작성 시 자격증명 / 거래 이력이 **stdout · 파일 · git** 어디에도 노출되지 않도록 주의.

---

## 10. 새 모듈 추가 시

| 종류 | 위치 | 비고 |
|---|---|---|
| 새 전략 | `src/strategy/plugins/` | auto-discovery |
| 새 지표 | `src/strategy/indicators.py` | 공통 라이브러리 확장 |
| 새 Reverse 정책 | `src/core/policies.py` 에 클래스 추가 + `_POLICY_REGISTRY` 등록 | 그 후 config의 `engine.reverse_signal_policy` 값으로 사용 |
| 새 거래소 | `src/execution/` 에 신규 executor + Broker 분기 | OKX 외 거래소 지원 시 |
| 새 데이터 소스 | `src/data/` 에 신규 feed | WebSocket 외 다른 소스 |
| 새 분류 모델 + Calibration | plugin에 `calibration_method` 파라미터 추가 + `_ensure_model`에서 `model_dir/calibrator_<method>.joblib` 자동 로드 + `generate_signal`에서 `raw_probs → calibrator.transform` 분기 (`src/strategy/plugins/ml_lightgbm.py` 참고) | `scripts/calibrate_models.py` 패턴으로 calibrator 학습 별도 진행 |
| 새 통계 메트릭 (Sharpe/Calmar/Bootstrap 외) | `src/ml/metrics_extended.py`에 함수 추가 + `tests/test_metrics_extended.py`에 단위 테스트 추가 | `scripts/analyze_results.py`가 자동 활용 |

엔진(`AbstractEngine`/`CoreEngine`/`BacktestEngine`) 자체 수정은 **공통
인프라 변경**일 때만. 전략 추가만으로 엔진을 건드린다면 추상화가 잘못된 것.

---

## 11. 디버깅 팁

### 11.1 로그 레벨

```yaml
logging:
  level: "DEBUG"
```

신호 생성·진입 검증 흐름 상세 출력. 로그 파일은 `logs/coinbot_YYYYMMDD_HHMMSS.log`.

### 11.2 합성 캔들로 빠른 검증

`tests/test_backtest_engine.py` 의 `_make_synthetic_candles` 패턴 참고. drift 인자로 강제 추세 만들어 SL/TP 체결 시점 통제 가능.

### 11.3 `inject_candles` 로 외부 데이터 사용

`HistoricalDataLoader` 를 우회해 임의 캔들 주입:

```python
eng.inject_candles({"15m": custom_df})
```

### 11.4 metrics.json 검사

백테 결과의 `by_strategy_name`·`by_exit_reason`·`by_direction` 분할로 어느
조건에서 손실/이익이 발생했는지 빠르게 파악.

### 11.5 백테 결과 정합성 검증 (CLAUDE.md 협업 규칙 10)

수익률이 의심스럽거나 fees/slippage 변경 효과를 빠르게 진단할 때, 직관적 가설(데이터 누출/lookahead 등)보다 데이터 단위 정합성을 먼저 점검:

```python
import json, pandas as pd
trades = pd.read_csv("path/to/trades.csv")
metrics = json.load(open("path/to/metrics.json"))["integrated"]
sum_pnl = float(trades["pnl"].sum())
total_pnl = float(metrics["total_pnl"])
assert abs(sum_pnl - total_pnl) < 0.01, f"trades.csv ↔ metrics.json 불일치: {sum_pnl} vs {total_pnl}"
```

`equity_curve.csv` 마지막 balance도 같이 비교 (`initial + sum_pnl`과 일치). 불일치 시 paper_executor·FeeModel·BacktestEngine.close_position 흐름 점검 — I-B012 같은 라이브-백테 일관성 위반 가능.

---

## 12. 더 알아보기

- 사용자 가이드 → `USER_GUIDE.md`
- 설계 사양 (단계별 진행 기록·잠재 이슈 트래커 포함) → `../00_Work_Report/PROTOTYPE_DESIGN_260425.md`
