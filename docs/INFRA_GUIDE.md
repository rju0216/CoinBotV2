# INFRA_GUIDE — CoinBot 백테 골격

새 퀀트 모델을 바로 구현할 수 있도록 **최소 개념 골격만 남긴 상태**의 단일 안내서.
과거의 특정 모델·학습/평가 파이프라인, 그리고 라이브/페이퍼 실거래·거래소 종속
인프라(OKX 주문·websocket·DB·알림)는 의도적으로 제거했다. 여기 담긴 것은 **어떤
모델에도 공통인 백테 실행 골격**과 그 구조상 주의점뿐이다.

---

## 1. 아키텍처 한눈에

- **플러그인 + 엔진 분리**: 엔진은 전략의 존재를 모른다. 전략은
  `src/strategy/plugins/<name>.py` 파일 1개 + `config/default.yaml` 섹션 1개로
  추가된다. **엔진 코드 수정 0** 이 원칙 — 수정이 필요하면 추상화가 잘못된 것.
- **거래 정책 = 모델 소유**: 진입 신호·사이징·SL/TP·reverse·진입 게이트는 전부
  `StrategyModule` 의 추상 메서드다. 엔진은 그 결정을 받아 *집행*만 하며 정책
  기본값을 갖지 않는다.
- **체결·정산**: 백테 엔진이 캔들 가격으로 체결을 시뮬하고 PnL 은
  `FeeModel.calc_pnl` 단일 공식으로 정산한다 (외부 브로커·거래소 없음, 순수 인메모리).

```
HistoricalDataLoader(캔들) → 마스터 TF 순회 → (보유) SL/TP 캔들 체결 검사
   → (보유) update_stop_loss / should_force_exit 훅
   → 봉 경계 TF 별 evaluate_strategies_on_bar
       → (슬롯 빔) generate_signal → allow_entry·compute_stop_loss/take_profit·
         compute_position_size (전부 모델) → Position 등록
       → (슬롯 참) should_reverse 로 청산·재진입
   → 청산 시 FeeModel 정산 → 인메모리 trades/equity 기록
종료 시 잔여 포지션 ENGINE_SHUTDOWN 강제 청산 → write_reports 3종 출력
```

## 2. 모듈 구조

```
src/
├── core/
│   ├── types.py     # Signal / Position / ExitDecision / AccountState / StrategyContext
│   └── enums.py     # SignalSide / PositionSide / PositionStatus / ExitReason
├── backtest/engine.py  # BacktestEngine: 봉마감 평가·진입·청산·SL/TP 시뮬 + 리포트 (엔진 전체)
├── strategy/
│   ├── base.py       # StrategyModule 추상 (필수 6 + 선택 훅)
│   ├── registry.py   # @register_strategy + auto-discovery + 활성화
│   └── plugins/      # ★ 신규 전략 파일을 여기에 (현재 비어 있음)
├── accounting/
│   ├── fee_model.py       # 수수료·PnL 단일 공식 (calc_pnl)
│   └── account_tracker.py # equity/peak/daily_pnl 계측 (정책 enforcement 없음)
├── data/historical.py  # 백테 캔들 로더 (CSV 캐시 + OKX 공개 API 병합, 동기)
└── utils/           # config_loader (순수 yaml) / logger

config/default.yaml          # 인프라 공통 설정 + (비어 있는) 전략 섹션 자리
src/main.py                  # CLI (backtest)
scripts/download_history.py  # 캔들 다운로드
tests/                       # 백테 골격 회귀 테스트 (픽스처는 tests/strategy_stub.py)
```

> **`src/research/` (모델 개발 오프라인 substrate)**: 위 백테 골격과 **별개**로, 새 퀀트
> 모델의 라벨·피처·학습·검증 파이프라인이 사는 곳. 엔진/플러그인과 분리(엔진 수정 0).
> - **Phase 0**(검증 substrate): `data/`(로더·audit)·`validation/`(regime·splitter·metrics·
>   baselines·harness·ledger)·`causality/`(leakage)·`normalize.py`
> - **Phase 1**(라벨): `labeling/` — `volatility`(ATR·YZ)·`triple_barrier`(3-class 삼중배리어
>   라벨·first_touch·N=label_horizon)·`distribution`(층1·층3 분포 게이트). **확정 라벨 = atr/w96/x3.0/N24**.
> - **Phase 2**(1층 피처): `features/` — `simple`(변동성변화·상대거래량·semi-dev·종가위치)·
>   `trend_strength`(ER·Hurst R/S)·`kalman`(robust KF slope+불확실성, Student-t 1-step 재가중)·
>   `build`(`build_features`→**X 8열**). 전부 OHLCV만·인과. 정규화는 harness 소유
>   (z-score 기본 / `RobustScaler` 드롭인, F-6).
> - **Phase 3 R1**(2층 모델·관문1): `models/` — `base`(`ProbaModel` 계약)·`tree_bench`(lightgbm)·
>   `mlp`(torch, 트렁크/헤드 분리, 단일태스크) + `experiments/r1_smoke`(관문1 스모크 러너·**사전등록
>   판정규칙**). **관문1 PASS**(mlp STRONG). `loader.resample_ohlcv`(1h→1d 파생, I-001 해소).
> - **Phase 3 R2**(창 순차탐색): `experiments/r2_window_search`(coordinate descent, MLP 선택+트리 참고)
>   + `features.hurst` 벡터화(F-13, 75x). 결과 = **기본창 유지**(창 튜닝 무gain). MasterPlan §12-4.
> - **Phase 3 R3**(모델 하이퍼): `models/mlp` 멀티태스크 헤드(reach·λ·expire마스킹, reach=None=단일태스크)
>   + `experiments/r3_multitask`(멀티태스크·정규화)·`r3_arch_kf`(구조·KF). 결과 = **R1 config 유지**
>   (멀티태스크·robust·용량↑·KF 무gain, **1h 튜닝저항 확정**). F-6 최종=z. §12-5. 다음=R4(TF/MTF).
> - 상세·진행은 `docs/00_Work_Report/QuantModel_MasterPlan.md`, 의존성은 `requirements-ml.txt`.

## 3. 새 모델(전략) 추가 방법

1. `src/strategy/plugins/my_strategy.py` 작성:
   ```python
   from src.core.enums import SignalSide
   from src.core.types import Signal, StrategyContext
   from src.strategy.base import StrategyModule
   from src.strategy.registry import register_strategy

   @register_strategy
   class MyStrategy(StrategyModule):
       name = "my_strategy"
       entry_timeframe = "15m"
       required_timeframes = ["15m"]
       sl_tp_fill_priority = "sl_first"   # 동시도달 처리 (필수 선언)

       # 필수 6 = 거래 정책 전부 모델 소유
       def generate_signal(self, ctx) -> Signal: ...
       def compute_stop_loss(self, ctx, signal) -> float | None: ...   # None=standing SL 없음
       def compute_take_profit(self, ctx, signal, sl) -> float | None: ...
       def compute_position_size(self, ctx, signal, sl) -> float: ...  # 사이징 공식
       def should_reverse(self, ctx, position, new_signal) -> bool: ...
       def allow_entry(self, ctx) -> bool: ...                          # 진입 게이트(리스크)
   ```
2. `config/default.yaml` 에 `my_strategy:` 섹션 추가 — **엔진이 강제하는 필수 키는
   없다.** 정책 임계값(예: `risk_per_trade_pct`)을 *모델이 자기 params 로* 정의하고
   위 메서드에서 읽는다.
3. `strategies.active` 리스트에 `"my_strategy"` 추가 (리스트 순서 = 우선순위).

### StrategyModule 인터페이스 — 거래 정책은 전부 모델 소유
- **필수 6 (abstract — 엔진에 기본값 없음)**: `generate_signal` /
  `compute_stop_loss`(None=standing SL 없음) / `compute_take_profit` /
  `compute_position_size`(사이징·레버리지·변동성) / `should_reverse` /
  `allow_entry`(DD/일일손실 등 진입 게이트) + 클래스 속성 `sl_tp_fill_priority`.
- **선택 훅 (기본 no-op, 엔진이 호출)**: `on_bar_close` · `update_stop_loss`(trailing) ·
  `should_force_exit`(regime/timeout 청산) · `on_position_opened` · `on_position_closed`.
- 모델 학습/추론·피처 엔지니어링·사이징 공식은 **전략이 자체 책임**. 인프라는
  피처/학습 파이프라인을 제공하지 않는다 (`generate_signal` 이 `Signal` 만 반환하면 됨).

## 4. 구조상 주의점 (구현 전 반드시 숙지)

0. **엔진=메커니즘 / 모델=정책 (핵심 원칙)**: 엔진은 거래 *정책*을 결정하지 않는다.
   진입 게이트·사이징·SL/TP·reverse·동시도달 우선순위는 전부 `StrategyModule` 의
   추상 메서드로 모델이 소유한다. 엔진의 `try_enter` 등은 그 메서드를 호출하고
   결과를 집행할 뿐이다. 남는 조건문은 슬롯 상태·actionable 여부·가격 교차 감지
   같은 *메커니즘*뿐이다.

1. **단일 전역 슬롯 (엔진 메커니즘)**: 동시에 1개 포지션만 보유(`_position` 단일
   슬롯). 여러 전략이 active 면 리스트 순서가 우선순위. 멀티 포지션이 필요하면
   엔진 슬롯 로직을 확장해야 한다 — 현재 가정에 잠겨 있음.

2. **봉 마감 기반 평가**: 진입 신호는 `entry_timeframe` **봉 마감 시점**에만
   생성된다. 틱 단위 의사결정 모델은 이 골격과 맞지 않는다.

3. **Lookahead 차단 (직접 손대지 말 것)**: `BacktestEngine._slice_candles(ts)` 가
   `df.index < ts` 로 진행 중 봉을 배제한다. 그래서 `ctx.candles[tf].iloc[-1]` 은
   항상 **직전 마감 봉**이고, 진입가는 현재 봉 open 으로 체결한다. 백테가 비현실적으로
   좋게 나오면 이 슬라이스가 깨졌는지부터 의심한다. (test_lookahead 로 회귀 방지.)

4. **SL/TP 는 모델이 거는 선언형 청산**: 값은 모델 소유(`compute_stop_loss` /
   `compute_take_profit` / `update_stop_loss`). `compute_stop_loss` 가 `None` 이면
   standing SL 없음 → 청산을 전적으로 `should_force_exit`(명령형)로 가져갈 수 있다.
   체결 시뮬은 `check_candle_sl_tp` 가 캔들 high/low 로 가격 교차를 감지하고, 동시
   도달 시 모델의 `sl_tp_fill_priority` 로 결정한다.

5. **PnL·수수료는 `FeeModel.calc_pnl` 단일 공식**. 백테 결과 의심 시 데이터 단위
   정합성부터 검증: `trades pnl 합` ↔ `metrics total_pnl` ↔ `equity_curve 변화량`
   (test_backtest_fees 가 이 invariant 를 검증).

6. **사이징·리스크 게이트는 모델 메서드**: `compute_position_size`(공식·레버리지·
   변동성) / `allow_entry`(DD·일일손실 등 진입 차단). 판단 재료는 `ctx.account`
   (`AccountState`: balance·equity·peak·daily_pnl·drawdown_pct). `account_tracker` 는
   이 값들을 *계측*만 하고 정책 enforcement 는 하지 않는다.

7. **결과는 인메모리 → 파일**: 백테는 trades/equity 를 메모리에 누적하고 종료 시
   `write_reports` 로 3종 파일(`trades.csv` / `equity_curve.csv` / `metrics.json`)만
   출력한다. DB·차트·거래소 연동은 없다.

## 5. 명령어

```bash
# 캔들 다운로드 (OKX 공개 API — 키 불필요)
python scripts/download_history.py --timeframe 15m --start 2024-01-01 --end 2024-12-31

# 백테스트
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
#   결과: data/backtest_reports/backtest_<start>_<end>/ 3종 (trades.csv, equity_curve.csv, metrics.json)

# 테스트
python -m pytest tests/ -q
```

`strategies.active: []` 인 현재 상태에서는 **무거래**로 동작한다 (엔진 골격만 도는
안전 상태). 전략을 추가해야 매매가 발생한다.
