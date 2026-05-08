# User Guide

CoinBot 뼈대 프로토타입의 사용자 가이드. 설치 → 환경 설정 → CLI 사용 →
백테 결과 해석 → 라이브 운영까지 다룬다.

---

## 1. 설치

### 1.1 Python

Python 3.11+ 권장. `python --version` 으로 확인.

### 1.2 의존성 설치

```bash
pip install -r requirements.txt
```

### 1.3 환경변수 (`.env`)

OKX API 자격증명은 `config/default.yaml` 에 두지 않고 `.env` 파일로 주입.
프로젝트 루트에 `.env` 작성:

```
OKX_API_KEY=your_api_key_here
OKX_SECRET=your_secret_here
OKX_PASSPHRASE=your_passphrase_here
```

`.env` 는 `.gitignore` 에 포함되어 git에 올라가지 않는다.

paper / backtest 모드만 사용한다면 자격증명 없어도 동작 (sandbox=false 라도
공개 캔들 API 만 사용). live 모드 시 필수.

---

## 2. CLI

세 가지 subcommand. config 미지정 또는 잘못된 인자 시 argparse 에러로 종료.

### 2.1 백테스트

```bash
python -m src.main backtest --config config/default.yaml --start 2024-01-01 --end 2024-12-31
```

- `--start` / `--end` 필수, ISO 또는 `YYYY-MM-DD` 형식
- 종료 후 `data/backtest_reports/00_Working/{tag}_backtest_{start}_{end}_{config_name}/{config_name}/` 에 결과 5종 파일 생성
- stdout 에 Summary 출력

### 2.2 페이퍼 모드 (시뮬레이션 실시간)

```bash
python -m src.main paper --config config/default.yaml
```

- 실제 가격으로 시뮬레이션, 거래소 주문은 보내지 않음
- `data/coinbot_paper.db` 에 거래·자산 영속 기록
- Ctrl+C 로 graceful shutdown

### 2.3 라이브 모드 (실거래)

```bash
python -m src.main live --config config/default.yaml
```

⚠️ **실제 자금이 거래되는 모드.** 시작 전 §5 라이브 운영 가이드 필독.

### 2.4 ML 모델 통합 평가 (`scripts/evaluate_models.py`)

5 모드 (Phase E-2 ~ E-2-3):

```bash
# 5 모델 × 6 분할 = 30 백테 + 베이스라인(B&H + macross 12) 일괄
python scripts/evaluate_models.py --mode full --eval-date 260502

# 특정 모델 + 분할
python scripts/evaluate_models.py --mode single --strategy ml_lightgbm --split 1

# 결과 수집만 (이미 백테 끝났을 때)
python scripts/evaluate_models.py --mode collect

# 슬리피지 sensitivity (5 모델 × {분할 1, Exp4} × 4 슬리피지 = 40)
python scripts/evaluate_models.py --mode sensitivity --eval-date 260503_sensitivity

# Calibration 백테 (4 분류 모델 × 분할 1 × {Platt, Isotonic} = 8)
python scripts/evaluate_models.py --mode calibration --eval-date 260503_calibration
```

- 출력: `data/backtest_reports/00_Working/eval_{날짜}/`
- multiprocessing.Pool(N=4) + 캔들 캐시 워밍업 자동 — 30 specs 약 5h
- 각 spec별 5종 파일 + 통합 `comparison.csv`

### 2.5 Calibrator 학습 (`scripts/calibrate_models.py`)

분류 모델 4개의 confidence calibration용. v001 모델 디렉토리에 `calibrator_platt.joblib` + `calibrator_isotonic.joblib` + `calibration_meta.json` 생성.

```bash
# 4 분류 모델 모두 (~8분)
python scripts/calibrate_models.py --strategy all --start 2020-01-01 --end 2024-12-31

# 특정 모델만
python scripts/calibrate_models.py --strategy ml_lightgbm --start 2020-01-01 --end 2024-12-31
```

- 학습 데이터로 walk-forward 26 folds 재실행 → OOS probabilities 수집 → calibrator 학습
- 기존 모델 파일(`model.txt`/`model.pth`)은 변경 0
- 백테 시 `config[strategy].calibration_method`로 적용 method 선택 (§3 참조)

### 2.6 결과 통계 분석 (`scripts/analyze_results.py`)

evaluate_models 결과 폴더에서 30 specs 매트릭스 메트릭 + 5 모델 pairwise bootstrap 검정 (Phase E-2-4 Step 2):

```bash
# 30 specs 매트릭스 + 20 pairwise bootstrap (분할 1, Exp4)
python scripts/analyze_results.py --eval-date 260503_baseline

# 매트릭스만 (메트릭 추출만, bootstrap 생략)
python scripts/analyze_results.py --eval-date 260503_baseline --metrics-only

# Bootstrap만 (매트릭스 생략)
python scripts/analyze_results.py --eval-date 260503_baseline --bootstrap-only --bootstrap-n 10000 --seed 42
```

- 입력: `eval_<날짜>/{strategy}_{split}/{config}/{trades.csv, equity_curve.csv, metrics.json}`
- 출력:
  - `analysis_metrics.csv` — 42 rows (30 specs + macross 6 + B&H 6) — Sharpe/Calmar/PF/MDD/Win rate
  - `bootstrap_pvalues.csv` — 20 rows (5 모델 pairwise × {분할 1, Exp4}) — p-value, CI
- annualization=365 (crypto 24/7 표준)
- B&H는 equity_curve 없음 → Sharpe NaN
- 실행 시간 ~30초 (Bootstrap n=10,000)

### 2.7 결과 시각화 (`scripts/plot_results.py`)

분할별 1 PNG × 6 — 5 모델 + macross + B&H equity overlay + BTC 가격 보조축:

```bash
# 6 분할 모두
python scripts/plot_results.py --eval-date 260503_baseline

# 특정 분할만
python scripts/plot_results.py --eval-date 260503_baseline --split 1
```

- 출력: `eval_<날짜>/equity_overlay_<split>.png` (16×9, 150 DPI, ~300KB)
- log scale Y축 (수익률 격차 5,000% ~ -5% 모두 표시)
- BTC 1d 보조 Y축 (시장 흐름 비교)
- B&H는 BTC 가격 시뮬레이션 (`initial × close/first_close`)
- 실행 시간 ~30초 (6 PNG)

---

## 3. config/default.yaml

```yaml
exchange:
  name: "okx"
  symbol: "BTC/USDT:USDT"
  sandbox: false
  leverage: 5

engine:
  reverse_signal_policy: "ignore"   # "ignore" | "reverse" | "same_strategy_only"

risk:
  max_daily_loss_pct: 0.05
  max_drawdown_pct: 0.35
  max_position_size_btc: 1.0
  max_concurrent_positions: 1

accounting:
  taker_fee_pct: 0.0005
  slippage_pct: 0.0
  funding_enabled: true

paper:
  initial_balance: 10000.0

data:
  history_bars: 300
  candle_dir: "data/candles"

database:
  path: "data/coinbot.db"

strategies:
  active: []                          # 활성 전략 이름 리스트 (순서 = 우선순위)

# 전략별 섹션은 strategy.name 과 일치하는 키
example_macross:
  risk_per_trade_pct: 0.01            # 필수
  max_leverage: 5                     # 필수
  ma_fast: 20
  ma_slow: 50
  ...
```

전략 추가는 `DEVELOPER_GUIDE.md` 참조.

### 분류 모델용 calibration_method (Phase E-2-3, I-B009)

`ml_lightgbm`/`ml_xgboost`/`dl_lstm`/`dl_transformer` 4 분류 모델은 confidence calibration 옵션 지원. PPO는 정책 모델이라 미적용.

```yaml
ml_lightgbm:
  ...
  calibration_method: "none"        # 기본: raw probability 그대로 사용
  # calibration_method: "platt"     # Platt scaling 적용 (효과 미미)
  # calibration_method: "isotonic"  # Isotonic regression (4 모델 모두 개선 — 추천)
```

`"none"` 외 값 사용 시 `model_dir/calibrator_<method>.joblib` 파일 필요. 없으면 `scripts/calibrate_models.py`로 사전 학습 (§2.5 참조). 분할 1 OOS 검증 결과: Isotonic이 4 모델 모두에서 raw 대비 개선 (Transformer +7.9%로 가장 큼).

### 전략 On/Off

```yaml
strategies:
  active: ["my_strategy"]             # On
  # active: []                        # Off (무거래)
  # active: ["a", "b"]                # 둘 다 On (선언 순서 = 우선순위)
```

---

## 4. 백테 결과 해석

### 4.1 출력 디렉토리 구조

```
data/backtest_reports/00_Working/
└── 260425_backtest_2024-01-01_2024-12-31_default/
    └── default/
        ├── trades.csv          ─ 모든 closed 거래
        ├── equity_curve.csv    ─ 시계열 잔고
        ├── metrics.json        ─ 통합 메트릭 + by_strategy_name/exit_reason/direction 분할
        ├── config_snapshot.yaml─ 사용된 config (자격증명 제거)
        └── equity_curve.png    ─ Equity + BTC 가격 + Drawdown 차트
```

### 4.2 Summary stdout

```
---- Backtest Summary ----
  initial_balance: 10000.0
  final_balance: 9694.32
  total_pnl: -305.68
  total_pnl_pct: -3.06
  num_trades: 12
  num_winners: 3
  num_losers: 9
  win_rate: 25.0
  max_drawdown_pct: 3.06
  reports: data\backtest_reports\00_Working\...
```

### 4.3 metrics.json 구조

```json
{
  "integrated": { "initial_balance", "final_balance", "total_pnl", "total_return_pct",
                  "total_trades", "winning_trades", "losing_trades", "win_rate_pct",
                  "gross_profit", "gross_loss", "profit_factor", "avg_win", "avg_loss",
                  "max_drawdown_pct" },
  "by_strategy_name": { "<name>": { "trades", "winning_trades", "losing_trades",
                                     "win_rate_pct", "pnl", "profit_factor" } },
  "by_exit_reason":  { "<reason>": {...} },
  "by_direction":    { "long"/"short": {...} }
}
```

### 4.4 결과 정합성 검증 (의심스러울 때)

수익률이 비현실적으로 높거나 fees/slippage 변경 효과가 안 보일 때, 직관 가설(데이터 누출 등)보다 데이터 단위 정합성 먼저 점검:

- **trades.csv pnl 합** == **metrics.json `integrated.total_pnl`**
- **initial_balance + sum(trades.pnl)** == **equity_curve.csv 마지막 row balance**

불일치 시 paper_executor·FeeModel·BacktestEngine.close_position 흐름 점검 (I-B012 같은 라이브-백테 일관성 위반 가능). DEVELOPER_GUIDE §11.5 참조. 자동 회귀 보호: `tests/test_backtest_fees.py`.

### 4.5 통계 분석 결과 해석 (`scripts/analyze_results.py` 출력)

#### `analysis_metrics.csv` 컬럼

| 컬럼 | 의미 |
|---|---|
| `sharpe_ratio` | 연환산 Sharpe (annualization=365). 수익률 대비 변동성. >2 양호, >5 매우 우수 |
| `calmar_ratio` | 연환산 수익률 / 최대 낙폭. >5 양호, >50 매우 우수 |
| `total_return_pct` | 백테 기간 전체 누적 수익률 |
| `profit_factor` | 총 이익 / 총 손실. >1.5 양호, >2 우수 |
| `max_drawdown_pct` | 누적 잔액의 최대 낙폭 |

#### `bootstrap_pvalues.csv` 컬럼

| 컬럼 | 의미 |
|---|---|
| `mean_pnl_diff` | model_a 평균 거래 PnL - model_b 평균 |
| `p_value` | 두 모델이 같은 분포에서 추출됐다는 가설(null)의 p-value. **<0.05 = 유의한 차이** |
| `ci_low_95` / `ci_high_95` | bootstrap 분포의 95% confidence interval |
| `significant_at_0.05` | p<0.05 여부 (true/false) |

해석:
- **p>0.05** → 두 모델 차이가 통계적으로 유의하지 않음 ("같은 수준")
- **p<0.05** → 차이가 진짜
- **p<0.001** → 매우 명확한 차이

예: `ml_lightgbm vs ml_xgboost p=0.61` → 두 GBDT 모델은 통계적으로 동등, 운영 시 둘 중 하나 선택 무관.

---

## 5. 다중 연도 병렬 백테 + 통합 리포트

### 5.1 7년치 병렬 백테 (Windows)

```bash
scripts\run_full_backtest.bat config/default.yaml
```

- Phase 1: `download_history.py` 가 1d/4h/15m 캔들을 미리 다운로드
- Phase 2: 7개 콘솔에서 2020 ~ 2026 연도별 백테 병렬 실행

### 5.2 통합 리포트 생성

```bash
scripts\merge_reports.bat 260425 default
```

- `260425_backtest_*_default` 패턴 디렉토리들을 시간순 정렬
- 복리 기준으로 trades · equity_curve 통합
- `260425_backtest_MERGE_{start}_{end}_default/default/` 에 통합 5종 파일 출력

---

## 6. 라이브 운영 가이드 ⚠️

라이브 모드는 **실제 자금이 거래되는 모드**. 다음 흐름을 반드시 준수한다.

### 6.1 라이브 진입 전 체크리스트

- [ ] 전략을 백테스트로 검증 — 최소 1년치, 합리적 PnL/MDD 확인
- [ ] paper 모드로 실시간 검증 — 최소 1~2일, 신호·체결 시점 정상 확인
- [ ] `.env` 의 OKX 자격증명 정확성 확인 (`OKX_API_KEY`/`OKX_SECRET`/`OKX_PASSPHRASE`)
- [ ] `config.exchange.sandbox: false` 설정 (실거래 환경)
- [ ] `config.exchange.leverage` 와 전략별 `max_leverage` 일치성 확인
- [ ] `config.risk.max_daily_loss_pct` / `max_drawdown_pct` / `max_position_size_btc` 확인
- [ ] OKX 계정 잔액·격리/교차 모드·레버리지 설정 확인
- [ ] 거래소에 기존 포지션이 있다면 → 라이브 진입 시 `_restore_state` 가
      DB·거래소 매칭으로 자동 입양. **DB가 깨끗하면 strategy_name="_unknown"
      orphan 처리**되므로 의도한 상태인지 확인

### 6.2 자동 입양 정책 (재시작 시)

| 거래소 포지션 | DB open trade | 활성 전략 | 결과 |
|---|---|---|---|
| ∅ | ∅ | 무관 | 정상 빈 슬롯 |
| ∅ | O | 무관 | DB trades 사후 closed 처리 |
| O | 무관 | **0개** | **RuntimeError 중단** (전략 없는 뼈대 상태) |
| O | 매칭 + active | ≥1 | 정상 OPEN 복원 |
| O | 매칭 + 전략 제거됨 | ≥1 | ORPHAN (엔진 SL/TP만 유효) |
| O | 매칭 실패 | ≥1 | strategy_name="_unknown" ORPHAN |

ORPHAN 상태에서는 전략의 `update_stop_loss`·`should_force_exit` 훅이 호출되지
않으므로 동적 SL 갱신·시간 기반 청산 같은 전략 특화 로직이 작동하지 않는다.

### 6.3 라이브 운영 중 위험 한도

자동으로 적용되는 안전장치:

- **일일 손실 한도** (`risk.max_daily_loss_pct`): daily_pnl 이 한도 초과 시 신규 진입 차단
- **최대 드로우다운 락** (`risk.max_drawdown_pct`): peak 대비 한도 초과 시 emergency brake.
  코드 수준에서 `RiskManager.unlock_drawdown(current_balance)` 호출로 수동 해제 필요
- **포지션 크기 상한** (`risk.max_position_size_btc`): 사이징이 이 값을 넘지 않음
- **동시 포지션 수** (`risk.max_concurrent_positions`): 신규 진입 차단

### 6.4 라이브 종료

`Ctrl+C` 시 graceful shutdown — broker/data_store close, 현재 포지션은 그대로
유지(거래소 SL/TP pending 주문이 청산을 담당).

### 6.5 라이브 운영 중 자동 인지·복구 (BL-2-4 hotfix 효과)

라이브 모드는 거래소(tick 기반) vs 엔진(봉 OHLC 기반)의 시간 해상도 차이 때문에
다음 case에서 시스템이 자동으로 거래소 상태를 ground truth로 동기화한다:

| Case | 자동 동작 |
|---|---|
| SL/TP **봉 내 도달** | 봉 마감 시 `check_candle_sl_tp` 인지 → 정상 close |
| SL/TP **spike만 도달** | 봉 마감 시 `broker.get_position()` 거래소 ∅ 인지 → `_sync_unexpected_close` 자동 호출 |
| 사용자 **manual close** (OKX 웹) | 동일 — 거래소 ∅ 인지 후 동기화 |
| 거래소 **강제 청산** (margin call) | 동일 |
| 재시작 시 거래소 ∅ + DB OPEN | `_restore_state` case 2 + `_fetch_actual_exit`로 정확한 청산 정보 복원 |

자동 동기화 시 거래소 trade history(`fetch_closed_orders`)에서 정확한
exit_price/pnl/reason fetch → DB close + 텔레그램 EXIT 알림 + daily_pnl 누적.
LONG/SHORT 모두 동일 동작.

### 6.6 라이브 운영 모니터링 로그

15m 봉 마감마다 자동 출력 (master_timeframe 기준):

```
[SIGNAL] ensemble HOLD probs=[S:0.05 H:0.92 L:0.03] conf=0.92 threshold=0.55 contributors=[...]
[ACCOUNT] balance=$X.XX equity=$X.XX unrealized=+/-$X.XX daily_pnl=+/-$X.XX dd=X.XX%
[POSITION] ensemble LONG/SHORT size=... entry=... current=... unrealized_pnl=... (XhYZm held)  # 보유 시
```

진입/청산 발생 시 추가:
```
[SIGNAL] ... → ENTRY contributors=[...]    # 진입 결정
INFO  Opened long/short X.XXXX BTC (XX contracts) market
INFO  SL set: sell/buy XX contracts @ XXXX.XX
INFO  TP set: sell/buy XX contracts @ XXXX.XX
... (보유 중) ...
INFO  close_position skipped: exchange position already closed (SL/TP triggered or external close)  # I-BL010 효과
INFO  [EXIT [ensemble] sl_hit/tp_hit] net_pnl=$+/-XX.XX | meta={...}
```

### 6.7 텔레그램 알림 종류

| 알림 | 발송 조건 |
|---|---|
| `ENTRY [strategy]` | 신규 진입 |
| `EXIT [strategy] reason: net_pnl=$X` | 청산 (TP/SL/timeout/외부 청산) |
| `Drawdown lock triggered` | dd ≥ max_drawdown_pct |
| `Daily loss limit reached` | daily_pnl ≤ -max_daily_loss_pct |
| `Circuit breaker OPEN` | API 5회 연속 실패 |
| `OOS decay [strategy]` | 적중률 < min_acc_threshold |

`.env`의 `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` 설정 + `config.live.notifications.channels: ["log", "telegram"]` 시 활성.
plain text 형식 (Markdown 사용 안 함, 모든 특수 문자 안전).

### 6.8 OKX 계정 권한 + 마진 모드

라이브 시작 전 OKX:
- API key의 `Trade` 권한 활성 (Read-only 불가)
- USDT-margined Perpetual swap 거래 활성
- `config.exchange.leverage` 값으로 margin mode 자동 cross 설정

`tdMode: cross`가 매 주문에 명시되므로 거래소 default와 무관.

---

## 7. 데이터 캐시

`data/candles/` 에 1m/5m/15m/1h/4h/1d CSV 캐시. 백테/라이브 시작 시
`HistoricalDataLoader.download_range_merged` 가:

1. 기존 CSV 로드
2. 요청 범위에서 부족한 부분만 OKX API 다운로드
3. 합쳐서 CSV 갱신
4. 메모리에 통합 DataFrame 주입

캐시가 충분하면 API 호출 0 (오프라인도 백테 가능).

수동으로 미리 받기:

```bash
python scripts/download_history.py --config config/default.yaml \
    --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-04-25
```

---

## 8. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `Strategy 'X' not found in registry` | `src/strategy/plugins/X.py` 가 없거나 `@register_strategy` 누락 |
| `Strategy 'X' config is missing required params` | config[X] 에 `risk_per_trade_pct` 또는 `max_leverage` 누락 |
| `Cannot run: no master timeframe or candles loaded` | `strategies.active: []` (무거래) 또는 캔들 로딩 실패 |
| `Exchange has open position but no active strategies` | 라이브 시작 시 거래소에 포지션 있는데 전략 0개 → 의도적 중단. 전략 추가 또는 거래소 포지션 수동 청산 필요 |
| 백테 결과가 매번 누적되어 보임 | (이미 해결됨) 단계 14에서 `BacktestEngine` 이 메모리만 사용하도록 분리 |
| `merge_yearly_reports.py` "매칭되는 연도별 리포트가 ... 개뿐입니다" | tag·config_name 패턴 불일치. `data/backtest_reports/00_Working/` 디렉토리명 확인 |

---

## 9. 더 알아보기

- 전략 작성·플러그인 인터페이스 → `docs/01_Guides/DEVELOPER_GUIDE.md`
- 설계 사양서 → `docs/00_Work_Report/PROTOTYPE_DESIGN_260425.md`
