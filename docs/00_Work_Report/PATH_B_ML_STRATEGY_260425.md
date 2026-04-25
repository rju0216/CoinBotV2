# 경로 B: ML/DL 기반 독립 전략 설계 (260425)

## 0. 문서 목적

이 문서는 CoinBot에 ML/DL 모델이 **전략 자체**가 되는 "경로 B"의 설계 사양과 단계별 작업 계획을 기록한다. 모델 출력이 곧 매매 신호이며, 기존 규칙 기반 전략의 보조가 아닌 독립 전략 플러그인으로 구현한다.

- 작성일: 2026-04-25
- 기반: `main` 브랜치 (프로토타입 뼈대 완성 상태, 커밋 `4d10f02`)
- 선행 문서: `PROTOTYPE_DESIGN_260425.md`

---

## 1. 설계 원칙

1. **엔진 코드 수정 = 0**: 모든 ML 전략은 기존 `StrategyModule` 인터페이스를 그대로 준수. 엔진·리스크·수수료 코드에 ML 관련 분기를 추가하지 않는다.
2. **학습/추론 분리**: 학습 코드(`scripts/train_*.py`, `src/ml/`)와 추론 코드(`src/strategy/plugins/*.py`)를 분리. 라이브 서버에 학습 의존성이 필요하지 않도록 한다.
3. **단계별 검증**: 각 모델은 이전 단계 대비 성과 개선을 정량적으로 증명해야 한다. 개선 없으면 해당 레이어는 복잡도만 높인 것으로 간주.
4. **기준선 비교 필수**: 모든 모델은 Buy & Hold + example_macross 대비 성과를 비교한다.
5. **의존성 분리**: ML 전용 패키지는 `requirements-ml.txt`로 분리. 기존 `requirements.txt`는 변경하지 않는다.

---

## 2. 전체 구조

```
[Phase 0: 공통 인프라]
  features.py / label_generator / walk_forward / feature_pipeline / models.py
      │
      ├── Phase B-1a: LightGBM 전략 (ML — Gradient Boosted Trees)
      ├── Phase B-1b: XGBoost 전략 (ML — 동일 계열, 모델만 교체)
      ├── Phase B-2a: LSTM 전략 (DL — 순차 시퀀스 모델)
      ├── Phase B-2b: Transformer 전략 (DL — Attention 기반, 다른 아키텍처)
      ├── Phase B-3: PPO 전략 (RL — 강화학습)
      │
      └── Phase E: 통합 평가 — 5개 모델 + 2개 기준선 비교
```

### 2.1 모델 범주 비교

| 범주 | 모델 | 입력 형태 | 핵심 차이 |
|------|------|-----------|-----------|
| ML (B-1) | LightGBM, XGBoost | 단일 시점 피처 벡터 (~35개) | 정형 데이터, 비선형 관계 학습. 두 모델은 같은 GBDT 계열로 구조 동일, 내부 알고리즘만 다름 |
| DL (B-2) | LSTM, Transformer | 시퀀스 (lookback×features) | 시간적 패턴 학습. LSTM은 순차 처리, Transformer는 Attention으로 전체 동시 참조. 근본적으로 다른 아키텍처 |
| RL (B-3) | PPO | 시퀀스 + 포지션 상태 | 진입/청산/관망을 직접 결정. `should_force_exit` 훅으로 청산 타이밍도 모델이 판단 |

### 2.2 공통 설계

- **레이블**: 3-class 방향 분류 (SHORT=0, HOLD=1, LONG=2) — 미래 N봉 수익률 기반
- **피처**: `src/strategy/features.py`에서 `indicators.py`를 조합한 ~35개 피처 벡터
- **SL/TP**: ATR 기반 (ML/DL 모델과 독립). RL은 `should_force_exit` 훅으로 자체 청산
- **워크포워드 검증**: 시간 기반 롤링 윈도우 (train 6M → test 2M → 슬라이드)
- **타임프레임**: config에서 entry_tf, required_timeframes 지정 → 15m/1h/4h/멀티 모두 테스트 가능

---

## 3. 신규 파일 구조

```
src/
├── strategy/
│   ├── features.py                 # 공통 피처 엔지니어링 (indicators.py 위에 구축)
│   └── plugins/
│       ├── ml_lightgbm.py          # B-1a 전략 플러그인 (추론 전용)
│       ├── ml_xgboost.py           # B-1b 전략 플러그인
│       ├── dl_lstm.py              # B-2a 전략 플러그인
│       ├── dl_transformer.py       # B-2b 전략 플러그인
│       └── rl_ppo.py               # B-3 전략 플러그인
│
├── ml/                             # ML 공용 (학습 + 추론 양쪽에서 사용)
│   ├── __init__.py
│   ├── feature_pipeline.py         # 피처 캐싱 (parquet) — 학습 전용
│   ├── label_generator.py          # 3-class 방향 레이블 — 학습 전용
│   ├── walk_forward.py             # 시간 기반 롤링 분할 — 학습 전용
│   ├── models.py                   # DL 모델 클래스 (LSTMClassifier, TransformerClassifier) — 학습+추론
│   └── env_trading.py              # B-3용 Gym 환경 — 학습 전용

scripts/
├── train_lightgbm.py               # B-1a 학습
├── train_xgboost.py                # B-1b 학습
├── train_lstm.py                   # B-2a 학습
├── train_transformer.py            # B-2b 학습
├── train_ppo.py                    # B-3 학습
└── evaluate_models.py              # 통합 비교 리포트

models/                             # 학습된 모델 아티팩트 (.gitignore)
├── lightgbm/
├── xgboost/
├── lstm/
├── transformer/
└── ppo/

data/features/                      # 피처 캐시 (.gitignore)

config/
├── ml_lightgbm.yaml
├── ml_xgboost.yaml
├── dl_lstm.yaml
├── dl_transformer.yaml
└── rl_ppo.yaml

requirements-ml.txt                 # ML 전용 의존성
```

총 신규 파일: **약 19개** (인프라 6 + 플러그인 5 + 학습 5 + 평가 1 + config 5 + requirements 1)

> Buy & Hold 베이스라인은 `evaluate_models.py`에서 직접 계산하므로 별도 플러그인 불필요.

---

## 4. 컴포넌트 상세 설계

### 4.1 공통 피처 엔지니어링 (src/strategy/features.py)

기존 `indicators.py`의 함수를 조합하여 ~35개 피처 벡터를 생성. 학습/추론 양쪽에서 동일하게 호출.

**피처 목록:**

| 범주 | 피처 | 개수 |
|------|------|------|
| 추세 | price_ema10_ratio, price_ema50_ratio, ema10_ema50_ratio, ema20_ema200_ratio | 4 |
| 모멘텀 | macd, macd_signal, macd_hist, rsi_14, rsi_7 | 5 |
| 변동성 | atr_pct, bb_width, bb_position | 3 |
| 추세 강도 | adx, plus_di, minus_di, di_diff, choppiness, efficiency_ratio | 6 |
| 거래량 | volume_ratio | 1 |
| 수익률 | return_1, return_5, return_10, return_20, volatility_20 | 5 |
| 캔들 구조 | body_ratio, upper_shadow, lower_shadow | 3 |
| **단일 TF 합계** | | **27개** |

멀티TF 사용 시: 상위 TF 피처에 `{tf}_` 접두사 → forward-fill merge.
예) entry_tf=15m + [1h, 4h] → 27 + 27 + 27 = 81개 피처.

**핵심 함수:**
- `compute_features(df) → pd.DataFrame` — 단일 TF 피처
- `compute_multi_tf_features(candles, entry_tf) → pd.DataFrame` — 멀티TF 병합
- `get_feature_names(entry_tf, extra_tfs) → list[str]` — 피처명 목록

### 4.2 레이블 생성 (src/ml/label_generator.py)

```
future_return = close.pct_change(horizon).shift(-horizon) × 100
  > +threshold_pct  → LONG (2)
  < -threshold_pct  → SHORT (0)
  그 외             → HOLD (1)
```

- `horizon`: 미래 N봉 (config에서 지정, 기본 10)
- `threshold_pct`: ±0.3% (config에서 지정)
- 마지막 horizon개 행은 미래 데이터 없으므로 NaN → 학습 시 제거

### 4.3 워크포워드 분할 (src/ml/walk_forward.py)

```
|--- Train (6M) ---|-- Embargo --|--- Test (2M) ---|
                                 |--- Train (6M) ---|-- Embargo --|--- Test (2M) ---|
```

- train_months, test_months, step_months는 config에서 지정
- embargo_bars = horizon (레이블 누출 방지: train 끝 horizon개 행 제거)
- 출력: `list[WalkForwardFold]` (fold_id, train_start/end, test_start/end)

### 4.4 피처 캐싱 (src/ml/feature_pipeline.py)

- `build_features()` — HistoricalDataLoader로 캔들 로드 → 피처 계산 → parquet 저장
- 캐시 존재 시 바로 로드, `--force-features` 옵션으로 재생성

### 4.5 DL 모델 클래스 (src/ml/models.py)

**LSTMClassifier:**
```
Input(seq_len, n_features) → LSTM(hidden=64) → Dropout(0.3)
→ Dense(32, ReLU) → Dense(3)    # logits, softmax는 loss에서
```
파라미터 ~40K (피처 40개 기준). CPU 추론 밀리초 단위.

**TransformerClassifier:**
```
Input(seq_len, n_features) → Linear(n_features → d_model=64)
→ PositionalEncoding → TransformerEncoder(2 layers, 4 heads, ff=128)
→ GlobalAveragePooling → Dense(32, ReLU) → Dense(3)
```
파라미터 ~80K. CPU 추론 가능한 경량 설계.

### 4.6 Gym 환경 (src/ml/env_trading.py)

PPO 학습용 트레이딩 시뮬레이터. CoinBot의 `FeeModel`과 동일한 수수료 적용.

| 항목 | 설계 |
|------|------|
| Observation | (lookback, n_features) — 최근 lookback개 봉의 스케일링된 피처 |
| Action | Discrete(3) — 0=HOLD, 1=LONG, 2=SHORT |
| Reward | 스텝별 PnL(%) - 수수료. 과매매는 수수료로 자연 억제 |
| 에피���드 | 연속 2000봉, 랜덤 시작점, 종료 시 강제 청산 |
| 수수료 | `taker_fee_pct + slippage_pct` (config와 동일) |

### 4.7 RL 전략의 엔진 훅 매핑

기존 엔진은 `generate_signal`을 슬롯이 빌 때만 호출하므로, RL의 "포지션 유지/청산" 판단을 위해 `should_force_exit` 훅을 활용한다.

```
엔진 호출 흐름:
1. check_candle_sl_tp      → SL/TP 안전장치 (넓게 설정)
2. check_strategy_exits
   └→ should_force_exit    → RL 정책이 현재 방향과 다른 action 선택 시 청산
3. evaluate_strategies_on_bar
   └→ generate_signal      → 슬롯 빌 때: RL 정책의 action으로 진입 결정
```

- SL: 넓은 ATR 기반 (atr_sl_mult=5.0) — 안전장치 역할
- TP: 매우 넓게 (reward_risk_ratio=10.0) — 정책이 직접 청산

---

## 5. Config 구조

모든 ML 전략 config는 동일한 구조를 공유한다. 공통 섹션(exchange, engine, risk, accounting, paper, data, database, logging)은 `config/default.yaml`과 동일.

**전략 고유 섹션 예시 (ml_lightgbm):**

```yaml
strategies:
  active: ["ml_lightgbm"]

ml_lightgbm:
  # 엔진 필수
  risk_per_trade_pct: 0.01
  max_leverage: 5

  # 모델
  model_path: "models/lightgbm/latest"
  confidence_threshold: 0.55

  # 타임프레임
  entry_timeframe: "15m"
  required_timeframes: ["15m", "1h", "4h"]

  # SL/TP
  atr_period: 14
  atr_sl_mult: 2.0          # ML/DL: 2.0, RL: 5.0 (넓게)
  reward_risk_ratio: 2.0     # ML/DL: 2.0, RL: 10.0 (넓게)

  # 학습 전용 (train_*.py가 참조, 추론 시 무시)
  train:
    horizon: 10
    threshold_pct: 0.3
    train_months: 6
    test_months: 2
    step_months: 2
    # 모델별 하이퍼파라미터 (lgb_params / xgb_params / hidden_size 등)
```

---

## 6. 환경 분리

| 환경 | 설치 | 용도 |
|------|------|------|
| 데스크탑 (NVIDIA GPU) | requirements.txt + requirements-ml.txt 전체 | 학습 + 백테스트 |
| 노트북 (CPU only) | requirements.txt + 사용 모델 패키지만 | 추론 + 라이브 운영 |

- DL 모델: GPU 학습 → `torch.save(model.cpu().state_dict())` → 노트북에서 `torch.load(map_location="cpu")` 추론
- LightGBM/XGBoost: CPU 학습/추론 모두 가능 (GPU 불필요)

### 6.1 requirements-ml.txt

```
# ML
lightgbm>=4.0.0
xgboost>=2.0.0
scikit-learn>=1.4.0

# DL
torch>=2.2.0

# RL
stable-baselines3[extra]>=2.3.0
gymnasium>=0.29.0

# 공통
pyarrow>=15.0.0
joblib>=1.3.0
```

---

## 7. 통합 평가 (Phase E)

### 7.1 비교 대상 (7개)

| # | 전략 | 유형 |
|---|------|------|
| 1 | Buy & Hold | 기준선 — 직접 계산 (최종가/시초가) |
| 2 | example_macross | 기준선 — EMA 크로스 백테스트 |
| 3 | ml_lightgbm | ML |
| 4 | ml_xgboost | ML |
| 5 | dl_lstm | DL |
| 6 | dl_transformer | DL |
| 7 | rl_ppo | RL |

### 7.2 비교 지표

| 지표 | 의미 | 출처 |
|------|------|------|
| Total Return (%) | 총 수익률 | metrics.json |
| Max Drawdown (%) | 최대 낙폭 | metrics.json |
| Sharpe Ratio | 위험 대비 수익 효율 | equity_curve.csv에서 계산 |
| Calmar Ratio | 연수익률 / MDD | 계산 |
| Win Rate (%) | 승률 | metrics.json |
| Profit Factor | 총이익 / 총손실 | metrics.json |
| Total Trades | 거래 횟수 (과매매 감지) | metrics.json |

### 7.3 유의성 검정

일별 수익률 차이에 대한 부트스트랩 검정 (n=10,000). p < 0.05이면 통계적으로 유의한 차이.

### 7.4 같은 계열 내 비교

- LightGBM vs XGBoost — 동일 피처·레이블·기간에서 직접 대결
- LSTM vs Transformer — 동일 입력·레이블에서 직접 대결
- 최고 ML vs 최고 DL vs RL — 접근 방식 자체의 유효성 평가

### 7.5 ⚠️ 학습/백테 기간 설정 — Phase E 진입 시 재수립 필요

**배경 (2026-04-25 Phase B-1a 검증에서 확인된 문제):**
- §9의 검증 명령은 학습 기간(2020-01-01 ~ 2024-12-31)에 백테 기간(2024-01-01 ~ 2024-12-31)이 **완전 포함**됨 → 데이터 누출
- B-1a end-to-end 검증에서 실제로 비현실적 결과 관측 (1년 5,177% 수익, 70% 승률, MDD 2.97%, 직선에 가까운 equity curve)
- PnL 계산식·사이징·수수료는 모두 정상 검증됨. 결과 자체가 누출 때문에 부풀려짐을 확인
- 모든 모델(B-1a/B-1b/B-2a/B-2b/B-3)에 동일하게 적용될 문제

**Phase E 진입 시 결정 필요한 사항:**
1. **시간적 분리 원칙**: 학습 종료일 < 백테 시작일 (strict). 두 기간 사이에 embargo(label horizon × bar_size)를 둘지 여부
2. **OOS 윈도우 길이 및 개수**: 단일 vs 다중. 다중일 경우 시장 체제(약세/강세/횡보) 별로 분할할지
3. **학습 데이터 양 vs OOS 길이 트레이드오프**: 캔들 보유 범위(2020-01 ~ 2026-04)에서 어떻게 분배할지
4. **Buy & Hold / example_macross 베이스라인 기간**: 동일 OOS 기간으로 비교

**참고 후보안 (Phase E 시점에 재논의):**
- (a) 단일 OOS: Train 2020-01 ~ 2023-12, Test 2024-01 ~ 2026-04 (2.3년)
- (b) 다중 OOS (체제별): 2022 약세장 / 2023 회복장 / 2024 강세장 / 2025-2026 최신 — 각각 train→test 페어
- (c) Walk-forward OOS 통합 평가: 학습 시 fold별 모델을 모두 저장 → 각 fold의 test 구간을 모두 합쳐 백테

**구현 영향:**
- `evaluate_models.py`(미작성)가 새 기간 설계를 수용하도록 작성 필요
- 각 모델의 학습 스크립트는 인자만 다르게 호출 가능 (현재 구조 유지)
- (c) 안 채택 시 학습 스크립트가 fold별 모델을 별도 저장하도록 수정 필요

---

## 8. 단계별 작업 계획

### 구현 순서 및 의존성

```
Phase 0 (공통 인프라)
  ↓
Phase B-1a (LightGBM)  ←  첫 번째 모델, 가장 빠른 결과 확인
  ↓
Phase B-1b (XGBoost)   ←  B-1a 복사 수준, 모델 API만 교체
  ↓
Phase B-2a (LSTM)      ←  Phase 0 + models.py 의존
  ↓
Phase B-2b (Transformer) ← B-2a와 파이프라인 동일, 모델 클래스만 교체
  ↓
Phase B-3 (PPO)        ←  Phase 0 + env_trading.py (가장 복잡)
  ↓
Phase E (통합 평가)     ←  위 Phase 중 1개 이상 완료 시 실행 가능
```

| Phase | 단계 | 주요 산출물 | 복잡도 |
|-------|------|-------------|--------|
| 0 | 공통 인프라 | features.py, ml/{feature_pipeline, label_generator, walk_forward, models}.py, requirements-ml.txt, .gitignore | 중 |
| B-1a | LightGBM | scripts/train_lightgbm.py, plugins/ml_lightgbm.py, config/ml_lightgbm.yaml | 중 |
| B-1b | XGBoost | scripts/train_xgboost.py, plugins/ml_xgboost.py, config/ml_xgboost.yaml | 낮 (B-1a 복사) |
| B-2a | LSTM | scripts/train_lstm.py, plugins/dl_lstm.py, config/dl_lstm.yaml | 중~높 |
| B-2b | Transformer | scripts/train_transformer.py, plugins/dl_transformer.py, config/dl_transformer.yaml | 낮 (B-2a 복사) |
| B-3 | PPO | ml/env_trading.py, scripts/train_ppo.py, plugins/rl_ppo.py, config/rl_ppo.yaml | 높 |
| E | 통합 평가 | scripts/evaluate_models.py | 중 |

---

## 9. 검증 기준

### Phase별 검증

| Phase | 검증 항목 |
|-------|-----------|
| 0 | pytest — features.py (NaN 처리, 컬럼 수), label_generator (분포, NaN), walk_forward (fold 수, 겹침 없음) |
| B-1a | 학습 실행 → models/lightgbm/v001_*/ 생성. 백테스트 → metrics.json의 total_trades > 0 |
| B-1b | 동일 (XGBoost 버전) |
| B-2a | GPU 학습 → CPU 추론 정상. 백테스트 → total_trades > 0 |
| B-2b | 동일 (Transformer 버전) |
| B-3 | Gym env reset/step 단위 테스트. 백테스트에서 FORCE_EXIT 사유 거래 존재 |
| E | comparison_table.csv에 7행 존재. equity_overlay.png 생성 |

### 최종 검증

- 다중 기간 교차: 2020~2022 학습 → 2023 테스트, 2020~2023 학습 → 2024 테스트
- 모든 모델의 OOS 성능이 random baseline(정확도 ~33%)을 초과하는지 확인

### 데스크탑(GPU) 환경 이관 후 수행할 검증

노트북 환경에 ML 패키지(lightgbm, torch 등)가 미설치 상태이므로 아래 항목은 데스크탑 이관 후 수행한다.

```bash
# 1. ML 의존성 설치
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 2. skip된 테스트 전체 실행 (torch + lightgbm 필요)
python -m pytest tests/ -v --tb=short
#   → skip 0건이 되어야 정상

# 3. LightGBM 학습 스크립트 end-to-end 실행
#    (사전에 캔들 데이터 다운로드 필요: scripts/download_history.py)
python scripts/train_lightgbm.py --config config/ml_lightgbm.yaml \
    --start 2020-01-01 --end 2024-12-31
#   → models/lightgbm/v001_*/ 디렉토리 + model.txt 생성 확인

# 4. 학습된 모델로 백테스트 실행
python -m src.main backtest --config config/ml_lightgbm.yaml \
    --start 2024-01-01 --end 2024-12-31
#   → metrics.json의 total_trades > 0 확인
```

---

## 10. 수정 대상 기존 파일

| 파일 | 변경 내용 |
|------|-----------|
| `.gitignore` | `models/`, `data/features/` 추가 (2줄) |
| 엔진/코어 코드 | **변경 없음** |
| `requirements.txt` | **변경 없음** |

---

## 11. 설계 점검 결과 (2026-04-25)

구현 착수 전 전체 계획을 코드 수준으로 점검. 4건의 수정 필요 사항 발견, 구현 시 반영 예정.

| # | 심각도 | 문제 | 해결 방안 |
|---|--------|------|-----------|
| 1 | 심각 | PPO Gym 환경의 observation_space가 2D `(lookback, n_features)` 이나 SB3 MlpPolicy는 1D 벡터 기대 | env의 observation_space와 `_get_obs()`에서 flatten하여 `(lookback * n_features,)` 1D로 통일 |
| 2 | 중간 | evaluate_models.py가 차트 단계에서 `run_backtest`를 이중 실행 | 1단계에서 report_dir를 dict로 캐싱하여 재사용 |
| 3 | 경미 | train_lstm.py에서 `model.state_dict().copy()`는 얕은 복사 — 이후 학습으로 best state 오염 가능 | `copy.deepcopy(model.state_dict())`로 교체 |
| 4 | 경미 | compute_features docstring "약 35개"이나 실제 27개 | docstring을 "27개"로 수정 |

**구조적 결함 없음. 구현 착수 가능.**

---

## 12. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-B001 | Phase 0 | pandas_ta가 데이터 부족 시 None 반환 → indicators.py 내부에서 AttributeError crash | src/strategy/features.py | Phase 0 | 해결 — features.py에서 None 체크 + try-except 방어 |
| I-B002 | Phase 0 점검 | `reindex(method="ffill")` pandas 3.0+ deprecated → FutureWarning | src/strategy/features.py | Phase 0 | 해결 — `.reindex().ffill()` 체인으로 교체 |
| I-B003 | Phase 0 점검 | PositionalEncoding 홀수 d_model 시 sin/cos 슬라이스 shape 불일치 | src/ml/models.py | Phase 0 | 해결 — div_term을 각 슬라이스 크기에 맞게 조정 |
| I-B004 | Phase B-1a 후 데스크탑 환경 검증 | `pip install -r requirements-ml.txt` 시 ale-py 휠 빌드 실패 (Python 3.14 + Visual C++ 컴파일러 부재). `stable-baselines3[extra]`가 Atari 환경용 ale-py를 끌어오는데, 본 프로젝트는 자체 트레이딩 Gym 환경(env_trading.py)을 사용 예정이라 Atari 의존성 불필요 | requirements-ml.txt | 데스크탑 환경 검증 단계 | 해결 — `stable-baselines3[extra]` → `stable-baselines3`로 변경하여 불필요한 부가 의존성(ale-py, OpenCV, Pygame 등) 제거 |
| I-B005 | Phase B-1a end-to-end 검증 | `python scripts/train_lightgbm.py` 실행 시 `ModuleNotFoundError: No module named 'src'`. download_history.py에는 있는 `sys.path.insert(...)` 라인이 train_lightgbm.py에 누락되어 프로젝트 루트가 sys.path에 추가되지 않음 | scripts/train_lightgbm.py | 데스크탑 환경 검증 단계 | 해결 — download_history.py와 동일한 sys.path 추가 라인 삽입. 향후 train_*.py / evaluate_models.py 신규 스크립트도 동일 패턴 적용 필요 |
| I-B006 | Phase B-1a end-to-end 검증 | `HistoricalDataLoader.download_range_merged`가 `[start_ms, end_ms]` 범위 슬라이싱 없이 캐시 CSV 전체를 반환. CSV 캐시가 요청 범위보다 길면 호출자가 `--end`를 지정해도 무시됨. 학습 데이터가 의도한 5년치(~175k행) 대신 6.2년치(217,817행)로 늘어남. 백테스트 엔진(engine.py:300)도 동일 영향. | src/data/historical.py | 데스크탑 환경 검증 단계 | 해결 — `download_range_merged` 반환 직전에 `[start_ms, end_ms]` 슬라이싱 추가. 캐시 CSV는 전체 보존, 반환값만 자름 |

---

## 13. 진행 기록

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| 0 | 설계 문서화 | 완료 | — | 본 문서 작성 |
| 0-a | 설계 점검 | 완료 | — | 4건 수정 사항 발견, §11 기록 |
| 1 | Phase 0: 공통 인프라 | 완료 | `da136e6` | requirements-ml.txt, .gitignore, features.py, ml/{__init__, label_generator, walk_forward, feature_pipeline, models}.py 신규. I-B001/B002/B003 발견/해결. 테스트 33건 추가 (23 passed + 10 skipped/torch 미설치) — 전체 119 passed + 10 skipped, 회귀 없음 |
| 2 | Phase B-1a: LightGBM | 완료 | `9accbd4` | plugins/ml_lightgbm.py, scripts/train_lightgbm.py, config/ml_lightgbm.yaml, tests/test_ml_lightgbm.py 신규. 테스트 9건 추가 (8 passed + 1 skipped/lightgbm) — 전체 127 passed + 11 skipped, 회귀 없음. ML 패키지 미설치 skip 테스트는 데스크탑 이관 후 검증 예정 (§9 참조) |
| 2-V | 데스크탑 환경 ML 의존성/skip 테스트 검증 | 완료 | — | 데스크탑(Python 3.14.3)에 requirements-ml.txt 설치 — I-B004 발견/해결. `pytest tests/ -v` 결과 **138 passed, 0 skipped, 0 failed** (이전 127+11 → 138+0). torch 2.11.0(cp314) 정식 휠 사용. §9의 ML 의존성 설치/skip 테스트 항목 완료 |
| 2-E | Phase B-1a end-to-end 검증 (학습+백테) | 완료 | (커밋 대기) | I-B005(sys.path 누락) / I-B006(`download_range_merged` 범위 미슬라이스) 발견·해결. 회귀 테스트 138 passed 유지. **학습**: 5년치 172,025행 / 81피처 / 26 folds, OOS Acc 0.6377·F1(macro) 0.6345, 약 45초. **백테**(2024 in-sample, 누출 있음): 910거래, 승률 69.89%, total_return 5,177%, MDD 2.97%. PnL 계산식·사이징·수수료 검산 모두 정상. 결과의 비현실성은 데이터 누출 때문임을 코드 레벨에서 확정. **§9의 LightGBM end-to-end 항목(`total_trades > 0`) 충족**. 모델 진짜 성능은 §7.5에 따라 Phase E에서 재평가 |
| — | Phase B-1b: XGBoost | 대기 | — | B-1a 검증 결과 학습/백테 파이프라인 정상 확인됨. B-1a 복사 후 모델 API 교체 |
| — | Phase B-2a: LSTM | 대기 | — | |
| — | Phase B-2b: Transformer | 대기 | — | |
| — | Phase B-3: PPO | 대기 | — | |
| — | Phase E: 통합 평가 | 대기 | — | |
