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
| ML (B-1) | LightGBM, XGBoost | 단일 시점 피처 벡터 (27개 / 멀티TF 81개) | 정형 데이터, 비선형 관계 학습. 두 모델은 같은 GBDT 계열로 구조 동일, 내부 알고리즘만 다름 |
| DL (B-2) | LSTM, Transformer | 시퀀스 (lookback×features) | 시간적 패턴 학습. LSTM은 순차 처리, Transformer는 Attention으로 전체 동시 참조. 근본적으로 다른 아키텍처 |
| RL (B-3) | PPO | 시퀀스 + 포지션 상태 | 진입/청산/관망을 직접 결정. `should_force_exit` 훅으로 청산 타이밍도 모델이 판단 |

### 2.2 공통 설계

- **레이블**: 3-class 방향 분류 (SHORT=0, HOLD=1, LONG=2) — 미래 N봉 수익률 기반
- **피처**: `src/strategy/features.py`에서 `indicators.py`를 조합한 27개 피처 벡터 (멀티TF 사용 시 27 × N개)
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
│   ├── sequence_utils.py           # (B-2a) 시퀀스 변환 (N, F) → (N-L+1, L, F) — DL/RL 학습+추론 공통
│   ├── models.py                   # DL 모델 클래스 (LSTMClassifier, TransformerClassifier) — 학습+추론
│   ├── calibration.py              # (E-2-3) MulticlassCalibrator — Platt OvR + Isotonic, plugin 추론 시 자동 로드/적용
│   └── env_trading.py              # B-3용 Gym 환경 — 학습 전용

scripts/
├── train_lightgbm.py               # B-1a 학습
├── train_xgboost.py                # B-1b 학습
├── train_lstm.py                   # B-2a 학습
├── train_transformer.py            # B-2b 학습
├── train_ppo.py                    # B-3 학습
├── evaluate_models.py              # 통합 비교 리포트 (5 mode: full/single/collect/sensitivity/calibration)
└── calibrate_models.py             # (E-2-3) 4 분류 모델 walk-forward 재실행 + calibrator 학습/저장

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

총 신규 파일: **약 24개** (인프라 7 + 플러그인 5 + 학습 5 + 평가 1 + config 5 + requirements 1)

> Buy & Hold 베이스라인은 `evaluate_models.py`에서 직접 계산하므로 별도 플러그인 불필요.

---

## 4. 컴포넌트 상세 설계

### 4.1 공통 피처 엔지니어링 (src/strategy/features.py)

기존 `indicators.py`의 함수를 조합하여 27개 피처 벡터를 생성. 학습/추론 양쪽에서 동일하게 호출.

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

### 4.8 Confidence Calibration (Phase E-2-3 — I-B009 해결 시도)

분류 모델(ml_lightgbm/ml_xgboost/dl_lstm/dl_transformer)의 raw probability가 정답 확률과 mismatch될 때 confidence_threshold(0.55) 필터링이 부정확해지는 문제 해결용. PPO는 정책 모델이라 적용 X.

**컴포넌트**:
- `src/ml/calibration.py::MulticlassCalibrator` — Platt scaling (OvR LogisticRegression) + Isotonic regression. 3-class softmax-normalize.
- `scripts/calibrate_models.py` — 학습 데이터로 walk-forward 26 folds 재실행 → 모든 fold OOS probabilities 수집 → MulticlassCalibrator 학습 → `calibrator_<method>.joblib` + `calibration_meta.json` 모델 디렉토리 저장. 모델(model.txt/model.pth)은 변경 0.

**plugin 통합**:
- config의 `calibration_method`: `"none"` (기본) / `"platt"` / `"isotonic"`
- plugin `_ensure_model`이 method≠"none"이면 `model_dir/calibrator_<method>.joblib` 자동 로드
- `generate_signal`에서 `raw_probs = model.predict()` → `calibrated = calibrator.transform(raw_probs)` → argmax + threshold

**Phase E-2-3 검증 결과 (분할 1, 2025 OOS)**:
- Isotonic이 4 모델 모두 개선 (Transformer +7.9% 가장 큼, ml_lightgbm +0.4%, ml_xgboost +2.5%, dl_lstm +4.3%)
- 모든 모델 PF/win_rate 동시 상승
- Platt 효과 미미 (~0%) — 비모수 Isotonic이 multi-class OvR에 더 적합
- 단 분류 모델 순위 변동 없음 (ml_lightgbm 1위 유지)

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
  calibration_method: "none"  # "none"/"platt"/"isotonic" (E-2-3 §4.8 — 분류 모델만)

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
5. **슬리피지 sensitivity 검증** (B-3 완료 후 추가 — I-B010): PPO가 5,390거래로 분류 모델의 6배 빈도. 슬리피지 0%/0.02%/0.05%/0.1% 변동 시 모든 모델의 수익률 변화 측정. PPO의 백테 우위가 라이브에서 유지되는지 검증
6. **Confidence calibration 검증** (B-2b 완료 후 추가 — I-B009): Platt scaling 또는 isotonic regression을 ML/DL 모델에 적용 후 재백테. Transformer의 OOS Acc 1위와 백테 최저 모순 해소 시도
7. **Lookahead bias 코드 추적** (B-1b 완료 후 추가 — I-B007): 백테 엔진의 generate_signal 호출 시점 + entry_price 결정 시점 정밀 검증. 모든 백테 결과의 신뢰성 기반 — **2026-04-27 Phase E-2-1에서 검증 완료**: 두 가지 lookahead 발견·수정. 영향 측정 결과 5개 모델 모두 ±5% 변화 — 학습/추론 분포 일치(features.iloc[i]가 시점 i의 close 포함하는 학습 패턴)로 인해 lookahead가 큰 영향이 아니었음. 기존 백테 결과 신뢰성 양호 확인

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

**기간 설정 원칙 (2026-04-25 B-1a 검증 후 추가):**
- **백테 기간은 학습 기간과 겹치지 않게 설정.** Phase별 동작 검증이라도 데이터 누출은 결과 해석을 어렵게 만듦
- 캔들 데이터 가용 범위: **2020-01-01 ~ 2026-04 (15m/4h)**, **~2026-04-10 (1h)**
- 권장 분할: 학습 2020-01-01 ~ 2024-12-31 / 백테 2025-01-01 ~ 2025-12-31 (1년 OOS)
- 모델 간 비교 일관성을 위해 모든 Phase B-* 백테는 동일 OOS 기간 사용 권장
- 진짜 성능 평가는 §7.5에 따라 Phase E에서 별도 수행

```bash
# 1. ML 의존성 설치
pip install -r requirements.txt
pip install -r requirements-ml.txt

# 1-a. (데스크탑 GPU 학습용) torch CUDA 빌드 설치 — I-B008 참조
#      pip 기본 인덱스에서 받은 torch는 CPU-only 빌드. DL 학습(LSTM/Transformer/PPO) 시 GPU 활용 위해 CUDA 빌드 필요.
#      cp314 + cu126 휠 정식 존재 확인됨 (Python 3.14 + CUDA 12.6).
pip uninstall torch -y
pip install --index-url https://download.pytorch.org/whl/cu126 torch
#   → 검증: python -c "import torch; print(torch.cuda.is_available())"  # True

# 2. skip된 테스트 전체 실행 (torch + lightgbm 필요)
python -m pytest tests/ -v --tb=short
#   → skip 0건이 되어야 정상

# 3. LightGBM 학습 스크립트 end-to-end 실행
#    (사전에 캔들 데이터 다운로드 필요: scripts/download_history.py)
python scripts/train_lightgbm.py --config config/ml_lightgbm.yaml \
    --start 2020-01-01 --end 2024-12-31
#   → models/lightgbm/v001_*/ 디렉토리 + model.txt 생성 확인

# 4. 학습된 모델로 백테스트 실행 (학습 기간과 분리된 OOS 구간)
python -m src.main backtest --config config/ml_lightgbm.yaml \
    --start 2025-01-01 --end 2025-12-31
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
| I-B007 | Phase B-1b end-to-end 검증 | 추론 시 `features.iloc[-1]`(가장 최근 봉)의 close가 피처로 들어가는지 미검증. 백테 엔진이 봉 진행 중에 `generate_signal`을 호출하고 같은 봉 close 가격으로 진입한다면 lookahead bias 발생 가능. B-1b 깨끗 OOS(2025)에서도 1년 4,411% 수익으로 비현실적으로 높은 결과가 lookahead 의심 사유. 데이터 누출과 별개의 문제 — 누출이 해소돼도 lookahead가 있으면 모든 백테 결과 부풀려짐 | src/strategy/plugins/ml_*.py 추론 시점 + src/backtest/engine.py `evaluate_strategies_on_bar` 호출 흐름 + entry_price 결정 시점 | Phase E-2-1 검증 단계 | **해결 (2026-04-27 Phase E-2-1)**: 코드 추적으로 두 가지 lookahead 확인 — ①`_slice_candles(ts)`가 `df.loc[:ts]`로 ts 봉(현재 close 포함)을 슬라이스에 포함, ②`current_price = close`로 봉 시작 시점에 봉 종가로 진입. 수정: `_slice_candles`를 `df[df.index < ts]`로 변경(ts 미만 슬라이스) + `current_price`를 `candle["open"]`(봉 시작 가격)으로 변경. **영향 측정** (B-1b·B-3 분할 1 백테): 모든 모델에서 결과 변화 ±5% 수준으로 작음 — 학습/추론 분포 일치 효과(학습 시 `features.iloc[i]`도 시점 i의 close 포함하는 패턴이라 lookahead가 있던 상태가 학습과 같은 분포였음). **결론**: 기존 백테 결과가 lookahead로 크게 부풀려지지 않음 — 백테 신뢰성 양호. 다만 라이브 재현성은 수정 후가 정확 |
| I-B011 | Phase E-2-1 lookahead 수정 후 백테 | I-B007 수정으로 첫 봉(ts0)에서 `candles_slice`가 빈 DataFrame이 되면서 features.py의 `compute_choppiness` 호출이 `'NoneType' object has no attribute 'rolling'` AttributeError. 원인: I-B001 해결 시 `compute_macd`/`compute_bbands`/`compute_adx` 호출은 try-except로 감쌌으나 `compute_choppiness`/`compute_efficiency_ratio` 호출은 누락 | src/strategy/features.py:156-159 | Phase E-2-1 검증 단계 | 해결 — features.py에 `compute_choppiness`/`compute_efficiency_ratio` 호출 try-except 추가. I-B001 해결 패턴 일관성 회복 |
| I-B008 | Phase B-2a end-to-end 검증 | `pip install -r requirements-ml.txt` 시 PyPI 기본 인덱스에서 받은 torch가 **CPU-only 빌드**(`torch-2.11.0+cpu`). `torch.cuda.is_available() == False` → LSTM 학습이 GPU(RTX 2060) 있는데도 CPU로 동작. 데스크탑 학습 환경에서 학습 시간 수십 배 늘어나는 문제. requirements-ml.txt에 단순 `torch>=2.2.0`만 명시되어 있어 빌드 종류 보장 안 됨 (CPU 빌드는 라이브 노트북 환경엔 적합) | requirements-ml.txt + 데스크탑 환경 설정 절차 | B-2a end-to-end 검증 단계 | 우회 적용 — 데스크탑에서 `pip uninstall torch -y && pip install --index-url https://download.pytorch.org/whl/cu126 torch`로 CUDA 12.6 빌드 수동 설치. cp314+cu126 휠 정식 존재 확인됨. requirements-ml.txt 자체는 CPU 기본값 유지(라이브 환경 호환), §9에 데스크탑 GPU 빌드 설치 가이드 추가 |
| I-B009 | Phase B-2b end-to-end 검증 | Transformer가 학습 OOS Acc는 4개 모델 중 최고(0.6569)인데 백테 결과는 최저(승률 57.98%, 수익 3,737%). 가설: confidence_threshold(0.55) 필터링 후 신호의 calibration이 잘못됨. 거래 수가 가장 많은데(959 vs B-1b 878) 그 중 정답 비율은 가장 낮음. 즉 모델이 자신감 표현(probability)이 정확도와 잘 매핑되지 않음 | src/strategy/plugins/dl_*.py 추론 + 학습 파이프라인의 calibration 단계 부재 | Phase E 또는 별도 검증 단계 | **부분 해결 (Phase E-2-3 Step 2/3)** — Platt scaling + Isotonic regression 양쪽 인프라 구축 (`src/ml/calibration.py` MulticlassCalibrator, `scripts/calibrate_models.py` walk-forward 26 folds 재실행 후 calibrator 학습/저장, plugin 4개에 자동 로드/적용 분기). v001 4 분류 모델에 calibrator 학습 (~150k OOS samples). 분할 1 백테 결과 (raw vs Platt vs Isotonic): **Isotonic이 4 모델 모두 개선** (ml_lightgbm 5,088→5,108%, ml_xgboost 4,910→5,032%, dl_lstm 4,663→4,865%, **dl_transformer 3,939→4,251% (+7.9%)**). 모든 모델 PF/win_rate 동시 상승. **Platt는 효과 미미** (~0%). I-B009 가설 (Transformer calibration 부정확)이 정확히 확인됨 — 단, 분류 모델 순위는 변동 없음 (ml_lightgbm 여전히 1위). 향후: train_*.py에 calibrator 자동 학습 옵션 추가 + BP-3 ensemble에 calibration 패턴 적용 |
| I-B010 | Phase B-3 end-to-end 검증 | PPO 백테가 5,390거래(분류 모델 ~900의 6배 빈도)로 매우 고빈도. 백테는 슬리피지 0% 가정인데 라이브에서는 슬리피지 발생 — 거래 빈도 6배라 수수료/슬리피지 영향 6배 증폭. Profit Factor 1.39로 안전 마진 좁음(분류 모델 PF 2~3 대비). 라이브 슬리피지 0.05% 추가만 되어도 거래당 비용 50% 증가 → 백테 우위(5,862% 수익) 라이브에서 유지 못할 가능성. 분류 모델은 거래 빈도 낮아 슬리피지 영향 제한적, PPO만 선택적으로 영향 | config/*.yaml의 accounting.slippage_pct (현재 0.0) + 백테 엔진의 슬리피지 모델 + 라이브 broker 실제 체결 가격 | Phase E 슬리피지 sensitivity 검증 단계 | **해결 (Phase E-2-3 Step 1)** — 5 모델 × 분할{1, Exp4} × 슬리피지{0%, 0.02%, 0.05%, 0.1%} = 40 백테 (eval_260503_sensitivity, wall time 6.78h). I-B012 fix 후 정확한 측정 결과: **PPO는 slip 0.02%만 추가돼도 1년 OOS break-even (205%→-5%), slip 0.05%에서 4년 OOS break-even (9,117%→69%) — 라이브 적용 불가**. 분류 모델은 slip 0.10%에서도 9,000~11,000% 유지 (4년 OOS) — **라이브 robust**. 라이브 슬리피지 0.05% 가정 시 분류 모델만 채택 가능. PPO는 confidence calibration 무관 (정책 모델) — Step 2/3에서도 제외 |
| I-B012 | Phase 0 (발생) / Phase E-2-3 Step 1 (발견) | `src/execution/paper_executor.py:_close_internal`이 round-trip fees(taker_fee + slippage)를 차감하지 않고 broker.balance에 gross_pnl만 반영. trades.csv는 net_pnl 정확 기록(엔진의 별도 FeeModel.calc_pnl 호출). 그러나 equity_curve.csv는 broker.get_balance() 기반이라 gross 누적 → metrics.json (total_pnl/total_return_pct/max_drawdown_pct) 부풀림. profit_factor/win_rate/total_trades/avg_win/avg_loss는 trades.csv 기반이라 정확. 라이브-백테 일관성(CLAUDE.md 정체성) 위반 — 라이브는 거래소가 fees 자동 차감하지만 paper는 자체 처리 누락. 모든 기존 백테 결과 영향 (Phase 0/B-1a/B-1b/B-2a/B-2b/B-3/E-1/E-2-2/E-2-2-OPT). | src/execution/paper_executor.py | Phase E-2-2-FIX | **해결** — ①PaperExecutor.__init__에 FeeModel.from_config(config) 주입, ②_close_internal에서 fees=fee_model.estimate_round_trip 차감 후 net_pnl을 balance에 반영. ③회귀 테스트 6건 신규 (tests/test_backtest_fees.py — invariant 1: initial+sum(trades.pnl)==equity_curve_final / invariant 2: sum(trades.pnl)==metrics.total_pnl / invariant 3: fees 증가 시 final 단조 감소). ④기존 85개 백테 폴더에 trades.csv 기반 재계산으로 metrics_corrected.json/equity_curve_corrected.csv/equity_curve_corrected.png 생성 (1회용 _recompute_metrics.py, 기존 파일 보존). ⑤CLAUDE.md 협업 규칙 10 신규 — 백테 결과 정합성 검증 우선 (trades↔metrics↔equity 일치성). pytest 207 passed (201+6). **부풀림 정량**: 분류 모델 1년 OOS ~900-1000%p, 4년 OOS ~2500%p; PPO 1년 ~4000%p, 4년 ~12000%p. 모델 순위 변동: Exp4 4년 OOS에서 PPO가 1위→5위로 하락 (21,403%→9,477%) |

---

## 13. 진행 기록

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| 0 | 설계 문서화 | 완료 | — | 본 문서 작성 |
| 0-a | 설계 점검 | 완료 | — | 4건 수정 사항 발견, §11 기록 |
| 1 | Phase 0: 공통 인프라 | 완료 | `da136e6` | requirements-ml.txt, .gitignore, features.py, ml/{__init__, label_generator, walk_forward, feature_pipeline, models}.py 신규. I-B001/B002/B003 발견/해결. 테스트 33건 추가 (23 passed + 10 skipped/torch 미설치) — 전체 119 passed + 10 skipped, 회귀 없음 |
| 2 | Phase B-1a: LightGBM | 완료 | `9accbd4` | plugins/ml_lightgbm.py, scripts/train_lightgbm.py, config/ml_lightgbm.yaml, tests/test_ml_lightgbm.py 신규. 테스트 9건 추가 (8 passed + 1 skipped/lightgbm) — 전체 127 passed + 11 skipped, 회귀 없음. ML 패키지 미설치 skip 테스트는 데스크탑 이관 후 검증 예정 (§9 참조) |
| 2-V | 데스크탑 환경 ML 의존성/skip 테스트 검증 | 완료 | — | 데스크탑(Python 3.14.3)에 requirements-ml.txt 설치 — I-B004 발견/해결. `pytest tests/ -v` 결과 **138 passed, 0 skipped, 0 failed** (이전 127+11 → 138+0). torch 2.11.0(cp314) 정식 휠 사용. §9의 ML 의존성 설치/skip 테스트 항목 완료 |
| 2-E | Phase B-1a end-to-end 검증 (학습+백테) | 완료 | `cddb875` | I-B005(sys.path 누락) / I-B006(`download_range_merged` 범위 미슬라이스) 발견·해결. 회귀 테스트 138 passed 유지. **학습**: 5년치 172,025행 / 81피처 / 26 folds, OOS Acc 0.6377·F1(macro) 0.6345, 약 45초. **백테**(2024 in-sample, 누출 있음): 910거래, 승률 69.89%, total_return 5,177%, MDD 2.97%. PnL 계산식·사이징·수수료 검산 모두 정상. 결과의 비현실성은 데이터 누출 때문임을 코드 레벨에서 확정. **§9의 LightGBM end-to-end 항목(`total_trades > 0`) 충족**. 모델 진짜 성능은 §7.5에 따라 Phase E에서 재평가 |
| 3 | Phase B-1b: XGBoost | 완료 | `f745ef4` | plugins/ml_xgboost.py, scripts/train_xgboost.py, config/ml_xgboost.yaml, tests/test_ml_xgboost.py 신규. B-1a 구조 그대로 + XGBoost API 교체 (objective=multi:softprob, DMatrix wrapping, model.json 저장). I-B005 교훈 반영(sys.path.insert 포함). 테스트 9건 추가 → pytest 147 passed. **학습** (2020-01 ~ 2024-12, 26 folds): OOS Acc 0.6364·F1 0.6334 (B-1a 0.6377/0.6345와 사실상 동률 — 같은 GBDT 계열, 피처 정보량 한계). 학습 시간 ~150초. **백테 (2025-01 ~ 2025-12 깨끗 OOS — §9 수정안 적용 첫 케이스)**: 878거래, 승률 **64.46%(=OOS Acc)**, MDD 4.9%, 1년 수익 4,411%, Profit Factor 2.73. equity curve가 1월 무거래·11월 BTC 급락 시 조정 등 **시장 반응 보이는 자연스러운 패턴** — B-1a 누출 백테(직선 우상향)와 질적으로 다름. PnL/사이징/수수료 검산 정상. **§9 검증(`total_trades > 0`) 충족**. Lookahead bias 가능성은 I-B007로 별도 등록 (Phase E에서 검증) |
| 3-T | 테스트 구조 리팩토링 (Phase 0 test_ml_infra.py 분리) | 완료 | `374af16` | B-2a 진입 직전 테스트 구조 일관성 정비. 변경: `tests/test_ml_infra.py` 삭제 → `test_features.py`, `test_label_generator.py`, `test_walk_forward.py`, `test_lstm.py`, `test_transformer.py` 5개로 분리. 기존 테스트 코드 그대로 이동(새 테스트 작성 아님). `_make_candles` 헬퍼는 필요 파일에 로컬 복사 (test_ml_lightgbm/xgboost와 동일 패턴). pytest **147 passed 유지** (회귀 없음). 결과: `src/ml/` 하위·`src/strategy/plugins/` 하위 모두 모듈별 1:1 패턴으로 일관성 회복. LSTM 모델 정의 테스트도 더 이상 `test_ml_infra.py` 안에 흩어지지 않고 `test_lstm.py` 단독 |
| 4 | Phase B-2a: LSTM | 완료 | `26279d2` | **결정사항**: lookback=60 / scaler=전체 train 1회 fit / epochs=50 + patience=5 / class weighting 미적용. **신규 6개**: `src/ml/sequence_utils.py`, `scripts/train_lstm.py`, `src/strategy/plugins/dl_lstm.py`, `config/dl_lstm.yaml`, `tests/test_sequence_utils.py`, `tests/test_dl_lstm.py`. I-B003 교훈(`copy.deepcopy(state_dict)`), I-B005 교훈(sys.path.insert) 반영. 테스트 21건 추가 → pytest **168 passed** (회귀 없음). **end-to-end 검증** (GPU torch 빌드 재설치 후 — I-B008 참조): 학습 26 folds, OOS Acc 0.6412 / F1 0.6392, 약 2분(GPU). 백테(2025 깨끗 OOS): 923거래, 승률 62.41%, MDD 4.9%, 1년 수익 4,418%. **B-1b(XGBoost) 대비 사실상 동률** (4,411% vs 4,418%, 승률 64.46% vs 62.41% — LSTM이 거래 빈도↑·승률↓로 결과적 동률). **Patience 진단 (patience=15 재학습)**: 모든 fold best_epoch이 2~4 범위 — patience=5가 충분했음 확정 (학습 부족 아님). **LSTM 한계 분석**: 멀티TF 피처(15m/1h/4h)가 이미 시퀀스 정보 일부 흡수 + label boundary 잡음 + 81피처 OHLCV 정보량 한계. lookback/hidden 변경으로는 큰 변화 어려움 — 진짜 개선은 추가 데이터(펀딩률/OI 등) 또는 라벨 개선 필요. §1.3 원칙상 단독 가치는 낮으나, **sequence_utils 인프라가 B-2b/B-3에서 재사용** + Phase E ensemble 후보 가치. v002(patience=15 진단 모델) 및 임시 config 삭제 완료, latest.json v001 복원 |
| 5 | Phase B-2b: Transformer | 완료 | `6b8f34a` | plugins/dl_transformer.py, scripts/train_transformer.py, config/dl_transformer.yaml, tests/test_dl_transformer.py 신규. B-2a 구조 + TransformerClassifier 교체. `transformer_params`(d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.3 — Phase 0 기본값). 학습 하이퍼파라미터는 B-2a와 동일(공정 비교). 테스트 11건 추가 → pytest 179 passed. **end-to-end 검증**: 학습 OOS Acc 0.6569·F1 0.6542 — **4개 모델 중 최고** (B-2a 0.6412 대비 +1.6%p, GBDT 대비 +1.9%p). 백테(2025 OOS): 959거래(가장 많음), 승률 57.98%, MDD 4.96%, 1년 수익 3,737%, Profit Factor 2.04 — **백테 기준 4개 모델 중 최저**. **⚠️ 모순적 발견: 학습 OOS 1위와 백테 1위가 정반대**. 가설: ①confidence calibration 문제(자신감 0.55 통과 신호 중 정답 비율이 GBDT 대비 낮음 — 거래 수 ↑·승률 ↓), ②더 복잡한 모델이 시장 분포 변화에 더 민감 (2025 분포가 학습 시기와 약간 다를 때 LSTM/GBDT보다 더 흔들림), ③walk-forward 평균(2020-2024) vs 단일 기간(2025) 측정 차이. §1.3 원칙상 단독 가치 낮으나 Phase E 앙상블에서 다양성 기여 가능. **핵심 시사점**: OOS Acc만으로 모델 선택 X — 백테 결과까지 같이 봐야 함. I-B009로 calibration 이슈 등록 |
| 6 | Phase B-3: PPO | 완료 | `0431800` | **결정사항**: episode_length=2000 / total_timesteps=200,000 / fee_pct=0.0005 / PPO SB3 기본값 / SL atr_mult=5.0·RR=10.0. **신규 6개**: `src/ml/env_trading.py`(Gym 환경), `scripts/train_ppo.py`, `src/strategy/plugins/rl_ppo.py`, `config/rl_ppo.yaml`, `tests/test_env_trading.py`, `tests/test_rl_ppo.py`. §11-#1 적용(1D flatten obs), `should_force_exit` hook 활용, mark-to-market reward. 테스트 22건 추가 → pytest **201 passed** (회귀 없음). **end-to-end 검증** (학습 ~10분, GPU): 모델 저장됨. 백테(2025 OOS): **5,390거래** (분류 모델 ~900의 6배), 승률 **45.57%**, MDD **3.65%**(가장 낮음), 1년 수익 **5,862%**(가장 높음), Profit Factor 1.39. **§9 검증 충족**: by_exit_reason의 force_exit 5,346건(99.2%) — RL 직접 청산 작동 증거. **거래 스타일이 분류 모델과 근본적으로 다름**: "큰 움직임 가끔 잡기" → "작은 움직임 자주 잡기"(평균 보유 6.5봉, 1.5시간). avg_win/avg_loss 1.66 — RR과 force_exit으로 낮은 승률 보상. equity curve가 가장 매끄러운 우상향 (11월 BTC 급락 시 영향 미미). **⚠️ 백테 vs 라이브 격차 위험**: 슬리피지 0% 가정. 라이브 슬리피지 0.05% 추가만 되어도 거래당 비용 50% 증가 — Profit Factor 1.39 안전 마진 좁음. 분류 모델(PF 2~3)은 거래 빈도 낮아 슬리피지 영향 제한적. I-B010으로 등록, Phase E 슬리피지 sensitivity 검증 필수 |
| 7 | Phase E-1: 다중 OOS 분할 학습 (Anchored + Expanding) | 완료 | `0cff54d` | **결정**: §7.5-#2를 (i) Anchored-window + (iii) Expanding OOS 결합으로 진행. 분할 통합 후 신규 학습 3세트 + 5년 정상 1세트(LightGBM만 — v002 정리 후 재학습) = 4분할 × 5개 모델 = **16회 학습** (~62분, GPU). **모델 디렉토리 구조 통일**: 5개 모델 모두 v001(5년)/v002(2년)/v003(3년)/v004(4년)로 일관. 학습 후 5개 latest.json 모두 v001로 복원 (라이브 호환성). 회귀 테스트 **201 passed 유지**. **학습 OOS Acc 결과 (PPO는 학습 OOS 측정 없음)**: 모델별 데이터 양 효과 명확 (2년 → 5년으로 갈수록 상승). 가장 큰 도약은 **3년→4년 사이**, 4년→5년은 거의 포화. **🔄 예상과 반대 발견**: DL 모델(LSTM/Transformer)이 작은 데이터(2년)에서 GBDT 대비 +2%p 더 좋음 — B-2a 한계 분석에서 추측한 "DL이 데이터 더 모으면 빛난다"는 **틀렸음**. 오히려 DL의 inductive bias가 데이터 부족 환경에서 더 강함. Transformer가 모든 학습 길이에서 일관된 1위 (2년 0.6263 / 3년 0.6262 / 4년 0.6565 / 5년 0.6569). 분할 3 (3년 학습)이 분할 2 (2년)보다 일부 모델에서 약함 — 2020~2022 학습 데이터의 시장 환경 특이성 또는 fold 14개 통계 노이즈 가능 |
| 8 | Phase E-2-1: Lookahead 수정 + evaluate_models.py 골격 | 완료 | `39d6422` | **결정**: (i) Anchored + (iii) Expanding 결합 → 6개 백테 분할 (모델 4개: v001 5년 / v002 2년 / v003 3년 / v004 4년 — 모델당 1~2개 OOS 백테). 다중 OOS 분할 매트릭스: 분할 1 (v001, OOS 2025-01~2025-12), Anchored A (v003, OOS 2023), Anchored B (v004, OOS 2024), Expanding 2 (v004, OOS 2024~2025), Expanding 3 (v003, OOS 2023~2025), Expanding 4 (v002, OOS 2022~2025). **I-B007 해결**: backtest engine `_slice_candles` ts 미만 + `current_price` open. 영향 ±5% (작음). **I-B011 해결**: features.py try-except 누락 추가. 회귀 201 passed 유지. **evaluate_models.py 골격 작성**: BacktestEngine 직접 호출 (subprocess 없음), config dict의 model_path 오버라이드, 결과를 별도 디렉토리(`data/backtest_reports/00_Working/eval_{YYMMDD}/`)에 저장 |
| 9 | Phase E-2-2: 분할 백테 + 베이스라인 | 완료 | (커밋 대기) | **42개 백테 완료** = 30 모델 (5 모델 × 6 분할) + 6 example_macross + 6 Buy & Hold. 결과: `data/backtest_reports/00_Working/eval_260427/comparison.csv`. **소요 시간 약 78시간** (예상 6시간의 13배 — E-2-2-OPT 필요성 확정). evaluate_models.py에 베이스라인 처리 추가 (compute_buy_and_hold + run_baseline_macross + 임시 config 자동 생성/삭제, +233줄). 회귀 테스트 **pytest 201 passed 유지**. **§9 검증 기준 충족**: `total_trades > 0` 42/42 모두 충족. **6대 핵심 발견**: ① **모든 ML 모델이 B&H + macross 압도** — 분할 1(2025) B&H -5.68%(약세장!) vs ML 5,000%+. example_macross는 거의 0% (룰 기반 무력). ML alpha 진짜 학습됨 명확한 증거. ② **PPO가 4년 OOS(Exp4)에서 21,403% 1위** + MDD 3.44% 최저. 단 슬리피지 미검증 (I-B010). ③ **Transformer 학습 OOS 1위 → 백테 최저 패턴 6/6 분할 모두 일관** (LightGBM 대비 -16~-29%). I-B009 calibration 문제 확정. ④ **데이터 양 효과**: 3년 학습→4년 학습 ~3x 도약, 4년→5년 미미 (Phase E-1 OOS Acc 패턴과 정확히 일치). ⑤ **모델 안정성 순위**: LightGBM(PF 3.21 평균) > XGBoost > LSTM > Transformer > PPO(MDD 1위 but PF 1.58). ⑥ **2025년이 약세장(B&H -5.68%)** — 새 발견. ML 모델이 강세장 의존 X, regime robust. **여전히 의심**: 슬리피지 0% 가정 + calibration 미적용 + lookahead 추가 검증 — Phase E-2-3 슬리피지 sensitivity가 라이브 적용 가능성 핵심 |
| 10 | Phase E-2-2-OPT: 백테 엔진 성능 최적화 | 완료 | (커밋 대기) | **종합**: 30 specs 78h → **5.12h (~15.2x)** 달성 — 목표 13-24x 범위 안. 결과 일치성: comparison.csv 42행 정확 일치 + sha256 sample 5/5 일치 (bit-perfect). pytest 201 passed 유지. **Step 0 (Profiling 1회용)**: `data/backtest_reports/00_Working/profiling/_profile_lightgbm_split1.py` (.gitignore 자동 제외). cProfile 측정: ml_lightgbm 분할 1 wall time 3454초, **compute_multi_tf_features cumtime 3251.7초 = 94.1%** — 핵심 bottleneck 확정 (3 TF × 11,473 호출 = 34,419회 처음부터 재계산). **Step 1 (features 사전계산)**: BacktestEngine.initialize에서 활성 entry_tf별 OOS 전체 features 1회 계산 → cache. helper 함수 `get_features_for_ctx(ctx, entry_tf)` 도입(features.py)으로 cutoff(`index < ctx.now`) 단일 출처화 — 5 plugin에는 1줄 교체만 (DRY). 라이브 모드는 `precomputed_features=None` → 즉시 계산 fallback (캡슐화). 변경: types.py(+4) + features.py(+17) + engine_base.py(+5) + backtest/engine.py(+15) + 5 plugin(각 ±2). 단독 효과: ml_lightgbm 분할 1 단일 백테 3454→227초 (~15x, cProfile 포함 baseline 기준). **Step 2 (모델 캐시) — 보류** (사안 C): split마다 model_dir 다름 → cache hit 0 + multiprocessing 워커당 1회 load라 효과 없음. 1.5h 절약. **Step 3 (multiprocessing)**: scripts/evaluate_models.py(+85)에 `Pool(N=4).imap_unordered(chunksize=1)` 도입. PyTorch thread thrashing 방지(`OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` import 전 설정), logger PID prefix, `_init_worker`로 eval_root 주입(BacktestSpec 미수정), 캔들 캐시 사전 워밍업(`_warmup_candle_cache` — 사안 F, 4 워커 race condition 방지). 베이스라인(B&H 6 + macross 6)은 main process 순차. 결과 출력 폴더 `eval_260502/` 분리(사안 G — baseline `eval_260427/` 보존). 단독 효과: ~2x (이론 4x). 효과 작은 이유: OOS 길이 차이 큼(1년~4년) — 마지막 4년 PPO/Exp4 끝날 때까지 다른 워커 일부 idle. **검증 결과 (사안 H)**: trades.csv sha256 5/5 정확 일치(ml_lightgbm/ml_xgboost/dl_lstm/dl_transformer/rl_ppo split 1) + comparison.csv 42행 diff 빈 출력 + 30 specs trades/return 불일치 0. multiprocessing 결정성 + lookahead 안전성 동시 검증. **신규 협업 규칙 8/9 도입 (CLAUDE.md)**: ⑧ 구조 점검 단계 — DRY/캡슐화/미래 확장성/1회용 코드 분리 자체 점검. ⑨ Phase 내부 Step 임시 보존 — 메모리 파일에 즉시 기록, Phase 종착 커밋 직후 삭제. **선행 조건**: E-2-3 (슬리피지 sensitivity + calibration) 80+ 백테가 5h 내 완료 가능 — 단축 효과로 다음 단계 부담 해소 |
| 11 | **Phase E-2-2-FIX**: I-B012 paper_executor fees 미반영 버그 수정 | 완료 | (커밋 대기) | **발견 경위**: Phase E-2-3 Step 1 (슬리피지 sensitivity) 진행 중 사용자가 comparison.csv 의심 ("슬리피지가 전혀 적용 안 되는 것 같다"). 정합성 검증(slip 4값 결과 비교)으로 trades.csv는 슬리피지 반영, equity_curve/metrics는 미반영 확인. **근본 원인**: `paper_executor._close_internal`이 broker.balance에 gross_pnl만 반영(fees 차감 누락). 엔진은 별도 FeeModel.calc_pnl로 trades.csv에 net_pnl 기록 (정확). equity_curve가 broker.get_balance() 기반이라 부풀림 누적. **Step 1 (코드 수정)**: PaperExecutor.__init__에 FeeModel.from_config(config) 주입 + _close_internal에서 round-trip fees 차감 후 net_pnl을 balance에 반영 (~10줄). CLAUDE.md 협업 규칙 10 신규 (백테 결과 정합성 검증 우선). **Step 2 (회귀 테스트)**: tests/test_backtest_fees.py 신규 (~150줄, 6 tests) — invariant 1 (initial+sum(trades.pnl)==equity_final, parametrize 4 fee 조합), invariant 2 (sum==metrics.total_pnl 디스크 검증), invariant 3 (fees 증가→final 단조감소). pytest 201→207 passed. **Step 3 (기존 결과 재계산)**: 1회용 _recompute_metrics.py (`data/.../00_Working/_phase_e22_fix/`, .gitignore 자동) — 85 폴더 자동 발견(eval_260427/eval_260502/eval_260502_sanity/eval_260503_sensitivity), trades.csv 기반 재계산 후 metrics_corrected.json/equity_curve_corrected.csv/equity_curve_corrected.png 생성 (기존 파일 보존, 사용자 비교 가능). **Step 4 (검증+커밋)**: 본 항목 + I-B012 §12 등록 + 커밋. **부풀림 정량 (충격)**: 분류 모델 1년 ~1000%p, 4년 ~2500%p. PPO 1년 ~4000%p (5,645%→1,633%), PPO 4년 ~12000%p (21,403%→9,477%). **모델 순위 변동 (Exp4)**: PPO 1위→5위로 하락. ml_lightgbm 4년이 새 1위(16,619%). I-B010 (PPO 슬리피지 우려)이 정량 확인됨 — slippage_pct=0이어도 taker_fee 누락만으로 PPO 거래수(22,038)에 비례한 부풀림. **발견 못한 이유 (자체 분석)**: ①trades↔metrics 정합성 검증 부재 ②CLAUDE.md "라이브-백테 일관성"이 명시인데 FeeModel 흐름 미추적 ③모든 백테가 같은 fee 설정이라 "정합"으로 착각 ④사용자가 여러 번 수익률 의심 시 "데이터 누출/lookahead" 가설로만 답함. 협업 규칙 10이 향후 동일 패턴 회귀 방지. **선행 조건 영향**: Phase E-2-3 Step 1 (40 백테) 부분 결과 폐기 → 수정된 코드로 재실행 필요 |
| 12 | **Phase E-2-3**: 슬리피지 sensitivity + Calibration | 완료 | (커밋 대기) | **종합**: I-B010 PPO 슬리피지 임계점 정량 답 + I-B009 부분 해결 (Isotonic regression이 4 분류 모델 모두에서 효과). **Step 1 (슬리피지 sensitivity, 40 백테)**: 5 모델 × 분할{1, Exp4} × 슬리피지{0%, 0.02%, 0.05%, 0.1%} = 40 specs (eval_260503_sensitivity, wall time 6.78h, multiprocessing N=4). 분할 1 결과: ml_lightgbm 5,088→2,936% (slip 0→0.10%), dl_transformer 3,939→1,483%, **rl_ppo 205→-5% (slip 0.02%부터 break-even)**. 분할 Exp4 결과: 분류 모델 16,584→11,401% / 14,713→9,227% 유지, **rl_ppo 9,117→-5% (slip 0.05%부터 break-even, 0.10%에선 적자)**. **결론**: PPO는 라이브 슬리피지 거의 모든 범위에서 손상 → 라이브 적용 불가. 분류 모델은 slip 0.10%에서도 9,000~11,000% robust. **추가 발견**: `metrics_corrected.json`(E-2-2-FIX 재계산본)도 진짜 정답 아님. trades 수는 같지만 size 부풀림 영향으로 pnl 다름. PPO 1년 1633%(corrected) → 205%(새 백테, 정확) — 1/8 수준 거짓 보고됐었음. 분류 모델은 영향 작음 (~1%). 진짜 정답 = 수정된 코드로 새로 실행한 결과. **Step 2 (Calibration 인프라, ~620줄 코드)**: `src/ml/calibration.py` 신규 (MulticlassCalibrator — Platt OvR + Isotonic, 90줄), `tests/test_calibration.py` 신규 (13 tests pass), `scripts/calibrate_models.py` 신규 (4 분류 모델 walk-forward 26 folds 재실행 + calibrator 학습/저장, 290줄). 4 plugin (lightgbm/xgboost/lstm/transformer) `_ensure_model`에 calibrator 자동 로드 + `generate_signal`에 transform 분기. 4 config에 `calibration_method: "none"` 기본값 추가. evaluate_models.py에 BacktestSpec.calibration_method + build_calibration_specs + `--mode calibration`. pytest 207→**220 passed**. **Step 2-실행 (~8분)**: 4 모델 v001에 calibrator_platt.joblib + calibrator_isotonic.joblib + calibration_meta.json 학습/저장 (~150k OOS samples each, label 분포 SHORT 28%/HOLD 42%/LONG 30%). **Step 3 (Calibration 8 백테, ~7분)**: 4 분류 모델 × 분할 1 × 2 알고리즘 (eval_260503_calibration). PPO 제외 — 정책 모델이라 calibration 무의미. **결과 (분할 1 raw vs Platt vs Isotonic)**: ml_lightgbm 5,088→5,122→5,108% (PF 3.47→3.51→3.64), ml_xgboost 4,910→4,940→5,032% (PF 3.19→3.24→3.61), dl_lstm 4,663→4,607→4,865% (PF 2.69→2.70→2.92), **dl_transformer 3,939→3,943→4,251% (+7.9% with Isotonic, PF 2.20→2.21→2.40)**. **Isotonic이 4 모델 모두에서 raw 대비 개선** (winrate +1.2~2.4%p, PF +5~8% 모두 상승). **Platt는 효과 미미** (~0%, 비모수 Isotonic이 multi-class OvR에 더 적합). **I-B009 가설 정확히 확인** — Transformer가 calibration으로 가장 큰 개선 (+7.9%) but 분류 모델 순위는 변동 없음 (ml_lightgbm 1위 유지). **Step 4 (본 항목)**: §12 I-B009/I-B010 갱신, 본 §13 행 12 신규, CLAUDE.md 협업 규칙 9 보강 (모든 Step 일관 적용 + 임의 면제 금지), pytest 220 passed 최종, 일괄 commit + push. **분할 A/B/Exp2/Exp3 baseline 재실행**: 사용자 결정으로 Phase E-2-4에 포함 (E-2-3 본 작업과 분리). **future work**: ①train_*.py에 calibrator 자동 학습 옵션 (현재 별도 calibrate_models.py 호출 필요) ②BP-3 ensemble에서 calibration 패턴 적용 ③Calibration이 Transformer에 가장 효과적이었던 원인 추가 분석 (학습 분포 vs 테스트 분포 차이?) |
| 13 | **Phase E-2-4**: Baseline 재실행 + 통계 분석 + 시각화 + 종합 | 완료 | (커밋 대기) | **종합**: 정확한 baseline 30 specs + 통계 분석 (Sharpe/Calmar/Bootstrap) + 6 PNG 시각화로 Phase E 본 검증 일단락. **6대 핵심 발견 정정/확정**. **Step 1 (Baseline 42 백테)**: `evaluate_models.py --mode full --eval-date 260503_baseline`, wall time 4.65h (16,757초). 사용자 결정으로 분할 A/B/Exp2/Exp3 재실행만이 아니라 전체 30 specs + 베이스라인 12 통합 폴더(`eval_260503_baseline/`). 정합성 검증 10/10 sha256 정확 일치(분할 1, Exp4 × 5 모델 vs `eval_260503_sensitivity/_slip0.0000`). N=4→6 시도(5.12→4.65h, 9.2% 단축)했으나 30%+ 기준 미달 → **N=4 복귀** (사용자 결정대로). 원인: tail effect (마지막 rl_ppo_Exp4 단독 처리 시 5 워커 idle), N에 무관한 bottleneck. **추가 발견**: macross_1 -5.36% (E-2-2-FIX `metrics_corrected` -6.72% 대비 1.36%p 차이) — corrected는 trades.csv 단순 cumsum이라 size 부풀림 잔존, **새 백테가 진짜 정답**. **Step 2 (통계 분석 인프라)**: ①`src/ml/metrics_extended.py` 신규 (~120줄, MulticlassCalibrator 패턴 — annualization=365 crypto 표준 + Calmar + Bootstrap n=10000 seed=42). ②`scripts/analyze_results.py` 신규 (~180줄, eval_260503_baseline 자동 스캔). ③`tests/test_metrics_extended.py` 신규 (20 tests pass — Sharpe edge case + Calmar + Bootstrap 동일 분포 vs 다른 분포 + 재현성). 미니 사안 (가)/(가)/(가) 결정: Bootstrap 그대로 / annualization 365 / B&H Sharpe 제외(equity_curve 없음). pytest 220→**240 passed**. 실행 결과: `analysis_metrics.csv` (42 rows: 30 specs + macross 6 + B&H 6) + `bootstrap_pvalues.csv` (5 모델 pairwise × {분할 1, Exp4} = 20 비교). **Step 3 (시각화)**: `scripts/plot_results.py` 신규 (~165줄, 옵션 B — BTC 1d 보조 Y축). 6 PNG (분할별 1) — log scale Y축 + 5 모델 + macross + B&H 시뮬레이션 (BTC 가격 기반). 출력 `eval_260503_baseline/equity_overlay_<split>.png`. **Step 4 (본 항목)**: §13 행 13 신규 + DEVELOPER_GUIDE §10 한 줄 + USER_GUIDE §2.6/§2.7/§4.5 신규 + 일괄 commit + push. **6대 핵심 발견 재평가**: ①ML이 베이스라인 압도 — **확정** (B&H -5.68% vs 분류 모델 5,088%, macross -5.36%로 모두 음수). ②**PPO 4년 OOS 1위는 무효** (size 부풀림으로 21,403%가 실제 9,117%로 1/2 수준). ③Transformer 학습 OOS 1위→백테 최저 — **확정** (calibration으로 +7.9% 개선했으나 여전히 분류 4개 중 최저). ④**데이터 양 효과 #4 정정 — "3년→4년 ~3x 도약"은 틀림**. 정확한 차이: ml_lightgbm 14,578→16,584 (**+13.8%**), 비슷한 수준. 데이터 양은 3년부터 포화. ⑤LightGBM 가장 안정 PF 3.21 — **확정** (정확 baseline에서도 PF 평균 ~3.2 유지, Sharpe 10.80 분할 1 1위). ⑥2025 약세장 — **확정** (B&H -5.68%). **통계 인프라 결과 추가**: Sharpe 분류 모델 1년 OOS 9.86~10.88 (매우 우수), 4년 OOS는 5.96~6.72로 감소(누적 시간 영향). Calmar ml_lightgbm 분할 1 = **810** 최고. **Bootstrap 핵심 결과 (20 비교)**: ①ml_lightgbm vs ml_xgboost p=0.61(분할 1), 0.65(Exp4) — **GBDT 두 모델 통계적 동등**, 운영 시 둘 중 하나 선택 무관. ②모든 모델 vs rl_ppo p<0.0001 — **PPO 명확히 열등** 확정. ③LGB/XGB vs Transformer 일관 p<0.01 (LGB/XGB 약간 우수). ④LSTM은 분할에 따라 갈림. **운영 권장**: ml_lightgbm 또는 ml_xgboost + calibration_method=isotonic. PPO는 라이브 부적합 확정. **Phase E-종착 별도 처리**: 사안 F 결정대로 본 commit과 분리, 별도 Phase로 PATH_B_PRODUCTION_*.md / PATH_B_LIVE_TRADING_*.md 신규 생성 + §14 정리 + §13 마지막 행 갱신 + commit `[경로B_종착]` |
| 14 | **Phase E 종착 (PATH_B_ML_STRATEGY 닫는 단계)** | 완료 | (커밋 대기) | **본 §13 마지막 갱신 + §14 closing 메모 + 신규 리포트 2개 한 묶음 커밋 `[경로B_종착]`**. ①본 §13 행 14 신규 (현재 항목). ②§14 상단에 "이 outline은 신규 리포트에 상세화됨, historical reference로 보존" 메모 추가 (사안 D-(나) 결정). ③`docs/00_Work_Report/PATH_B_PRODUCTION_260503.md` 신규 생성 — BP-1/2/3 상세 계획 (~280줄, 사안 A-(가) 결정). I-BP001 (funding_fee 백테 미반영) + I-BP002 (백테 엔진 warmup 미구현) carry-over 등록 (사안 B-(나) 결정). ④`docs/00_Work_Report/PATH_B_LIVE_TRADING_260503.md` 신규 생성 — BL-1/2 상세 계획 (~250줄). ⑤진행 우선순위 명시 (BP-1+2 → BP-3 → BL-1 → BL-2, 사안 C-(가) 결정). ⑥pytest 240 passed 회귀 확인 후 commit + push. **Phase E 종합 결과** (E-2-2/2-2-OPT/2-2-FIX/2-3/2-4 통합): ①정확한 baseline 30 specs 확보 (eval_260503_baseline). ②**ml_lightgbm 또는 ml_xgboost + isotonic = 운영 권장 후보** (Sharpe 10.42-10.88, Calmar 240-810, GBDT 두 모델 통계적 동등 p>0.6). ③**PPO 라이브 부적합** 확정 (slip 0.05% break-even + 모든 모델 vs PPO p<0.0001). ④데이터 양 효과 #4 정정 — "3년→4년 +13.8%만, 3년부터 포화". ⑤2025 약세장 (B&H -5.68%) 모든 ML 모델 압도. ⑥신규 협업 규칙 8 (구조 점검) + 9 (Step 임시 보존) + 10 (정합성 검증) 도입. **Phase E commit 체인**: a208255(E-2-2) → cf80846(OPT) → dd53987(FIX) → 743baec(E-2-3) → 74d0c96(E-2-4) → 본 commit (E-종착). PATH_B_ML_STRATEGY는 본 commit으로 닫고, 후속 작업은 PATH_B_PRODUCTION/LIVE_TRADING로 이어짐 |

---

## 14. Phase B 후속 로드맵 (Phase E 종착 후 진행)

> ✅ **CLOSING 메모 (2026-05-03 갱신, Phase E 종착 시점)**:
> Phase E 종착으로 본 §14의 outline은 **신규 리포트에 상세화 완료**:
> - §14.2 outline → `docs/00_Work_Report/PATH_B_PRODUCTION_260503.md` (Phase BP-1/2/3 상세 계획)
> - §14.3 outline → `docs/00_Work_Report/PATH_B_LIVE_TRADING_260503.md` (Phase BL-1/2 상세 계획)
>
> 본 §14는 **historical reference로 보존** (Phase E 진행 중 사전 outline의 발견 시점·맥락 기록 가치). 후속 작업은 신규 리포트에서 진행.
>
> §14.1의 TODO는 §13 행 14 (Phase E 종착)로 이행 완료 (커밋 `[경로B_종착]`에서 처리).

**🔔 작성 시점:** 2026-04-28 (Phase E-2 진행 중에 사전 작성). Phase E 종착 시 본 §14를 갱신하여 닫는 메모(작업 완료 보고)로 정리하고, 상세 계획은 신규 리포트로 이전.

Phase B (ML 전략 비교 연구)는 Phase E로 종착되며, 이후 작업은 두 개의 별도 리포트에서 진행 — 모두 **"경로 B"의 연속**.

### 14.1 Phase E 종착 시 수행할 작업 (TODO — 한 묶음 커밋)

커밋 메시지 예: `[경로B_종착] Phase E 마무리 + 후속 로드맵 + 신규 리포트 2개`

1. **본 §13 마지막 갱신** — Phase E-2-2/2-3/2-4의 결과 + 커밋 ID 모두 채움
2. **본 §14 닫는 정리** — "이 작업은 PATH_B_PRODUCTION/LIVE_TRADING으로 이어졌음" 식의 클로징 메모로 변환
3. **`PATH_B_PRODUCTION_{날짜}.md` 신규 생성** — §14.2의 outline을 기반으로 상세 계획
4. **`PATH_B_LIVE_TRADING_{날짜}.md` 신규 생성** — §14.3의 outline을 기반으로 상세 계획
5. 회귀 테스트 1회 통과 확인 후 커밋

### 14.2 PATH_B_PRODUCTION_*.md 구조 (Phase BP-1, BP-2, BP-3)

**목적:** 데이터 확장 + 운영 인프라 + 모델 결합·강화 — **라이브 운영 가능한 base 구축**

#### Phase BP-1: 데이터 확장
- **펀딩률·OI 데이터 수집 + 피처 추가** (가장 큰 alpha source — 우리 모델의 사각지대 해소)
  - OKX API에서 펀딩률 history + open interest 수집
  - 피처에 funding_rate, oi_change 등 추가 (현재 81 → ~85개)
- **펀딩률 백테 정확도 검증** (수익률 의심 분석에서 추가 — 2026-04-28)
  - config의 `accounting.funding_enabled: true`이지만 실제 백테에서 funding_fee가 0인지 확인 필요
  - 백테 엔진의 `_record_trade_close` 내 funding_fee 처리 로직 점검
  - OKX `fetch_funding_history` API 호출 정상 작동 여부
  - 라이브 시 funding_fee 누락 시 long 포지션 보유 시 평균 -0.03%/일 추가 비용 발생 가능
- **2020 이전 BTC 현물 데이터 추가** (학습 데이터 +50%)
  - 2017-2019 BTC/USDT 현물 데이터 (OKX 또는 Bitstamp)
  - 학습 시 현물 + 선물 병합 처리
  - 시기별 거래량 정규화 (현물/선물 단위 차이)

#### Phase BP-2: 운영 인프라
- **Live OOS monitoring 시스템** (alpha decay 자동 감지)
  - 실시간 OOS Acc 측정 (예: 최근 100봉 정확도)
  - 임계 미달 시 자동 정지 또는 사이징 축소
- **동적 사이징** (변동성 기반 risk_per_trade 조정)
  - ATR 평균 대비 현재 변동성 비율로 사이징 스케일
- **백테 엔진 warmup 자동화** (2026-04-28 추가 — "백테 초반 무거래" 분석에서)
  - 현재: `_load_candles`가 [start_dt, end_dt]만 로드 → 4h EMA200 워밍업 33일 동안 모든 모델 HOLD (피처 NaN)
  - 검증: B-1b 분할 1 백테에서 첫 거래가 2025-02-03 (백테 시작 + 33일)으로 정확히 일치
  - 영향: OOS 짧을수록 워밍업 손실 비율 큼 (1년 분할 9% vs 4년 분할 2.3%)
  - 개선 방안: `_load_candles`가 가장 긴 indicator warmup(현재 4h EMA200 = 800시간) 만큼 start_dt를 앞당겨 캔들 미리 로드, 백테 루프는 원래 start_dt부터 진입 평가
  - 또는 config의 `data.history_bars`를 백테 엔진에서도 활용
  - 라이브에서는 history_bars=300으로 자연 처리됨 (4h × 300 = 50일 ≥ 33일)

#### Phase BP-3: 모델 결합
- **Confidence calibration 본격 적용** (I-B009 후속)
  - Phase E-2-3 검증 결과 기반 production 적용
  - Platt scaling 또는 isotonic regression
- **Ensemble** (5개 모델 결합)
  - Phase E 결과로 다양성 확인됨
  - voting / stacking / weighted average 등 비교
- **Triple-barrier 라벨** (label noise 해소 — OOS 64% 한계 돌파 시도)

**이슈 ID 체계:** I-BP001~

### 14.3 PATH_B_LIVE_TRADING_*.md 구조 (Phase BL-1, BL-2)

**목적:** Regime 검증 + 학술 검증 + 실거래 전환

#### Phase BL-1: Regime + 추가 검증
- **Regime-matched 백테** (사용자 제안)
  - 학습 2022~2024 (약세+회복) → 백테 2020-2021 (강세) — regime 매칭 검증
  - 또는 HMM/stylized facts로 현재 시장과 비슷한 과거 시기 식별
- **Walk-forward 통합 OOS** (가장 엄밀한 평가)
  - 학습 시 fold별 모델 모두 저장 → 통합 백테
- **Lookahead bias 추가 점검** (I-B007 후속)
  - 캔들 데이터 자체의 미래 정보 누출 등 다른 경로 점검
  - Indicator forward-bias (pandas_ta 등 라이브러리 검증)
  - Walk-forward의 embargo 누설 가능성
- **호가창/유동성 모델링 정교화** (2026-04-28 추가 — 슬리피지 정교화의 일부)
  - 현재: open 가격 즉시 체결 가정
  - 개선: OKX 호가창 snapshot 데이터 활용해 유동성 시뮬레이션
  - 사이즈 대비 호가창 깊이로 실효 체결 가격 산정
- **Multi-hypothesis 보정** (2026-04-28 추가 — 수익률 의심 분석에서)
  - 5개 모델 × 6개 분할 + Phase E-2-3 슬리피지·calibration = 50+개 비교
  - Bonferroni 보정 (α=0.05/N) 또는 False Discovery Rate(FDR) 적용
  - Phase E-2-4의 부트스트랩 검정 결과를 다중 가설 보정으로 재평가
  - "p < 0.05라도 실제론 우연일 확률" 정량화

#### Phase BL-2: 실거래 전환
- **Paper → 소액 실거래 점진 전환**
  - 단일 모델로 시작 → 점진 확장
  - Fail-safe / 자동 알림 (텔레그램 등)
- **다중 거래소** (Binance + OKX)
- **Survivorship bias 점검** (학술적 검증)
- **거래소 다운/API 지연 대응 시뮬레이션** (2026-04-28 추가 — 백테 vs 라이브 격차 분석)
  - 거래소 API rate limit, 일시적 다운, 응답 지연 시 모델 신호 처리
  - 신호 발생 ~ 실제 진입 사이 1~10초 지연 시 가격 변동 영향
  - Order rejection 시 retry 로직
  - Circuit breaker (자동 정지 + 알림)
  - 라이브 운영 안전망의 핵심 — Paper trading에서 시뮬레이션 가능

**이슈 ID 체계:** I-BL001~

### 14.4 모든 후속 리포트가 PATH_B_ML_STRATEGY에서 계승

- **5개 모델 base** (Phase B-1a/b/2a/b/3) — 코드 + 학습된 모델 (v001~v004 디렉토리)
- **평가 인프라** — `evaluate_models.py`, `walk_forward`, `sequence_utils` 등
- **잠재 이슈 트래커** I-B001~I-B011 (모두 해결 또는 검증 완료)
- **설계 원칙** — 엔진 코드 수정 = 0, 학습/추론 분리, 단계별 검증, 솔직한 검증 분류 등 (CLAUDE.md 협업 규칙 그대로)

### 14.5 명명 규칙

| 식별자 | 의미 |
|---|---|
| `PATH_B_ML_STRATEGY` | 경로 B 시작 — ML 전략 비교 (본 문서) |
| `PATH_B_PRODUCTION` | 경로 B Production Readiness — 다음 Phase 1·2 묶음 |
| `PATH_B_LIVE_TRADING` | 경로 B Live Trading — 다음 Phase 3·4 묶음 |
| Phase ID `BP` / `BL` | Production / Live |
| 이슈 ID `I-BP` / `I-BL` | 새 리포트의 잠재 이슈 prefix |
