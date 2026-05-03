# PATH_B_PRODUCTION (경로 B Production 전환 단계)

작성 시점: 2026-05-03 (Phase E 종착 시점)
선행: `PATH_B_ML_STRATEGY_260425.md` (Phase 0 ~ E-2-4 완료, commit `74d0c96`로 닫음)
후속: `PATH_B_LIVE_TRADING_260503.md` (Regime 검증 + 실거래 전환)

---

## 0. 문서 목적

CoinBot의 "경로 B" — ML/DL 모델이 전략 자체가 되는 흐름 — 의 **production base 구축** 단계. PATH_B_ML_STRATEGY에서 확립한 5개 모델 base + 평가 인프라 위에 **데이터 확장 + 운영 인프라 + 모델 결합**을 추가하여 라이브 운영 가능한 base를 만든다.

---

## 1. PATH_B_ML_STRATEGY 계승 사항

### 1.1 코드 자산 (commit `74d0c96` 시점)

- **5개 모델 base** (v001~v004): ml_lightgbm, ml_xgboost, dl_lstm, dl_transformer, rl_ppo
- **평가 인프라**: `evaluate_models.py` (5 mode: full/single/collect/sensitivity/calibration), `analyze_results.py`, `plot_results.py`
- **Calibration 인프라**: `src/ml/calibration.py` (MulticlassCalibrator: Platt OvR + Isotonic), `scripts/calibrate_models.py`
- **통계 인프라**: `src/ml/metrics_extended.py` (Sharpe, Calmar, Bootstrap)
- **회귀 테스트 240건** (test_backtest_fees / test_calibration / test_metrics_extended 포함)

### 1.2 잠재 이슈 트래커 carry-over

PATH_B_ML_STRATEGY §12의 I-B001~I-B012 모두 해결 또는 검증 완료. 단 다음 2건은 BP에서 처리할 영역으로 carry-over (사안 B 결정):

- **I-BP001 (carry from I-B007 후속)**: funding_fee 백테 미반영 — BP-1에서 fix 필수
- **I-BP002 (carry from §14.2)**: 백테 엔진 warmup 미구현 — BP-2에서 fix

상세는 §6 잠재 이슈 트래커 참조.

### 1.3 설계 원칙 (PATH_B_ML_STRATEGY §1과 동일)

- 엔진 코드 수정 = 0 (전략 추가 시)
- 학습/추론 분리 (학습은 별도 스크립트, 추론은 plugin)
- 단계별 검증 + 솔직한 검증 분류
- CLAUDE.md 협업 규칙 1~10 그대로 적용

### 1.4 Phase E 운영 권장 (라이브 적용 후보)

E-2-4 분석 결과 기반:
- **운영 후보 1순위**: `ml_lightgbm` 또는 `ml_xgboost` + `calibration_method=isotonic`
  - 통계적 동등 (Bootstrap p=0.61), Sharpe 10.42-10.88, Calmar 240-810
- **PPO 라이브 부적합** 확정: slip 0.05% break-even, 모든 모델 vs PPO p<0.0001

---

## 2. 전체 구조 (Phase BP-1, BP-2, BP-3)

```
Phase BP-1 (데이터 확장)
├─ 펀딩률·OI 수집 + 피처 추가
├─ 펀딩률 백테 정확도 검증 (I-BP001 fix)
└─ 2020 이전 BTC 현물 데이터

Phase BP-2 (운영 인프라)
├─ Live OOS monitoring
├─ 동적 사이징
└─ 백테 엔진 warmup 자동화 (I-BP002 fix)

Phase BP-3 (모델 결합)
├─ Confidence calibration 본격 적용
├─ Ensemble (5개 모델 결합)
└─ Triple-barrier 라벨
```

### 2.1 진행 우선순위 (사안 C 결정)

> ⚠️ **알파벳 순서 아님 — 의존성·영향력 기반 순서**

1. **BP-1 + BP-2 (데이터 + 운영)** ← 가장 시급. 라이브 가기 전 필수
2. BP-3 (모델 결합) ← 데이터·운영 안정 후
3. (이후) PATH_B_LIVE_TRADING의 BL-1 → BL-2

---

## 3. Phase BP-1: 데이터 확장

### 3.1 목적

5개 모델은 OHLCV 81 피처(15m/1h/4h × 27)만 사용. **펀딩률/OI는 BTC 무기한 선물의 핵심 alpha source인데 누락**. 추가하여 모델 정보량 확장.

### 3.2 작업 항목

#### 3.2.1 펀딩률·OI 데이터 수집 + 피처 추가

**현재 상태** (코드 점검 완료):
- `src/data/historical.py` `HistoricalDataLoader`: 캔들만 다룸 (`download`, `download_range_merged`). 펀딩률/OI 메서드 부재.
- ccxt OKX 지원: `fetch_funding_rate_history`, `fetch_open_interest` 가능

**작업**:
1. `HistoricalDataLoader`에 메서드 신규:
   - `download_funding_rate_history(start_ms, end_ms)` → `data/funding/{symbol}_funding.csv`
   - `download_open_interest_history(start_ms, end_ms, timeframe)` → `data/oi/{symbol}_oi_{tf}.csv`
2. `src/strategy/features.py`에 새 피처 함수:
   - `compute_funding_features(funding_df)` — 평균/MA/Z-score
   - `compute_oi_features(oi_df, candles)` — OI 변화율, 거래량 비율 등
3. `compute_multi_tf_features` 확장 — 펀딩률/OI 합쳐 81 → ~85개 피처
4. 피처 캐시 (`src/ml/feature_pipeline.py`) 무효화 — 새 피처 추가 시 재생성
5. 5개 train_*.py 스크립트가 자동 새 피처 흡수 (피처 이름 동적 — 기존 패턴)
6. plugin 추론 (`compute_multi_tf_features` 호출)도 자동 반영

#### 3.2.2 펀딩률 백테 정확도 검증 (I-BP001)

**현재 상태**:
- `src/accounting/fee_model.py:57-64` `estimate_funding`이 `funding_enabled=True`라도 **0 반환** (TODO 주석 있음)
- `src/core/engine_base.py:close_position`에 `funding_fee=0.0` 하드코드

**영향**:
- LONG 포지션 평균 -0.01%/8h 펀딩률 가정 시 1년 보유 ~10% 추가 비용 발생 가능
- 단 우리 모델은 평균 보유 6.5봉 (1.6시간) → 영향 미미 가능. 그러나 검증 필요.

**작업**:
1. `FeeModel.estimate_funding` 구현 — 펀딩률 history × 보유 시간 비례 계산
2. `BacktestEngine`이 close_position 시 보유 기간의 펀딩률 합산 → close_position에 funding_fee 인자 전달
3. 라이브 LiveExecutor도 동일 — 거래소 체결 응답의 funding 자동 반영 확인
4. 회귀 테스트 신규: `tests/test_funding.py` — funding_fee 정합성 검증 (test_backtest_fees.py 패턴)
5. 정확한 baseline 재실행 (eval_260503_baseline 30 specs를 funding 적용으로 재계산하여 영향 정량)

#### 3.2.3 2020 이전 BTC 현물 데이터 추가

**현재 상태**:
- 학습 데이터 2020-01-01 ~ 2024-12-31 (5년, v001 기준)
- 2017-2019 BTC가 주요 강세장 (2017 bull, 2018 crash) — regime 다양성 ↓

**작업**:
1. 2017-2019 BTC/USDT 현물 데이터 수집 (OKX/Bitstamp)
2. 현물 vs 선물 차이 (펀딩률, 만기, 단위) 정규화
3. 학습 시기별 단가 차이 처리 (2017 BTC ~$1k vs 2024 ~$70k → 가격 정규화 또는 returns로만 학습)
4. v005 모델 학습 (8년 학습 데이터) → Phase E-1 패턴으로 재평가

### 3.3 결정 사안 (BP-1 진입 시 결정)

#### A. 펀딩률 데이터 시간 단위
- (가) 8시간 (OKX 표준 펀딩 주기) — 단순
- (나) 15분 봉별 보간 (forward-fill) — 피처 일관성

#### B. OI 데이터 timeframe
- (가) 15m (entry_timeframe과 일치)
- (나) 1h (변동성 ↓, 노이즈 ↓)

#### C. 새 피처 추가 후 v005 학습 vs 기존 v001-v004 fine-tune
- (가) v005 신규 학습 (clean)
- (나) fine-tune (학습 시간 ↓)

### 3.4 검증 기준

- 펀딩률 적용 전후 30 specs 결과 비교 (수익률 변화 정량)
- 신규 피처 효과: v005(85 피처) vs v001(81 피처) — 같은 5년 학습 + 분할 1 OOS
- pytest 240 + 신규 funding 테스트 통과

---

## 4. Phase BP-2: 운영 인프라

### 4.1 목적

라이브 운영 시 필수 안전망 + 효율성 인프라. 학습-라이브 격차 줄이기.

### 4.2 작업 항목

#### 4.2.1 Live OOS monitoring 시스템

**목적**: alpha decay 자동 감지

**작업**:
1. `src/live/oos_monitor.py` 신규
   - 라이브 운영 중 최근 N봉(예: 100봉) 정확도 측정
   - `model.predict` 결과 vs 실제 future return 비교 (라벨은 `generate_direction_labels` 동일 패턴)
   - 임계 미달 (예: OOS Acc < 학습 OOS Acc - 10%p)시 알림 또는 자동 정지
2. `CoreEngine` 통합 — bar 마감 시점 자동 호출
3. config 옵션: `live.oos_monitoring.enabled` / `min_acc_threshold` / `alert_method`

#### 4.2.2 동적 사이징

**목적**: 변동성 기반 risk_per_trade 조정 — 변동성 높을 때 size ↓

**현재 상태**: `src/risk/manager.py` `calculate_position_size`가 고정 `risk_per_trade_pct` 사용

**작업**:
1. `RiskManager.calculate_position_size` 확장:
   - 인자: `volatility_factor` (현재 ATR / 평균 ATR)
   - size = base_size × min(1.0, target_volatility / current_volatility)
2. plugin이 ctx.candles에서 변동성 계산 후 risk_manager에 전달
3. config 옵션: `risk.volatility_targeting.enabled` / `target_atr_pct`

#### 4.2.3 백테 엔진 warmup 자동화 (I-BP002)

**현재 상태**:
- `src/backtest/engine.py:_load_candles`가 `[start, end]`만 로드
- 4h EMA200 = 800시간 ≈ 33일 워밍업 → 백테 초반 무거래
- 1년 OOS는 9% 손실 (33/365), 4년 OOS는 2.3%

**작업**:
1. `BacktestEngine._load_candles` 변경 — config의 `data.history_bars` (default 300) 활용
   - 가장 큰 indicator window (4h EMA200 = 800시간) 계산
   - start_ms를 `start - max_warmup_bars * tf_ms`로 앞당겨 캔들 미리 로드
   - 백테 루프(`run`)는 원래 start_dt부터 진입 평가 시작
2. master_df slice는 `master_df.loc[start:end]`로 그대로 (warmup 캔들은 features 계산용으로만)
3. 회귀 검증: B-1b 분할 1 (2025) 결과 변화 — warmup 적용 후 첫 거래 시점 2025-01-01 직후로 변경 (이전 2025-02-03)

### 4.3 결정 사안 (BP-2 진입 시 결정)

#### D. Live OOS monitoring 임계값 처리
- (가) 자동 정지 (안전 우선)
- (나) 알림만 (사용자 결정)
- (다) 사이징 자동 축소

#### E. 동적 사이징 변동성 metric
- (가) ATR 평균 (단순)
- (나) 일일 returns std (Sharpe와 일관)
- (다) GARCH 기반 (정교)

### 4.4 검증 기준

- Live OOS monitoring: paper trading 1주일 시뮬레이션
- 동적 사이징: 백테 결과 변동성 (변동성 높은 시기 size 감소 확인)
- Warmup: B-1b 분할 1 첫 거래 시점이 2025-01-01 직후로 이동 + total_trades 증가

---

## 5. Phase BP-3: 모델 결합

### 5.1 목적

Phase E-2-3에서 검증된 calibration + Phase E의 5 모델 결과 차이를 활용하여 단일 모델 한계 돌파.

### 5.2 작업 항목

#### 5.2.1 Confidence calibration 본격 적용 (I-B009 후속)

**현재 상태**: 인프라 완비 (`src/ml/calibration.py`, `scripts/calibrate_models.py`, plugin 4개 분기)

**작업**:
1. v005 (BP-1 학습) 모델에 calibrator 학습 + 저장 (calibrate_models.py 그대로)
2. config 기본값을 `calibration_method: "isotonic"`으로 변경 (현재 default "none")
3. 회귀 검증: 30 specs 재실행 (calibrated baseline 확보)

#### 5.2.2 Ensemble (5개 모델 결합)

**현재 상태**: (C) 배타적 경합 정책 — 같은 시점 1 모델만 진입. Ensemble 미구현.

**작업**:
1. `src/strategy/plugins/ensemble.py` 신규 — 새 plugin
   - 내부에 5 모델 로드
   - generate_signal: 5 모델 각각 raw_probs 계산 → voting 또는 weighted average
   - voting 방식: majority vote, soft vote (probability 평균)
   - weighted: bootstrap p-value로 신뢰도 가중
2. config 옵션: `ensemble.method: "soft"/"hard"/"weighted"` / `ensemble.models: [list]`
3. 백테 평가 (eval_260503_baseline 30 specs와 비교)
4. PPO 포함 여부 — 기본 제외 (BL 부적합 확정)

#### 5.2.3 Triple-barrier 라벨

**현재 상태**: `src/ml/label_generator.py:generate_direction_labels` 단순 horizon-pct 방식

**작업**:
1. `generate_triple_barrier_labels(df, upper_barrier_pct, lower_barrier_pct, time_barrier_bars)` 신규
2. SHORT/HOLD/LONG → SHORT/HOLD/LONG/timeout (4-class) 또는 binary로 단순화
3. Label noise 감소 → OOS Acc 64% 한계 돌파 시도
4. 5개 모델 v006 학습 (Triple-barrier label) → 비교

### 5.3 결정 사안

#### F. Ensemble voting 방식
- (가) Soft vote (probability 평균) ← 단순, 일반적으로 효과 좋음
- (나) Hard vote (majority class)
- (다) Weighted (Bootstrap p-value 기반)

#### G. Triple-barrier 4-class vs binary
- (가) 4-class (SHORT/HOLD/LONG/timeout)
- (나) binary (Profit/Loss) — label noise 더 적음

### 5.4 검증 기준

- Calibrated 30 specs vs raw 30 specs: 평균 수익률 + Sharpe 변화
- Ensemble vs 단일 최고 모델 (ml_lightgbm): 분할 1 OOS, Bootstrap p-value
- Triple-barrier vs direction_labels: v006 vs v001 OOS Acc 비교

---

## 6. 진행 기록 (Phase BP-1/2/3)

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| (대기) | Phase BP-1: 데이터 확장 | 대기 | — | 펀딩률·OI 수집 + 피처 추가 + I-BP001 fix + 2020 이전 데이터 |
| (대기) | Phase BP-2: 운영 인프라 | 대기 | — | Live OOS monitoring + 동적 사이징 + I-BP002 fix |
| (대기) | Phase BP-3: 모델 결합 | 대기 | — | Calibration 본격 적용 + Ensemble + Triple-barrier |

---

## 7. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BP001 | Phase B-3 / E-2-2-FIX (carry from PATH_B_ML_STRATEGY §14.2) | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환 (line 57-64). `engine_base.close_position`에 `funding_fee=0.0` 하드코드. config의 `accounting.funding_enabled=true`이지만 실제 백테에 미반영. 라이브 LONG 포지션 평균 -0.01%/8h 펀딩률 추가 비용 발생 가능 (보유 시간 짧으면 미미할 수 있음) | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | Phase BP-1 | 미해결 — BP-1에서 `estimate_funding` 구현 + 라이브-백테 일관성 검증 + 정확 baseline 재실행 |
| I-BP002 | Phase E-2-2-OPT 분석 (carry from §14.2) | `BacktestEngine._load_candles`가 `[start, end]`만 로드 → 4h EMA200 800시간(33일) 워밍업으로 백테 초반 무거래 (1년 OOS 9% / 4년 OOS 2.3% 손실). config의 `data.history_bars`(300) 백테 엔진에서 미참조. 라이브에서는 history_bars=300으로 자연 처리됨 | src/backtest/engine.py:_load_candles | Phase BP-2 | 미해결 — BP-2에서 `_load_candles` 수정으로 warmup 캔들 미리 로드. 백테 루프는 원래 start_dt부터 진입 평가 |

신규 이슈는 작업 진행 시 I-BP003~ 형태로 추가.

---

## 8. 후속 로드맵

PATH_B_PRODUCTION 종착 후 → `PATH_B_LIVE_TRADING_260503.md` Phase BL-1 (Regime + 추가 검증) → BL-2 (실거래 전환).
