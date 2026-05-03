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

- **I-BP001 (carry from I-B007 후속)**: funding_fee 백테 미반영 — ~~BP-1에서 fix 필수~~ → **BL carry-over** (사안 H로 BP-1 스킵, BL-1 §3.2.6에서 재진입 여부 확인. 미진입 시 BL 종착까지 carry-over). 보유 시간 평균 1.6시간이라 영향 미미 추정.
- **I-BP002 (carry from §14.2)**: 백테 엔진 warmup 미구현 — ~~BP-2에서 fix~~ → **해결 (BP-2-1)**. `_load_candles`가 `data.history_bars` 활용

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
Phase BP-1 (데이터 확장) ⚠️ 스킵 (사안 H, BL-1 §3.2.6에서 재진입 여부 확인 [조건부])
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

1. ~~**BP-1 + BP-2 (데이터 + 운영)**~~ → **BP-2 (운영)** 단독 ← BP-1은 사안 H로 스킵 (데이터 수집 인프라 미준비)
2. BP-3 (모델 결합) ← 운영 인프라 안정 후
3. (이후) PATH_B_LIVE_TRADING의 BL-1 (Regime 검증 + **BP-1 재진입 여부 확인** [조건부, 마지막]) → BL-2

---

## 3. Phase BP-1: 데이터 확장 ⚠️ 스킵 (사안 H)

> **스킵 결정 (2026-05-03)**: 데이터 수집 인프라 미준비로 BP-1 전체 스킵. BL-1 §3.2.6에서 재진입 여부 확인. 진입 결정 시 본 §3 내용을 그대로 활용한다. v005 모델 번호는 본 단계에서 학습할 모델용으로 예약 (BP-3 Triple-barrier 학습은 v006로 명명).
>
> **사유**:
> - 3.2.1 펀딩률·OI 수집 — ccxt OKX API 다운로드 인프라 미준비
> - 3.2.2 I-BP001 fix — 펀딩률 데이터 없으면 검증 불가 → BL carry-over
> - 3.2.3 2020 이전 BTC 데이터 — 외부 거래소(Bitstamp 등) 수집 인프라 미준비
> - I-BP001은 보유 시간 평균 1.6시간이라 백테 영향 미미 추정 (§3.2.2 자체 평가)

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

#### H. BP-1 진행 여부 ✅ 결정됨 (2026-05-03)
- (가) BP-1 완전 스킵 → BP-2 → BP-3 (v001-v004 기반) ← **선택**
- (나) BP-1 부분 진행 (3.2.2 코드 fix만, 검증 보류)
- (다) BP-2 → BP-3 → BP-1 (라이브 직전 재방문)

선택 사유: 데이터 수집 인프라 미준비. BP-3에서 v005 의존을 v001-v004 치환 가능 (calibration 인프라 PATH_B_ML_STRATEGY E-2-3에서 완성). I-BP001은 보유 시간 평균 1.6시간이라 영향 미미 추정. BL-1 §3.2.6에서 진입 여부 확인. 미진입 시 BL 종착까지 carry-over.

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

## 4. Phase BP-2: 운영 인프라 ✅ 완료 (2026-05-03)

### 4.1 목적

라이브 운영 시 필수 안전망 + 효율성 인프라. 학습-라이브 격차 줄이기.

### 4.2 작업 항목

#### 4.2.1 Live OOS monitoring 시스템 ✅ 완료 (BP-2-3)

**목적**: alpha decay 자동 감지

**완료 작업**:
1. `src/live/oos_monitor.py` 신규 — `LiveOOSMonitor` 클래스
   - `record_prediction(strategy_name, entry_tf, ts, side, entry_close)` → buffer에 push
   - `evaluate_pending(now, close)` → horizon 도달 시 actual label 분류 + hit 채점
   - window 도달 시 적중률 < `min_acc_threshold`이면 logger.warning 알림
2. `CoreEngine.initialize`에서 monitor 초기화 (enabled=true 시), `_on_bar_closed`에서 evaluate_pending
3. `AbstractEngine`에 `_record_oos_signal` helper (evaluate_strategies_on_bar의 슬롯 빔/reverse 경로에서 push). 순환 import 방지로 `Any | None` 필드만 보유
4. config 옵션: `live.oos_monitoring.{enabled, window, horizon, threshold_pct, min_acc_threshold, cooldown_bars, alert_method}`
5. 단위 테스트 12건 (`tests/test_oos_monitor.py`)

#### 4.2.2 동적 사이징 ✅ 완료 (BP-2-2)

**목적**: 변동성 기반 risk_per_trade 조정 — 변동성 높을 때 size ↓

**완료 작업**:
1. `RiskManager.calculate_position_size`에 `volatility_factor: float = 1.0` 인자 추가
   - 공식: `adjustment = min(1.0, 1.0 / factor)` (사안 J 가: 축소만)
   - default 1.0이면 비활성과 동일
2. `engine_base._compute_volatility_factor(strategy, ctx)` helper 신규
   - entry_timeframe candles에서 ATR(lookback) 산출 → factor = current_atr_pct / target_atr_pct
   - 캡슐화: ATR 계산은 engine_base, RiskManager는 단순 곱셈 (indicators 의존 누출 없음)
3. `try_enter`에서 helper 호출 → RiskManager에 전달 (백테/라이브 양 엔진 자동 적용)
4. config 옵션: `risk.volatility_targeting.{enabled, target_atr_pct, lookback}`. default 비활성
5. Fallback 6단계 (enabled false/target≤0/봉수 부족/ATR series 이상/close≤0/factor≤0 → factor=1.0)
6. 단위 테스트 9건 (RiskManager 5 + engine_base helper 4)

#### 4.2.3 백테 엔진 warmup 자동화 (I-BP002) ✅ 완료 (BP-2-1)

**완료 작업**:
1. `BacktestEngine._load_candles` 변경 — `data.history_bars` (default 300)만큼 TF별 start_ms 앞당김. 라이브 backfill과 동일 키 재사용 (사안 I 가)
2. `run` 루프의 master_df slice는 변경 없음 — warmup 캔들은 `_build_features_cache`에서만 사용
3. **회귀 검증 결과**: ml_xgboost 2025-01-01~2025-03-31 백테에서 첫 거래 시점 **2025-01-01 09:30 UTC** (warmup 미적용 33일 → 9.5시간으로 단축)
4. 단위 테스트 3건 (`test_load_candles_includes_warmup`, `test_load_candles_respects_custom_history_bars`, `test_run_loop_slices_to_original_range_after_warmup`)

### 4.3 결정 사안 (BP-2 진입 시 결정)

#### D. Live OOS monitoring 임계값 처리 ✅ 결정됨 — **(나) 알림만**
- (가) 자동 정지 (안전 우선)
- (나) 알림만 (사용자 결정) ← **선택**
- (다) 사이징 자동 축소

#### E. 동적 사이징 변동성 metric ✅ 결정됨 — **(가) ATR 평균**
- (가) ATR 평균 (단순) ← **선택**. 이미 `compute_atr` / `atr_pct` 인프라 존재. 라이브-백테 일관성 자동
- (나) 일일 returns std (Sharpe와 일관)
- (다) GARCH 기반 (정교) — 별도 학습 인프라 필요. ML 모델과 중복 우려

#### I. 백테 warmup 캔들 산정 방식 ✅ 결정됨 — **(가) `data.history_bars` 재사용**
- (가) `data.history_bars` (현재 300) 재사용 ← **선택**. 라이브와 일치 (DRY)
- (나) 백테 전용 신규 키
- (다) 코드에서 자동 산출

#### J. 동적 사이징 조정 방향 ✅ 결정됨 — **(가) 축소만**
- (가) **축소만** (factor>1일 때만 size 감소) ← **선택**. 라이브 운영 안전 우선
- (나) 양방향 (잔잔할 때 size 증가) — 잔잔→폭증 regime shift에 약함

#### K. ATR 산출 timeframe ✅ 결정됨 — **(가) entry_timeframe**
- (가) **entry_timeframe** ← **선택**. plugin과 일관된 시간 척도
- (나) 4h 고정
- (다) config로 명시

#### L. `target_atr_pct` 기본값 ✅ 결정됨 — **(가) 0.005 (15m 기준 0.5%)**
- (가) **0.005** ← **선택**. 학습 기간 평균 ATR_pct 근처
- (나) 0.003 (보수적)
- (다) 0.010 (느슨)

#### M-Q 추가 default 결정 (BP-2-3 구현 중)

| ID | 사안 | 채택 default |
|---|---|---|
| M | OOS prediction capture 방식 | `evaluate_strategies_on_bar`에서 `generate_signal` 결과 직후 push (옵션 B, plugin 무영향) |
| N | HOLD 신호 처리 | buffer에 push 안 함 (방향 예측 없음) |
| O | 다중 strategy buffer | `dict[str, deque]` strategy_name별 분리 |
| P | min_acc_threshold default | 0.50 (보수적 baseline) |
| Q | 알림 cooldown | 10봉 (노이즈 방지) |

### 4.4 검증 기준 + 결과

| 작업 | 검증 방식 | 결과 |
|---|---|---|
| Live OOS monitoring | 단위 테스트 12건 + paper 시연 | ✅ 단위 테스트 통과. paper 1주일 시연은 BL-2 §4.2.1로 carry (default 비활성이라 라이브 영향 0) |
| 동적 사이징 | 백테 회귀 (ml_xgboost 2025-01~03) | baseline 672.61% / MDD 6.28% → 동적 612.28% / MDD 6.23%. 거래 수·승률 동일. **수익률 -9.0%, MDD ≈ 동일** → 변동성 안정 regime에선 효과 미미. BL-1 §3.2.1 regime-matched에서 본격 평가 (I-BP003 등록) |
| 백테 warmup | 단위 테스트 3건 + 회귀 백테 | ✅ ml_xgboost 2025-01-01 첫 거래 시점 09:30 UTC (33일 → 9.5시간) |

전체 회귀: 240 → **264 passed** (신규 24건, 회귀 0건)

---

## 5. Phase BP-3: 모델 결합

### 5.1 목적

Phase E-2-3에서 검증된 calibration + Phase E의 5 모델 결과 차이를 활용하여 단일 모델 한계 돌파.

### 5.2 작업 항목

#### 5.2.1 Confidence calibration 본격 적용 (I-B009 후속)

**현재 상태**: 인프라 완비 (`src/ml/calibration.py`, `scripts/calibrate_models.py`, plugin 4개 분기)

**작업** (BP-1 스킵 반영):
1. ~~v005 (BP-1 학습) 모델에 calibrator 학습 + 저장~~ → **기존 v001-v004 모델 calibrator는 PATH_B_ML_STRATEGY E-2-3에서 학습 완료**. 추가 학습 불필요.
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

**작업** (BP-1 스킵 반영, v005 번호는 BL-1 BP-1 재진입용으로 예약):
1. `generate_triple_barrier_labels(df, upper_barrier_pct, lower_barrier_pct, time_barrier_bars)` 신규
2. SHORT/HOLD/LONG → SHORT/HOLD/LONG/timeout (4-class) 또는 binary로 단순화
3. Label noise 감소 → OOS Acc 64% 한계 돌파 시도
4. 5개 모델 **v006** 학습 (81 피처 + Triple-barrier label, v005는 BL-1 BP-1 재진입용으로 비워 둠) → v001 대비 비교

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
| 2026-05-03 | Phase BP-1: 데이터 확장 | **스킵 (사안 H)** | — | BL-1 §3.2.6에서 재진입 여부 확인 (조건부). I-BP001은 BL-1/BL 종착까지 carry-over |
| 2026-05-03 | **Phase BP-2: 운영 인프라** | **✅ 완료** | (이번 커밋) | 운영 인프라 3건. 회귀 240→264 |
| 2026-05-03 | └ BP-2-1 백테 warmup (I-BP002 fix) | 완료 | (이번 커밋) | 단위 3건 + 회귀 백테 (첫 거래 33일→9.5시간) |
| 2026-05-03 | └ BP-2-2 동적 사이징 (사안 J/K/L 가) | 완료 | (이번 커밋) | enabled=false default. 회귀 백테 -9% pnl / MDD ≈ 동일 → I-BP003 등록 |
| 2026-05-03 | └ BP-2-3 Live OOS monitor (사안 D 나) | 완료 | (이번 커밋) | enabled=false default. 단위 12건 |
| (대기) | Phase BP-3: 모델 결합 | 대기 | — | Calibration 본격 적용(v001-v004 기반) + Ensemble + Triple-barrier(v006) |

---

## 7. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BP001 | Phase B-3 / E-2-2-FIX (carry from PATH_B_ML_STRATEGY §14.2) | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환 (line 57-64). `engine_base.close_position`에 `funding_fee=0.0` 하드코드. config의 `accounting.funding_enabled=true`이지만 실제 백테에 미반영. 라이브 LONG 포지션 평균 -0.01%/8h 펀딩률 추가 비용 발생 가능 (보유 시간 짧으면 미미할 수 있음) | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | ~~Phase BP-1~~ → **BL-1 §3.2.6** (조건부, 미진입 시 BL 종착까지 carry-over) | 미해결 — BL-1 §3.2.6 BP-1 재진입 결정 시 `estimate_funding` 구현 + 라이브-백테 일관성 검증 + 정확 baseline 재실행. 미진입 시 보유 시간 평균 1.6시간으로 영향 미미 추정 유지, 라이브 운영 전 재검증 |
| I-BP002 | Phase E-2-2-OPT 분석 (carry from §14.2) | `BacktestEngine._load_candles`가 `[start, end]`만 로드 → 4h EMA200 800시간(33일) 워밍업으로 백테 초반 무거래 (1년 OOS 9% / 4년 OOS 2.3% 손실). config의 `data.history_bars`(300) 백테 엔진에서 미참조. 라이브에서는 history_bars=300으로 자연 처리됨 | src/backtest/engine.py:_load_candles | **Phase BP-2-1** | **✅ 해결 (BP-2-1)** — `_load_candles`가 `data.history_bars` 활용. ml_xgboost 2025-01~03 회귀 백테에서 첫 거래 시점 2025-01-01 09:30 UTC 확인. 단위 3건 + 회귀 백테로 검증 |
| I-BP003 | Phase BP-2-2 회귀 백테 분석 | 동적 사이징 (사안 J 가 축소만)이 변동성 안정 regime에서 수익률만 깎고 MDD 보호 효과 미미 (ml_xgboost 2025-01~03: 수익률 -9.0%, MDD -0.05%p ≈ 동일). 추정 원인: 변동성 spike 부족 + 모델의 ATR 기반 SL이 이미 변동성 적응적 | src/risk/manager.py + src/core/engine_base.py | BL-1 §3.2.1 (Regime-matched 백테) | 미해결 — default `enabled: false` 유지. BL-1 regime-matched 백테 (2018 약세 / 2020-2021 강세)에서 효과 재평가. target_atr_pct 튜닝 또는 사안 J (나) 양방향 재고 가능성 |

신규 이슈는 작업 진행 시 I-BP004~ 형태로 추가.

---

## 8. 후속 로드맵

PATH_B_PRODUCTION 종착 후 → `PATH_B_LIVE_TRADING_260503.md` Phase BL-1 (Regime + 추가 검증 + **BP-1 재진입 여부 확인** [조건부, 마지막]) → BL-2 (실거래 전환).
