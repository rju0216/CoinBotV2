# PATH_B_LIVE_TRADING (경로 B 실거래 전환 단계)

작성 시점: 2026-05-03 (Phase E 종착 시점)
선행: `PATH_B_PRODUCTION_260503.md` (Phase BP-1/2/3 — 데이터 확장 + 운영 인프라 + 모델 결합)
선행 (root): `PATH_B_ML_STRATEGY_260425.md` (Phase 0 ~ E-2-4 완료, commit `74d0c96`)

---

## 0. 문서 목적

Phase BP 종착 후 라이브 실거래 전환을 위한 **Regime 검증 + 학술 검증 + 실거래 전환** 단계. PATH_B_PRODUCTION에서 확장된 데이터·인프라·모델 결합을 기반으로, 실 거래소에서 안전하게 운영 가능한 시스템 완성.

---

## 1. PATH_B_PRODUCTION 계승 사항

### 1.1 가정 상태 (BL-1 진입 시점)

PATH_B_PRODUCTION 종착 시점에 다음 인프라 완비 가정:
- ~~펀딩률·OI 피처 통합 (BP-1)~~ → **BP-1 스킵 (사안 H, PATH_B_PRODUCTION §3.3)**. BL-1 §3.2.6에서 재진입 여부 확인.
- ~~I-BP001 (funding_fee 백테)~~ → **carry-over** (BL-1 §3.2.6 또는 BL 종착까지). I-BP002 (백테 warmup) fix는 BP-2에서 완료.
- Live OOS monitoring + 동적 사이징 + 백테 warmup 자동화 (BP-2)
- Calibration 본격 적용 (config 기본 isotonic) + Ensemble plugin (BP-3)
- 5개 모델 v006 (81 피처 + BP-3 Triple-barrier label). v005 번호는 BP-1 재진입 시 사용 예약.

### 1.2 잠재 이슈 트래커 carry-over

PATH_B_PRODUCTION 종착 시점 carry-over:
- **I-BP001** (funding_fee 백테 미반영): BL-1 §3.2.6 BP-1 재진입 결정 시 처리. 미진입 시 BL 종착까지 carry-over (보유 시간 평균 1.6시간으로 영향 미미 추정 — 라이브 운영 전 재검증).

추가 carry-over는 BP-2/BP-3 종착 시 갱신.

### 1.3 설계 원칙

PATH_B_ML_STRATEGY §1 + PATH_B_PRODUCTION §1.3 그대로 — CLAUDE.md 협업 규칙 1~10 적용.

### 1.4 운영 권장 (Phase E 종착 시점 기준, BP에서 갱신될 수 있음)

- `ml_lightgbm` 또는 `ml_xgboost` + `calibration_method=isotonic`
- BP-3 ensemble 결과에 따라 단일 모델 또는 ensemble 채택

---

## 2. 전체 구조 (Phase BL-1, BL-2)

```
Phase BL-1 (Regime + 추가 검증)
├─ Regime-matched 백테 (사용자 제안: 학습 2022~2024 / 백테 2020-2021)
├─ Walk-forward 통합 OOS
├─ Lookahead bias 추가 점검
├─ 호가창/유동성 모델링 정교화
├─ Multi-hypothesis 보정 (Bonferroni / FDR)
└─ (조건부, 마지막) BP-1 재진입 여부 확인 — 데이터 수집 인프라 준비 시

Phase BL-2 (실거래 전환)
├─ Paper → 소액 실거래 점진 전환
├─ 다중 거래소 (OKX + Binance)
├─ Survivorship bias 점검
└─ 거래소 다운/API 지연 대응 시뮬레이션
```

### 2.1 진행 우선순위 (사안 C 결정 — PATH_B_ML_STRATEGY phase_b_followup_plan.md 인용)

> ⚠️ **알파벳 순서 아님 — 의존성·영향력 기반 순서**

PATH_B_PRODUCTION 종착 후:
1. **BL-1 (Regime + 검증)** ← 학술 검증 후 실거래 가능
   - BL-1 내부 순서: Regime/WF/Lookahead/호가창/Multi-hypothesis (5개 작업) → **BP-1 재진입 여부 확인 (마지막, 조건부)**
2. **BL-2 (실거래 전환)** ← 마지막

---

## 3. Phase BL-1: Regime + 추가 검증

### 3.1 목적

E-2-4의 30 specs baseline 결과는 2020-2025 데이터로 **단일 거시 환경**에서 학습/평가됨. 실거래 전환 전에:
- 다른 regime (강세/약세/횡보)에서도 robust한지 검증
- Walk-forward 통합 OOS로 가장 엄밀한 평가
- Lookahead/호가창/multi-hypothesis 등 학술적 의심 점검

### 3.2 작업 항목

#### 3.2.1 Regime-matched 백테 (사용자 제안)

**가설**: 모델이 2020-2024 학습이라 2025 OOS 결과가 좋아도, 2017-2019 (다른 regime)에서는 다를 수 있음

**작업**:
1. **학습 2022~2024 (약세+회복)** → **백테 2020-2021 (강세 bull)**
   - 5 모델 (v001-v004 + BP-3 v006, BP-1 재진입 시 v005 추가) × 1 분할 = 5+ 백테
   - 비교: 같은 모델의 2025 OOS 결과 vs 2020-2021 OOS 결과
2. (옵션) HMM/stylized facts로 regime 식별 → 현재 시장과 비슷한 과거 시기 자동 매칭
3. 결과: regime 변화에 모델이 얼마나 robust한지 정량

#### 3.2.2 Walk-forward 통합 OOS

**현재 상태** (코드 점검 완료):
- `train_*.py` 5개가 walk-forward 26 folds 학습하지만 **마지막 fold 모델만 저장** (best_model = model 마지막 fold)
- 26 folds OOS predictions는 메모리에서 사용 후 폐기

**작업**:
1. `train_*.py` 5개 수정 — 모든 fold 모델 저장 (`models/<type>/v00X/folds/fold_<id>/`)
2. 신규 백테 모드: `evaluate_models.py --mode walkforward`
   - 각 fold의 OOS 기간에 해당 fold 모델 사용
   - 26 folds OOS 모두 통합 → 통합 OOS 백테 결과
3. 단일 fold 평가보다 엄밀 (가장 큰 학술적 검증 표준)

#### 3.2.3 Lookahead bias 추가 점검

**현재 상태**: I-B007 (E-2-1에서 해결) — `_slice_candles(ts)`를 ts 미만으로, current_price를 open으로 변경. 단 다른 경로 가능성 점검 필요.

**작업**:
1. **Indicator forward-bias 점검** — pandas_ta 라이브러리 검증
   - 매 indicator (EMA, RSI, MACD, ADX, ...) 결과의 행 i가 행 i+1, i+2, ... 의 데이터를 사용하지 않는지
   - 합성 데이터로 단위 테스트 (현재 시점 이후 데이터 변경 시 행 i 결과 불변 확인)
2. **Walk-forward embargo 점검** — train 끝 horizon개 행 제거(I-B005 해결)는 충분한지
   - horizon=10 (15분 × 10 = 2.5시간) embargo로 라벨 누설 차단
   - 더 보수적: horizon=20-30
3. **데이터 수집 자체의 누설 점검** — OKX `fetch_ohlcv` 응답이 미완성 봉(현재 진행 중) 포함하는지

#### 3.2.4 호가창/유동성 모델링 정교화

**현재 상태**: `paper_executor`가 open 가격 즉시 체결 가정 (slippage_pct로만 단순 비용)

**작업**:
1. OKX 호가창 snapshot 수집 (별도 인프라)
2. 사이즈 대비 호가창 깊이로 실효 체결 가격 산정
   - 작은 size: 첫 호가 즉시 체결
   - 큰 size: 호가 여러 단계 침투 → market impact
3. `paper_executor.open_position` 확장 — book depth 인자 추가
4. 백테 시 fee_model.slippage_pct 대신 실 호가창 시뮬

#### 3.2.5 Multi-hypothesis 보정

**현재 상태**: E-2-4에서 단일 비교 Bootstrap p-value 계산. 단 30 specs + sensitivity 40 + calibration 8 = 78+ 비교 → 다중 가설 문제

**작업**:
1. `src/ml/metrics_extended.py`에 helper 추가:
   - `bonferroni_correction(p_values, alpha=0.05)` — α/N
   - `fdr_correction(p_values, alpha=0.05)` — Benjamini-Hochberg
2. `analyze_results.py`의 bootstrap_pvalues.csv에 보정 p-value 컬럼 추가
3. Phase E-2-4 결과 재평가:
   - "p<0.0001로 PPO vs 다른 모델 차이 명확" 결론이 다중 보정 후에도 유지되는지
   - "GBDT 두 모델 동등 p=0.61" 결론은 변동 없음 (대각선)

#### 3.2.6 BP-1 재진입 여부 확인 (조건부, 마지막)

**현재 상태 (2026-05-03)**: PATH_B_PRODUCTION 사안 H로 BP-1 전체 스킵. 데이터 수집 인프라 미준비.

**진입 조건** (다음 중 하나 충족 시):
1. 펀딩률·OI 데이터 수집 인프라 준비 (ccxt OKX API + 저장 파이프라인)
2. 2020 이전 BTC 데이터 수집 인프라 준비 (Bitstamp 등)
3. BL-1 5개 검증 결과 I-BP001 영향이 무시 불가로 판명

**작업 (진입 결정 시)**: PATH_B_PRODUCTION §3 전체 참조 — 펀딩률·OI 수집 + 피처 추가 + I-BP001 fix + 2020 이전 데이터. PATH_B_PRODUCTION §3.3 결정 사안 A/B/C는 본 단계에서 결정.

**미진입 결정 시**:
- I-BP001은 BL 종착까지 carry-over (영향 미미 추정 유지)
- v005 번호는 비워 둠
- 라이브 운영 시 펀딩률 실 비용 모니터링 (BL-2 §4.2.1 paper trading + 소액 실거래 단계)
- 전체 경로 B 종착 후 별도 PATH로 데이터 확장 재방문 가능

**검증 기준 (진입 시)**: PATH_B_PRODUCTION §3.4 그대로

### 3.3 결정 사안 (BL-1 진입 시 결정)

#### A. Regime-matched 백테 — 학습/백테 기간
- (가) 학습 2022-2024 / 백테 2020-2021 (사용자 제안, 강세 regime 검증)
- (나) HMM 기반 자동 매칭 (정교, 추가 인프라)

#### B. Walk-forward 통합 OOS — fold 모델 저장 시점
- (가) 모든 train_*.py에 옵션 추가 (사용자 명시 시 저장)
- (나) 별도 train_walkforward_*.py 신규

#### C. 호가창 모델링 — 데이터 수집 시기
- (가) BL-1 진입 시 즉시 수집 (수 주 소요)
- (나) BL-2 진입 직전 수집 (paper trading 시 함께)

#### D. Multi-hypothesis 보정 알고리즘
- (가) Bonferroni만 (보수적)
- (나) FDR만 (덜 보수적, 일반적)
- (다) 둘 다 비교

### 3.4 검증 기준

- Regime-matched 백테 결과: 2020-2021 OOS vs 2025 OOS 차이 정량
- Walk-forward 통합 OOS: 30 specs vs 26 folds 통합 결과 비교
- Lookahead 추가 점검: 새 발견 0건이 이상적
- Multi-hypothesis 보정 후 유의 비교 수 변화

---

## 4. Phase BL-2: 실거래 전환

### 4.1 목적

학술 검증 완료된 모델을 실 거래소에서 점진적으로 운영. **자금 보호 + 안전망 우선**.

### 4.2 작업 항목

#### 4.2.1 Paper → 소액 실거래 점진 전환

**현재 상태**: `src/live/engine.py` `CoreEngine` 존재. `LiveExecutor` (OKX ccxt 사용) 구현되어 있음.

**작업**:
1. **Paper trading 1주일** — 라이브 가격 + paper executor로 모델 검증
2. **소액 실거래 1주일** — 운영 자금의 1-5%만 사용
3. **점진 확장** — 결과 모니터링 후 자금 비중 증가
4. Fail-safe: 일일 손실 한도 + 자동 정지 + 텔레그램/이메일 알림
5. 모델 1개로 시작 (Phase E-2-4 권장 ml_lightgbm + isotonic) → 점진적으로 ensemble 확장

#### 4.2.2 다중 거래소 (Binance + OKX)

**현재 상태**: ccxt 4.4+ 의존성, OKX만 구현. Broker abstraction 있음.

**작업**:
1. `src/execution/binance_executor.py` 신규 — LiveExecutor 패턴
2. `Broker` 분기 — config의 `exchange.name`으로 OKX/Binance 자동 선택
3. 거래소별 fee/funding 차이 → FeeModel 분기 또는 거래소별 config
4. Multi-exchange 백테 (다른 거래소 데이터로 동시 검증)

#### 4.2.3 Survivorship bias 점검

**작업**:
1. 학습 데이터 BTC만 사용 — 다른 코인 (ETH, SOL 등)에서도 robust한지
2. 거래쌍 확장 가능성 검토
3. 학술적 보고 작성 (limitations 문서)

#### 4.2.4 거래소 다운/API 지연 시뮬레이션

**현재 상태**: `LiveExecutor`에 retry 로직 일부 있음 (`_retry_api`, MAX_RETRIES=3)

**작업**:
1. **API rate limit 시뮬레이션** — 단위 테스트
2. **응답 지연 1-10초 시뮬레이션** — 신호 발생 ~ 실제 진입 사이 지연 시 가격 변동 영향
3. **Order rejection retry 로직 강화**
4. **Circuit breaker** — N회 연속 실패 시 자동 정지 + 알림
5. **거래소 다운 시 fallback** — 다중 거래소 활용 (BL-2.2 연계)

### 4.3 결정 사안 (BL-2 진입 시 결정)

#### E. 소액 실거래 시작 자금 비중
- (가) 1% (가장 보수적)
- (나) 5% (사용자 위험 선호)

#### F. 다중 거래소 우선순위
- (가) OKX 메인 + Binance 백업
- (나) 동시 운영 (양쪽 자금 분산)

#### G. Circuit breaker 자동 정지 임계값
- (가) 연속 실패 3회 (보수적)
- (나) 연속 실패 5회

### 4.4 검증 기준

- Paper trading 1주일: 백테와 실 결과 차이 (수익률, 거래 수, fees, 슬리피지)
- 소액 실거래 1주일: 자금 안전 + 모델 신호 정확
- Multi-exchange: 동일 신호에 두 거래소 체결 결과 비교
- Circuit breaker: 인위적 실패 주입 시 정지/알림 확인

---

## 5. 진행 기록 (Phase BL-1/2)

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| (대기) | Phase BL-1: Regime + 추가 검증 | 대기 | — | Regime-matched + Walk-forward 통합 OOS + Lookahead 추가 점검 + 호가창 + Multi-hypothesis + (조건부) BP-1 재진입 여부 확인 |
| (대기) | Phase BL-2: 실거래 전환 | 대기 | — | Paper → 소액 실거래 + 다중 거래소 + Survivorship + Circuit breaker |

---

## 6. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BP001 | PATH_B_PRODUCTION carry-over (사안 H로 BP-1 스킵) | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환. `engine_base.close_position`에 `funding_fee=0.0` 하드코드. 백테에 펀딩률 미반영. (PATH_B_PRODUCTION §7 동일) | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | BL-1 §3.2.6 (재진입 시) / 미진입 시 BL 종착까지 carry-over | 미해결 — 보유 시간 평균 1.6시간으로 영향 미미 추정. 라이브 운영 전 재검증 필요 |

PATH_B_PRODUCTION 종착 시 신규 carry-over 후보 ID는 I-BL001~ 형태로 등록.

---

## 7. 종착 후

PATH_B_LIVE_TRADING BL-2 종착으로 "경로 B" 모든 작업 완료. 라이브 운영 안정화 후:
- 다른 거래쌍 확장 (Survivorship bias §4.2.3 후속)
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도)
- 새 모델 paradigm 탐색 (LLM 기반 등 — 새 PATH 시작 가능)
