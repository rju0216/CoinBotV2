# PATH_B_LIVE_EXTENSION (라이브 운영 확장)

작성 시점: 2026-05-05 (BL-1 종착 + BL-2 재구성 직후)
선행: `PATH_B_LIVE_TRADING_260503.md` (BL-2 종착 — 라이브 거래 시작)
선행 (root): `PATH_B_PRODUCTION_260503.md`, `PATH_B_ML_STRATEGY_260425.md`

---

## 0. 문서 목적

PATH_B 라이브 거래 시작 (BL-2 종착) 후 **운영 확장 작업**을 다루는 별도 PATH.

PATH_B_LIVE_TRADING이 "라이브 진입 + 안정화"에 집중하기 위해 BL-1 종착 시점에 분리됨 (사용자 결정 2026-05-05). 본 문서의 작업들은 라이브 거래 자체에는 필요 없으나, 라이브 안정화 후 robust성·확장성·정확도 향상을 위한 영역.

---

## 1. 계승 사항 (BLE-1 진입 시점 가정)

### 1.1 BL-2 종착 시점 인프라
- 라이브 운영 중인 모델: dl_transformer v007 + isotonic (또는 ensemble) — BL-2-3 paper 비교 후 결정
- Fail-safe: notifier (텔레그램/이메일) + Circuit breaker (BL-2-1)
- 호가창 인프라: paper_executor book depth + OKX snapshot 누적 (BL-2-2)
- I-BP001 funding_fee 모니터링 데이터 (BL-2-3 paper 누적)
- OOS monitor (BP-2-3) 라이브 적중률 추적 데이터

### 1.2 잠재 이슈 carry-over
- **I-BP001** (funding_fee 백테 미반영): BL-2 paper trading에서 영향 정량 측정. 라이브 자체엔 영향 없음 (`_close_with_funding`로 자동 반영). 단 백테 결과 정확도를 위해 fix 검토는 BLE-4 (BP-1 데이터 확장) 시점
- **I-BL002** (ensemble walkforward 미수행): BLE-2에서 해결
- 추가 carry는 BL-2 종착 시 갱신

### 1.3 설계 원칙
PATH_B_ML_STRATEGY §1 + PATH_B_PRODUCTION §1.3 + PATH_B_LIVE_TRADING §1.3 그대로 — CLAUDE.md 협업 규칙 1~10 적용.

### 1.4 운영 권장 (BL-2 종착 시점, 추후 갱신)
BL-2-3 paper trading 결과 기반 결정 (R'' 사안). 본 문서는 진입 시점에 운영 권장 갱신.

---

## 2. 전체 구조 (Phase BLE-1, BLE-2, BLE-3, BLE-4)

```
Phase BLE-1 (다중 거래소)
├─ src/execution/binance_executor.py 신규 (LiveExecutor 패턴)
├─ Broker 분기 (config.exchange.name으로 OKX/Binance 자동 선택)
├─ 거래소별 fee/funding 차이 처리
└─ Multi-exchange 백테 검증

Phase BLE-2 (Ensemble walkforward — I-BL002 해결)
├─ ensemble.py + evaluate_models.py 확장 — fold 모델별 sub_params model_path 오버라이드
├─ ensemble walkforward 평가 4 모델 26 folds 통합
└─ 운영 권장 ensemble vs dl_transformer 학술 비교 정량

Phase BLE-3 (Survivorship bias — 다른 코인)
├─ ETH/USDT, SOL/USDT 등 다른 거래쌍 학습
├─ BTC robust성과 비교
└─ 학술적 한계 (limitations) 문서화

Phase BLE-4 (BP-1 데이터 확장 carry, 조건부)
├─ 펀딩률·OI 데이터 수집 인프라 (ccxt OKX API)
├─ 2020 이전 BTC 현물 데이터 수집 (Bitstamp 등)
├─ I-BP001 funding_fee 백테 정확 처리
└─ v005 학습 (8년 + 펀딩률·OI 피처)
```

### 2.1 진행 우선순위 (사용자 자유 결정)

각 step 독립적이라 순서 자유. 운영 권장 우선순위:

1. **BLE-2 (Ensemble walkforward)** ← 학술 robust성 보강. 코드 작업만 (사용자 GPU 없음). 최소 작업
2. **BLE-1 (다중 거래소)** — OKX 의존성 분산. 자금 안정성 강화
3. **BLE-4 (BP-1 데이터 carry)** — I-BP001 정확 처리 + 모델 정보량 확장. 데이터 인프라 준비 시
4. **BLE-3 (Survivorship)** — 다른 코인 학습. 가장 큰 작업 (사용자 GPU + 데이터)

또는 사용자 우선순위 (예: 자금 안전 우선이면 BLE-1)에 따라 자유 진행.

---

## 3. Phase BLE-1: 다중 거래소 (Binance + OKX)

### 3.1 목적
OKX 단일 거래소 의존성 분산. Binance 백업 또는 동시 운영으로 거래소 다운/제재 등 리스크 완화.

### 3.2 작업 항목

#### 3.2.1 binance_executor.py 신규
**현재 상태**: `src/execution/live_executor.py` (OKX ccxt 사용). Broker abstraction 있음.

**작업**:
1. `src/execution/binance_executor.py` 신규 — LiveExecutor 패턴 그대로
2. ccxt binance 사용 (perpetual futures)
3. `_retry_api` 동일 사용

#### 3.2.2 Broker 분기
- `src/execution/broker.py` — `config.exchange.name`으로 "okx"/"binance" 분기
- 기존 OKX 분기 + Binance 추가

#### 3.2.3 거래소별 fee/funding 차이
- FeeModel 분기 또는 거래소별 config 섹션 (`exchange.taker_fee_pct` 별도)
- funding rate 주기 차이 (OKX 8h, Binance 8h 동일하지만 시점 다름) 검증

#### 3.2.4 Multi-exchange 백테
- 동일 신호에 두 거래소 데이터로 동시 백테 → 결과 차이 정량

### 3.3 결정 사안 (BLE-1 진입 시)

#### F. 다중 거래소 우선순위
- (가) OKX 메인 + Binance 백업 (OKX 다운 시 fallback)
- (나) 동시 운영 (양쪽 자금 분산 — 자금 50:50 또는 사용자 비율)

### 3.4 검증 기준
- binance_executor 단위 테스트 (LiveExecutor 패턴 동일)
- Multi-exchange 백테: OKX 결과 vs Binance 결과 차이 (수수료/슬리피지 영향)
- 실 라이브: 동일 신호에 두 거래소 체결 결과 비교

---

## 4. Phase BLE-2: Ensemble walkforward (I-BL002 해결)

### 4.1 목적
PATH_B_LIVE_TRADING BL-1-3 단일 모델 walkforward (4 모델 26/26 positive) 검증 시 ensemble 평가는 누락됐음 (I-BL002). ensemble 자체가 fold 모델별 sub_params model_path 오버라이드 메커니즘 부재.

운영 권장 ensemble (BP-3-2 단일 OOS 1118%) vs dl_transformer (walkforward + Regime 일관 1위)의 학술 비교 정량 보강.

### 4.2 작업 항목

#### 4.2.1 ensemble plugin 확장
- `src/strategy/plugins/ensemble.py`에 fold-aware sub-plugin 로드 옵션
- 또는 ensemble.yaml의 sub_params model_path를 동적으로 fold_dir로 오버라이드하는 evaluate_models 메커니즘

#### 4.2.2 evaluate_models walkforward — ensemble 지원
- `--mode walkforward --strategy ensemble` 시 4 sub-model 각각의 fold 모델로 ensemble 추론
- 각 sub-model fold_dir 매핑 (4 sub-model × 26 folds)
- aggregate_walkforward_results 그대로 활용

#### 4.2.3 walkforward 평가 + 비교
- Ensemble walkforward 결과 (26 folds 통합)
- 단일 모델 walkforward (BL-1-3) 결과와 비교
- 운영 권장 정정/강화

### 4.3 결정 사안 (BLE-2 진입 시)

#### S''. I-BL002 처리 시점
- (가) BLE-1 (다중 거래소) 후
- (나) BLE-1 전 (학술 robust성 우선)
- (다) BL-2 paper 병행 (별도 GPU 사용, 시간 절약)

### 4.4 검증 기준
- Ensemble walkforward 26/26 positive fold 여부
- dl_transformer 단독 vs ensemble 비교 (Calmar/MDD/PF)
- 운영 권장 정정 (ensemble이 walkforward에서도 우수하면 1순위 강화)

---

## 5. Phase BLE-3: Survivorship bias 점검

### 5.1 목적
학습 데이터 BTC만 사용. 다른 코인 (ETH, SOL 등)에서도 모델 robust한지 검증 → 학술적 한계 (Survivorship bias) 처리.

### 5.2 작업 항목

#### 5.2.1 다른 코인 데이터 수집
- ETH/USDT:USDT, SOL/USDT:USDT 캔들 다운로드 (OKX ccxt)
- 기간: BTC와 동일 (2020-2024 또는 가능한 만큼)

#### 5.2.2 다른 코인 학습
- 동일 4 모델 (lightgbm/xgboost/lstm/transformer)을 ETH/SOL로 학습
- features.py 그대로 사용 (가격 절대값 의존 없음 — returns/ratio 위주)
- 사용자 GPU 작업 (8 모델 × 학습 시간)

#### 5.2.3 평가
- BTC vs ETH vs SOL Sharpe/Calmar/PF 비교
- regime robust성 확인 (강세/약세/횡보)

#### 5.2.4 학술적 한계 문서
- `docs/02_Limitations/SURVIVORSHIP_<날짜>.md` 신규
- BTC 결과의 일반화 가능성 + 다른 코인 결과 + 한계 명시

### 5.3 결정 사안 (BLE-3 진입 시)
- 어떤 코인 학습? — ETH/SOL 외 추가 (예: BNB, XRP)?
- 학습 데이터 기간 — BTC와 동일 vs 코인별 가용 데이터

### 5.4 검증 기준
- 다른 코인에서도 모델 양수 수익 + MDD 통제
- BTC vs 다른 코인 결과 차이 정량 (수수료/유동성 영향)

---

## 6. Phase BLE-4: BP-1 데이터 확장 carry (조건부)

### 6.1 목적
PATH_B_PRODUCTION에서 사안 H로 BP-1 스킵 + BL-1-5에서 미진입 확정. **데이터 수집 인프라가 준비된 시점에 재방문**.

### 6.2 진입 조건
다음 중 하나 충족 시:
1. ccxt OKX 펀딩률·OI 자동 수집 인프라 준비
2. 2020 이전 BTC 외부 거래소 (Bitstamp 등) 수집 인프라 준비
3. 라이브 운영에서 I-BP001 영향이 명확히 무시 불가 판명

### 6.3 작업 항목 (진입 결정 시)
PATH_B_PRODUCTION §3 그대로 활용 — 펀딩률·OI 수집 + 피처 추가 + I-BP001 fix + 2020 이전 데이터.

### 6.4 v005 모델 학습
- v005는 BP-1 재진입 예약 번호 (BP-3, BL-1-5에서 영구 비워둠 정책 유지)
- BLE-4 진입 시 v005 학습 (8년 데이터 + 펀딩률·OI 피처)
- BP-3 calibration + BL-1-3 walkforward 동일 인프라 활용 (--save-all-folds)

### 6.5 검증 기준 (PATH_B_PRODUCTION §3.4 그대로)
- 펀딩률 적용 전후 30 specs 결과 비교
- 신규 피처 효과: v005 (85 피처) vs v007/v001 (81 피처)
- I-BP001 라이브-백테 일관성 검증

---

## 7. 진행 기록 (Phase BLE-1/2/3/4)

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| (대기) | Phase BLE-1: 다중 거래소 | 대기 | — | binance_executor + Broker 분기 |
| (대기) | Phase BLE-2: Ensemble walkforward (I-BL002) | 대기 | — | 학술 robust성 보강 |
| (대기) | Phase BLE-3: Survivorship | 대기 | — | 다른 코인 학습 + 비교 |
| (조건부) | Phase BLE-4: BP-1 데이터 carry | 조건부 | — | 데이터 인프라 준비 시 |

---

## 8. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BL002 | BL-1-3 종착 (PATH_B_LIVE_TRADING §6) | Ensemble walkforward 평가 미수행. ensemble.py가 fold 모델별 sub_params 오버라이드 메커니즘 부재 | src/strategy/plugins/ensemble.py + scripts/evaluate_models.py | **BLE-2** | 미해결 — 단일 모델 walkforward 모두 robust 검증되어 우선순위 낮음 |

PATH_B_LIVE_TRADING §6의 I-BP001은 BL-2 §4.2.3 paper trading 모니터링 + BLE-4 (조건부 fix) 영역.

신규 carry-over 후보 ID는 I-BLE001~ 형태로 등록.

---

## 9. 종착 후

PATH_B_LIVE_EXTENSION 모든 step 종착 시 "경로 B" 모든 작업 완료. 그 후:
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도) — 별도 PATH 가능
- 새 모델 paradigm 탐색 (LLM 기반 등) — 새 PATH 시작 가능
