# PATH_B_LIVE_EXTENSION (라이브 운영 확장)

작성 시점: 원작 2026-05-05 (BL-1 종착 + BL-2 재구성 직후) / 갱신 2026-05-09 (PATH_B_LIVE_TRADING 종착 반영, BLE-5/6 신설) / 갱신 2026-05-10 (BLE-7 운영 모니터링 강화 신설)
선행: `PATH_B_LIVE_TRADING_260503.md` (✅ 종착 2026-05-09 — 라이브 거래 시작 본 목적 달성)
선행 (root): `PATH_B_PRODUCTION_260503.md`, `PATH_B_ML_STRATEGY_260425.md`

---

## 0. 문서 목적

PATH_B 라이브 거래 시작 (BL-2 종착) 후 **운영 확장 작업**을 다루는 별도 PATH.

PATH_B_LIVE_TRADING이 "라이브 진입 + 안정화"에 집중하기 위해 BL-1 종착 시점에 분리됨 (사용자 결정 2026-05-05). 본 문서의 작업들은 라이브 거래 자체에는 필요 없으나, 라이브 안정화 후 robust성·확장성·정확도 향상을 위한 영역.

---

## 1. 계승 사항 (BL-2 종착 2026-05-09 시점)

### 1.1 BL-2 종착 시점 인프라
- **라이브 운영 중인 모델: ensemble v010** (4 sub-plugin: ml_lightgbm + ml_xgboost + dl_lstm + dl_transformer + isotonic, 학습 cutoff 2026-04-01) — BL-2-4 라이브 거래 시작 시점 채택, Trade 4 (manual close 시연) e2e 검증 통과
- Fail-safe: notifier (텔레그램 plain text) + Circuit breaker + RiskManager EventBus (BL-2-1)
- 호가창 인프라: paper_executor book depth + OKX snapshot 누적 (BL-2-2)
- 외부 청산 자동 동기화 (`_sync_unexpected_close`, hotfix-K) + 재시작 복원 4 case + SL/TP conditional order 검증/재등록 (hotfix-H/I/K)
- daily_pnl 자동 reset (`maybe_reset_for_new_day`, hotfix-N) — 백테/페이퍼/라이브 일관 적용
- OOS monitor (BP-2-3) 라이브 적중률 추적 데이터
- 라이브 운영 펀딩률 모니터링 데이터 (`_close_with_funding`로 라이브 자동 반영, BLE-5 baseline)

### 1.2 잠재 이슈 carry-over (BL-2 종착 시점 갱신)
- **I-BP001** (funding_fee 백테 미반영): BLE-5 신설로 carry. 라이브 운영 1.6h 평균 보유 → 1건당 ~0.002% PnL 영향, 미미. 라이브 펀딩률 모니터링 데이터 누적 후 처리
- **I-BL002** (ensemble walkforward 미수행): BLE-2에서 해결 — 그대로 유지
- I-BL003~I-BL018 (BL-2-3/2-4 hotfix 18건): **모두 ✅ 해결** (PATH_B_LIVE_TRADING §6 참조). carry 0
- 신규 carry-over 후보 ID는 I-BLE001~ 형태로 등록

### 1.3 설계 원칙
PATH_B_ML_STRATEGY §1 + PATH_B_PRODUCTION §1.3 + PATH_B_LIVE_TRADING §1.3 그대로 — CLAUDE.md 협업 규칙 1~10 적용. 라이브-백테 일관성 (CLAUDE.md 핵심 설계 원칙 4) 유지.

### 1.4 운영 권장 (BL-2 종착 결과 반영)
- **모델: ensemble v010 채택 확정** (라이브 운영 중)
- 자금: 시작 자금 ~$3,370 유지, 점진 확장은 사용자 자율 (1-2주 안정 운영 후 결정)
- 라이브 모니터링: 사용자 직접 — POSITION/ACCOUNT 로그 (15m 주기) + 텔레그램 ENTRY/EXIT/Drawdown/Daily loss/Circuit breaker 알림
- 거래 패턴: 진입 시 SL/TP conditional order 거래소 등록 (안전장치) + 봉 마감 시 외부 청산 자동 sync

---

## 2. 전체 구조 (Phase BLE-1 ~ BLE-7)

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
├─ v005 학습 (8년 + 펀딩률·OI 피처)
└─ (I-BP001 fix는 BLE-5 분리)

Phase BLE-5 (I-BP001 funding_fee 백테 통합 — 신설)
├─ FeeModel.estimate_funding 구현 (ccxt OKX funding history fetch)
├─ engine_base.close_position에 funding_fee 통합 (백테에서도 적용)
├─ 라이브 펀딩률 모니터링 데이터 vs 백테 추정 비교
└─ 백테 결과 정확성 보강 (BLE-4 와 분리 가능, 펀딩률 데이터 베이스 공유)

Phase BLE-6 (라이브-백테 정합성 검증 — 신설)
├─ 라이브 누적 거래 vs 같은 기간 백테 결과 비교
├─ paper-실 거래 격차 정량 (BL-2-4 본 검증 항목 후속)
├─ 거래별 fee/slippage/funding 차이 분석
└─ baseline 정합성 리포트 (다른 BLE 진행 시 영향 측정 baseline)

Phase BLE-7 (운영 모니터링 강화 — 신설, 2026-05-10)
├─ BLE-7-1 콘솔/파일 로그 정보량 보강 (sub_probs / bar 컨텍스트 / 위험 한도 거리 / SL/TP 거리)
├─ BLE-7-2 텔레그램 알림 정보량 보강 (콘솔 로그와 일관 유지)
├─ (향후) BLE-7-3+ 일일 리포트 자동 발송, OOS Decay 자동 알람 임계 강화 등
└─ 라이브 운영 누적 시 의미 ↑ 영역 누적
```

### 2.1 진행 우선순위 (사용자 자유 결정)

각 step 독립적이라 순서 자유. 운영 권장 우선순위:

1. **BLE-7 (운영 모니터링 강화)** ← 라이브 운영 즉시 가치 ↑. 사용자 직접 모니터링 가시성 보강. 콘솔/파일 로그 (BLE-7-1) → 텔레그램 (BLE-7-2) 단계 진행
2. **BLE-2 (Ensemble walkforward)** ← 학술 robust성 보강. 코드 작업만 (사용자 GPU 없음). 최소 작업
3. **BLE-5 (I-BP001 백테 통합)** — 작은 hotfix. 백테 정확성 보강. 라이브 펀딩률 모니터링 데이터 누적 후
4. **BLE-1 (다중 거래소)** — OKX 의존성 분산. 자금 안정성 강화
5. **BLE-6 (라이브-백테 정합성 검증)** — 라이브 거래 누적 (~1개월 / 거래 ≥ 30건 권장) 후 baseline 의미 ↑. 다른 BLE 진행 시 회귀 영향 측정 baseline 으로 재활용 가능
6. **BLE-4 (BP-1 데이터 carry)** — 데이터 인프라 준비 시. v005 학습 (사용자 GPU)
7. **BLE-3 (Survivorship)** — 다른 코인 학습. 가장 큰 작업 (사용자 GPU + 데이터)

또는 사용자 우선순위 (예: 자금 안전 우선이면 BLE-1, 백테 정확성 우선이면 BLE-5)에 따라 자유 진행.

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
PATH_B_PRODUCTION에서 사안 H로 BP-1 스킵 + BL-1-5에서 미진입 확정. **데이터 수집 인프라가 준비된 시점에 재방문**. 모델 정보량 확장 (펀딩률·OI 피처 + 2020 이전 데이터).

(I-BP001 funding_fee 백테 통합 자체는 BLE-5 로 분리 — 펀딩률 데이터 베이스 공유, 작업 결이 다름)

### 6.2 진입 조건
다음 중 하나 충족 시:
1. ccxt OKX 펀딩률·OI 자동 수집 인프라 준비
2. 2020 이전 BTC 외부 거래소 (Bitstamp 등) 수집 인프라 준비

### 6.3 작업 항목 (진입 결정 시)
PATH_B_PRODUCTION §3 그대로 활용 — 펀딩률·OI 수집 + 피처 추가 + 2020 이전 데이터. (I-BP001 fix 부분은 BLE-5 별도 처리)

### 6.4 v005 모델 학습
- v005는 BP-1 재진입 예약 번호 (BP-3, BL-1-5에서 영구 비워둠 정책 유지)
- BLE-4 진입 시 v005 학습 (8년 데이터 + 펀딩률·OI 피처)
- BP-3 calibration + BL-1-3 walkforward 동일 인프라 활용 (--save-all-folds)

### 6.5 검증 기준 (PATH_B_PRODUCTION §3.4 그대로)
- 신규 피처 효과: v005 (85 피처) vs v007/v001 (81 피처)
- 2020 이전 데이터 추가 시 walkforward fold 수 증가 + robust성 비교

---

## 7. Phase BLE-5: I-BP001 funding_fee 백테 통합 (신설)

### 7.1 목적
`FeeModel.estimate_funding` 정의되어 있으나 `funding_enabled=True` 라도 0 반환 + `engine_base.close_position` 의 `funding_fee=0.0` 하드코드 → 백테에 펀딩률 미반영. PATH_B_LIVE_TRADING 종착 시 BLE-5 로 carry 결정 (보유 시간 평균 1.6h → 1건당 ~0.002% PnL 영향, 미미하지만 백테 정확성 보강).

라이브에선 `_close_with_funding` 으로 자동 반영되므로 영향 없음. 백테-라이브 일관성 (CLAUDE.md 핵심 설계 원칙 4) 차원에서 fix.

### 7.2 작업 항목

#### 7.2.1 FeeModel.estimate_funding 구현
- ccxt OKX `fetch_funding_rate_history(symbol, since, limit)` 활용
- 진입 시각 ~ 청산 시각 사이 funding rate event 합산
- 또는 캐시 layer 도입 (백테 다회 실행 시 fetch 비용 회피)

#### 7.2.2 engine_base.close_position에 funding_fee 통합
- 현재 `close_position(funding_fee=0.0 ...)` 하드코드를 caller 가 추정값 전달하도록
- 백테: 봉 마감 시점 funding rate 추정 (ccxt fetch 또는 사전 다운로드 캐시)
- 라이브: 기존 `_fetch_funding_since_entry` 그대로 (변경 없음)

#### 7.2.3 라이브 vs 백테 비교
- 라이브 펀딩률 모니터링 데이터 (BL-2-4 이후 누적) vs 백테 추정 비교
- 1건당 ~0.002% 영향 추정 검증

#### 7.2.4 BLE-4 와의 의존성
- BLE-4 의 펀딩률·OI 데이터 수집 인프라 활용 가능 (둘 다 펀딩률 데이터 기반)
- BLE-5 만 단독 진행 가능 (ccxt fetch_funding_rate_history 만 사용)
- BLE-4 와 묶음 진행 시 데이터 인프라 공유

### 7.3 결정 사안 (BLE-5 진입 시)

#### T'. fetch 전략
- (가) 매 백테 실행 시 ccxt fetch (느림, 정확)
- (나) 사전 다운로드 + 캐시 (빠름, 캐시 갱신 주기 결정 필요)
- (다) BLE-4 데이터 인프라와 묶어 진행 (BLE-4 와 함께)

### 7.4 검증 기준
- 펀딩률 적용 전후 백테 결과 비교 (PnL/MDD 차이)
- 라이브 누적 funding cost 측정값 vs 백테 추정값 일치성 (~$0.5 오차 허용)
- 회귀 테스트 466 pass 유지 (또는 신규 테스트로 fee_model 회귀 영역 강화)

---

## 8. Phase BLE-6: 라이브-백테 정합성 검증 (신설)

### 8.1 목적
BL-2-4 본 검증 항목 "paper-실 거래 격차 정량" 후속. 라이브 누적 거래와 같은 기간 백테 결과를 비교해 fee/slippage/funding 영향 정량 측정. **다른 BLE phase 진행 시 baseline 으로 재활용** (BLE-5 적용 효과 측정, BLE-1 다중 거래소 결과 비교 등).

### 8.2 진입 조건
- 라이브 거래 누적 ≥ ~30건 (통계적 의미)
- 라이브 운영 ≥ ~1개월 (다양한 시장 국면 카버)
- 또는 사용자 판단 (적은 거래수로도 정합성 점검 가능)

### 8.3 작업 항목

#### 8.3.1 라이브 거래 데이터 추출
- `data/coinbot_*.db` 의 trades 테이블 export
- 진입/청산 시각·가격·size·pnl·fee 등 추출

#### 8.3.2 같은 기간 백테 실행
- 라이브 운영 시작 시점 ~ 분석 시점 백테
- 동일 config (ensemble v010 + risk + accounting 동일)

#### 8.3.3 거래별 비교 분석
- trade-by-trade 매칭 (시각/방향 기준)
- entry_price / exit_price / pnl / fee 차이 정량
- 슬리피지 추정 (라이브 실거래 - 백테 추정 가격 차이)
- 외부 청산 (사용자 manual close 등) 분류별 영향

#### 8.3.4 정합성 리포트
- `docs/02_Limitations/LIVE_BACKTEST_PARITY_<날짜>.md` 신규
- 평균/중앙값 차이 + 표준편차 + outlier 분석
- baseline 으로 보존 (다른 BLE 변경 시 회귀 측정용)

### 8.4 결정 사안 (BLE-6 진입 시)

#### U'. 비교 metric
- (가) trade-by-trade (각 거래 1:1 매칭)
- (나) aggregate (전체 PnL/Sharpe/Calmar 비교만)
- (다) 둘 다

#### V'. baseline 보존 정책
- (가) 한 번 측정 후 정적 baseline
- (나) 주기적 갱신 (월 1회 등)

### 8.5 검증 기준
- trade-by-trade 매칭 ≥ 90% (시각 ±15분 + 방향 일치)
- 평균 PnL 차이 < 0.5% (절대값)
- 외부 청산·매칭 실패 case 모두 분류 확인

---

## 9. Phase BLE-7: 운영 모니터링 강화 (신설, 2026-05-10)

### 9.1 목적
사용자 라이브 모니터링 가시성 강화. BL-2-4 hotfix-K 의 e2e 검증 시 사용자가 ensemble probs 변동이 가격 변동에 비해 작아 보이는 현상 점검 중 발견 — 현재 로그가 sub-plugin 별 probs / bar 컨텍스트 / 위험 한도 거리 / SL/TP 거리 등 핵심 진단 정보를 제공 안 함.

이 phase 는 라이브 운영 자체에 필요 없는 *가시성* 영역이라 EXTENSION 분류. 단 라이브 직접 모니터링 시 즉시 가치 ↑ 라 1순위 (§2.1).

### 9.2 작업 항목

#### 9.2.1 BLE-7-1 콘솔/파일 로그 보강 (2026-05-10 진행)
[SIGNAL] / [ACCOUNT] / [POSITION] 3개 라인 정보량 강화. 새 라인 추가 없음 (가독성 유지).

추가 정보:
- **sub_probs** — 4 sub-plugin 별 풀 [S:H:L] 표기 → 변동 출처 모델 식별
- **bar 컨텍스트** — close + Δ% (직전 봉 대비) + range% (high-low) → probs 변동을 가격 변동과 짝지어 해석
- **daily_pnl 한도 거리** — 한도 절대값 ($) + 도달 % → 일일 손실 한도까지 여유 가시화
- **DD 락 거리** — 현재 DD %·절대값 + 락 한도 %·절대값 + 도달 % → 락까지 거리 가시화
- **SL/TP 거리** — 가격 + Δ% (현재가 대비) → 청산 임박도 즉답

코드 변경: `src/strategy/plugins/ensemble.py` (meta sub_probs) + `src/live/engine.py` (3 log 함수) + `src/core/engine_base.py` (호출처 시그니처).

#### 9.2.2 BLE-7-2 텔레그램 알림 보강 (별건, BLE-7-1 안정화 후)
콘솔/파일 로그와 일관성 유지하며 텔레그램 메시지 정보량 보강. I-BL014 회귀 영역이라 plain text 정합성 검증 필수 (parse_mode 미사용 유지).

진입 조건: BLE-7-1 1-2일 라이브 모니터링 안정화 + 사용자 텔레그램 사용 패턴 파악 후.

#### 9.2.3 향후 BLE-7-3+ (선택, 운영 누적 후 결정)
- 일일 리포트 자동 발송 (텔레그램 / 이메일)
- OOS Decay 자동 알람 임계 세분화
- 모델 stale 감지 + 자동 재학습 trigger
- 라이브 metrics 추출 (Grafana 등 대시보드 연동)

이 영역들은 운영 누적 (1-2개월+) 후 필요성 명확해질 때 별도 step 으로 추가.

### 9.3 결정 사안 (BLE-7 진입 시)

#### W'. BLE-7-1 출력 형식 (2026-05-10 결정 완료)
- E-1=가 시그니처 변경 (close, prev_close 인자 추가)
- E-2=가 sub_probs 풀 [S:H:L] 표기
- E-3=가 bar 컨텍스트 (close+Δ%+range%)
- E-4=가 daily 한도 + DD 락 모두 % + 절대값 ($) 표기
- E-5=가 SL/TP 가격 + Δ%

#### X'. BLE-7-2 텔레그램 보강 시점
- BLE-7-1 안정화 1-2일 후 진입
- 사용자 텔레그램 사용 패턴 파악 결과 반영

### 9.4 검증 기준
- BLE-7-1: 단위 테스트 4-5건 (sub_probs 노출 / 출력 포맷 / SL/TP 거리 / 한도 도달 % / 시그니처 호환). 회귀 466 → 470+ pass
- BLE-7-2: 텔레그램 plain text 정합성 (I-BL014 회귀 방지) + 메시지 길이 관리

---

## 10. 진행 기록 (Phase BLE-1 ~ BLE-7)

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| (대기) | Phase BLE-1: 다중 거래소 | 대기 | — | binance_executor + Broker 분기 |
| (대기) | Phase BLE-2: Ensemble walkforward (I-BL002) | 대기 | — | 학술 robust성 보강. 코드 작업만 (GPU 불필요) |
| (대기) | Phase BLE-3: Survivorship | 대기 | — | 다른 코인 학습 + 비교 |
| (조건부) | Phase BLE-4: BP-1 데이터 carry | 조건부 | — | 데이터 인프라 준비 시. v005 학습 |
| (대기) | Phase BLE-5: I-BP001 funding_fee 백테 통합 (신설) | 대기 | — | PATH_B_LIVE_TRADING 종착 시 carry. 라이브 펀딩률 모니터링 데이터 누적 후 처리 |
| (대기) | Phase BLE-6: 라이브-백테 정합성 검증 (신설) | 대기 | — | 라이브 거래 ≥ 30건 / 운영 ≥ 1개월 누적 후 baseline 측정. 다른 BLE 진행 시 baseline 재활용 |
| 진행 중 | Phase BLE-7: 운영 모니터링 강화 (신설, 1순위) | 진행 중 | — | BLE-7-1 ✅ 완료 / BLE-7-2 대기 / BLE-7-3+ 향후 |
| 2026-05-10 | └ BLE-7-1: 콘솔/파일 로그 보강 | ✅ 완료 | (이번 커밋) | ensemble.py meta sub_probs 추가 + `_log_signal_status` 시그니처 확장 (bar_context dict) + `_log_position_status` SL/TP 거리 + `_log_account_status` daily 한도/DD 락 거리 (% + 절대값). 단위 6건 신규 추가 (TestBLE71*), 회귀 466→472 pass |
| (대기) | └ BLE-7-2: 텔레그램 알림 보강 | 대기 | — | BLE-7-1 안정화 1-2일 후 진행. I-BL014 회귀 영역 (plain text 정합성) |

---

## 11. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BL002 | BL-1-3 종착 (PATH_B_LIVE_TRADING §6) | Ensemble walkforward 평가 미수행. ensemble.py가 fold 모델별 sub_params 오버라이드 메커니즘 부재 | src/strategy/plugins/ensemble.py + scripts/evaluate_models.py | **BLE-2** | 미해결 — 단일 모델 walkforward 모두 robust 검증되어 우선순위 낮음 |
| I-BP001 | PATH_B_PRODUCTION carry-over → PATH_B_LIVE_TRADING § 6 → BLE-5 | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환. `engine_base.close_position`에 `funding_fee=0.0` 하드코드. 백테에 펀딩률 미반영 | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | **BLE-5** | 미해결 — 보유 시간 평균 1.6h → 1건당 ~0.002% PnL 영향, 미미. 라이브 펀딩률 모니터링 데이터 누적 후 fix |

신규 carry-over 후보 ID는 I-BLE001~ 형태로 등록.

---

## 12. 종착 후

PATH_B_LIVE_EXTENSION 모든 step (BLE-1 ~ BLE-7) 종착 시 "경로 B" 모든 작업 완료. 그 후:
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도) — 별도 PATH 가능
- 새 모델 paradigm 탐색 (LLM 기반 등) — 새 PATH 시작 가능
- BLE-7-3+ (일일 리포트, OOS Decay 자동 알람, 모델 자동 교체 등) — 운영 누적 후 필요성 재검토 시 BLE-7 안에 step 추가 또는 별건 PATH 가능
