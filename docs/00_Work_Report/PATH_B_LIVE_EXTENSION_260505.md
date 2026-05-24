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
├─ BLE-6-1 라이브 DB OKX sync 인프라 (prerequisite, 2026-05-23 진행)
│   ├─ trades 테이블에 entry_order_id/exit_order_id/synced_at 컬럼
│   ├─ open/close 시 ccxt order id 추출·저장
│   ├─ TradeSyncer (src/live/trade_sync.py) — 미sync batch 처리
│   └─ CoreEngine._close_with_funding 끝에 sync trigger (라이브 전용)
└─ BLE-6-2 라이브 vs 백테 정합성 비교 (BLE-6-1 종착 후)
    ├─ 라이브 누적 거래 vs 같은 기간 백테 결과 비교
    ├─ paper-실 거래 격차 정량 (BL-2-4 본 검증 항목 후속)
    ├─ 거래별 fee/slippage/funding 차이 분석
    └─ baseline 정합성 리포트 (다른 BLE 진행 시 영향 측정 baseline)

Phase BLE-7 (운영 모니터링 강화 — 신설, 2026-05-10)
├─ BLE-7-1 콘솔/파일 로그 정보량 보강 (sub_probs / bar 컨텍스트 / 위험 한도 거리 / SL/TP 거리)
├─ BLE-7-2 텔레그램 알림 정보량 보강 (콘솔 로그와 일관 유지)
├─ BLE-7-3 잔고 입금 처리 가이드 (CLAUDE.md + data/deposits.json 메모 인프라)
├─ (향후) BLE-7-4+ 일일 리포트 자동 발송, OOS Decay 자동 알람 임계 강화 등
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

#### 8.3.0 BLE-6-1 라이브 DB OKX sync 인프라 (prerequisite, 2026-05-23 진행)

BLE-6 본 비교의 baseline 정확성을 위해 라이브 DB 의 거래 수치를 OKX 실값으로 sync 하는 인프라 우선 구축.

**배경**: $341 누락 발견 — trades.pnl 합 (+$165) vs 실제 잔액 변화 (-$176) 차이. 원인: DB 의 entry_price/exit_price/trading_fee 가 엔진 추정값 (봉 close 가격 + config taker_fee_pct 추정) 이고 OKX 실 체결가/fee 와 다름.

**해결**: TradeSyncer 인프라 신설 — 라이브 close 직후 미sync trade 일괄 처리. 신규 거래는 ccxt order id 로 정확 매칭, 기존 15건은 시간/방향/size/reduceOnly fallback 매칭 후 OKX id 도 같이 저장 (이후 id 기반).

**결정 사안 (확정)**:
- D=가: trades 테이블 신규 컬럼 3개 (entry_order_id / exit_order_id / synced_at)
- E=가: close 직후 batch sync (미sync 전체)
- F=가, L=가: 기존 15건은 자동 sync 메커니즘에 자연 흡수 (단일 메커니즘)
- G=나: CoreEngine._close_with_funding 끝에 sync trigger (라이브 전용, 백테/페이퍼 영향 0)
- H=가: await sync_all_unsynced 동기 호출
- I=가: 시간 매칭 margin ±60초
- J=가: 매칭 실패 시 WARNING + synced_at NULL 유지
- K=가: sync 함수 시작점에 broker.is_live 가드
- M=가: USDT fee 만 sync, 다른 currency 면 WARNING + 0 처리
- N=가': size tolerance ±0.005 BTC (1 contract 의 half, round 영향 안전 흡수)

#### 8.3.1 BLE-6-2 라이브 거래 데이터 추출 (BLE-6-1 종착 후)
- `data/coinbot_*.db` 의 trades 테이블 export (sync 완료 데이터)
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

#### 9.2.3 BLE-7-3 잔고 입금 처리 가이드 (2026-05-23 진행)

라이브 운영 중 사용자가 OKX 잔고 *추가 입금* 시 발생할 수 있는 영역 정리:
- PnL/daily_pnl 인식 안 됨 검증 (외부 입금은 거래 이벤트 아님)
- peak_equity 재시작 시점 자동 갱신 (`_restore_state` 의 `update_equity(balance)`)
- DB `bot_meta.initial_balance` 수동 update (수익률 기준 재설정)
- 입금 메모 영구 기록 (`data/deposits.json`, git untracked)

구체 절차는 `CLAUDE.md` 의 "잔고 입금 처리 가이드" sub-section 참조 (Phase 1 Claude 자동 snapshot / Phase 2 사용자 수행 / Phase 3 Claude 자동 기록·검증).

#### 9.2.4 향후 BLE-7-4+ (선택, 운영 누적 후 결정)
- 일일 리포트 자동 발송 (텔레그램 / 이메일)
- OOS Decay 자동 알람 임계 세분화
- 모델 stale 감지 + 자동 재학습 trigger
- 라이브 metrics 추출 (Grafana 등 대시보드 연동)
- 입금 기록 자동화 (deposits 테이블 + 자동 marker)

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
| 진행 중 | Phase BLE-6: 라이브-백테 정합성 검증 (신설) | 진행 중 | — | BLE-6-1 ✅ 완료 / BLE-6-2 대기 |
| 2026-05-23 | └ BLE-6-1: 라이브 DB OKX sync 인프라 | ✅ 완료 | 271f8a1 | trades 테이블에 entry_order_id/exit_order_id/synced_at 컬럼 추가 + ccxt order id 추출 흐름 + TradeSyncer 모듈 (src/live/trade_sync.py) + CoreEngine._close_with_funding 끝 sync trigger. id 매칭 + 시간/방향/size fallback (±60s, ±0.005 BTC). 단위 6건 신규 (TestTradeSync), 회귀 475→481 pass. 라이브 다음 close 시점에 기존 15건 자동 retroactive sync 예정 |
| 2026-05-24 | └ I-BLE002 fix: BLE-6-1 sync 알고리즘 보강 | ✅ 완료 | (이번 커밋) | 4개 원인 모두 해결 — A(amount contracts × contract_size BTC 변환), B(fillPnl 합 ≠ 0 → reduce_only), C(`_fetch_all_fills` pagination: PAGE_LIMIT=100 × MAX_PAGES=10 + id dedup), D(자연 skip). 단위 6건 갱신 + 신규 2건 (pagination basic + dedup) = 8건. 회귀 481→**483 pass**. **진단 결과 (1회용 _tmp_sync_match_sim.py 실행, 결과만 보존)**: DB 16건 중 **15건 entry+exit 정확 매칭 가능** (Trade 2~16, size BTC ±0.001 + ts ±60s 안). **Trade 1 만 entry 매칭 영구 실패** — entry buy 7.5 contracts 가 OKX `fetch_my_trades` 응답에 누락 (since earliest-5분 으로도 미흡수, OKX API 영역 추정). 사용자 결정 (AC=나): entry/exit 둘 다 매칭만 sync 정책 유지 → **Trade 1 영구 unsynced**, 향후 수동 sync 방안은 I-BLE003 후속 작업. exit margin 확대/exit 독립 매칭은 미적용 |
| (대기) | └ BLE-6-2: 라이브 vs 백테 정합성 비교 | 대기 | — | I-BLE002 fix 완료 → 라이브 재시작 후 다음 close 시 15건 retroactive sync 확인 → BLE-6-2 진입 가능 (라이브 거래 ≥ 30건 / 운영 ≥ 1개월 누적 baseline 측정) |
| (carry) | └ I-BLE001 (예정, 2순위): 라이브 재시작 daily_pnl 복원 + close_at 컬럼 + sync 후 메모리 정정 | 미해결 | — | I-BLE002 fix 완료 (이번 커밋) + sync 정상 검증 후 라이브 정지 시점에 진행 (Z=가). 견적 ~95분. 상세 plan §11 I-BLE001 entry 참조 |
| (carry) | └ I-BLE003 (예정, 3순위): Trade 1 수동 sync 방안 모색 | 미해결 | — | I-BLE002 fix 로 15/16 정상 sync 되지만 Trade 1 entry 가 OKX `fetch_my_trades` 응답에서 누락. 후보 방안: ① OKX 웹 거래 내역 직접 조회 + DB 수동 update (CSV export 등) ② 다른 ccxt API (`fetch_orders`, `fetch_order(id)` — order id 모름이라 어려움) 시도 ③ 영구 unsynced 유지 + 정합성 분석 시 DB 추정값으로 처리. 진행 트리거: I-BLE001 완료 후 또는 사용자 우선순위 결정 |
| 진행 중 | Phase BLE-7: 운영 모니터링 강화 (신설, 1순위) | 진행 중 | — | BLE-7-1 ✅ 완료 / BLE-7-2 대기 / BLE-7-3+ 향후 |
| 2026-05-10 | └ BLE-7-1: 콘솔/파일 로그 보강 | ✅ 완료 | 25e1b41 | ensemble.py meta sub_probs 추가 + `_log_signal_status` 시그니처 확장 (bar_context dict) + `_log_position_status` SL/TP 거리 + `_log_account_status` daily 한도/DD 락 거리 (% + 절대값). 단위 6건 신규 추가 (TestBLE71*), 회귀 466→472 pass |
| 2026-05-10 | └ BLE-7-1 보강: 가독성 + conf class | ✅ 완료 | 99f002d | 라이브 며칠 운영 후 발견 — 한 줄 출력이라 가독성 ↓ + conf 가 어느 class(S/H/L) 점수인지 불명확. 멀티라인 (\n + prefix 별 9/10/11 space 들여쓰기) + 라인 사이 빈 줄 + conf=H:0.92 형식 (probs argmax 기반 — signal.side ≠ argmax 가능한 threshold 미달 case 도 직관). 단위 3건 신규 (TestBLE71ConfClassLabel) + 기존 1건 흡수, 회귀 472→475 pass |
| (대기) | └ BLE-7-2: 텔레그램 알림 보강 | 대기 | — | BLE-7-1 안정화 1-2일 후 진행. I-BL014 회귀 영역 (plain text 정합성) |
| 2026-05-23 | └ BLE-7-3: 잔고 입금 처리 가이드 | ✅ 완료 | (이번 커밋) | CLAUDE.md "잔고 입금 처리 가이드" sub-section 신설 (Phase 1 Claude 자동 snapshot / Phase 2 사용자 수행 / Phase 3 자동 기록·검증). `data/deposits.json` 메모 인프라 (git untracked). 향후 입금 자동 처리 흐름 확립 |

---

## 11. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BL002 | BL-1-3 종착 (PATH_B_LIVE_TRADING §6) | Ensemble walkforward 평가 미수행. ensemble.py가 fold 모델별 sub_params 오버라이드 메커니즘 부재 | src/strategy/plugins/ensemble.py + scripts/evaluate_models.py | **BLE-2** | 미해결 — 단일 모델 walkforward 모두 robust 검증되어 우선순위 낮음 |
| I-BP001 | PATH_B_PRODUCTION carry-over → PATH_B_LIVE_TRADING § 6 → BLE-5 | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환. `engine_base.close_position`에 `funding_fee=0.0` 하드코드. 백테에 펀딩률 미반영 | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | **BLE-5** | 미해결 — 보유 시간 평균 1.6h → 1건당 ~0.002% PnL 영향, 미미. 라이브 펀딩률 모니터링 데이터 누적 후 fix |
| I-BLE001 | BLE-7-3_fix2 (peak_equity 정정) 후속 점검 시 발견 (2026-05-23) | **3개 영역 묶음**: ① 라이브 재시작 시 `RiskManager.daily_pnl` 복원 메커니즘 부재 — `_restore_state` 가 `data_store.get_daily_pnl()` 호출 안 함 → 오늘 누적 거래 손익 영구 손실 (clean startup case). ② `trades.timestamp` = open 시각 — `close_at` 컬럼 부재 → `get_daily_pnl` 쿼리가 *open 기준* 으로 부정확 (어제 open + 오늘 close 등 자정 경계 case). ③ BLE-6-1 sync 후 메모리 `daily_pnl` 미정정 → DB `trades.pnl` 합 (OKX 실값) 과 메모리 (엔진 추정 합) 어긋남 | src/data/store.py + src/core/engine_base.py + src/live/engine.py + tests | **BLE 후속 단발 fix (예정, 2026-05-23 결정)** | 미해결 — 결정 사안 확정 (V=가 새 잠재 이슈 등록 / W=가 hotfix-L case 2 add_pnl 제거 후 일원화 / X=가 sync 후 daily_pnl 재정렬 / Y=가 `COALESCE(closed_at, timestamp)` fallback). 진행 트리거 Z=가 (다음 close 후 BLE-6-1 sync 결과 검증 + 라이브 정지 시점). **상세 7-step 계획**: ① `store.py` 스키마 마이그레이션 (`closed_at TEXT`) + `close_trade` 시그니처 확장 + `get_daily_pnl` 쿼리 변경 (`COALESCE(closed_at, timestamp) LIKE 'YYYY-MM-DD%'`) — ~20분. ② `engine_base._record_trade_close` + 호출처 (live engine / backtest engine) 에 `closed_at=now` 전달 — ~10분. ③ `_restore_state` 잔액 복구 직후 `daily_pnl = await data_store.get_daily_pnl()` + INFO 로그 — ~10분. ④ `_close_with_funding` 의 `sync_all_unsynced` 호출 직후 `synced_count > 0` 시 `daily_pnl = get_daily_pnl()` 재정렬 — ~10분. ⑤ `_restore_state` case 2 의 `risk_manager.add_pnl(actual_pnl)` 제거 (I-BL016 hotfix-L 영역) — ③에 일원화. close_trade 만 호출 — ~5분. ⑥ 단위 테스트 5건 (스키마 / get_daily_pnl COALESCE 쿼리 / _restore_state 복원 / sync 후 정정 / case 2 영역 일관성) — ~30분. ⑦ 작업 보고서 §10 갱신 + I-BLE001 ✅ 표기 + 단위 회귀 481+5=486 pass 확인 — ~10분. 견적 합 ~95분. **잠재 영역**: hotfix-L case 2 의 add_pnl 제거 후 단위 테스트 `test_case2_same_day_exit_accrues_daily_pnl` 등 갱신 필요. close_at NULL 인 기존 15건은 BLE-6-1 sync 시 OKX exit ts 활용 가능하나 본 작업에서는 timestamp fallback 으로 충분 |

| I-BLE002 | BLE-6-1 sync 첫 trigger 시 발견 (2026-05-24 19:30) — Trade 16 close 후 `Trade sync: 0 synced, 16 failed` | **BLE-6-1 sync 알고리즘이 OKX 응답과 mismatch 로 전건 매칭 실패**. 4개 영역: **A. `amount` 단위 = contracts (정수/소수), BTC 아님** — `fetch_my_trades` 응답의 `amount` 가 OKX 의 `fillSz` (contracts). 본 시스템 contract_size=0.01 BTC. 시스템 db_size(BTC) vs OKX amount(contracts) 직접 비교 → 100배 차이로 tolerance 0.005 무관 fail. **B. `reduceOnly` 필드 OKX info 에 없음** (`reduceOnly_raw: null`, `info.reduceOnly: null`) — 대신 `info.fillPnl` 필드 사용 가능 (entry fill 은 fillPnl=0, close fill 은 fillPnl=실현PnL). 현재 알고리즘은 모든 fill 을 `reduce_only=False` 로 인식 → exit 매칭 영역 전건 fail. **C. fetch_my_trades limit=200 요청해도 100 fills 만 반환** (OKX V5 `/api/v5/trade/fills` 한도) — 18일 데이터 + 다중 fill 큰 거래 (예: 24 fills 1 order) 시 잘림. pagination 필요. **D. 사용자 외 거래 섞임** — DB Trade 1 (7.5 contracts) 외 다른 거래 (2.35 contracts 등) 가 응답에 포함. 매칭 자연 skip 으로 처리 (영향 없음, 정보용) | src/live/trade_sync.py + tests/test_trade_sync.py + scripts/diagnose_okx_fills.py | **2026-05-24 fix 완료** | ✅ 해결 — 5-step 모두 완료 (이번 커밋). **변경 영역**: ① `_group_and_aggregate` 에 `contract_size` 인자 + `total_amount_contracts × contract_size = BTC` 변환 후 `by_order[oid]["amount"]` 저장. `total_fill_pnl 합 ≠ 0 → reduce_only=True` 판별. ② `sync_all_unsynced` 내부에서 `broker.executor.contract_size` 추출 (paper 가드 후 LiveExecutor 보장). ③ `_fetch_all_fills(exchange, symbol, since_ms)` 신규 함수 — PAGE_LIMIT=100 × MAX_PAGES=10, id dedup, last_ts 기반 forward iteration, 무한 루프 차단. ④ 단위 테스트 6건 갱신 (amount contracts 단위 + info.fillPnl 추가 + executor.contract_size mock) + 신규 2건 (test_pagination_basic / test_pagination_dedup) = 총 8건. ⑤ 회귀 **481 → 483 pass**. **진단 정량 (1회용 scripts/_tmp_sync_match_sim.py 실행 결과 2026-05-24, 스크립트 삭제, 결과 보존)**: DB 16건 vs OKX 100 fills (32 unique orders) 매칭 시뮬 — Trade 2~16 (15건) entry+exit 모두 ±60s ± 0.005 BTC tolerance 안에서 정확 매칭. Trade 1 만 entry (long, 0.075 BTC = 7.5 contracts buy @ 12:15:01) 가 OKX 응답에 누락 (since earliest - 5분 으로도 미흡수). Trade 1 exit (sell 7.5 contracts @ 81707.30, fillPnl=-34.71) 는 매칭 가능하나 사용자 결정 (AC=나: entry/exit 둘 다 매칭만 sync) 으로 **Trade 1 영구 unsynced**. 수동 sync 방안은 I-BLE003 후속. **검증 안내**: 사용자 라이브 재시작 (`python -m src.main live --config config\ensemble.yaml`) → 다음 close 시점 `Trade sync: 15 synced, 1 failed (errors=1)` 로그 기대 (failed=Trade 1) |
| I-BLE003 | I-BLE002 fix 진행 시 발견 (2026-05-24) — Trade 1 entry buy 7.5 contracts 가 OKX `fetch_my_trades` 응답에 누락 | Trade 1 (long, 0.075 BTC, 2026-05-06T12:15:01 entry) 의 entry order 가 since=earliest-5분 pagination 전체 fetch 에서도 응답 누락. 가능 원인: ① OKX API 영역 한계 (특정 시점 fill 누락) ② 라이브 첫 진입 시 instrument 설정 (margin mode 등) 직후라 응답 차이 ③ 다른 endpoint (closed_orders) 에는 있을 가능성. DB 의 entry_price/trading_fee 가 엔진 추정값 유지 → trades.pnl 합 (DB) vs 실 잔액 변화 차이 영역에 ~$40 정도 영향 | src/live/trade_sync.py 보강 또는 별도 수동 sync 스크립트 | **별도 후속 작업 (예정)** | 미해결 — 후보 방안: ① OKX 웹 거래 내역 수동 export + DB 직접 UPDATE 스크립트 (가장 robust) ② `fetch_closed_orders` 시도 (BLE-6-1 의 `_fetch_actual_exit` 패턴 차용) ③ 영구 unsynced 유지 + 정합성 분석 (BLE-6-2) 시 DB 추정값으로 처리. 진행 트리거: I-BLE001 완료 후 또는 사용자 우선순위 결정 |

신규 carry-over 후보 ID는 I-BLE004~ 형태로 등록.

---

## 12. 종착 후

PATH_B_LIVE_EXTENSION 모든 step (BLE-1 ~ BLE-7) 종착 시 "경로 B" 모든 작업 완료. 그 후:
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도) — 별도 PATH 가능
- 새 모델 paradigm 탐색 (LLM 기반 등) — 새 PATH 시작 가능
- BLE-7-3+ (일일 리포트, OOS Decay 자동 알람, 모델 자동 교체 등) — 운영 누적 후 필요성 재검토 시 BLE-7 안에 step 추가 또는 별건 PATH 가능
