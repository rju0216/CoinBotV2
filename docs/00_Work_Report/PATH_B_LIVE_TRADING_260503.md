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

### 1.4 운영 권장 (BL-2-3 진입 시점, 2026-05-05 갱신)

BL-1 본 검증 통과 + BL-2-1/2 인프라 + OOS warm-up 완비 → paper 1일 검증 후 라이브 진입.

**라이브 채택 결정 (사용자, 2026-05-05)**:
- **모델**: **v009 재학습 후 사용** (학습 cutoff 2024-12-31 → 약 17개월 alpha decay 우려 해소)
- **Strategy**: **ensemble** (4 모델 v009 + isotonic) — 단일 OOS 1순위 (BP-3-2 1118%) + 다양성
- **활성 옵션** (paper 진입 직전): OOS monitor + 호가창 + Telegram 알림 모두 ON
- I-BL002 (ensemble walkforward 미수행)는 PATH_B_LIVE_EXTENSION으로 carry

**참고 — 단일 모델 선택 시 1순위**: `dl_transformer` (walkforward + Regime 일관 1위)
- BL-1-3 walkforward: Calmar 74.3, MDD 5.68%, 26/26 positive
- BL-1-4 Regime: Calmar 200, MDD 3.72%, SHORT win 76.21%
- 2순위: dl_lstm (walkforward PF 4.12)
- 3순위: ml_lightgbm/ml_xgboost (GBDT 통계적 동등)

PPO 라이브 부적합 확정 (E-2-4 + BL-1-2 Multi-hypothesis 보정 후에도 유지)

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

#### 3.2.1 Regime-matched 백테 ✅ 완료 (BL-1-4)

**완료 작업** (사안 A=가, I'=나 triple_barrier, O'=나 evaluate_models, P'=가 4 단일, Q'=가 --save-all-folds 없이):
1. 4 모델 v008 학습 (2022-2024, label_method=triple_barrier) + calibration
2. `evaluate_models.py` SPLIT_DEFINITIONS에 `("Regime", "v008", "2020-01-01", "2021-12-31")` 추가
3. `MODEL_VERSIONS["v008"]` 매핑. `split_to_oos_years["Regime"]=2.0`
4. 4 모델 single 모드 백테 (2020-2021, 강세 bull regime)

**결과 (4 모델, calibrated)**:

| 모델 | trades | win | total_return | MDD | PF | Calmar(연환산) | LONG/SHORT |
|---|---|---|---|---|---|---|---|
| ml_lightgbm | 2,234 | 67.55% | 6752.99% | 5.73% | 2.75 | 127 | 1116/1118 (균형) |
| ml_xgboost | 2,260 | 67.30% | 6674.45% | 5.55% | 2.71 | 134 | 1173/1087 |
| dl_lstm | 2,340 | 66.79% | 6985.69% | 4.00% | 2.82 | 187 | 1234/1106 |
| **dl_transformer** | 2,198 | **69.29%** | **7092.72%** | **3.72%** | **2.90** | **200** ⭐ | 1374/824 |

**핵심 발견**:
1. 4 모델 모두 강세 bull에서 robust (regime lottery 아님)
2. **dl_transformer 일관 1위** (walkforward + Regime 모두) — 라이브 채택 최우선
3. **dl_transformer SHORT win 76.21%** (다른 모델 66-67% 대비 +10%p, PF 4.63 vs 2.6-2.9) — 강세장에서도 SHORT 정확
4. **단순 long-bias 우려 해소**: 4 모델 모두 LONG/SHORT 거래 균형, 양쪽 win rate 비슷
5. GBDT 동등성 (E-2-4) Regime에서도 재확인

#### 3.2.2 Walk-forward 통합 OOS ✅ 완료 (BL-1-3)

**완료 작업** (사안 B=가, H'=가 nested, K'=가 reset, L'=가 calibration none, M'=나 v007 triple_barrier):
1. 4 train_*.py에 `--save-all-folds` 인자 추가. nested `v00X/folds/fold_NN/` 구조 (model + feature_names + scaler[DL] + per-fold meta + model_arch[DL])
2. `scripts/evaluate_models.py`에 `--mode walkforward --strategy <name>` 신규. fold 자동 스캔 → 26 spec 백테 → 통합 metrics 산출
3. **walkforward 결과 (4 모델, 26 folds × ~4.5년 OOS)**:

| 모델 | mean_return/fold | max_DD | Calmar (추정) | win_rate | PF | positive folds |
|---|---|---|---|---|---|---|
| ml_lightgbm | 399.29% | 7.66% | 52.1 | 71.51% | 3.81 | 26/26 |
| ml_xgboost | 386.23% | 9.08% | 42.5 | 70.36% | 3.57 | 26/26 |
| dl_lstm | 399.59% | 7.42% | 53.9 | 72.26% | **4.12** | 26/26 |
| **dl_transformer** | **422.19%** | **5.68%** | **74.3** ⭐ | 71.77% | 3.84 | 26/26 |

**핵심 발견**:
- 4 모델 모두 26/26 positive fold — 4.5년 시점 lottery 무관 양의 수익. 모델 robust성 매우 강함
- dl_transformer가 walkforward 최우수 (MDD 가장 낮음 + Calmar 가장 높음). 운영 권장 1순위 (안정성)에 추가
- GBDT 동등성 (E-2-4 결론) walkforward에서도 재확인
- ml_xgboost MDD 9.08%로 단일 시점 변동성 노출 큼 (4 모델 중 가장 약한 안정성)
- ensemble walkforward 미수행 → **I-BL002** carry (§6 참조)

#### 3.2.3 Lookahead bias 추가 점검 ✅ 완료 (BL-1-1)

**완료 작업**: `tests/test_lookahead.py` 신규 (3 클래스 14 케이스)
1. **Indicator forward-bias** (10건): EMA/SMA/MACD/RSI/ATR/BBands/BB_Width/ADX/Choppiness/EfficiencyRatio 모두 backward-only 확인 — 합성 데이터 마지막 50행 교체 → 앞 250행 indicator 결과 동일성 검증
2. **Walk-forward embargo 시간 격차** (2건): horizon=10/30 모두 train 끝(embargo 적용 후) ↔ test 시작 사이 ≥ horizon 봉 격차 보장
3. **OHLCV fetch fresh-bar** (2건, mock): ccxt 진행 중 봉 가능성은 plugin/엔진 측 차단(I-B007 + 라이브 BAR_CLOSED + 백테 _slice_candles)으로 자연 처리됨 확인

**결과**: **Lookahead 새 발견 0건**. 3중 방어 (엔진 + indicators backward-only + WF embargo) 검증 완료.

#### 3.2.4 호가창/유동성 모델링 정교화 ⚠️ BL-2 carry (사안 C 나)

**결정**: BL-1에서 미수행. BL-2 §4.2.1 paper trading 단계와 함께 수집·구현.

**사유**:
- BL-1은 학술 검증 위주 (모델 자체 robust성). 호가창은 운영 단계 인프라
- 호가창은 시점별 변동 큼 — BL-1에서 수개월 전 수집 데이터는 BL-2 진입 시점에 outdated 가능
- paper trading 시 라이브 가격 받아오는 동안 호가창 동시 수집이 자연스러움
- 우리 자금 규모에서는 size 작아 호가창 충격 미미할 가능성 (자금 확장 시 핵심)

**작업 (BL-2 §4.2.1로 이관)**:
1. OKX 호가창 snapshot 수집 (paper trading 동시)
2. 사이즈 대비 호가창 깊이로 실효 체결 가격 산정
3. `paper_executor.open_position` 확장 — book depth 인자
4. 백테 시 fee_model.slippage_pct 대신 실 호가창 시뮬

#### 3.2.5 Multi-hypothesis 보정 ✅ 완료 (BL-1-2)

**완료 작업** (사안 D=다 둘 다):
1. `src/ml/metrics_extended.py`에 `bonferroni_correction` + `fdr_correction` (Benjamini-Hochberg, 단조 보정 + 인덱스 복원) 신규
2. `scripts/analyze_results.py`의 bootstrap_pvalues.csv에 4 컬럼 추가 (`p_value_bonferroni`, `significant_at_0.05_bonf`, `p_value_fdr`, `significant_at_0.05_fdr`)
3. eval_260503_baseline 재처리 → **N=20 비교 중 raw=14 / FDR=12 / Bonferroni=10 유의** (alpha=0.05)

**E-2-4 결론 재검증**:
- ✅ "PPO vs 모든 모델 p<0.0001" — 8/8 비교 모두 raw=bonf=fdr=0 유지
- ✅ "GBDT 두 모델 동등 p=0.61" — 보정 후에도 비유의 (변동 없음)
- ✅ Exp4에서 dl_transformer가 GBDT와 다름 — FDR 유지 (Bonferroni만 사라짐)
- ⚠️ split=1의 약한 신호 2건 (lightgbm vs lstm, lstm vs transformer) — 양 보정 후 사라짐 (다중 검정 노이즈)

핵심 운영 결정 (PPO 제외, GBDT 동등) 모두 보정 후에도 robust.

#### 3.2.6 BP-1 재진입 여부 확인 ⚠️ **미진입 확정 (BL-1-5, 2026-05-05)**

**결정 사유**:
- 데이터 수집 인프라 미준비 (ccxt 펀딩률·OI 자동 수집 + 2020 이전 BTC 외부 거래소 수집 모두 미구축)
- BL-1-1/2/3/4 5개 검증 결과 I-BP001 영향이 명확히 무시 가능 추정 (보유 시간 평균 1.6시간)
- BL-1 본 검증 (Lookahead/Multi-hyp/Walkforward/Regime) 모두 통과 — BP-1 재진입 없이도 라이브 채택 가능

**미진입 처리**:
- **I-BP001**은 BL 종착까지 carry-over (라이브 운영 전 BL-2 paper trading 시점에 펀딩률 실 비용 모니터링으로 재검증)
- **v005 번호 영구 비워둠** (BP-1 재진입 시 사용 예약 그대로)
- 펀딩률·OI 피처 + 2020 이전 데이터는 PATH_B 종착 후 별도 PATH로 데이터 확장 재방문 가능
- PATH_B_PRODUCTION §3 (펀딩률·OI 수집 + 2020 이전 데이터) 작업 항목은 **재방문용 참조 자료**로 보존

### 3.3 결정 사안 (BL-1 진입 시 결정)

#### A. Regime-matched 백테 — 학습/백테 기간 ✅ 결정됨 — **(가) 학습 2022-2024 / 백테 2020-2021**
- (가) **학습 2022-2024 / 백테 2020-2021** ← 선택 (단순, 강세 bull 명확 검증)
- (나) HMM 자동 매칭 — 추가 인프라 부담

#### B. Walk-forward 통합 OOS — fold 모델 저장 시점 ✅ 결정됨 — **(가) 4 train_*.py에 옵션 추가**
- (가) **train_*.py에 `--save-all-folds` 옵션 추가** ← 선택 (DRY)
- (나) 별도 train_walkforward_*.py — 코드 중복

#### C. 호가창 모델링 — 데이터 수집 시기 ✅ 결정됨 — **(나) BL-2 진입 직전**
- (가) BL-1 즉시 수집 — oudated 위험
- (나) **BL-2 진입 직전 (paper trading 시 함께)** ← 선택 (fresh 데이터)

#### D. Multi-hypothesis 보정 알고리즘 ✅ 결정됨 — **(다) 둘 다 비교**
- (가) Bonferroni만 — 너무 보수적
- (나) FDR만
- (다) **Bonferroni + FDR 둘 다** ← 선택 (학술 robust성)

#### H'. (신규, BL-1-3) fold 모델 디렉토리 ✅ 결정됨 — **(가) nested**
- (가) **`models/<type>/v00X/folds/fold_NN/` (nested)** ← 선택. 단일 학습 산출물
- (나) 별도 디렉토리

#### I'. (신규, BL-1-4) v007 라벨 ✅ 결정됨 — **(나) triple_barrier**
- (가) direction
- (나) **triple_barrier** ← 선택. v006과 라벨 동일, 운영 권장 1순위 일관

#### J'. (신규, BL-1-3) fold 모델 calibration ✅ 결정됨 — **(다) raw**
- (가) fold별 calibrator 학습 — 인프라 부담 큼
- (나) latest calibrator 재사용 — lookahead 약간
- (다) **raw 사용** ← 선택. walkforward 본 목적 (모델 자체 평가)

#### K'. (신규, BL-1-3) fold 백테 자금 처리 ✅ 결정됨 — **(가) reset**
- (가) **각 fold initial=$10K reset** ← 선택. 통계적 평균
- (나) 이어받기 — 마지막 fold 가중치 쏠림

#### L'. (신규, BL-1-3) walkforward calibration 강제 ✅ 결정됨 — **(가) "none" 강제**
- (가) **"none" 강제** ← 선택. fold 디렉토리에 calibrator 없음 + J'=다와 일관
- (나) config 따름

#### M'. (신규, BL-1-3) 사용자 GPU 학습 모델 ✅ 결정됨 — **(나) v007 (Triple-barrier)**
- (가) v001 (direction)
- (나) **v007 (triple_barrier)** ← 선택. 운영 권장 1순위와 일관

#### N'. (신규, BL-1 Step A) 모델 학습 version 결정 ✅ 결정됨 — **(나) max+1**
- (가) 카운트+1 — 구 동작, 충돌 위험
- (나) **`max(existing v) + 1`** ← 선택. v005 비어둠 정책 자동 호환

### 3.4 검증 기준 + 결과

| 작업 | 검증 결과 |
|---|---|
| BL-1-1 Lookahead 추가 점검 | ✅ 새 발견 0건. 단위 테스트 14건 (10 indicator + 2 embargo + 2 OHLCV mock) |
| BL-1-2 Multi-hypothesis 보정 | ✅ E-2-4 핵심 결론 (PPO 부적합, GBDT 동등) Bonferroni/FDR 보정 후 유지. raw 14 / FDR 12 / Bonferroni 10 유의 (N=20) |
| BL-1-3 Walk-forward 통합 OOS | ✅ 4 모델 26/26 positive fold. dl_transformer Calmar 74.3 최우수. ml_xgboost MDD 9.08% 최약. ensemble walkforward는 I-BL002 carry |
| BL-1 Step A (덮어쓰기 방지) | ✅ next_model_version (max+1) + resolve_unique_dir (postfix). 단위 테스트 12건 |
| BL-1 Step B (I-BL001 fix) | ✅ calibrate_models.py가 build_labels_from_config 사용. label_method 자동 분기 |
| BL-1-4 Regime-matched 백테 | ✅ 4 모델 강세 bull 모두 robust. dl_transformer 일관 1위 (Calmar 200, MDD 3.72%, SHORT win 76.21%) |
| BL-1-5 BP-1 재진입 여부 | ⚠️ **미진입 확정** (데이터 인프라 미준비 + I-BP001 영향 미미 확인). I-BP001은 BL 종착까지 carry-over |

전체 회귀: 264 → **329 passed** (BL-1-1 신규 14 + BL-1-2 신규 11 + BL-1-3 신규 6 + Step A 신규 12 = 43, 회귀 0)

---

## 4. Phase BL-2: 라이브 진입 (2026-05-05 재구성)

### 4.1 목적

학술 검증된 모델 (BL-1 통과)을 실 거래소에 안전하게 진입. **BL-2 종착 = 라이브 거래 시작 = "경로 B" 본 목적 달성**.

확장 작업 (다중 거래소, Survivorship 등)은 별도 PATH `PATH_B_LIVE_EXTENSION_260505.md`로 분리.

### 4.2 작업 항목 (4 step)

#### 4.2.1 Fail-safe 강화 (BL-2-1) ✅ 완료 (2026-05-05)

**완료 작업** (사안 G=나 5회, T''=가 log+telegram, U''=나 trade 일시 중단, V'' POSITION on, W''/X'' default):
1. `src/utils/notifier.py` 신규 — `Notifier` abstract + LogNotifier/TelegramNotifier/EmailNotifier/CompositeNotifier + `build_notifier_from_config` factory. Telegram은 urllib + asyncio.to_thread, Email은 smtplib. 미설정 시 LogNotifier fallback (warning 1회)
2. `src/execution/live_executor.py`에 `CircuitBreaker` 클래스 + `CircuitBreakerOpen` 예외 + `_retry_api` 통합. LiveExecutor `_call` helper 추출 + 14곳 호출 일괄 치환
3. `src/risk/manager.py` `attach_event_bus` + `_enter_drawdown_lock` / daily loss → publish (cooldown 1회). `EventType.CIRCUIT_BREAKER_OPEN` + `EventType.OOS_DECAY` 신규
4. `src/live/engine.py` `_setup_notifier_subscriptions` (6 이벤트 → notifier 라우팅, levels config). `_on_bar_closed`에서 circuit breaker 상태 감시 + publish
5. `src/core/engine_base.py` `try_enter`에 `_circuit_breaker_open` 차단
6. `src/utils/config_loader.py` — `.env`의 `TELEGRAM_BOT_TOKEN`/`CHAT_ID`/`EMAIL_SMTP_*` 자동 주입
7. 단위 테스트 33건 (notifier 14 + CircuitBreaker 11 + RiskManager EventBus 4 + config_loader 4)

#### 4.2.2 호가창 인프라 (BL-2-2, BL-1 §3.2.4 carry) ✅ 완료 (2026-05-05)

**완료 작업** (사안 Y''=가 BAR_CLOSED, Z''=나 depth 20, AA'''=가 인프라만, BB'''=가 parquet, CC'''=가 silent fallback):
1. `src/data/orderbook.py` 신규 — `compute_market_impact(orderbook, side, size)` VWAP helper + `OrderBookCollector` (ccxt fetch_order_book + 일별 parquet append) + `row_to_ccxt` 역변환
2. `src/execution/paper_executor.py` `open_position`/`close_position`에 `orderbook` 인자 (옵션). 가용 시 VWAP 침투 가격, 미가용 시 fill_price fallback
3. `src/execution/broker.py` mode=="live"면 orderbook 무시 (거래소 자동 처리), paper만 전달
4. `src/core/engine_base.py` try_enter/close_position에 `_latest_orderbook` 전달 (BacktestEngine은 None 유지)
5. `src/live/engine.py` initialize에서 collector 초기화 (DataFeed.exchange 재사용). `_on_bar_closed` master_timeframe BAR_CLOSED만 fetch + 캐시
6. `config/default.yaml` `live.orderbook` 섹션 (enabled=false, depth=20)
7. 단위 테스트 22건 (compute_market_impact 9 + Collector 4 + paper_executor 8 + row_to_ccxt 1)

#### 4.2.X OOS Monitor Warm-up (BL-2 추가 step) ✅ 완료 (2026-05-05)

**결정 사안** (DD''=가 학습 cutoff 이후 전체 + EE''=yes 격차 알림):
1. `src/live/oos_monitor.py` `warmup_from_history` 신규 — bars + signal_iter 받아 cutoff 이후 record_prediction + 매 봉 evaluate. 결과 dict (`samples`, `accuracy`, `gap`, `learned_oos_acc`) 반환
2. `src/live/engine.py` `_warmup_oos_monitor` (4 단일 모델 루프, ensemble skip) + `_warmup_one_strategy` (train_meta cutoff/oos_acc 추출 + historical 다운로드 + signal_iter ctx 빌드) + `_extract_train_meta`. CoreEngine.initialize 마지막에 호출 (try/except 안전)
3. `config/default.yaml` `oos_monitoring.warmup_decay_threshold_pct: 0.10` 추가
4. 단위 테스트 6건 (TestWarmup)

**효과**: 라이브/paper 시작 시 학습 cutoff 이후 데이터로 buffer 사전 채움 → 즉시 적중률 보유 + 학습 OOS Acc(약 75%) 대비 격차 ≥ 10%p이면 oos_decay publish → 텔레그램 알림으로 alpha decay 사전 감지

#### 4.2.3 Paper trading (BL-2-3, 사용자 1일 운영) — 진입 직전

**진입 결정 (2026-05-05)**:
- **모델**: v009 재학습 후 사용 (학습 cutoff 갱신 — 가능한 최신 데이터까지)
- **Strategy**: ensemble (4 모델 v009 + isotonic)
- **활성 옵션 모두 ON**: OOS monitor + 호가창 + Telegram 알림
- **검증 기간**: 1일 (통계적 검증/적중률은 1주로도 부족 — paper의 본 가치는 코드 통합 흐름 + 운영 절차 검증)

**사용자 작업 가이드** (paper 진입 직전):

1. **데이터 다운로드 (최신까지)**:
```bash
python scripts/download_history.py --config config/default.yaml --timeframe 1d,4h,15m --start 2020-01-01 --end 2026-04-30
```

2. **v009 학습 (4 모델, --save-all-folds 옵션)**:
```bash
python scripts/train_lightgbm.py    --config config/ml_lightgbm.yaml    --start 2020-01-01 --end 2026-04-30 --save-all-folds
python scripts/train_xgboost.py     --config config/ml_xgboost.yaml     --start 2020-01-01 --end 2026-04-30 --save-all-folds
python scripts/train_lstm.py        --config config/dl_lstm.yaml        --start 2020-01-01 --end 2026-04-30 --save-all-folds
python scripts/train_transformer.py --config config/dl_transformer.yaml --start 2020-01-01 --end 2026-04-30 --save-all-folds
```
→ `next_model_version` 자동 v009 (max=8+1)

3. **v009 calibrator 학습**:
```bash
python scripts/calibrate_models.py --strategy all --start 2020-01-01 --end 2026-04-30
```

4. **(선택) 짧은 sanity 백테** (4개월 OOS):
```bash
python -m src.main backtest --config config/ml_xgboost.yaml --start 2026-01-01 --end 2026-04-30
```

5. **config/ensemble.yaml 활성 옵션 변경**:
```yaml
live:
  oos_monitoring:
    enabled: true            # warm-up 자동 실행
  orderbook:
    enabled: true            # 호가창 수집 + paper VWAP
  notifications:
    channels: ["log", "telegram"]   # 텔레그램 활성 (.env에 TOKEN/CHAT_ID 있음)

# ensemble.yaml의 sub_params 안 model_path는 latest 그대로 (v009 자동 사용)
# 단 sub_params 안 calibration_method가 isotonic인지 확인
```

6. **Paper trading 1일 실행**:
```bash
python -m src.main paper --config config/ensemble.yaml
```

**검증 포인트** (1일 paper):
- DataFeed 라이브 가격 수신 안정성
- 봉 마감 트리거 → 신호 생성 → 진입/청산 흐름
- 텔레그램 알림 동작 (POSITION_OPEN/CLOSE on)
- OOS monitor warm-up 즉시 적중률 (학습 OOS 약 75% vs 현재 격차)
- 호가창 fetch + parquet 저장 (15m마다)
- DB 기록 정확성
- Circuit breaker (자연 발생 시)
- 첫 거래 발생 (~5-10건/일 예상)

**실제 운영 결과 (2026-05-05 ~ 2026-05-06, 조기 종료)**:

학습 cutoff 결정은 **2026-04-01** (3-5/일 거래 빈도 추정 시 cutoff 후 가용 봉으로 OOS warm-up window 100 충족 보장. v009 → v010으로 재학습). paper 1차 시도(v010, 21:35 KST)에서 5시간 운영 중 다수 잠재 이슈 노출 → 7건 hotfix 후 재실행 반복.

| 검증 포인트 | 상태 | 비고 |
|---|---|---|
| DataFeed 라이브 가격 수신 안정성 | ✅ 충족 | 1006 disconnect 발생 시 watchdog timeout(I-BL006 fix)으로 자동 복구 |
| 봉 마감 → 신호 생성 흐름 | ✅ 충족 | Phase 3-D fix 후 매 봉 정상 [SIGNAL] 출력 (gap=0 일관) |
| OOS monitor warm-up 즉시 적중률 | ✅ 충족 | samples=100, accuracy=0.8200, learned_oos_acc=0.7674, gap=-0.0526 (alpha decay 없음, ensemble 효과 +5.26%p) |
| 호가창 fetch + parquet 저장 | ✅ 충족 | I-BL005 fix 후 봉당 1회 정상 (이전 25,376행 폭증 → 정상화) |
| DB 기록 정확성 | ✅ 충족 | equity 봉 마감마다 정상 기록 |
| 4 sub-plugin 정상 voting | ✅ 충족 | contributors 4/4 일관 |
| 학습-추론 cycle 일관성 (Phase 3-D) | ✅ 충족 | 라이브 path도 `ts < now` 적용 — backtest와 정확 동일 cycle 명시 |
| 텔레그램 ENTRY/EXIT 알림 | ⏸ 미검증 | 거래 0건 (paper 운영 중 시장 횡보 — `H:0.94+` 강한 HOLD 신호 지속). paper 시작 전 텔레그램 자체 검증 ✅ + 단위 테스트(notifier 14건) ✅로 우회 검증 |
| 거래 진입/청산 흐름 | ⏸ 미검증 | 거래 0건. 단위 테스트(paper_executor 8건, engine_base 등)로 우회 검증 ✅ |
| Circuit breaker 자연 발생 | ⏸ 미검증 | 자연 발생 안 함. 단위 테스트 11건으로 검증 ✅ |
| OOS decay 알림 | ⏸ 미발생 | gap=-0.05라 임계 미달 (정상) |

**조기 종료 결정 근거 (2026-05-06)**:
- 핵심 검증 포인트 7개 충족 — paper의 본 목적(코드 통합 흐름 + 운영 절차 + 학습-추론 cycle 일관성) 달성
- 미검증 3개는 **시장 변동성/거래 발생 의존** — paper 더 운영해도 보장 X
- 미검증 영역은 단위 테스트로 우회 검증 + BL-2-4 진입 시 자연 검증 가능
- 시간 비용 vs 추가 검증 가치 판단

**Phase 3-D 본질 fix 효과**:
- 라이브 운영의 신호 결정 cycle이 backtest 결과를 그대로 재현 보장
- gap=0/1 변동성 사라짐 → 동일 시점 동일 데이터에 동일 신호 → **매매 일관성 결정론적 보장**
- BL-2-4 (소액 실거래) 진입 시 backtest 기대값 그대로 적용 가능

#### 4.2.4 소액 실거래 점진 전환 (BL-2-4, 사용자 운영)

**작업** (BL-2-3 paper 1일 안정 확인 후):
1. **소액 실거래** — 사안 E 결정 자금 비중 (default 1% 추천, 또는 0.5%)
2. paper trading vs 실 거래 결과 차이 정량 (수익률, 거래 수, fees, 슬리피지)
3. **점진 확장** — 1-2주 안정 후 자금 비중 증가
4. 라이브 운영 안정화 → **경로 B 본 목적 달성**

### 4.3 결정 사안 (각 step 진입 시)

#### E. 소액 실거래 시작 자금 비중 (BL-2-4 진입 시)
- (가) 1% (가장 보수적, 추천)
- (나) 5% (사용자 위험 선호)

#### G. Circuit breaker 자동 정지 임계 (BL-2-1 진입 시)
- (가) 연속 실패 3회 (보수적, 추천)
- (나) 연속 실패 5회

#### R''. (신규) 라이브 운영 모델 선택 (BL-2-3 진입 시)
- (가) dl_transformer 단독 (walkforward + Regime 일관 1위)
- (나) ensemble (단일 OOS 1위, walkforward 미검증)
- (다) 둘 다 paper에서 비교 후 결정 — 추천

#### F. 다중 거래소 우선순위 → **EXTENSION (BLE-1) 진입 시 결정**
#### S''. I-BL002 ensemble walkforward 처리 → **EXTENSION (BLE-2) 진입 시 결정**

### 4.4 검증 기준 + 결과

| Step | 검증 기준 | 결과 |
|---|---|---|
| BL-2-1 Fail-safe | notifier + CircuitBreaker + RiskManager EventBus + config_loader env 주입 | ✅ 단위 33건 통과 |
| BL-2-2 호가창 | compute_market_impact + Collector + paper_executor VWAP | ✅ 단위 22건 통과 |
| BL-2 추가 step OOS warm-up | warmup_from_history (cutoff/window/gap/안전망) | ✅ 단위 6건 통과 |
| BL-2-3 Paper trading 1일 | 코드 통합 흐름 + 운영 절차 + 치명적 버그 | 대기 (사용자) |
| BL-2-4 소액 실거래 | 자금 안전 + paper-실 거래 격차 정량 + 점진 확장 안정성 | 대기 (사용자) |

전체 회귀: 329 → **391 passed** (BL-2-1/2 + warm-up 신규 62, 회귀 0)

### 4.5 EXTENSION carry-over

다음 작업은 PATH_B_LIVE_EXTENSION으로 분리:
- 다중 거래소 (Binance + OKX) — BLE-1
- Ensemble walkforward (I-BL002 해결) — BLE-2
- Survivorship bias (다른 코인 학습) — BLE-3
- BP-1 데이터 확장 carry (펀딩률·OI + 2020 이전, 인프라 준비 시) — BLE-4

---

## 5. 진행 기록 (Phase BL-1/2)

| # | 단계 | 상태 | 커밋 | 비고 |
|---|------|------|------|------|
| 2026-05-05 | **Phase BL-1: Regime + 추가 검증** | **✅ 완료** | (이번 커밋) | 본 검증 5건 모두 통과. BL-2 진입 준비 |
| 2026-05-04 | └ BL-1-1 Lookahead 추가 점검 | ✅ 완료 | d18c37e | tests/test_lookahead.py 14건. 새 발견 0건 |
| 2026-05-04 | └ BL-1-2 Multi-hypothesis 보정 | ✅ 완료 | d18c37e | bonferroni + fdr helper. eval_260503_baseline 재처리. E-2-4 핵심 결론 유지 |
| 2026-05-05 | └ BL-1 Step A (덮어쓰기 방지) | ✅ 완료 | d18c37e | path_utils.py + 4 위치 적용. 단위 테스트 12건 |
| 2026-05-05 | └ BL-1 Step B (I-BL001 fix) | ✅ 완료 | d18c37e | calibrate_models.py가 build_labels_from_config 사용 |
| 2026-05-05 | └ BL-1-3 Walk-forward 통합 OOS | ✅ 완료 | d18c37e | 4 train --save-all-folds + evaluate_models walkforward. 4 모델 26/26 positive. dl_transformer 최우수 (Calmar 74.3) |
| 2026-05-05 | └ BL-1-4 Regime-matched 백테 | ✅ 완료 | (이번 커밋) | 4 모델 v008 학습 (2022-2024) + Regime 백테 (2020-2021 강세 bull). dl_transformer 일관 1위 (Calmar 200, MDD 3.72%, SHORT win 76.21%) |
| 2026-05-05 | └ BL-1-5 BP-1 재진입 여부 | ⚠️ 미진입 확정 | (이번 커밋) | 데이터 인프라 미준비 + I-BP001 영향 미미. I-BP001 BL 종착까지 carry-over |
| 진행 중 | **Phase BL-2: 라이브 진입** (재구성) | 진행 중 | — | BL-2-1/2 + warm-up 코드 완료. BL-2-3/4 사용자 대기 |
| 2026-05-05 | └ BL-2-1 Fail-safe (notifier + CircuitBreaker + RiskManager event_bus) | ✅ 완료 | (이번 커밋) | 단위 33건. 텔레그램 .env 통합 |
| 2026-05-05 | └ BL-2-2 호가창 인프라 (paper VWAP) | ✅ 완료 | (이번 커밋) | 단위 22건. ccxt fetch_order_book + parquet append |
| 2026-05-05 | └ BL-2 추가 step OOS warm-up (DD''=가, EE''=yes) | ✅ 완료 | (이번 커밋) | 학습 cutoff 이후 buffer 사전 채움 + 격차 ≥ 10%p 알림 |
| 2026-05-06 | └ BL-2-3 Paper trading | ✅ 완료 (조기 종료) | 467b00c | v010 ensemble paper 운영 + 7개 hotfix(I-BL003~I-BL007). 핵심 6개 검증 충족, 거래 흐름은 횡보장으로 미발생 — 단위 테스트로 우회 검증. §4.2.3 결과 참조 |
| 2026-05-06 | └ BL-2-4 사전준비: I-BL008 fix | ✅ 완료 | 84d3708 | LiveExecutor `_call` 무한 재귀 → 라이브 즉시 crash 차단. 단위 2건 (391→436 진행 중) |
| 2026-05-06 | └ BL-2-4 hotfix-G: I-BL009 + [ACCOUNT] 로그 | ✅ 완료 | 93e1fe9 | `set_margin_mode` lever 파라미터 + 매 15m 재정 상태 모니터링. 단위 4건 (440 pass) |
| 2026-05-06 | └ BL-2-4 hotfix-H: I-BL010~I-BL013 첫 SL 청산 사고 묶음 | ✅ 완료 | 0655acf | close_position skip + 강건성 + conditional order 검증 + DB recovery. 단위 10건 (450 pass) |
| 2026-05-07 | └ BL-2-4 hotfix-I: I-BL013 본질 fix (fetch_closed_orders) | ✅ 완료 | 4fd250e | fetch_my_trades → fetch_closed_orders. PnL 정확도 9배 향상. 단위 3건 추가 (453 pass) |
| 2026-05-07 | └ BL-2-4 hotfix-J: I-BL014 텔레그램 plain text | ✅ 완료 | 7854116 | parse_mode 제거 → EXIT 알림 정상 도착. 단위 2건 (455 pass) |
| 2026-05-08 | └ BL-2-4 hotfix-K: I-BL015 외부 청산 동기화 | ✅ 완료 | 841790c | 거래소 외부 청산(SL/TP spike, manual, 강제) 자동 동기화. LONG/SHORT 모두 커버. 단위 4건 (459 pass) |
| 2026-05-09 | └ BL-2-4 hotfix-M: I-BL017 DEVELOPER_GUIDE 갱신 | ✅ 완료 | (이번 커밋) | 9 영역 갱신 (§3 ML+DL/§4.3 hook 행/§4.5 진단 컨벤션/§5.1 봉 마감/§5.3 청산/§5.4 재시작 복원 신규/§5.5 cycle 일관성 신규/§10 알림+거래소 패턴/§11.6 라이브 디버깅). 라인 420→544 (+124). 인용 클래스/함수 18개 grep 검증. 단위 회귀 459 pass 유지 (코드 변경 0). USER_GUIDE §6 cross-link 3곳 추가 |
| 진행 중 | └ BL-2-4 라이브 운영 (사용자) | 진행 중 | — | 자금 점진 확장은 사용자 자율. 1-2주 안정 후 BL-2-4 종착 결정 |
| (carry) | └ I-BL016: _restore_state case 2 daily_pnl 누적 | 미해결 | — | I-BL015 후속 (낮은 우선순위) |
| (확장) | PATH_B_LIVE_EXTENSION (별도 PATH) | 대기 | — | 다중 거래소 + ensemble walkforward + Survivorship + BP-1 데이터 carry |

---

## 6. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BP001 | PATH_B_PRODUCTION carry-over (사안 H로 BP-1 스킵) | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환. `engine_base.close_position`에 `funding_fee=0.0` 하드코드. 백테에 펀딩률 미반영. (PATH_B_PRODUCTION §7 동일) | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | ~~BL-1 §3.2.6~~ → **BL 종착까지 carry-over (BL-1-5 미진입 확정)** | 미해결 — 보유 시간 평균 1.6시간으로 영향 미미 추정. **BL-2 §4.2.1 paper trading 시 펀딩률 실 비용 모니터링으로 재검증 필수** |
| I-BL001 | BL-1-2 잠재 이슈 발견 (Multi-hypothesis 보정 step에서 calibrate_models.py 점검 중 발견) | `scripts/calibrate_models.py`가 `train.label_method` 미참조 — `generate_direction_labels`만 하드코드 사용. BP-3-3에서 `train.label_method=triple_barrier`로 학습된 v006 모델에 대해 calibrate_models.py가 direction labels로 calibrator 학습 → 모델 출력(barrier hit 확률)과 calibrator 학습 라벨(direction)의 의미적 mismatch | scripts/calibrate_models.py | **BL-1 Step B** | **✅ 해결 (BL-1 Step B + 사용자 v006/v007 재calibration)** — `build_labels_from_config` helper 사용. label_params 기록. v006/v007 모두 사용자가 재학습/재calibration 완료 |
| I-BL002 | BL-1-3 종착 시 발견 (walkforward 평가가 4 단일 모델만 수행) | Ensemble plugin (BP-3-2)이 walkforward 평가 미수행. 단일 모델 walkforward 결과 (4 모델 26/26 positive)만으로 ensemble robust성 추정 불가. ensemble은 sub-plugin 인스턴스를 latest로 로드하므로 fold 모델별 평가 인프라가 필요 | src/strategy/plugins/ensemble.py + scripts/evaluate_models.py | BL-2 진입 전 별도 step 또는 BL-1-4 후속 | 미해결 — BL-2 paper 단계에서 ensemble 라이브 적용 전 walkforward 검증 권장. 단일 모델 walkforward에서 모두 robust 검증되어 우선순위 낮음. **EXTENSION (BLE-2)로 carry** |
| I-BL003 | BL-2-3 paper 시작 직후 발견 | ensemble 단독 활성 시 OOS warm-up이 ensemble buffer를 채우지 않아 paper 운영에서 warm-up 효과 없음. paper record는 strategy.name="ensemble" 1개라 sub-plugin warmup도 무용 | src/strategy/base.py + src/strategy/plugins/ensemble.py + src/live/engine.py | **BL-2-3 hotfix-A** | ✅ 해결 — `extract_train_meta` hook으로 plugin 캡슐화. 단위 10건 추가 (391→401 pass). paper 재실행 시 e2e 검증 (samples=100, gap=-0.0526) |
| I-BL004 | BL-2-3 paper 1차 시도 후 (warmup 약 1시간) | `_warmup_one_strategy`가 매 ts마다 81 피처를 처음부터 재계산 (O(N²)). BacktestEngine은 이미 features 1회 계산 후 ctx.precomputed_features에 주입하는 패턴 보유 | src/live/engine.py | **BL-2-3 hotfix-B** | ✅ 해결 — `_features_cache` 임시 주입 + try/finally cleanup. paper 재실행 시 warmup 시간 1시간 → 1분 34초 단축 e2e 검증 |
| I-BL005 | BL-2-3 paper 5시간 운영 후 (호가창 25,376행/4.5h, 기대치 ~18) | `_on_bar_closed`의 호가창 fetch가 `_should_process_bar` 검사 **앞**에 위치 — ccxt가 봉 진행 중 close 변동마다 _on_bar_closed 트리거 시 매번 fetch 발생 | src/live/engine.py | **BL-2-3 hotfix-C** | ✅ 해결 — 위치 이동 (단순). paper 재실행 시 호가창 정상 적재 |
| I-BL006 | BL-2-3 paper 5시간 운영 후 (17:15 UTC 이후 새 봉 마감 미수신 ~50분 stuck) | ccxt.pro `watch_ohlcv`가 1006 disconnect 후 reconnect는 일부 성공했지만 일정 시간 후 hang 발생. except 진입 못 해 reconnect 루프 미진입 | src/data/feed.py | **BL-2-3 hotfix-D** | ✅ 해결 — `asyncio.wait_for` 120초 timeout + TimeoutError 별도 처리. 단위 3건 추가 |
| I-BL007 | BL-2-3 paper 운영 중 발견 (ensemble HOLD 신호의 conf=1.00 빈발 → 분석 결과 라이브 추론이 backtest와 다른 cycle 가능) | sub-plugin이 진행 중 봉의 NaN row를 dropna로 우연히 제외하는 메커니즘에 의존. gap=0/1 변동성 발생 가능 → 동일 시점 동일 데이터에 다른 신호 가능 (매매 일관성 위협). 또한 sub-plugin 추론 실패 시 어떤 sub가 어떤 사유로 실패했는지 진단 정보 부재 (UX상 conf=1.00이 "확률 100% HOLD 확신"으로 오해 소지) | src/strategy/features.py + 4 sub-plugin + src/strategy/plugins/ensemble.py + src/live/engine.py | **BL-2-3 hotfix-F** (Phase 1 + 3 + 3-C + 3-D) | ✅ 해결 — Phase 1: sub-plugin fail_reason + nan_by_tf 진단. Phase 3: compute_multi_tf_features에서 진행 중 sub_tf 봉 제외. Phase 3-C: helper로 dropna 패턴 통일 (ML/DL 모두 학습-추론 일관). Phase 3-D: get_features_for_ctx 라이브 path도 `ts < now` 적용 — backtest와 정확 동일 cycle 명시 보장 (gap 변동성 제거). 단위 18건 추가 |
| I-BL008 | BL-2-4 진입 직전 라이브 모드 코드 점검 중 발견 | `LiveExecutor._call` 메서드가 자기 자신을 무한 재귀 호출 (`return await self._call(...)`) → 라이브 모드 시작 시 첫 ccxt API 호출(`set_leverage`)에서 즉시 RecursionError. paper 모드는 LiveExecutor 미사용이라 미발견. circuit_breaker 단위 테스트도 `_retry_api` 모듈 함수 직접 호출만 검증해 `_call` path 미검증 | src/execution/live_executor.py | **BL-2-4 진입 직전 hotfix** | ✅ 해결 — `self._call` → `_retry_api` 모듈 함수 호출로 정정. 단위 테스트 2건 추가 (mock으로 `_call` → `_retry_api` 호출 검증, 434→436 pass) |
| I-BL009 | BL-2-4 라이브 첫 가동 시 발견 | `LiveExecutor.initialize`의 `set_margin_mode("cross", symbol)` 호출이 OKX API 요구사항(`lever` 파라미터 1-125) 미준수 → WARNING 발생. 매 진입 주문이 `tdMode: cross` 명시하므로 거래 동작 자체엔 무영향이지만 코드 정확성 위해 fix | src/execution/live_executor.py | **BL-2-4 hotfix-G** | ✅ 해결 — `params={"lever": str(self.leverage)}` 추가. 단위 테스트는 ccxt mock 부담 > 실효성으로 e2e 검증(라이브 재시작 시 WARNING 사라짐)으로 대체 |
| I-BL010 | BL-2-4 첫 SL 청산 시 발견 | 거래소가 SL 자동 청산한 후 봉 마감 시점에 엔진의 `broker.close_position`이 redundant reduceOnly 주문 발송 → OKX 51169 reject ("don't have positions to reduce") → ExchangeError. 매 봉 마감마다 동일 에러 반복 발생 가능 | src/execution/live_executor.py | **BL-2-4 hotfix-H (Step 1)** | ✅ 해결 — `LiveExecutor.close_position`에 사전 `get_position` 확인 추가. 거래소 ∅이면 redundant 호출 skip. 단위 테스트 2건 추가 |
| I-BL011 | I-BL010과 함께 발견 | 라이브 재시작 시 거래소 conditional order(SL/TP) 생존 검증 메커니즘 부재. 거래소가 자동 cancel했거나 사용자가 수동 cancel한 경우 SL/TP 없이 운영 → 자금 위험 노출 | src/live/engine.py | **BL-2-4 hotfix-H (Step 4)** | ✅ 해결 — `_verify_and_restore_sl_tp` helper 추가. `_restore_state` 끝에서 `fetch_open_orders`로 SL/TP order 검증, 누락 시 `place_stop_loss`/`place_take_profit` 재등록. 단위 테스트 3건 |
| I-BL012 | I-BL010 분석 중 발견 (사용자 OKX 사고 traceback) | `engine_base.close_position`이 broker exception 시 시스템 상태 정리(self._position=None, DB close, event publish) 모두 미실행. 라이브에서 거래소가 이미 청산한 상태에서 redundant 호출이 reject되면 매 봉마다 재시도 + DB는 OPEN 유지 + 텔레그램 알림 누락 + daily_pnl 누락 | src/core/engine_base.py | **BL-2-4 hotfix-H (Step 2)** | ✅ 해결 — broker.close_position을 try/except로 감싸 exception 시 거래소 포지션 검증. 거래소 ∅이면 시스템 상태 정리 진행, 거래소 O이면 propagate. 단위 테스트 2건 |
| I-BL013 | I-BL010 분석 중 발견 (DB recovery 부정확) | `_restore_state` case 2(거래소 ∅ + DB O)가 모든 trade를 exit_price=entry_price, pnl=0, ENGINE_SHUTDOWN으로 close. 실제 SL 청산 손실(예: -$40.85)이 DB에 누락되어 통계/분석 부정확 | src/live/engine.py | **BL-2-4 hotfix-H (Step 3) + 본질 fix (hotfix-I)** | ✅ 해결 — `_fetch_actual_exit` helper. **초기 fetch_my_trades 사용 시 OKX 응답에 reduceOnly 식별 누락으로 fallback 발동(부정확) → 사용자 지적으로 본질 fix**: `fetch_closed_orders`로 변경 (reduceOnly + average 정확 제공). PnL = (exit-entry)×size - 진입_fee - 청산_fee 추정. 사용자 OKX 표시값과 ~$0.79 (~2%) 오차 — taker_fee_pct 정확 차이. 단위 테스트 6건 (정상 path, reduceOnly 필터, side 필터, paper, fetch 실패, SHORT TP) |
| I-BL014 | BL-2-4 두 번째 SL 청산 시 발견 | `TelegramNotifier`가 `parse_mode=Markdown` 사용. 메시지 내 특수 문자(`-`, `$`, 긴 float 등) + Markdown V1 파서의 까다로움으로 EXIT 메시지(예: `net_pnl=$-40.71` + meta의 `pnl=-40.71368847497047`)에서 텔레그램 API 400 Bad Request reject. ENTRY는 정상 작동. 결과: 사용자가 청산 알림 누락 — 라이브 모니터링 가시성 저하. 자금/DB는 정상 | src/utils/notifier.py | **BL-2-4 hotfix-J** | ✅ 해결 — `parse_mode` 제거 + plain text 사용. meta 포맷 plain key=value. 모든 특수 문자 안전 처리. 단위 테스트 2건 추가 (453→455 pass) |
| I-BL015 | BL-2-4 SHORT TP spike 청산 시 발견 (12:05 사례) | 거래소가 우리 모르게 포지션 청산하는 4가지 case (LONG/SHORT 모두 동일 패턴): SL/TP spike 청산(봉 OHLC 범위 밖 짧은 가격 spike), 사용자 manual close, 거래소 강제 청산(margin call). 봉 OHLC 기반 `check_candle_sl_tp`가 인지 못함 → 매 봉마다 self._position 살아있다고 판단, 거래소엔 ∅ → 잘못된 [POSITION]/[ACCOUNT] 로그, DB OPEN 누적, 텔레그램 EXIT 누락, daily_pnl 누락, **새 진입 기회 손실** | src/live/engine.py | **BL-2-4 hotfix-K** | ✅ 해결 — `_on_bar_closed`에 master_tf 봉 마감 시 `broker.get_position()` 동기화 추가. 거래소 ∅ 인지 시 `_sync_unexpected_close` 호출 → I-BL013 `_fetch_actual_exit` 재사용으로 정확한 청산 정보 fetch → `_close_with_funding` 정상 close 흐름 (I-BL010 skip + DB close + 알림 + daily_pnl 누적). LONG/SHORT 자동 커버 (broker.get_position이 side 무관). 단위 4건 추가 (455→459 pass). **라이브 e2e 검증 (2026-05-09 15:30 KST manual close 시연, LONG entry=$79,687.60 → exit=$80,268.30, net_pnl=+$33.97)** — I-BL010/I-BL013/I-BL014/I-BL015 5 hotfix 동시 효과 확인. reason="tp_hit" 분류 (manual close 도 LONG exit>entry 이면 TP 로 추정 — 거래소 API 한계로 진짜 manual/SL spike/TP spike 구분 불가, 시스템 동작·PnL·DB·알림은 정확) |
| I-BL016 (carry) | I-BL015 분석 중 발견 | `_restore_state` case 2(거래소 ∅ + DB O 재시작 시점)에서 DB는 close되지만 `risk_manager.daily_pnl`에 누적 안 됨 → daily_loss_limit 정확성 영향. 같은 날 재시작 vs 다른 날 재시작 정책 결정 필요 | src/live/engine.py + src/risk/manager.py | **BL-2-4 hotfix-L (예정)** | 미해결 — I-BL015 후속 작업. I-BL015의 `_sync_unexpected_close`는 정상 close 흐름 통해 daily_pnl 자동 누적 (별도 fix 필요 없음). 단 _restore_state case 2 영역만 carry |
| I-BL017 | BL-2-4 hotfix-K commit 직후 발견 (DEVELOPER_GUIDE 점검) | `docs/01_Guides/DEVELOPER_GUIDE.md`가 BL-2-1 ~ BL-2-4 hotfix들을 반영 못함. 봉 마감 흐름(§5.1)·청산 흐름(§5.3) 내용이 hotfix 전 상태. 재시작 복원·cycle 일관성·진단 패턴·새 모듈 추가 시 패턴 등 다수 누락. 새 plugin/알림채널/거래소 추가 시 가이드 부족 | docs/01_Guides/DEVELOPER_GUIDE.md | **BL-2-4 hotfix-M** | ✅ 해결 — 2 묶음 Step (§5 라이브 흐름 / §3·§4·§10·§11 인터페이스·확장)으로 진행. §5.4 재시작 복원·§5.5 cycle 일관성 신규 + §3 ML/DL train_meta·§4.3 extract_train_meta hook·§4.5 진단 컨벤션·§10 새 알림 채널 신규+거래소 패턴 확장·§11.6 라이브 cycle 디버깅 신규 추가. 라인 420→544 (+124). 인용 클래스/함수 18개 grep 검증. 단위 회귀 459 pass 유지 (코드 변경 0). USER_GUIDE §6 cross-link 3곳 추가 |

신규 carry-over 후보 ID는 I-BL018~ 형태로 등록.

---

## 7. 종착 후

### BL-1 종착 (2026-05-05) — BL-2 진입 준비 완료

**본 검증 통과 항목**:
- ✅ Lookahead 새 발견 0건 (BL-1-1)
- ✅ Multi-hypothesis 보정 후에도 E-2-4 핵심 결론 유지 (BL-1-2)
- ✅ 4 모델 26/26 positive walkforward (BL-1-3, 4.5년 OOS)
- ✅ 4 모델 강세 bull regime 모두 robust (BL-1-4)
- ✅ 인프라: 덮어쓰기 방지 (Step A) + I-BL001 fix (Step B)

**Carry-over 이슈** (BL-2에서 처리):
- I-BP001 funding_fee 백테 미반영 (BL-2 §4.2.1 paper trading 모니터링)
- I-BL002 ensemble walkforward 미수행 (BL-2 진입 전 권장)

**라이브 채택 권장 (BL-2 시작 시)**:
- 1순위: dl_transformer v007 + isotonic (walkforward + Regime 일관 1위)
- 1순위 (다양성): ensemble (단일 OOS 1위, walkforward 미검증)

### BL-2-3 종착 (2026-05-06) — BL-2-4 진입 준비 완료

**핵심 검증 통과**:
- ✅ DataFeed + WebSocket 안정성 (watchdog 자동 복구)
- ✅ 봉 마감 → 신호 생성 흐름 (gap=0 일관)
- ✅ OOS warm-up baseline (gap=-0.0526, ensemble 적중률 82%)
- ✅ 호가창/equity/DB 기록 정상
- ✅ 4 sub-plugin 일관 voting
- ✅ 학습-추론 cycle 100% 일관 (Phase 3-D)

**hotfix 7건 통합 완료** (I-BL003 ~ I-BL007):
- A: ensemble warmup buffer
- B: warmup features cache (1시간 → 1.5분)
- C: 호가창 fetch 위치
- D: WebSocket watchdog
- E: 신호/포지션 모니터링 hook
- F: sub-plugin 추론 실패 진단 + dropna 패턴 + multi-TF 진행 중 봉 제외
- G(=Phase 3-D): 라이브 path `ts < now` 명시 cycle 통일

**Carry-over** (BL-2-4 또는 EXTENSION으로):
- I-BP001 (funding_fee 백테): BL-2-4 paper-실거래 비교 시 재검증
- I-BL002 (ensemble walkforward): EXTENSION (BLE-2)로 carry
- 텔레그램 ENTRY/EXIT + 거래 흐름 e2e: BL-2-4 진입 시 자연 검증

**미검증 영역 우회 보장**:
- 텔레그램: paper 시작 전 1줄 직접 테스트 + 단위 14건
- 거래 흐름: paper_executor 8건 + engine_base 단위 테스트
- Circuit breaker: 11건 단위 테스트

**라이브 채택 결정 (BL-2-4 시작 시 그대로 사용)**:
- 모델: v010 (4 sub-plugin 학습 cutoff 2026-04-01)
- Strategy: ensemble (4 모델 + isotonic)

### BL-2-4 진행 상황 (2026-05-06 ~ 진행 중)

**라이브 운영 시작**: 2026-05-06 18:43 KST. OKX 계정 USDT 자동 조회 (시작 자금 ~$3,370).

**거래 발생 + 자동 처리 검증**:
- Trade 1: LONG entry $82,159.50 → SL $81,707.20 청산 → pnl=-$40.85 (OKX 정확값 -$40.71 시스템 기록)
- Trade 2: 두 번째 SL 청산 → 텔레그램 EXIT 알림 누락 발견 (I-BL014 trigger)
- Trade 3: SHORT TP spike 청산 ($79,139.10 / +$60.39) → 외부 청산 인지 누락 발견 (I-BL015 trigger)
- Trade 4: LONG entry $79,687.60 (14h45m held) → 2026-05-09 15:30 KST 사용자 manual close 시연 → exit=$80,268.30 / net_pnl=+$33.97 (hotfix-K 라이브 e2e 검증, 아래 참조)

**라이브 e2e 검증 (Trade 4 시연으로 동시 확인된 hotfix 효과 5건, 2026-05-09)**:
- I-BL015 (hotfix-K): master_tf 봉 마감 시 `broker.get_position()` ∅ 인지 → `_sync_unexpected_close` 자동 호출 (`Position closed externally — syncing: exit=80268.30 reason=tp_hit` 로그 확인)
- I-BL013 (hotfix-I): `_fetch_actual_exit` 의 `fetch_closed_orders` 가 정확한 exit_price=80268.30 fetch 성공
- I-BL010 (hotfix-H Step 1): `LiveExecutor.close_position` 사전 `get_position()` 검사로 redundant 호출 skip (`close_position skipped: exchange position already closed` 로그 확인)
- I-BL014 (hotfix-J): 텔레그램 EXIT 알림 plain text 정상 수신
- daily_pnl 자동 누적: 정상 close 흐름 통해 +$33.97 누적 (15:30:03 ACCOUNT 로그 `daily_pnl=+33.97` 확인)

reason 분류 한계(별건 추적 영역): `_fetch_actual_exit` 의 reason 추정은 LONG `exit>entry → TP_HIT` / `<entry → SL_HIT` 로 가격 비교로만 분류. 거래소 API 가 manual/SL spike/TP spike/사용자 close 를 구분 안 해주므로 진짜 청산 사유는 추정 불가. 시스템 동작·PnL·DB·알림은 정확.

**hotfix 진행 흐름**:
- BL-2-4 진입 직전: I-BL008 fix
- 라이브 첫 가동: I-BL009 (set_margin_mode WARNING) + `[ACCOUNT]` 로그 추가
- 첫 SL 청산 사고: I-BL010~I-BL013 묶음 (close_position skip + 강건성 + DB recovery)
- 본질 fix: I-BL013을 fetch_my_trades → fetch_closed_orders로 정확도 강화
- 텔레그램: I-BL014 (Markdown → plain text)
- SHORT TP spike 사고: I-BL015 (거래소 외부 청산 자동 동기화)
- 문서 갱신: hotfix-M (I-BL017, DEVELOPER_GUIDE 9 영역 갱신, 라인 420→544)

**잠재 이슈 carry-over**:
- I-BL016: `_restore_state` case 2의 daily_pnl 누적 (낮은 우선순위, 같은 날 재시작 시점에만 의미)

**종착 결정 권장 시점** (사용자 판단):
- 1-2주 안정 운영 + 거래 ≥ 20-30건 누적 + 이상 알림 0
- 또는 자금 점진 확장 후 안정 검증 완료 시점

### PATH_B 종착 후 (BL-2 종착 이후)

PATH_B_LIVE_TRADING BL-2 종착 = **라이브 거래 시작 = 경로 B 본 목적 달성**.

라이브 안정화 후 진행:
- **PATH_B_LIVE_EXTENSION** — 다중 거래소 / Ensemble walkforward / Survivorship / BP-1 데이터 carry
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도) — 별도 PATH 가능
- 새 모델 paradigm 탐색 (LLM 기반 등) — 별도 PATH 가능
