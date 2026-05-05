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

### 1.4 운영 권장 (BL-1-3 종착 시점, 2026-05-05 갱신)

BL-1-3 walkforward 통합 OOS (26 folds × 4.5년) 결과 반영:

**1순위 (안정성 우선)**: **`dl_transformer` v007 + isotonic**
- walkforward MDD 가장 낮음 (5.68% — 다른 모델보다 1.7-3.4%p 낮음)
- mean_return per fold 422.19% (4 모델 중 최고)
- Calmar 추정 74.3 (2위 dl_lstm 53.9보다 +38%)
- 26/26 positive fold

**1순위 (다양성/단일 OOS 우수)**: **`ensemble` (4 모델 v007 + isotonic)**
- BP-3-2 단일 OOS 1118.51% / MDD 5.27% / Calmar 212.2
- 단 walkforward 미수행 — I-BL002로 carry

**2순위**: dl_lstm (PF 4.12, Calmar 53.9, MDD 7.42%)
**3순위**: ml_lightgbm/ml_xgboost (GBDT 통계적 동등, Calmar 42-52, MDD 7.66-9.08%)

PPO 라이브 부적합 확정 (E-2-4 + BL-1-2 Multi-hypothesis 보정 후에도 유지)

원래 Phase E-2-4 권장 (참고): ml_lightgbm/xgboost + isotonic 통계적 동등 (Sharpe 10.42-10.88)

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

#### 3.2.4 호가창/유동성 모델링 정교화

**현재 상태**: `paper_executor`가 open 가격 즉시 체결 가정 (slippage_pct로만 단순 비용)

**작업**:
1. OKX 호가창 snapshot 수집 (별도 인프라)
2. 사이즈 대비 호가창 깊이로 실효 체결 가격 산정
   - 작은 size: 첫 호가 즉시 체결
   - 큰 size: 호가 여러 단계 침투 → market impact
3. `paper_executor.open_position` 확장 — book depth 인자 추가
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
| BL-1-4 Regime-matched 백테 | 대기 |
| BL-1-5 BP-1 재진입 여부 (조건부) | 대기 |

전체 회귀: 264 → **329 passed** (BL-1-1 신규 14 + BL-1-2 신규 11 + BL-1-3 신규 6 + Step A 신규 12 = 43, 회귀 0)

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
| 진행 중 | **Phase BL-1: Regime + 추가 검증** | **진행 중** | — | BL-1-1/2/3 + Step A/B 완료. BL-1-4/5 대기 |
| 2026-05-04 | └ BL-1-1 Lookahead 추가 점검 | ✅ 완료 | (이번 커밋) | tests/test_lookahead.py 14건. 새 발견 0건 |
| 2026-05-04 | └ BL-1-2 Multi-hypothesis 보정 | ✅ 완료 | (이번 커밋) | bonferroni + fdr helper. eval_260503_baseline 재처리. E-2-4 핵심 결론 유지 |
| 2026-05-05 | └ BL-1 Step A (덮어쓰기 방지) | ✅ 완료 | (이번 커밋) | path_utils.py + 4 위치 적용. 단위 테스트 12건 |
| 2026-05-05 | └ BL-1 Step B (I-BL001 fix) | ✅ 완료 | (이번 커밋) | calibrate_models.py가 build_labels_from_config 사용 |
| 2026-05-05 | └ BL-1-3 Walk-forward 통합 OOS | ✅ 완료 | (이번 커밋) | 4 train --save-all-folds + evaluate_models walkforward. 4 모델 26/26 positive. dl_transformer 최우수 (Calmar 74.3) |
| (대기) | └ BL-1-4 Regime-matched 백테 | 대기 | — | 코드 작업 (SPLIT_DEFINITIONS + MODEL_VERSIONS) + 사용자 GPU (v007 또는 v008 학습) |
| (조건부) | └ BL-1-5 BP-1 재진입 여부 | 조건부 | — | 데이터 수집 인프라 준비 시 |
| (대기) | Phase BL-2: 실거래 전환 | 대기 | — | Paper → 소액 실거래 + 다중 거래소 + Survivorship + Circuit breaker |

---

## 6. 잠재 이슈 트래커

| ID | 발생 단계 | 이슈 | 대상 컴포넌트 | 해결 단계 | 상태 |
|---|---|------|---|---|---|
| I-BP001 | PATH_B_PRODUCTION carry-over (사안 H로 BP-1 스킵) | `FeeModel.estimate_funding`이 `funding_enabled=True`라도 0 반환. `engine_base.close_position`에 `funding_fee=0.0` 하드코드. 백테에 펀딩률 미반영. (PATH_B_PRODUCTION §7 동일) | src/accounting/fee_model.py + src/core/engine_base.py + src/backtest/engine.py | BL-1 §3.2.6 (재진입 시) / 미진입 시 BL 종착까지 carry-over | 미해결 — 보유 시간 평균 1.6시간으로 영향 미미 추정. 라이브 운영 전 재검증 필요 |
| I-BL001 | BL-1-2 잠재 이슈 발견 (Multi-hypothesis 보정 step에서 calibrate_models.py 점검 중 발견) | `scripts/calibrate_models.py`가 `train.label_method` 미참조 — `generate_direction_labels`만 하드코드 사용. BP-3-3에서 `train.label_method=triple_barrier`로 학습된 v006 모델에 대해 calibrate_models.py가 direction labels로 calibrator 학습 → 모델 출력(barrier hit 확률)과 calibrator 학습 라벨(direction)의 의미적 mismatch | scripts/calibrate_models.py | **BL-1 Step B** | **✅ 해결 (BL-1 Step B + 사용자 v006/v007 재calibration)** — `build_labels_from_config` helper 사용. label_params 기록. v006/v007 모두 사용자가 재학습/재calibration 완료 |
| I-BL002 | BL-1-3 종착 시 발견 (walkforward 평가가 4 단일 모델만 수행) | Ensemble plugin (BP-3-2)이 walkforward 평가 미수행. 단일 모델 walkforward 결과 (4 모델 26/26 positive)만으로 ensemble robust성 추정 불가. ensemble은 sub-plugin 인스턴스를 latest로 로드하므로 fold 모델별 평가 인프라가 필요 (`evaluate_models.py walkforward`가 ensemble.sub_params의 model_path를 fold_dir로 오버라이드하는 mechanism 부재) | src/strategy/plugins/ensemble.py + scripts/evaluate_models.py | BL-2 진입 전 별도 step 또는 BL-1-4 후속 | 미해결 — BL-2 paper trading 단계에서 ensemble 라이브 적용 전 walkforward 검증 권장. 단일 모델 walkforward에서 모두 robust 검증되어 우선순위 낮음 |

신규 carry-over 후보 ID는 I-BL003~ 형태로 등록.

---

## 7. 종착 후

PATH_B_LIVE_TRADING BL-2 종착으로 "경로 B" 모든 작업 완료. 라이브 운영 안정화 후:
- 다른 거래쌍 확장 (Survivorship bias §4.2.3 후속)
- 다른 시간프레임 추가 (1m/5m 고빈도 또는 1h/4h 저빈도)
- 새 모델 paradigm 탐색 (LLM 기반 등 — 새 PATH 시작 가능)
