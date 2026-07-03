# 01 — BTC 퀀트 모델 v1 설계

이 문서는 BTC 무기한 1h **HMM 레짐 모델**의 설계·구현 작업 보고서다.
Phase 단위로 진행 기록·결정·잠재 이슈를 누적한다.

- **모델 스펙**: [btc_quant_spec_v1.md](../99_Brainstorming_Docs/00_btc_quant_model_brainstorming/btc_quant_spec_v1.md)
- **브레인스토밍 전문**: [btc_quant_session_transcript.md](../99_Brainstorming_Docs/00_btc_quant_model_brainstorming/btc_quant_session_transcript.md)
- **인프라 구조**: [INFRA_GUIDE.md](../01_Guide_Docs/INFRA_GUIDE.md)

---

## 확정 방향 — (가) 선택적 재구성

기존 Base 인프라를 **3계층으로 갈라** 취급한다 (D1 검증으로 타당성 확인):

| 계층 | 대상 | 처리 |
|---|---|---|
| **A 메커니즘** | lookahead 차단·FeeModel PnL·단일슬롯·봉마감루프·SL/TP 시뮬·백테 리포트·indicators·회귀테스트 | **재사용** (엔진 수정 0~최소) |
| **B 정책 계약 표면** | 필수6 메서드 형태·should_reverse·플러그인 구조·helpers | **모델 맞춤 재설계** (엔진 수정 허용) |
| **C 라이브 OKX 레이어** | live_executor·trade_sync·feed·orderbook·notifier (~2,700줄) | **보류** (라이브 단계까지 미사용·미폐기) |

근거: 마찰은 전부 B(정책 표면)에 집중. 재구현 시 가장 위험한 부분(lookahead·PnL)은
정작 충돌 없는 A → 재구현은 순수 리스크. "계약을 모델에 맞춰 바꾼다"가 핵심.

---

## 진행 기록표

| Phase | 내용 | 상태 | 일자 | 커밋 |
|---|---|---|---|---|
| D1 | 메커니즘 A 독립 검증 + 스펙↔인프라 갭 분석 | ✅ 완료 | 2026-06-29 | d76d3cc |
| D2 | 정책 계약 표면 B 재설계 + O-2·O-4·O-5 결정 | ✅ 완료 | 2026-06-30 | (미커밋) |
| D3 | 학습 파이프라인 설계 (데이터분할·feature·t-HMM·K선택/매핑·walk-forward) + O-1·O-3·O-6 결정 | ✅ 완료 | 2026-07-01 | 5f8a288 |
| D3.5 | 구현 전 점검 (코드검증·환경·DD-1) — GO | ✅ 완료 | 2026-07-01 | 5f8a288 |
| D4 | 구현 (D4-1·D4-2·D4-3 ✅ / D4-3.5 진행중 / D4-4~5 예정) | 진행중 | 2026-07-01~ | D4-3 b14f175 / 백테분석·τ스윕 f90ae56·5841f6e / logic·churn가드 eb9d8e1·216612f |

---

## Phase D1 — 메커니즘 검증 & 갭 분석 (완료)

### 결론: 메커니즘 A 재사용 확정 (엔진 수정 0)
독립 fresh-eyes 검증(협업규칙 14) + 사전 갭 분석이 두 핵심 난점에서 **독립 수렴**.
치명 결함(미래 데이터 누출, PnL 불일치) 없음. lookahead 차단·PnL 정합성·단일슬롯·
SL/TP 교차 판정 모두 코드상 정확, 백테/라이브가 동일 helper 공유(DRY·캡슐화 양호).

### 핵심 결정사항
- **D-1 진입/청산가 = "신호 봉 마감 직후 = 다음 봉 시초가"** (종가 아님).
  - 레짐은 양쪽 다 직전 마감 봉(T)으로 계산 → 신호 동일·lookahead 없음. 체결가만
    `close_T`(스펙 원안) vs `open_{T+1}`(엔진). 라이브 피드가 새 봉 첫 tick(open≈close)
    발행 → 백테 `open_{T+1}` = 라이브 `close(첫tick)` = 동일 의미. 엔진 쪽이 실행지연
    반영해 더 정직. **조건**: 모델은 SL/TP·사이징을 실제 체결가(`ctx.current_price`)
    기준으로 계산. **인지**: 백테에 진입 슬리피지(close→next open)가 현실 비용으로 포함.
- **D-2 레짐 전환 스왑 = `should_force_exit` + 빈슬롯 진입** (should_reverse 미사용).
  - reverse 경로는 후보 신호가 actionable해야만 청산 → 저confidence 적대전환 시 청산
    못 하는 불일치. 스펙의 "적대성 먼저 처리 → 비면 새 진입" 2단계 원칙과 어긋남.
    각 전략이 자기 레짐 종료를 force_exit로 청산 → 같은 봉 빈 슬롯 경쟁이 정합. → **D2에서 should_reverse 인터페이스/엔진 흐름 제거**.
- **D-3 공유 regime 모듈** — 2-플러그인이 HMM을 봉당 2번 돌리는 중복 차단. 발견→매핑
  (HMM·contract)을 봉당 1회 계산·캐시, 추세/평균회귀 매매 로직이 소비. (스펙 "전략
  완전 분리"는 매매 로직 분리지 레짐 계산 분리가 아님 → 모순 없음.)
- **D-4 HMM filtering** — *(D4-2 정정: incremental → sliding-window)* 매 봉 **직전 W봉(100)
  sliding-window filter**(forward-only). 원안 incremental(α 상태 carry)은 라이브 재시작 시
  validity 시작부터 재현 불가 → 백테 불일치(H4 위반). sliding-window 는 백테·라이브가 같은
  W봉을 filter → 동일 γ. O(W·T)(O(T²) 아님), contract 모델당 1회라 스윕 재사용. 엔진 수정 불필요.

### 인지사항 (메커니즘 수정 불필요, 해석/구현 시 감안)
- **백테 낙관 편향**: 백테는 봉 full high/low로 SL/TP 시뮬, 라이브는 시초 한 점 +
  거래소 conditional order 의존 → 백테가 SL/TP를 더 정확/자주 체결. + 백테 funding=0
  stub. 트레일링·장기보유(추세) 모델이라 백테-라이브 갭 요인.
- **회귀 테스트 부재 4건** (코드는 맞으나 미보장) → I-002, D4 백로그.

---

## Phase D2 — 정책 계약 표면(B) 재설계 (완료)

### Step 1 — 플러그인 구조: (가) 단일 디스패처
```
RegimeQuantStrategy (plugin 1개, entry_tf=1h)
├── RegimeService   # 발견+매핑: HMM 로드·sliding-window filtering·Contract 산출(봉당 timestamp 캐시)
│                   #   Contract = {type, direction, confidence(γ), volatility(ATR24)}
├── TrendLogic      # 추세 매매 (별도 파일/클래스)
└── RangeLogic      # 평균회귀 매매 (별도 파일/클래스)
```
- 레짐 상호배타 → 단일 슬롯 자연 점유. 전역 싱글톤 회피. 스펙 "전략 완전 분리"는 매매
  로직(TrendLogic/RangeLogic 파일 분리)으로 충족(레짐 계산은 공유가 정합).
- 대안 (나) 2-독립플러그인+공유 캐시는 전역 상태 비용으로 기각.

### Step 2 — 인터페이스 변경 (B, 코드는 D4에서 일괄 구현)
| 변경 | 파일 | 내용 |
|---|---|---|
| 제거 | base.py | `should_reverse` abstract |
| 제거 | engine_base.py | reverse flow(`evaluate_strategies_on_bar` step3 L308-335) + 주변 死코드(L333-335 `if None return`, pyramid `held_strategy` 재취득) 정리 |
| 제거 | strategy/helpers/reverse.py | should_reverse 전용 헬퍼(dead) |
| 제거 | core/enums.py | `ExitReason.REVERSE_SIGNAL` — reverse flow 제거 후 死코드(enums:46+engine_base:328 2곳뿐, 검증완료) |
| 추가 | core/enums.py | `ExitReason.REGIME_EXIT` (적대 전환 청산 리포트) |
| **추가** | base.py + engine_base.py | **`update_take_profit` 옵션 훅** — `update_stop_loss`와 대칭. `check_strategy_exits`에서 매 봉 `position.take_profit` 갱신 → 평균회귀 **이동 중심선 TP** 추적(봉내 정확 체결) |

**무변경(기존 메커니즘/컨벤션 충족)**: Signal↔contract(`Signal.meta`+`confidence`),
적대청산(`should_force_exit`), 트레일링 동결(`update_stop_loss` None).

**스펙 전환정책 → 훅 매핑**:
| 스펙 규칙 | 구현 위치 |
|---|---|
| 적대적 전환 → 즉시 청산 | `should_force_exit` → `ExitDecision(REGIME_EXIT)` |
| trend→range → 트레일링 동결·보유 | `update_stop_loss` None(동결) + `should_force_exit` None |
| 청산 후 새 레짐 진입 | 빈슬롯 `generate_signal`(새 레짐) — 같은 봉 |
| 우호적 전환 → TP/SL 유지 | 훅 무동작(보유) |

**명확화 5건 (D4 준수)**:
1. reverse 제거 = 주변 死코드 정리 포함(단순 삭제 아님).
2. contract는 `RegimeService.get_contract(candles)`에서 **봉당 timestamp 캐시**(한 봉에 여러
   훅이 호출해도 1회 계산). filter 는 직전 W봉 sliding-window (D4-2 정정: incremental carry 아님 — 라이브 재시작 일관성).
3. 포지션↔로직 소속 + **contract→position.meta**: 엔진이 `signal.meta`를 `position.meta`로 자동복사 안 함 → `on_position_opened`(성공 진입, signal 미수신)에서 인스턴스 stash로 수동 기입, `on_position_closed`에서 해제.
4. `sl_tp_fill_priority` 단일값 `"sl_first"` — 추세 TP 없음→동시도달 불가, 평균회귀만 의미.
5. `allow_entry` → True (confidence 게이트는 `generate_signal`에).

### Step 3 — 트레이딩 결정
- **O-2 = (가) 리터럴 비례**: `size = risk_based_size(진입, sl=k_sl·ATR, balance, risk%, max_leverage) × confidence`.
  - confidence가 게이트(θ)를 이미 통과 → 경계에서 추가 축소(나)는 이중 보수화라 기각.
  - risk% = **직교적 리스크 예산**(SL 거리 불변, 사이즈만 조절·손익 진폭). **스윕(0.5~2~3%)**,
    평가축 = 수익률 **+ max_drawdown**(추세=낮은승률→연속손실→DD). 천장=감내 가능 최대낙폭.
  - max_leverage = 저변동/타이트SL에서만 무는 **꼬리위험(SL 실패·청산) 가드레일**, 정상변동성엔
    inert. `≤ exchange.leverage(=5)`.
- **O-5 = (가)**: gen1 백테 funding=0 수용 + 명시. 실보유기간·funding 노출 측정 후 2세대 판단(I-001 유지).
- **D3로 이관**: O-1(type 임계값) · O-3(walk-forward 방식) — 발견층이라 학습 파이프라인과 함께.

### 구조 점검 (협업규칙 8) 요약
- DRY: 적대청산 force_exit 1곳, 레짐 계산 RegimeService 1곳.
- 캡슐화: reverse flow 제거로 전환 정책이 전부 전략(force_exit)으로 수렴, 엔진=메커니즘만.
- 미래 확장성: Contract 4필드 고정 → HMM 교체 영향 국소. should_reverse 제거는 단일 플러그인 전제라 무관.
- 1회용 코드: 해당 없음(인터페이스 설계).

---

## Phase D3 — 학습 파이프라인(발견층) 설계 (완료)

> D3 = 설계·방법·결정. 실제 학습·K확정·임계값·스윕은 **D4 실행**.

### Step 0 — 데이터 분할 전략
- **Anchored(확장) walk-forward**: train 시작 2020-01 고정, N개월 재추정.
- **Dev 2020-01~2022-12**(튜닝 전용) / **OOS 2023-01~2026-06**(무튜닝 정직 평가) / **2018 스트레스 홀드아웃**(학습 미사용).
- **3계층 검증**: K·임계 method(Dev) / HMM params(walk-forward 재추정) / 트레이딩계수(Dev 스윕, OOS 고정).
- 근거: anchored → 2022 약세장이 모든 train 윈도우 포함 → trend/short 커버리지 보존. 2022(유일 지속형 약세)는 시간순상 Dev.
- 양 블록 3레짐 보유 확인. 한계 I-004: OOS 하락=단일 에피소드(2025-10~2026-02 ~47%), 2018로 보완.

### Step 1 — feature 파이프라인
- 3 코어: 로그수익률(1봉) / **실현변동성 std(24봉 로그수익률)** (S1-1=가) / Kaufman ER(48봉, indicators.py:67).
- z-score: train 윈도우 params만, **artifact 동행**, test/live 적용(재계산 금지=누수차단).
- 혼동 주의: feature 변동성(정규화·HMM 입력) ≠ contract.volatility(ATR24·가격·SL/TP).
- **윈도우 경계 체크리스트(I-005, D4 단위테스트)**: H1 first-valid=48·min_periods full / H2 현재봉 close 포함(causal)·shift 미적용 / H3 walk-forward warmup lookback / H4 sliding-window 일관(백테·라이브가 직전 W봉 동일 filter → 동일 γ·ATR도 bounded 슬라이스) / H5 rolling std ddof·min_periods 일관 / H6 모델 스왑은 valid_period 선택+sliding-window 로 자동(별도 α 재init 불필요).

### Step 2 — t-HMM 모델 + 구현경로
- 다변량 **Student's-t emission**(μ_k,Σ_k,ν_k) + transition A + π. EM(Baum-Welch, t=scale-mixture, ν=digamma root-find). 추론=**forward-only filtering**(sliding-window), log-space.
- 명확화: **EM은 forward-backward OK(학습)**, **추론만 forward-only(신호)**. D4 과잉적용 금지.
- **O-6=(나) 단계 구현**: Gaussian 스캐폴드(plumbing 검증) → emission만 t 교체. **gen1 최종=t(필수 교체)**, K·임계 최종값은 t-모델. **검증=Gaussian·t 모두 합성데이터 복원 + analytic spot-check** (hmmlearn 은 Py3.14 빌드 불가·Gaussian 한정이라 미사용 — D4-1 결정).

### Step 3 — K 선택 + 매핑
- K=2~6, **3렌즈**: holdout LL(elbow) + BIC(ν 포함) + **커버리지 진단표**(trend/long·short·range/none). 값은 D4.
- 매핑(offline·train·raw 단위): type=|mean_return_k|/vol_k vs τ; direction=sign(trend)/none(range); **confidence=γ 집계**(동일 contract 매핑 상태들의 γ_t 합); volatility=ATR24.
- 추론: γ_t → contract별 집계 → 최대 질량 contract*. 매핑표 artifact 고정.
- **O-1=(가)** 손 고정 τ + 스윕(anchored 전 윈도우 전역 동일).

### Step 4 — walk-forward + artifact + RegimeService
- **O-3=(가) 오프라인 사전학습**: 윈도우별 artifact 선학습, RegimeService는 로드만. causal·백테빠름·라이브일관. 재추정 N=3~6개월(D4 스윕).
- **artifact 스키마**(윈도우당): K·emission_kind / π·A·{μ_k,Σ_k,ν_k}(z공간) / zscore{μ_f,σ_f}×3 / feature_config / mapping(state→type,dir) / τ·raw_state_stats / valid_period / meta. 직렬화 **단일 JSON**(모델 작아 npz 불필요·pickle 회피).
- **스왑·일관 (D4-2 정정)**: sliding-window(직전 W봉 filter)가 burn-in 을 매 봉 내장 → 별도 cold 재init 불필요. 모델 스왑은 valid_period 선택으로 자동. warmup 미충족만 None.
- **RegimeService.get_contract(candles)**: ts 모델선택 → 직전(W+warmup)봉 feature(causal)+zscore → warmup시 None → **직전 W봉 sliding-window filter** → γ 집계 + **bounded ATR**(직전 W+24봉) → Contract. 봉당 timestamp 캐시(H4). (백테=라이브 동일 입력→동일 출력.)
- **디렉토리**: training `src/strategy/regime/training/`(또는 scripts) / runtime `src/strategy/regime/` / plugin `plugins/regime_quant.py`+`regime/{trend,range}_logic.py` / artifact `data/regime_models/`(untracked).

### D3 결정 요약
S1-1(가 실현변동성) · O-1(가 손고정 τ+스윕) · O-3(가 오프라인 사전학습) · O-6(나 Gaussian→t 단계).

---

## 구현 전 점검 (D3→D4 게이트) — ✅ GO

독립 fresh-eyes 2개(엔진 수정 가능성 / 재사용·지표) + 환경·데이터 + 자체 설계 재검토(협업규칙 14). **블로커 없음.**

### 검증 결과
- **엔진 수정 4건 전부 가능·저위험·테스트 무파손**: should_reverse 제거(base:81-87 + stub:35 + helpers/reverse.py + test_helpers reverse 섹션) / reverse flow 제거(engine_base:308-335 + 338 held 재취득 정리, 엔진 reverse 회귀 테스트 부재) / `update_take_profit`(check_candle_sl_tp가 매 봉 `position.take_profit` 직독:656 → **이동 TP 성립 확인**) / `REGIME_EXIT`(by_exit_reason 동적 집계라 무문제).
- **재사용 확정**: 6 메서드·`Signal.confidence`+`.meta`·`Position.meta`·AccountState 5필드·`risk_based_size×confidence`·지표 ATR/BB/RSI/ER. → **엔진 수정 0으로 모델 구현 가능**.
- **환경**: scipy 1.17·pandas_ta·sklearn·matplotlib ✅(Py3.14). ⚠️ hmmlearn 빌드 불가(MS C++ 빌드툴 필요) → 드롭(합성복원+analytic 대체) / 2018 데이터 없음(2019-12부터).

### DD-1 — 추세 진입 정책 (결정)
**관대 진입**: `flat ∧ 레짐=trend ∧ direction ∧ conf≥θ_trend → 진입`. 전환 봉 한정 X, 에피소드당 횟수 제한 X(전환 시 conf<θ여도 이후 conf≥θ면 진입). 구현 단순(에피소드 상태 불필요).
- 리스크: 트레일링 손절 직후 재진입 churn(휩쏘) → gen1 무가드, **D4 측정(I-006)**, 2세대 가드 판단(스펙 "측정 후 방어" 철학).

### t-HMM 검증 (hmmlearn 드롭 — D4-1)
hmmlearn 은 Py3.14 빌드 불가 + Student's-t 미지원(Gaussian 한정) → **드롭**. 대신 **Gaussian·t 모두 합성데이터 복원 테스트**(알려진 HMM 생성→적합→파라미터·posterior 복원) + **analytic spot-check**(소형 예제 forward 확률 손계산)로 검증. self-contained(numpy/scipy), ground-truth 대조 → t까지 커버.

### D4 변경 표면 (확정)
1. **indicators 신규 2건**: `compute_log_returns`, `compute_realized_vol`.
2. **엔진 수정 4건**(위) + 미사용 `REVERSE_SIGNAL` enum 제거.
3. **정리 동반**: helpers/reverse.py, test_helpers.py reverse 섹션, strategy_stub.py:35.
4. **신규 모델**: RegimeService·feature·filtering·Contract·디스패처·Trend/Range 로직·학습 파이프라인·artifact.
5. **contract→position.meta**: on_position_opened 수동 기입(signal.meta 자동복사 없음).
6. **requirements**: scipy 추가(HMM 핵심). hmmlearn 미사용(빌드 불가·드롭).
7. **회귀 테스트**: I-002(4종)·I-005(경계 H1~H6)·합성복원(Gaussian+t)+analytic spot-check·I-006(churn 측정).
8. **2018 스트레스 데이터**: 스트레스 테스트 단계에 다운로드.

---

## Phase D4 — 구현 (진행중)

> 백테 실행·스윕은 사용자 직접 수행(백테 정책) — 커맨드 가이드 제공. 각 단계 Phase 단위 커밋.

### 단계 구성
| 단계 | 내용 | 게이트 |
|---|---|---|
| **D4-1** ✅ | 인프라 prep: 엔진 4수정·indicators 2·정리·requirements | 회귀 258 + update_take_profit·트레일링 e2e |
| **D4-2** ✅ | 발견층(Gaussian): feature·EM·sliding-window filter·K선택·매핑·artifact·RegimeService | 회귀 294 + 합성복원·analytic·causal·M-step 참조대조 |
| **D4-3** ✅ | 매매층: 디스패처+Trend/Range(DD-1·None) + 빌드도구·Dev 백테 가이드 | 회귀 358 + E2E·I-002④·churn(I-006)·정합성 |
| **D4-3.5** 진행중 | Gaussian 계수 스윕(τ·θ·k·churn가드) + range 원인분석(v2 프로파일링) | trend 수익전환 확정 ✅ + range 유효성 진단 |
| D4-4 | Gaussian→t emission 교체 | 합성복원(t) + Dev 재백테 |
| D4-5 | walk-forward OOS + 계수 스윕 + 커버리지 최종 + 2018 스트레스 + 낙관편향/funding 해석 | OOS 정직 평가 |
| (범위 밖) | 라이브 통합(C): RegimeService 상태 복원·OKX 배선 | 라이브 단계 |

### D4-1 — 인프라 prep (완료)
- **엔진 수정 4건**: reverse flow·should_reverse·REVERSE_SIGNAL 제거 / update_take_profit·REGIME_EXIT 추가.
- **indicators 2건**: compute_log_returns·compute_realized_vol.
- **정리**: reverse.py 삭제 + reverse 전수 스윕 9곳(src·docs) + stub·test_helpers + CLAUDE/INFRA_GUIDE/README 인터페이스(필수 6→5, update_take_profit 추가).
- **requirements**: scipy 추가, hmmlearn 드롭(Py3.14 빌드 불가).
- **회귀 258 통과** (253 + 동적 SL/TP 신규 5: update_take_profit 갱신·None유지·update_stop_loss 갱신·트레일링 SL e2e=I-002③·이동 TP e2e).
- **잔여**: I-002 ①②(lookahead 절단·진입가)는 D4-2/D4-3 실데이터 통합 시 작성(기존 메커니즘, 258에 간접 포함).

### D4-2 — 발견층 Gaussian (완료)
- **신규 모듈** `src/strategy/regime/`: contract·features·hmm·mapping·selection·artifact·service·training.
- **HMM**: Emission ABC(m_step 자기완결 → t 수용) + GaussianEmission(full cov+reg) + log-space EM(다중 init·best LL) + **forward-only filter**.
- **추론(RegimeService)**: **sliding-window filter**(직전 W봉) + **bounded ATR**(직전 W+24봉) + feature 도 직전(W+warmup)봉만 계산(O(T²)→O(T)) → **백테=라이브 동일(H4)**. 봉당 timestamp 캐시.
- **매핑**: raw 로그수익률 per-state 통계 type=|μ|/std vs τ·direction·confidence=γ집계. **K선택**: holdout LL+BIC+커버리지 진단·추천.
- **artifact**: 단일 JSON (emission type-agnostic, t-swap 대비).
- **검증 (독립 fresh-eyes + 자체)**: 실버그·lookahead **0**. EM=참조 Baum-Welch atol 1e-10 일치. 추론 causal(smoothed 학습 격리·음성 가드 테스트). **회귀 294** (D4-2 +36: features7·hmm9·mapping6·selection3·artifact2·service9).
- **설계 정정 2건(H4)**: ① D-4 incremental→sliding-window(라이브 재시작 일관) ② ATR 전체RMA→bounded 슬라이스.
- **잔여**: I-007(EM 학습 성능) → D4-5. I-002 ①②는 D4-3 통합 시.

### D4-3 — 매매층 (완료, 커밋대기)

> 4 Step 분할, 각 Step 종착 후 **독립 fresh-eyes 교차검증(규칙14)** + 자체 비판 재점검.
> 치명 결함 0. **회귀 358** (D4-2 294 + D4-3 64).

- **S1 매매 로직 (34 테스트)** — `regime/trend_logic.py`(TrendLogic) · `regime/range_logic.py`(RangeLogic):
  - Trend(스펙§3): 관대진입(DD-1) · 초기SL(∓k_trend_sl·ATR) · TP None · 트레일링(동방향 trend만·유리방향만, trend→range·적대는 동결) · force_exit(적대→REGIME_EXIT) · 사이징(risk_based_size×conf, O-2).
  - Range(스펙§4): 셋업/트리거 상태기계(IDLE/ARMED_LONG/ARMED_SHORT, 만료3종: 레짐이탈·거리만료·RSI해제) · 이동 BB중심선 TP · SL(∓k_range_sl·ATR) · force_exit(적대).
  - **I-008 신규**: RangeLogic BB/RSI 를 직전 indicator_window 봉(bounded)으로 계산 → 히스토리 길이 무관(백테=라이브, H4). RSI(Wilder RMA)·ATR 의 D4-2 정정② 와 동종 문제를 동일 패턴으로 해소.
- **S2 디스패처 (13)** — `plugins/regime_quant.py`(RegimeQuantStrategy):
  - RegimeService+Trend/Range wiring, artifact 로딩(`model_dir`), 필수5+선택훅을 `position.meta["logic"]`/`signal.meta["logic"]` 로 디스패치, contract→position.meta stash(원시 dict, DB직렬화 안전).
  - config `regime_quant:` 섹션(17키), `active: []` 뼈대 유지(규칙7).
  - 독립검증 반영: `_HOLD` 싱글톤 제거(mutable meta 공유 함정) · `ctx.candles` 접근 `.get()` 통일.
- **S3 E2E (7)** — `tests/test_regime_quant_e2e.py`, 실 BacktestEngine + tiny Gaussian artifact:
  - 정합성(규칙10, trades pnl합↔metrics↔equity 3자일치) · **I-002④ 레짐스왑 같은봉**(청산→같은 ts 재진입) · **I-006 churn 계측** · range 적대청산 · **service 통합 스모크(실 HMM filtering→매매)** · trend 트레일익절 · range TP체결.
  - §5.2 전환순서(적대먼저→비면 진입) · force_exit vs SL 평가순서 실엔진 확인.
- **S4 빌드도구 (10)** — `training.make_anchored_windows` + `scripts/build_regime_artifacts.py`:
  - anchored walk-forward 윈도우(train_end=valid_start−1h 누수차단, tz-aware) · CLI(빌드 시 항상 window_*.json 정리) · Dev 백테 커맨드 가이드.
  - 독립검증: walk-forward 미래참조·train/valid 누수·tz 경계 **전부 정확** 확인.
- **게이트 충족**: E2E ✅ · I-002④ ✅ · churn(I-006) 계측 ✅ · 정합성 ✅.
- **신규 파일**: `regime/{trend,range}_logic.py` · `plugins/regime_quant.py` · `scripts/build_regime_artifacts.py` + 테스트 5(trend·range·plugin·e2e·training=64). **수정**: `regime/training.py`(make_anchored_windows) · `config/default.yaml`(regime_quant) · `.gitignore`(data/regime_models/).

### 백테 전 재검증 (D4-3 후 게이트) — ✅ GO

독립 fresh-eyes 3개(R1 D4-1·R2 D4-2·R3 D4-3, 규칙14) + 통합 검증(V1 파이프라인·V2 회귀·V3 코드↔문서). **3 커밋(b83a9f7·192c2fe·b14f175) 변경 파일 전수 매핑**(추가·수정·삭제 누락 0). **백테 블로커 0.**

- **R1 D4-1**: 엔진 4수정 메커니즘만(정책 누출 0) · reverse 죽은참조 0 · indicators causal · scipy↔import 배선 · test_engine_base tautology 아님.
- **R2 D4-2**: 추론 `filter`-only(smoothing 누출 0, 추론경로 죽은참조조차 없음) · z-score train-only 누수차단 · service bounded H4 일관 · EM 참조 atol 1e-10 · artifact round-trip 무손실.
- **R3 D4-3**: 스펙§3/§4 정합 · 디스패치(`logic` 위임) · walk-forward 누수/tz · config 17키 양방향 · S1~S4 누적 보강 전부 반영 · regime 통합 100 테스트.
- **V1**: 발견→매핑→매매 service 통합 스모크 완주 · Contract 4필드 frozen 고정(HMM 교체 무관). **V2**: 회귀 **358 passed**. **V3**: `INFRA_GUIDE:44` 필수6→5 stale 수정(D4-1 전수스윕 누락분), requirements·config·트래커 일치.
- **경미(전부 비블로커, 결과영향 0/의도)**: `realized_vol` ddof=1 명시부재 · 진입가 동일성 전용회귀 부재 · M-1 n_init 비교 stale LL(저장값 재계산 정확) · M-2 빈상태 range 매핑(보수) · valid_end 경계봉 1개 HOLD(의도된 무거래).
- **백테 추적 매트릭스**: **정합성(규칙10) 최우선**(trades pnl합↔metrics↔equity) + I-001(추세 평균보유봉)·I-004(by_direction trend/short 표본)·I-006(진입간격·SL→재진입)·I-007(빌드 wall-clock)·I-005-H3(윈도우 경계봉 contract None).

### D4-3 Dev 백테 baseline 분석 (Gaussian, 2021-2022)

> 사용자 직접 실행(백테 정책). 결과 텍스트 보존(규칙8-4), 분석 스크립트는 1회용(미커밋).

**실행 설정**: 백테 2021-01-01~2023-01-01 (train anchored 2020-01~, walk-forward 3개월 재추정 8윈도우) / Gaussian K=3·τ=0.5·n_init2·n_iter50 / θ_trend=θ_range=0.6·k_trend_sl2.0·k_trend_trail2.5·k_range_sl1.5·risk1%·BB(20,2)·RSI(14,30/70). ※ Dev 설계(2020-2022) 중 2020은 train warmup, 실 평가는 2021-2022.

**결과**: initial 10,000 → final 4,884.19, **total_pnl -5,115.81 (-51.16%)**, 315거래, 승률 43.81%, max_dd 56.35%.

**검증 — 결과 신뢰 가능 (구현 결함 0)**
- 정합성(규칙10): trades pnl합 = metrics total = equity변화 = **-5,115.81 완전 일치**.
- 부호: 방향-부호 불일치 4/315(수수료 경계). SHORT 하락→이익·LONG 상승→이익 정상.
- 사이징: `risk_based_size`(balance×risk%÷SL거리)가 거래당 balance 대비 정확히 risk%(1%) 손실 보장(코드·test_helpers). notional/balance 중앙 0.48x 정상 레버리지. (거래당 pnl_pct 중앙 -1.003%는 **notional 기준** 손익 규모 — balance 기준 risk%와 분모가 달라 직접 동일시 안 함.)

**근본 원인 — 매핑 τ 스케일 미스매치**: artifact 8윈도우×3상태 = **24개 전부 range/none**. per-state \|μ\|/σ=0.003~0.109인데 **τ=0.5**라 trend 판정 불가 → **추세매매 0회(`trend_logic` dead), 100% 평균회귀**(regime_exit 0, tp_hit 149=range).

**심층 분석**
- 연도별: 2021 -2,506(승률 47.9%) / 2022 -2,610(승률 **39.5%**) — 하락추세일수록 승률 급락.
- exit: sl_hit 166(평균 -80.3) / tp_hit 149(평균 +55.2) / regime_exit 0. 손익비 **PF 0.62**(avg_win 60.5 / avg_loss -76.0, 손익분기 승률 55.7% ≫ 43.81%).
- 보유기간 중앙 **5시간**(레짐 지속 17~48봉과 무관하게 조기 청산). churn(I-006): 진입간격 중앙 38h·연속<2h 2.5% → 이번 경미(range 셋업대기). max_dd 56%: 과베팅 아님(notional/balance 0.48x), 승률열위 복리누적.

**발견층 진단 — 건강 (매핑 τ만 병목)**: self-transition 0.94~0.98 → 상태 지속 0.7~2일(매매호흡 정합). **상태1: 저변동 상승(mean +0.00056/봉, \|μ\|/σ=0.109, ~20봉 지속)** = 방향성 뚜렷한데 τ=0.5가 range로 오분류.

**시장 특성 (Dev 2020-2022 실캔들)**: 분기 ER 대부분 <0.1(전체 0.002)·분기수익 ±40~168% = **"방향 크나 경로 지그재그인 변동성 큰 추세"**(clean trend/range 아님). 진짜 횡보(작은수익+낮은 ER)는 2021Q4·2022Q1·2022Q3 3분기뿐. → trend/range 구분 자체가 어려움(τ 0.02~0.11 좁은구간 민감 원인), 지그재그 추세는 트레일링 휩쏘 위험.

**해석 (정정)**: "평균회귀 로직 실패"가 아니라 **"레짐 오분류로 변동성 큰 방향성 장에 평균회귀 강제적용"**. 평균회귀 순수성능은 횡보구간(Dev에 적음)에서만 판단 가능 → 이 백테로 미판단. I-001(funding=0): 보유 5h ≪ 8h → 이번 영향 미미. risk% 증가: 수익성 불변·진폭만 → PF<1에서 무의미(파산 가속), PF>1 후 수익-DD 조정 노브.

**결론·다음 단계**
- **τ 실증**: 상태1 등 방향성 레짐을 trend로 재라벨 → 추세매매 기계적 작동 + 전략적 유효성(지그재그 추세서 수익) 검증. 주의: τ 안정구간 확인(스펙§6)·trend 거래 분리분석·I-006 churn 재관측.
- **구간 (나) 현 2021-2022 유지**(2021Q1 상승+2022 하락 추세검증 충분, train 2020 안정, 코로나(2020-03) train 격리). 2020 대상승 확장은 t-emission(D4-4) 후.
- risk% 조정 보류(PF>1 확인 후).

### D4-3 τ 스윕 + trend 거래 분해 (2021-2022)

> baseline 의 "τ 실증" 실행. HMM 1회 학습 캐시 → τ별 mapping 재빌드 → τ별 백테(1회용 탐사, `data/tau_sweep/` 미커밋). τ 후보 0.01~0.10 (\|μ\|/σ 분포 0.003~0.109 기반). θ_trend=0.6·나머지 계수 baseline 고정.

**τ 스윕 결과** (표면):

| τ | trend상태(L/S) | 거래(t/r) | 전체pnl% | dd% | trend wr | trend pf |
|---|---|---|---|---|---|---|
| 0.01 | 18(12/6) | 1213(1188/25) | -97.9 | 98.0 | 24.4 | 0.49 |
| 0.03 | 8(8/0) | 663(572/91) | -89.6 | 90.1 | 18.9 | 0.53 |
| 0.05 | 4(4/0) | 428(238/190) | -66.9 | 68.8 | 19.3 | 0.58 |
| 0.06~0.08 | 3(3/0) | 391(172/219) | -64.0 | 66.1 | 20.9 | 0.58 |
| 0.10 | 1(1/0) | 332(37/295) | -54.3 | 56.9 | 29.7 | 0.83 |

- 표면: τ↓ trend↑일수록 전체 pnl 악화(-54→-98%), 안정구간 없음(단조). trend/short 는 τ≤0.02만.

**trend 거래 분해 (τ=0.05, long 238)** — 표면 결론의 **정정**:

| 지표 | 값 |
|---|---|
| 승률 | 19.3% |
| avg_win / avg_loss | +130.9 / -53.7 → **손익비 2.44** |
| 보유 중앙 / max | 11h / **1571h(65일)** |
| max_win / 손실 | +533 / 균일 -105(초기 SL) |
| exit | 전부 sl_hit(트레일링) |

- **핵심 정정**: 추세매매는 **부적합이 아니라 설계대로(스펙 §3.4 낮은승률+큰손익비) 작동**. 손익비 2.44, 트레일링이 큰 추세 포착(65일 보유·+533). 손실은 초기 SL 로 -105 균일 제한.
- **진짜 문제**: 승률 19.3% < **손익분기 승률 29.1%**(= 53.7/(131+53.7)). 딱 10%p 부족 → 근소 손실.
- **원인**: 관대진입(DD-1)이 conf≥θ만 되면 무제한 진입 → 지그재그에 가짜 추세 진입 남발 + churn(진입 연속<2h **25%**, I-006 실발현).

**결론 (방향 전환)**:
- 이전 baseline 분석의 "추세매매 부적합·근본재고" **오판을 정정**. 추세매매 엔진(트레일링)은 건강.
- **(가) 진입 필터 스윕이 정답 경로**: θ_trend 상향(0.6→0.7/0.8/0.9)·관대진입 churn 가드로 **승률을 손익분기(29%) 위로** → 손익비 2.44 유지 시 수익 전환.
- 기간 변경(2020 clean 상승 등)은 (가) 성공 후 강건성 검증(순서).
- 신규 **I-010**(진입 승률<손익분기). **DD-1**(관대진입) 재검토 대상.

### D4-3.5 계수 스윕 + range 원인분석 (진행중)

> D4-3 baseline·τ스윕에 이은 Gaussian 기반 계수 스윕·레짐 규명. **t 교체(D4-4) 전에 발견층 근본(range 정의) 문제를 먼저 규명**. 스크립트는 1회용 탐사(`data/tau_sweep/` 미커밋, 결과 텍스트만 보존).

**① churn 가드 도입 (I-010·DD-1 재검토) — 커밋 216612f·eb9d8e1**
- RegimeQuantStrategy 에 `cooldown_bars`·`entry_on_transition_only`(전환봉 한정). 기본값=관대(하위호환). `trades.csv` `logic` 컬럼 추가(trend/range 분리 분석).
- 실증(τ=0.05): churn 연속<2h **25%→0%**, 전체 pnl **-66.9%→-12.4%**(cd24).

**② confidence 분포 — θ_trend 무력**
- trend contract confidence **median 0.997**(τ=0.05). θ 0.6→0.9 올려도 진입 봉 97%→83% → **θ_trend 진입필터 무력**. 승률·손익비 노브는 θ가 아니라 **k_trend_sl/trail**.

**③ k_trail × k_sl × 가드 그리드 스윕 (36조합, τ=0.05 고정)**
- 최선 **cd24·k_trail 6.0·k_sl 3.0**: 전체 -6.8%, **trend margin +10.7**(승률 32.3 > 손익분기 21.6). k_trail 넓힐수록 손익비↑(3.0→4.65, 지그재그 휩쏘 완화).
- **trend/range 분리**: trend n62 **pnl +2,434(수익!)** / range n92 **pnl -3,115(손실)**. → **추세매매 수익 전환 확정, range 가 발목**.

**④ range 손실 원인 — "추세장 탓" 기각 (두 번째 오판 정정)**
- range 분기별 분해: **횡보 분기(2022Q1 -351 wr0% / 2022Q3 -920 wr35%)에서도 손실** → 시장국면 무관.
- **근본 가설(I-011)**: mapping 이 range 를 "저 \|μ\|/σ(저방향성)"로만 정의하고 **mean-reversion 미검증**. train(2020-2022 추세지배)에 진짜 박스권 없어 HMM range 상태가 "추세 중 되돌림"으로 오염 → 평균회귀 역주행. K=3 은 임의 고정(K 미탐색).

**⑤ range 원인분석 v2 계획 (독립검증 F-1~F-6 반영)**
- **1단계 프로파일링(계수 무관)**: 전체 Dev 단일 학습 × **K=2~6**, 상태별 \|μ\|/σ + **중심선(BB mid) 대비 편차의 half-life/AR(1)**(로그가격 절대값 아님 = F-1 치명 반영) + 밴드폭 내 복귀비율. **지속구간(run) 내부 측정** + 표본수·신뢰구간(F-2) → 진짜 range 후보 식별.
- **2단계 진단(대안가설 분기, F-4)**: (A)특정 K에 저\|μ\|/σ+편차 mean-revert 존재 → 오염(해상도) / (B)전 상태 half-life 발산 → BTC 1h 평균회귀 부재 → range 포기 / (C)후보없고 박스권 희소 → 데이터 빈약(§1.7 train 확장) / (D)mean-revert 양호한데 백테 손실 → 계수 미최적.
- **3단계 매매-특화(A·D 시)**: 후보 상태 range 백테+계수.
- **4단계 설계**: mapping **3분류 trend/range/none**(none 은 별도임계 아닌 mean-reversion 지표로 결정, F-5) + K 재선택(**LL·BIC·커버리지·mean-reversion 4렌즈**, trend 동시 재평가, F-3).

---

## 잠재 이슈 트래커

| ID | 이슈 | 발생 | 상태 | 해결 경로 |
|---|---|---|---|---|
| I-001 | 백테 funding=0 → 추세 장기보유 비용 과소평가 (라이브 갭) | D1 | known limitation | O-5(가): gen1 수용·측정 후 2세대 판단 |
| I-002 | 회귀 테스트 부재 4건: ①엔진측 lookahead 절단(`_build_ctx`/`_slice_candles`) ②진입가 백테=라이브 동일성 ③트레일링 SL end-to-end ④레짐 스왑 시퀀스 | D1 | 부분해소 | **③④ D4-3 S3 E2E 해소**(트레일익절·스왑같은봉) / ①② 기존 메커니즘(`test_lookahead` 간접), 실데이터 통합 시 명시 |
| I-003 | 2-플러그인 시 HMM 레짐 봉당 중복 계산 | D1 | 해소(설계) | D2 (가) 단일 디스패처+RegimeService 캐시로 차단 |
| I-004 | OOS 하락 = 단일 에피소드(2025-10~2026-02) → trend/short OOS 표본 작음 | D3 | known limitation | 2018 스트레스로 보완·결과 해석 시 감안 |
| I-005 | 윈도우 경계·인덱싱 정확성 (H1~H6) | D3 | 대부분 해소(D4-2) | features 테스트(H1·H2·H5)·service sliding-window(H4·H6). 잔여 H3 walk-forward warmup 은 D4-3/5 |
| I-006 | 추세 관대진입의 재진입 churn (트레일링 손절 후 즉시 재진입 휩쏘) | D3.5 | 해소 | τ 스윕서 실발현(연속<2h 25%) → **churn 가드(cooldown/transition)로 해소**(D4-3.5①, 연속<2h 25%→0%) |
| I-007 | EM 학습 forward/backward Python 루프 성능 (50k봉×K×init×iter×윈도우 느릴 수 있음) | D4-2 | OPEN | D4-5 실학습 전 최적화(벡터화/numba/init·iter 축소) |
| I-008 | RangeLogic BB/RSI 히스토리 길이 의존(RSI Wilder RMA) → 백테≠라이브 미세 불일치 | D4-3 | 해소(S1) | bounded 윈도우(`indicator_window`) 계산 — RegimeService H4(ATR bounded) 패턴 차용 |
| I-009 | 매핑 τ 스케일 미스매치: per-state \|μ\|/σ ~0.01-0.1 vs τ=0.5 → 24상태 전부 range | D4-3 백테 | 스윕완료 | τ 스윕 완료(trend 살림). **range 손실은 τ 아니라 mean-reversion 미검증(I-011)이 근본** → type 최종값은 v2 3분류와 함께. `build_regime_artifacts` 커버리지 경고 잔여 |
| I-010 | 추세 진입 승률(19%) < 손익분기(29%) — 관대진입(DD-1) 가짜진입+churn (손익비 2.44·트레일링은 건강) | D4-3 τ스윕 | 해소 | θ_trend 무력(conf≈1) → **k_trail 6.0 + cooldown24 로 trend margin +10.7·pnl +2,434 수익전환**(D4-3.5③) |
| I-011 | range 손실 근본: mapping 이 "저 \|μ\|/σ=range"로 정의(mean-reversion 미검증) + train 추세지배로 range 상태 오염(미검증 가설). K=3 임의 | D4-3.5 | 분석중 | v2 프로파일링(K=2~6, 중심선편차 half-life)로 진짜 range 유무 진단 → 3분류(trend/range/none) or range 포기 |

---

## 결정 항목 (O-NNN)

| ID | 항목 | 결정 | 상태 |
|---|---|---|---|
| O-1 | type 경계 임계값 | **(가) 손 고정 τ + 스윕** (전역 동일) | ✅ D3 |
| O-2 | Size ∝ confidence 형태 | **(가) 리터럴 비례** `size=risk_based_size×confidence` | ✅ D2 |
| O-3 | walk-forward 재추정 방식 | **(가) 오프라인 사전학습** (N=3~6개월 D4 스윕) | ✅ D3 |
| O-4 | 적대청산 매핑 | **force_exit + 빈슬롯**(should_reverse 미사용) | ✅ D2 |
| O-5 | gen1 펀딩 처리 | **(가) 백테 funding=0 수용 + 측정 후 결정** | ✅ D2 |
| O-6 | t-HMM 구현 경로 | **(나) Gaussian 스캐폴드 → t 교체** (gen1=t). 검증=합성복원+analytic, hmmlearn 드롭(D4-1) | ✅ D3 |
| S1-1 | 변동성 feature 정의 | **(가) 실현변동성** std(24봉 로그수익률) | ✅ D3 |
| DD-1 | 추세 진입 정책 | **관대 진입 → churn 가드 도입** (`cooldown_bars`·`entry_on_transition_only`) | ✅ D4-3.5 (가드로 trend 수익전환 +2,434) |
| O-7 | range 매핑 정책 | **(보류) 3분류 trend/range/none + mean-reversion 검증** (v2 진단 후 확정) | 진행중 D4-3.5 |

---

## 변경 기록 (문서·코드)

- **2026-06-29**: "종가 진입" → "다음 봉 시초가 진입" 정정 (결정 D-1). 커밋 d76d3cc.
  - `spec_v1.md` L131·136·175 인라인 갱신.
  - `transcript.md` 상단 정정 배너 추가 (verbatim 대화 텍스트는 보존).
  - 코드(src/): 해당 기록 없음 (플러그인 비어 있음).
- **2026-06-30**: docs 구조 재편(00_Work_Reports/01_Guide_Docs/99_Brainstorming_Docs) +
  경로·파일트리 정합화(CLAUDE.md/README.md). 커밋 d76d3cc.
- **2026-06-30**: Phase D2 설계 확정(플러그인 구조·인터페이스 변경·O-2/O-4/O-5).
  **코드 변경은 D4에서 일괄** (현재까지 src/ 무변경).
- **2026-07-01**: Phase D3 설계 확정(데이터분할·feature·t-HMM·K선택/매핑·walk-forward/artifact·
  O-1/O-3/O-6/S1-1). **코드 변경은 D4** (src/ 무변경 유지).
- **2026-07-01**: 구현 전 점검(D3.5, fresh-eyes 2 + 환경/데이터). **GO** — 엔진 수정 4건·재사용
  검증 완료. 결정 DD-1(관대진입)·hmmlearn 추가·2018 스트레스단계 다운로드·REVERSE_SIGNAL 제거.
  신규 I-006. **src/ 무변경 유지(코드는 D4)**.
- **2026-07-01**: **D4-1 인프라 prep 구현** — 엔진 4수정(reverse flow·should_reverse·
  REVERSE_SIGNAL 제거 / update_take_profit·REGIME_EXIT 추가) + indicators 2 + reverse 전수 정리
  (src·tests·CLAUDE/INFRA_GUIDE/README) + requirements(scipy 추가, **hmmlearn 드롭**: Py3.14
  빌드 불가 → 합성복원+analytic 검증으로 대체). **회귀 258 통과**. 커밋 b83a9f7.
- **2026-07-01**: **D4-2 발견층 Gaussian 구현** — `src/strategy/regime/` 8개 모듈(contract·
  features·hmm·mapping·selection·artifact·service·training). 독립 fresh-eyes + 자체 검증:
  실버그·lookahead 0, EM 참조 Baum-Welch atol 1e-10 일치, 추론 causal. **설계 정정 2건(H4)**:
  D-4 incremental→sliding-window, ATR 전체RMA→bounded. 신규 I-007. **회귀 294** (D4-1 258 + D4-2 36).
  커밋 192c2fe.
- **2026-07-01**: **D4-3 매매층 구현** (4 Step, 각 Step 독립 fresh-eyes 교차검증) — S1 매매로직
  (`regime/{trend,range}_logic.py`, **신규 I-008** bounded 지표) + S2 디스패처(`plugins/regime_quant.py`,
  `_HOLD` 싱글톤 제거·`.get()` 통일) + S3 E2E 7(정합성·I-002④ 스왑·I-006 churn·service통합·트레일익절·range TP)
  + S4 빌드도구(`training.make_anchored_windows`·`scripts/build_regime_artifacts.py`). 독립검증:
  미래참조·누수·tz 전부 정확. **회귀 358** (294 + 64). config `regime_quant` 섹션·`.gitignore`
  data/regime_models/ 추가. `active: []` 유지(규칙7). 커밋 b14f175 (+보고서 ID 26a2482).
- **2026-07-01**: **백테 전 재검증 (D4-3 후 게이트) — GO**. 독립 fresh-eyes 3(R1·R2·R3) +
  통합(V1 파이프라인·V2 회귀358·V3 코드↔문서), 3커밋 변경파일 전수 매핑. **백테 블로커 0**.
  `INFRA_GUIDE.md:44` 필수6→5 stale 수정(D4-1 reverse 전수스윕 누락분). 경미 5건 전부 비블로커.
- **2026-07-02**: **D4-3 Dev 백테 baseline 분석**(2021-2022, Gaussian K=3·τ=0.5). total -51.16%,
  승률 43.81%, max_dd 56.35%. 정합성 3자일치·부호 정상 → **구현 결함 0**. 근본원인: **τ=0.5 스케일
  미스매치로 24상태 전부 range → 추세매매 0회, 100% 평균회귀**. 발견층(self-trans 0.94~0.98, 상태1
  \|μ\|/σ=0.109)은 건강. 시장=지그재그 큰 방향성(ER<0.1). 신규 **I-009**. 다음: τ 실증·스윕(구간 나 유지).
- **2026-07-02**: **D4-3 τ 스윕 + trend 분해**. τ 0.01~0.10 스윕(θ_trend 고정): τ↓ trend↑일수록
  전체 악화(-54~-98%). **단 trend 거래 분해 시 손익비 2.44·트레일링 정상(65일 보유·max_win +533)**
  — 직전 '추세매매 부적합' **오판 정정**(추세엔진 건강). 진짜 문제=승률 19%<손익분기 29%(관대진입
  가짜진입+churn). 신규 **I-010**, I-006 실발현·I-009 스윕완료·DD-1 재검토. 다음: **(가) θ_trend·진입가드 스윕**.
- **2026-07-02~03**: **D4-3.5 계수 스윕 + range 원인분석** (단계 재편 D4-3.5 신설). churn 가드
  (cooldown/transition, 커밋 216612f)+logic 컬럼(eb9d8e1) → **k_trail 6.0·cd24 로 trend 수익전환
  (+2,434, I-010·I-006 해소, DD-1 확정)**. θ_trend 무력(conf median 0.997). range 는 횡보서도 손실
  → "추세장 탓" 기각(2번째 오판). 근본가설 **I-011**(range 저방향성 정의·mean-reversion 미검증·train 오염).
  **v2 계획**(K=2~6 중심선편차 half-life 프로파일링→진단→3분류, 독립검증 F-1 치명 반영). 신규 결정 **O-7**(range 3분류 보류).
