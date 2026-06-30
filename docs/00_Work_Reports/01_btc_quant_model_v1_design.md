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
| D4 | 구현 (D4-1·D4-2 ✅ / D4-3~5 예정) | 진행중 | 2026-07-01~ | D4-1 b83a9f7 / D4-2 미커밋 |

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
| D4-3 | 매매층: 디스패처+Trend/Range(DD-1·None) + Dev 백테(Gaussian) | E2E + I-002④ + churn(I-006) + 정합성 |
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

---

## 잠재 이슈 트래커

| ID | 이슈 | 발생 | 상태 | 해결 경로 |
|---|---|---|---|---|
| I-001 | 백테 funding=0 → 추세 장기보유 비용 과소평가 (라이브 갭) | D1 | known limitation | O-5(가): gen1 수용·측정 후 2세대 판단 |
| I-002 | 회귀 테스트 부재 4건: ①엔진측 lookahead 절단(`_build_ctx`/`_slice_candles`) ②진입가 백테=라이브 동일성 ③트레일링 SL end-to-end ④레짐 스왑 시퀀스 | D1 | OPEN | D4 테스트 추가 |
| I-003 | 2-플러그인 시 HMM 레짐 봉당 중복 계산 | D1 | 해소(설계) | D2 (가) 단일 디스패처+RegimeService 캐시로 차단 |
| I-004 | OOS 하락 = 단일 에피소드(2025-10~2026-02) → trend/short OOS 표본 작음 | D3 | known limitation | 2018 스트레스로 보완·결과 해석 시 감안 |
| I-005 | 윈도우 경계·인덱싱 정확성 (H1~H6) | D3 | 대부분 해소(D4-2) | features 테스트(H1·H2·H5)·service sliding-window(H4·H6). 잔여 H3 walk-forward warmup 은 D4-3/5 |
| I-006 | 추세 관대진입의 재진입 churn (트레일링 손절 후 즉시 재진입 휩쏘) | D3.5 | 측정 대상 | D4 백테 측정, 2세대 가드 판단 |
| I-007 | EM 학습 forward/backward Python 루프 성능 (50k봉×K×init×iter×윈도우 느릴 수 있음) | D4-2 | OPEN | D4-5 실학습 전 최적화(벡터화/numba/init·iter 축소) |

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
| DD-1 | 추세 진입 정책 | **관대 진입** (flat∧trend∧conf≥θ, 횟수 무제한) | ✅ D3.5 |

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
