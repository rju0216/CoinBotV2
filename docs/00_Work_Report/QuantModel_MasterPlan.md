# QuantModel MasterPlan — 개발·검증·결정 관리 문서

> **성격**: "**어떻게 만들고·검증하고·결정하나**". 모델 정체("무엇을")는
> `docs/BTCUSDTperp_QuantModel_Design_vDraft.md` 가 정본이며 이 문서는 그것을
> **참조만** 한다(중복 서술 금지). 이 문서는 개발 계획·학습 스케줄·리뷰어
> 플래그·결정 로그(D-NNN)·잠재 이슈(I-NNN)·진행 기록표를 담는다.
>
> **현재 시점**: **Phase 0~2 + Phase 3 R1·R2 완료**(2층 트리·MLP, **관문1 PASS**; R2 창탐색 = 무material gain → **기본창 유지**, §12-3/§12-4). 다음 = **R3**(멀티태스크·정규화·구조·KF, 1h).

---

## 0. 근거 확인 (계획의 전제 — 실측)

| 항목 | 실태 | 함의 |
|---|---|---|
| BTC 퍼프 캔들 | 15m/1h/4h/1d 모두 **2020~2026.6 (~6.5년)** 보유 (1h 56.9k행, 1m도 있음) | 2020폭락·21불·22베어·23회복·24불 포함 → 독립 매크로 국면 확보 |
| 의존성 | **torch 2.11(CUDA)·sklearn·lightgbm·xgboost 이미 설치** | 2층 MLP + 트리 벤치 **신규 의존성 0** |
| 펀딩 데이터 | **없음** | 비용모델 펀딩 = 설계대로 "보수적 상수"(또는 후속 다운로드) → F-3 |
| 기타 종목 | ETH/SOL/XRP/DOGE 4h 존재 | 국면검증 이후 멀티심볼 강건성 재활용 가능(보너스) |
| TF별 종료일 | 1h=6/16, 4h=6/28, 1d=6/23 **불일치** | MTF 결합 시 겹치는 구간 정렬 필요 → F-4 |

**인프라 매핑 확인**: 삼중배리어 청산이 기존 엔진 인터페이스에 그대로 붙음 —
하단배리어=`compute_stop_loss`, 상단=`compute_take_profit`, 시간한계 N=`should_force_exit`,
고정1단위=`compute_position_size` 상수, θ진입게이트=`allow_entry`/`generate_signal`.
→ **엔진 코드 수정 0** 원칙 유지하며 경제성 검증(관문2)을 `BacktestEngine` 으로 실행 가능.

---

## 1. 빌드 순서 — "순진한 아래→위"의 함정

**엄격한 bottom-up 은 성립하지 않음**: 1층 창 길이 순차 탐색(coordinate descent)이
2층 예측력을 목적함수로 쓰므로, 2층 없이는 1층을 "완성"할 수 없다. 또한 라벨과
피처는 서로 독립이고 둘 다 2층 입력이다. 실제 의존 그래프:

```
데이터(有) ─┬─ [Phase1] 라벨(삼중배리어)  ──┐
            └─ [Phase2] 1층 피처(기본창)    ──┴─ [Phase3] 2층 MLP+트리 (창·λ 튜닝 = 1층으로 역류)
[Phase0] 검증하네스·국면태깅 (전 단계 관통) ─────────────────→ [Phase4] 멍청한 3층 (엔진 경유 경제성)
                                                                → [Phase5] 최종 3층 (엣지 확인 후만)
```

**채택 = 얇은 수직 슬라이스 먼저, 그 다음 확장 (D-001)**: 1h 에 기본 파라미터로
라벨→피처→트리벤치+작은MLP→멍청3층까지 최단 관통 → ① 인과성/통합 버그 조기 발견
② 엣지 유무 신호 최속 확인 → 그 위에서 창 튜닝·TF/MTF 확장. 설계 "엣지 정찰 우선"과 정합.

---

## 2. Phase 계획 (개념·논리)

| Phase | 산출물 | 검증 종류 | 관문 |
|---|---|---|---|
| **0 검증 인프라** | walk-forward 하네스, **국면 태깅(규칙기반)**, 인과성/누수 테스트 스위트, z-score train-only, 지표계산(균형정확도·MCC·로그손실·semi-dev), 결과 정합성 검증(CLAUDE 10) | 소프트웨어 정합성 | — |
| **1 라벨** ✅ | 삼중배리어 생성기(ATR·YZ, x·N, 인과) + **라벨분포 검증**(층1 학습가능성·층3 국면일관성) — **완료: N=24·atr/w96/x3.0 확정** | **후보1 라벨분포**(성과 안 봄) | 분포 게이트 **통과** |
| **2 1층 피처** ✅ | 6축 피처(robust KF·ER/Hurst·변동성변화·상대거래량·semi-dev·종가위치), **기본창** — **완료: build_features→X 8열, 인과·행동 검증** | 소프트웨어 정합성(인과·정상성) | — |
| **3 2층 MLP** 🔶 | 공유트렁크 멀티태스크 MLP + 트리 벤치. 여기서 **창 순차탐색·λ·정규화·TF/MTF 확장** — **R1·R2 완료(관문1 PASS; R2 창탐색 무gain→기본창 유지)**, R3~R4 남음 | **후보2 예측력** | **관문1 통계적엣지** |
| **4 멍청한 3층** | 방향·θ진입·고정사이즈·배리어청산 + 비용모델, **기존 엔진 경유** | **후보3 경제성** | **관문2 경제적엣지** |
| **5 최종 3층** | 엣지 성격에 맞춘 형태(직접정책최적화 유력) | — | — |

- **Phase 1~3 은 엔진 무관 순수 오프라인 연구 코드**(관문1 = 라벨 대비 예측 평가, 엔진 불필요).
  **Phase 4 에서만 플러그인/엔진 배선** → 엔진 수정 0 보장.
- **상시 검증**: 인과성/누수 하네스를 1급 산출물로 격상(z-score train-only, MTF 완성봉만,
  배리어 폭 진입시점 정보만 — 각각 회귀 테스트). 단위→회귀→시연(CLAUDE 4),
  미커밋 Step 메모리 임시보존(CLAUDE 9), 비자명 변경 후 fresh-eyes(CLAUDE 14).

**substrate 계약 (Phase 0 확정 → 하류 준수, 규칙 18)** — Phase 1~3 은 아래 인터페이스에 맞춘다:
- **모델 콜러블** `fit(X,y)`/`predict(X)`/`predict_proba(X)→DataFrame[classes]`·`classes_`, 산출 `FoldPrediction` → **Phase 3** 트리벤치·MLP 가 구현(baseline 이 참조 구현) **(실현: `src/research/models` — `ProbaModel` 계약, `TreeBench`(lightgbm)·`SmallMLP`(torch, 트렁크/헤드 분리, R1 단일태스크). D-020~023)**
- **라벨** = 클래스 Series(X.index 정렬), **N=label_horizon** 이 splitter purge/꼬리예약 구동 → **Phase 1** 삼중배리어가 이 형태 + N 공급 **(실현: `src/research/labeling` — `BarrierLabels`(labels·first_touch·label_horizon), `LABEL_CLASSES=("up","down","expire")`, N=24 확정)**
- **피처** = X DataFrame, **정규화는 harness 소유**(폴드 train-only) → **Phase 2** 피처는 자기정규화 불필요, `BaseScaler` 교체(F-6) **(실현: `src/research/features` — `build_features`→X 8열 [vol_change·relative_volume·semi_dev·close_position·er·hurst·kf_slope·kf_uncertainty], 전부 OHLCV만·인과. F-6 해소: `RobustScaler`(median/MAD) 드롭인 제공, z vs robust 선택은 Phase 3)**
- **regime** = 모델 TF(forward_fill) → Phase 3/4 국면집계(계약 F-8) · **실행** = `run_walk_forward` + `RunLedger`(비교계수 F-1) → Phase 3 R1~R5

---

## 3. 코드 구조 (D-002)

- **오프라인 연구모듈**(신설, 예: `src/research/`): 라벨·1층·2층 학습 → 모델 아티팩트 생성.
- **얇은 플러그인**(`src/strategy/plugins/`): 아티팩트 로드 → 1층 추론 + 2층 추론 + 3층 결정 → `Signal`.
- **경제성 검증**: 기존 `BacktestEngine` 경유(삼중배리어 = SL/TP/force_exit 매핑).
- **원칙**: 엔진 코드 수정 0(INFRA_GUIDE §4-0). 학습/추론 경계를 오프라인↔라이브로 명확 분리.

---

## 4. 국면(regime) 프레임 (D-003)

**두 개념 분리**:
- **(A) 태그 카테고리/세그먼트**: 규칙 기반 사후 태깅 산출물. 카테고리 개수는 설계 선택,
  **시간 세그먼트 수는 데이터가 정함(고정 아님)**. 결과를 슬라이스하는 축.
- **(B) 독립 매크로 국면 (≈6~8, 데이터 성질)**: 시장이 근본적으로 바뀐 거대 에피소드 수.
  파라미터가 아니라 세는 값. **일반화 독립 시험이 몇 번뿐인지**를 말하는 희소 자원 →
  홀드아웃 대신 "비교개수 제한 + 국면 일관성" 방법론의 근거(F-1·F-2).

**엣지 판정과의 관계**: 엣지 유무는 walk-forward 지표(관문1/2)가 채점. 매크로 국면은
그 위에 얹는 **일관성 렌즈**(방식1 교차검증·방식2 국면별 분리집계). "6~8"은 채점 격자가
아니라 독립 일관성-시험의 상한(해석용).

**채택 방식 (가 뼈대 + 다 오버레이)**:
- **(가) 기계적 규칙**이 경계·개수를 정함(선별·게이트에 쓰는 국면은 오직 이것, 무재량·재현).
  후보 규칙: 추세(장기 MA 기울기 부호·강도) × 변동성(실현변동성 백분위) →
  소수 버킷, **히스테리시스 + 최소 지속기간**으로 잔진동 흡수(가짜 국면 방지 + 표본 하한 강제).
- **(다) 명명된 에피소드**(예: 22 베어)는 **해석·보고용 대조로만**(선별엔 절대 미사용).
- **주의(하한)**: 너무 잘게 쪼개면 국면별 소수클래스 표본 부족 → 지표 노이즈(라벨 층1 "최소 절대수"와 동형).
  목표는 "많이 쪼개기"가 아니라 "통계적으로 의미 있는 최소 단위로, 재현 가능하게".
- **구체 수치** → **Step 0.1에서 확정**(6.5년 태깅 리뷰): `ma_len=100·slope_k=20·deadband=0.02·vol_len=30·vol_hi_pct=0.5·min_duration=14`. 결과 28 세그먼트 → ~7-8 매크로 에라(추세 up1042/down689/flat521).

---

## 5. 학습 실행 스케줄 (어느 조합·몇 개·언제)

제약: 설계 "홀드아웃 대신 비교개수 제한". **분포 기반 선별은 무제한(싸고 성과비교 아님),
성과 기반 비교는 집중·계수.**

| 순번 | 시점 | 무엇을 | 규모 | 성격 |
|---|---|---|---|---|
| **R0 라벨분포** ✅ | Phase1 직후 | {ATR/YZ}×{창}×{x}×{N}, **분포만** 평가 | **128 config 실행** | **완료**: 51/128 통과 → **atr/w96/x3.0/N24 1개** 선택(§12-1) |
| **R1 스모크** ✅ | Phase3 착수 | 1h·라벨1개·기본창·기본하이퍼, 트리벤치+작은MLP(5seed) | **8판(비교 2)** | **완료: 관문1 PASS**(mlp STRONG — ll<Prior·MCC0.13·BA0.41·일관성0.91). §12-3 |
| **R2 창 순차탐색** ✅ | R1 통과 후 | 1h 고정, 피처 창 coordinate descent(MLP 5seed 선택 + 트리 참고) | **19판(예산 60)** | **완료: 창 튜닝 무material gain**(Δll −0.0034 노이즈수준) → **기본창 유지**. §12-4 |
| **R3 하이퍼 소그리드** | R2 후 | 1h, λ(낮게 2~3)×정규화(2~3)×깊이너비(2~3) | 소수 | 좁게 시작 |
| **R4 TF/MTF 확장** | **1h 관문1 통과 후에만** | 4h·1d·15m 단독 + MTF 1~2조합 | TF당 튜닝 파이프 재실행, 제한 | 확증·강건성. **best-of-4 복권 금지**(F-2) |
| **R5 경제성** | Phase4 | 멍청3층, 엔진 경유, θ 안정성 스윕 + 국면분리 | 생존자만 | 관문2 |

**요지**: 처음부터 전TF×전조합 폭발 금지. 1h 에서 파이프라인 입증 후 확장.

- **R1 관문1 순진 기준선** = **Prior/Uniform**(Phase 0 확정, `baselines.py`) + 트리벤치(Phase 3). 실행·비교계수는 `run_walk_forward`+`RunLedger`(Phase 0 완성).
- **R2 전제**: **F-13 해소**(R2.0a) — hurst 벡터화(75배, build 67.7s→9.2s). **F-12 해소**(D-025, 방식=config 하이퍼).
- **R4 전제**: **I-001 remediation 완료** — R1 이 regime 에 1d 를 써 **선행 해소**(`resample_ohlcv` 1h→1d 파생, §12-3/I-001). MTF 도 이 파생 헬퍼 재사용.

---

## 6. 컨셉 무결성 8기둥 (정본 — 트리아지 C 판정 기준)

CLAUDE.md 규칙 15의 C 등급은 아래 기둥을 약화·모순·재도입하는 결정을 말한다.
(모델 컨셉 진화 시 이 목록을 갱신 — 유일 정본.)

1. **지도학습 우선 엣지정찰** (예측→정책 순서)
2. **인과성/무누수** (봉마감·그 시점 정보만·완성봉만)
3. **선별 vs 확인 분리** (선별은 예측력·분포로, 성과는 최종 확인만)
4. **1층 정성가공 → 하류 단순화** (시간압축→벡터·MLP·래그0)
5. **무가정·장식금지** (데이터가 정함, 개선 못하면 탈락)
6. **팻테일 대응** (robust KF + semi-dev)
7. **멀티태스크 결합** (배리어결과 + 도달시간, 같은 사건의 두 투영)
8. **검증 방법론** (walk-forward·홀드아웃 없음·비교개수 제한·국면 일관성)

---

## 7. 리뷰어 플래그 트래커 (F-NNN)

| ID | 내용 | 발생 | 상태 |
|---|---|---|---|
| **F-1** | R2 창 순차탐색 = 홀드아웃 없는 검증셋 성과 선별 → **검증셋 과적합 위험**. 방어: 비교개수 사전등록 + 선택 창을 국면분리에서 재확인. | 계획 §5 | **R2 적용**(예산 사전등록 19/60 + 국면분리 재확인 5/6국면·14/22폴드 우세; 개선 미미→기본창 유지로 노이즈 미채택). 열림(R3/R4 동일 규율 지속) |
| **F-2** | 선별 2단 중첩(라벨=분포/TF=예측력). 다른 TF는 "4개 중 최고 뽑기(부풀림)" 아니라 **확증·강건성**으로. | 계획 §5 | 열림(Phase3 대비) |
| **F-3** | 펀딩 데이터 부재 → 상수 근사가 실비용/엣지 왜곡 가능. Phase4 전 실펀딩 다운로드 옵션. | 계획 §0 | 열림(Phase4 대비) |
| **F-4** | TF별 종료일 불일치 → MTF 결합 시 겹치는 구간 정렬 필수. | 계획 §0 | 열림(Phase3 MTF 대비) |
| **F-5** | purge/embargo = 라벨 지평 N에 결합(교차-Phase). splitter가 N을 파라미터로 받아야 하고, walk-forward T·V·S 도 피처/라벨 창에 번인 결합 → 수치는 Phase1~2 후 확정. | 계획 §11 | **해소(Phase1)**: **N=24 확정** → T·V·S = train_min 8760·val_size 2160·N 24·**22폴드**(step=val). embargo 0 유지 |
| **F-6** | 설계 §7이 **z-score** 지정했으나 robust 대비 **논증 없음**(근거는 train-only 인과성뿐). 기둥6(팻테일) 견주면 robust(median/MAD)이 이상치에 강함. **방식**: Phase2에서 ①분포 진단 먼저(비교예산 무관)→②왜곡 유의시만 예측력 비교(F-1 계수). Step0.3 scaler는 `BaseScaler` 교체 인터페이스(robust 드롭인). | Step0.3 | **해소(Phase2 Step2.4)**: 실 1h 진단 → 팻테일 축(semi_dev·kf_uncertainty 초과첨도~73, kf_uncertainty std/robustScale≈5.0, \|z\|>5가 정규 1만배) z-score 왜곡 유의 → `RobustScaler` 드롭인 제공. **최종 z vs robust 선택은 Phase3 예측력**(F-1). 기본 z-score 유지 |
| **F-7** | 겹치는 val 창(step<val_size) → per-regime 봉단위 귀속 **이중계수**. 현재 non-overlap 기본이라 미발현(문서화됨). | fresh-eyes | 열림(Phase3 rolling/overlap 시 dedup·가중) |
| **F-8** | per-regime 귀속 계약: regime_tags는 **모델 TF 정렬** 필수(원시 상위TF 주면 대부분 드롭), n은 "귀속 가능 봉만"(NaN 태그 제외). 문서화됨. | fresh-eyes | 열림(Phase3 커버리지 경고 추가 고려) |
| **F-9** | `forward_fill_completed`·`ZScoreNormalizer.transform`이 unique/complete 컬럼 무가드 전제. | fresh-eyes | 열림(Phase3 MTF 재사용 시 가드) |
| **F-10** | `run_walk_forward`가 fold의 y NaN을 드롭 안 함(harness). 삼중배리어 라벨은 동시터치→NaN을 중간에 낼 수 있어(실측 ~0.08%) Phase3 모델 학습 시 NaN 클래스 유입 가능. R0(Phase1)는 분포만·모델 fit 안 함이라 미발현. | Phase1 seam | **해소(Phase3 R1 Step3.1)**: `run_walk_forward` 가 **정규화 前** train·val 각각 X∪y NaN 드롭 + 커버리지 로깅(`WalkForwardResult.coverage`). val NaN 채점 제외. 회귀 5. 실 R1 val 드롭 50봉 |
| **F-11** | 라벨 참조가=close_i·스캔=후속 high/low(D-007)인데 엔진 진입은 현재봉 open(INFRA §4-3). 라벨-실행 미세 불일치 + 봉내 동시터치(현 NaN 제외)는 1m/15m 인트라바로 복원 가능. | Phase1 | 열림(**Phase4 플러그인 배선 시** 정합·인트라바 복원 검토) |
| **F-12** | KF Q/R/dof 를 Phase3 에서 어떻게 인과적합하나 — (a)폴드 train-only MLE 재적합(정규화 동형) vs (b)창길이처럼 config 하이퍼 coordinate descent. 설계 §7 문구는 (a) 뉘앙스, R2 파이프라인은 (b) 정합. Phase2 는 고정 기본값이라 무관. | Phase2 Step2.3 | **해소(R2, D-025)**: **(b) config 하이퍼** 채택 — D-019(피처=폴드무관 순수함수·전역X, harness 슬라이스) **보존**. (a)는 폴드내 KF 재계산이라 D-019 붕괴. KF 실 튜닝은 **R3**. |
| **F-13** | hurst(R/S) `rolling.apply(python)` 성능 — 실 56.9k봉 build_features 67.7s(hurst 병목). Phase3 R2 창 순차탐색(다수 config×폴드) 전 벡터화 필요. Phase2 스코프=정합성이라 미해결. | Phase2 Step2.2/2.4 | **해소(R2.0a)**: `sliding_window_view` 벡터화(sub-window R/S 전창 동시). 스칼라와 **수치 동치**(실 1h 56.9k 완전 일치·NaN포함), hurst 48.2s→0.64s(75x), build 67.7s→9.2s. NaN 창=pandas min_periods 매칭, 퇴화행만 폴백. 회귀 박제. |

---

## 8. 결정 로그 (D-NNN)

CLAUDE.md 규칙 15 트리아지 적용. 등급 L/S/C, 건드린 기둥, 상태.

| ID | 결정 | 등급 | 기둥 | 상태 |
|---|---|---|---|---|
| **D-001** | 진행 방식 = **1h 수직 슬라이스 후 확장** (전TF 병렬·엄격 bottom-up 기각) | S | 1,8 | 확정 |
| **D-002** | 코드 구조 = **오프라인 연구모듈 + 얇은 플러그인 + 경제성 엔진경유** | S | 2,4 | 확정 |
| **D-003** | 국면 = **규칙기반(가) 뼈대 + 명명 에피소드(다) 해석 오버레이**. 6~8 강제 아님. 수치는 Phase0. | S | 8 | **확정**(수치 Step0.1 완료: ma100/slope20/db0.02/vol30/pct0.5/mindur14) |
| **D-004** | 결정 관리 = **3단 트리아지(L/S/C) + D-NNN 로그 + 고정 5문** 채택 → CLAUDE.md 규칙 15 | S | — | 확정 |
| **D-005** | walk-forward 창 = **expanding 주 + 최소 학습창 하한 + rolling 3차 확인**(rolling은 선별 아닌 최근성 확인용) | C | 8 | 확정(수치 Phase1~2 후) |
| **D-006** | 국면 평가 = **A(연속 walk-forward) 하드 선별 게이트 / B(방식1 국면격리) 등급형 강건성 진단**(관문2 국면일관성 구체화, **사전등록 규칙**, 붕괴·부호반전 시에만 킬) / **봉단위 국면 귀속**(경계 흐림 무해) | C | 8 | 확정(임계 Phase3~4) |
| **D-007** | 라벨 참조가 = **close_i, 배리어 스캔 = 후속 봉 high/low** (엔진 open 진입과 미세 불일치는 F-11) | S | 2 | 확정 |
| **D-008** | 3-class 인코딩 **{up,down,expire}** (대칭·방향편향 없음 — 실측 |up−down|≤0.01 확증) | L | 5 | 확정 |
| **D-009** | vol estimator = **공통 콜러블 인터페이스**(ATR·YZ 병렬 탐색축, `VOL_ESTIMATORS`) | S | — | 확정 |
| **D-010** | 변동성 = **분수(가격대비) 정규화** → 배리어 `close·(1±x·σ)`, x 추정기간 비교가능 | L | 6,7 | 확정 |
| **D-011** | 단순피처 정의: vol_change=ln(σ_t/σ_{t-lag})·relative_volume=ln(vol_t/MA_past)·close_position flat→0.5·semi_dev=metrics 재사용 | L | 5 | 확정 |
| **D-012** | Hurst 추정 = R/S(Rescaled Range). DFA/VR 은 Phase3 예측력 요구시 격상 | S | 6 | 확정 |
| **D-013** | 계열2 모듈명 `trend_strength`(features 안 "regime" 미사용 — 방화벽 개념 혼동 회피) | L | 2 | 확정 |
| **D-014** | 기본창/기본파라미터 = Phase2 placeholder, Phase3 R2 튜닝(기둥5) | L | 5 | 확정 |
| **D-015** | F-6 해소 = `RobustScaler`(median/MAD) 드롭인 제공, 선택은 Phase3 예측력 | S | 6 | 확정 |
| **D-016** | KF 상태공간 = 선형 local linear trend(level+slope), F=[[1,1],[0,1]] (비선형 미도입→EKF/UKF 불요) | S | 4,6 | 확정 |
| **D-017** | KF robust = Student-t 1-step IRLS 재가중, 단일 KF(IMM 보류). dof→∞ Gaussian 복원 | S | 6 | 확정 |
| **D-018** | KF Q/R/dof = Phase2 고정 기본값(하이퍼), 튜닝 Phase3 **R3**(방식=config 하이퍼 D-025; R2는 창만) | S | 2 | 확정 |
| **D-019** | 피처 = OHLCV→X 순수함수(online 전방필터 폴드무관), 폴드로직 harness 소유(D-002) | S | 2 | 확정 |
| **D-020** | MLP 프레임워크 = torch (2층 신경망) | L | — | 확정 |
| **D-021** | 모델 모듈 = `src/research/models/`(`ProbaModel` 계약 base). baseline 과 동일 duck-typed 계약이나 학습모델 족 분리(argmax 소량 중복, Phase0 미변경) | S | 2,4 | 확정 |
| **D-022** | 트리 벤치 = lightgbm 단일(xgboost 보류) | L | — | 확정 |
| **D-023** | 트리·MLP 기본 하이퍼 = R1 placeholder(MLP 트렁크2×32·dropout0.1·Adam·시간순 early stop / tree 200·leaves31), 튜닝 R2/R3 | L | 5 | 확정 |
| **D-024** | R2 창탐색 방식 = **MLP 5seed 직접 선택**(목적함수 log_loss) + **트리 참고**(is_comparison=False, "창=정보량 모델무관" 가설 순위상관 검증). 트리 프록시 탐색 기각(사용자). config 캐시·예산 사전등록 | S | 3,8 | 확정 |
| **D-025** | F-12 해소 = KF Q/R/dof **config 하이퍼**(coordinate descent 축)로, 폴드 MLE 재적합(D-019 붕괴) 기각. KF 튜닝은 R3 | S | 2,4 | 확정 |
| **D-026** | R2 결론 = 창 튜닝 무material gain(Δll −0.0034 노이즈수준, 국면 5/6·폴드 14/22 약우세) → **기본창 유지**(선택창 미채택). 1h 엣지 약함은 튜닝부족 아닌 실제 신호강도(기둥5 장식금지) | S | 5,3 | 확정 |

---

## 9. 진행 기록표

| 시점 | 단계 | 결과 | 커밋 |
|---|---|---|---|
| 2026-07-11 | 전체 계획 수립 | Phase0~5·R0~R5·F1~4·D001~004 확정. CLAUDE.md 규칙15 추가, 본 문서 생성 | 67a1f3c |
| 2026-07-11 | Phase 0 상세계획 | 컴포넌트 A~G·Step 0.1~0.5 확정. 창 방식(D-005)·국면 평가 구조(D-006)·F-5 등록. §11 신설 | (미커밋) |
| 2026-07-11 | Phase 0 Step 0.1~0.5 구현 | `src/research/` substrate 전 컴포넌트 + `tests/research/` — 데이터로더·audit(오프그리드)·regime·splitter(purge)·누수하네스·normalize·metrics·baselines·harness·ledger. I-001 발견 | (미커밋) |
| 2026-07-11 | 통짜 통합 검증 | full-stack 커밋 테스트 + fresh-eyes 독립 재스캔 → **실결함 F1/F2/F6 수정+회귀**, 저심각도 F-7~9 문서화·등록. 핵심 뼈대 독립 검증 clean. **143 테스트 통과** | (미커밋) |
| 2026-07-11 | Phase 0 종착: 보고서 통합 | 교차-Phase 전파(규칙18) 반영: substrate 계약·D-003 확정·§11-4 상태·F-6~9·I-001·R4 게이트·discoverability 포인터 | 794f31f |
| 2026-07-11 | Phase 1 Step 1.1~1.4 구현 | `src/research/labeling/`: volatility(ATR·YZ 인과)·triple_barrier(3-class·first_touch·유한지평 N)·distribution(층1·층3). 인과 3중 규율. tests 신규 41 | (미커밋) |
| 2026-07-11 | Phase 1 Step 1.5 R0 스윕 | 1h 128 config(성과 안 봄) → 51 통과 → **atr/w96/x3.0/N24 1개 확정**. F-5 해소(T·V·S). up/down 대칭 실측 | (미커밋) |
| 2026-07-11 | Phase 1 종착 검증 | 통짜 full-stack(규칙17, 확정라벨 게이트통과 박제)+층3↔regime seam(규칙16) + fresh-eyes(규칙14) **CRITICAL/HIGH 0**, LOW#4 NaN가드 수정+회귀. **188 테스트 통과**. 교차Phase전파(규칙18): F-10/11·D-007~010 | (미커밋) |
| 2026-07-12 | Phase 2 Step 2.1~2.3 구현 | `src/research/features/`: simple(4축)·trend_strength(ER/Hurst R/S)·kalman(robust KF). build_features→X 8열. 행동검증(스파이크/지속·불확실성비상수·dof→∞). 신규 42 test | (미커밋) |
| 2026-07-12 | Phase 2 Step 2.4 통합·종착 | seam(X↔harness/splitter/normalize)·recursive KF 폴드슬라이싱 인과·정규화후 불확실성 생존. F-6 해소(진단→RobustScaler). fresh-eyes(규칙14) KF **CRITICAL/HIGH 0**. 실 full-stack smoke. F-13 등록. **244 통과** | 903bd14 |
| 2026-07-12 | Phase 3 R1 Step 3.1~3.2 | `harness` F-10 NaN 위생(정규화前 드롭·커버리지) + `src/research/models/`(ProbaModel·TreeBench(lightgbm)·SmallMLP(torch, 단일태스크·다중seed)). 신규 harness+5·models+13 | (미커밋) |
| 2026-07-12 | I-001 remediation(3.3 선행) | 손상 실범위 = 04-06~05-19 흩어진 14일 값손상(audit 미검출 클래스, 트래커보다 넓음). `resample_ohlcv`(1h→1d 완전버킷) 헬퍼로 regime 1d 파생. 파생 audit green·clean날 공식 일치. 신규 loader+4 | (미커밋) |
| 2026-07-12 | I-002 해소 | `multiclass_log_loss` 비정렬 labels proba 오정렬(sklearn 정렬가정 위배, Phase0 잠복→R1 발각). labels 정렬 수정+회귀(비정렬 known-answer). blast radius=R1 ledger만 | (미커밋) |
| 2026-07-12 | Phase 3 R1 Step 3.3 종착 | `experiments/r1_smoke`(사전등록 판정규칙 박제) 8판 실행 → **관문1 PASS**(mlp STRONG). 인과 self-check green·판정규칙 테스트+8. 규칙19 추가. **275 통과** | 2b34548 |
| 2026-07-12 | Phase 3 R2.0a~R2.1 | F-13 hurst `sliding_window_view` 벡터화(수치동치·75x, build 67.7s→9.2s) + `experiments/r2_window_search` coordinate descent 러너(MLP선택+트리참고·config캐시·예산등록). 신규 trend_strength+5·r2+2 | 52a60ac |
| 2026-07-12 | R2.2 스모크 | 기본창 1config 실데이터 → R1 MLP 정확 재현(ll 1.0673·mcc0.128·ba0.413) → 파이프라인 검증. 1config=8.9분 | 52a60ac |
| 2026-07-13 | R2.3 창탐색 실행 | 19판(예산60). 선택창(er96·rv96·lag12) Δll −0.0034. 트리vsMLP Spearman 평균 0.39(hurst −0.5 불일치). D-024/025 확정 | (미커밋) |
| 2026-07-13 | R2.4 국면분리 재확인·종착 | 선택창 국면 5/6·폴드 14/22 우세이나 미미(down\|low 악화)·노이즈수준 → **기본창 유지**(D-026). 문서 종합 갱신·§12-4 | (미커밋) |

---

## 10. 잠재 이슈 트래커 (I-NNN)

| ID | 내용 | 발생 | 상태 |
|---|---|---|---|
| **I-001** | 1d 캔들 손상 — audit 는 **off-grid 5봉**(04-08·09·10·12·14 16:00)만 검출했으나 **실범위는 04-06~05-19 흩어진 14일 값손상**(on-grid 부분봉 = audit **미검출** 클래스, Phase3 R1서 전수스캔으로 확정). 1h clean. | Step 0.1(범위확정 Phase3 R1) | **해소(Phase3 R1 3.3 선행)**: `resample_ohlcv`(loader)로 감사-clean 1h→1d 파생(완전버킷만·00:00그리드), regime·MTF 공용. 파생 1d audit green, clean날 공식 1d와 정확 일치 검증 |
| **I-002** | `multiclass_log_loss` 가 비정렬 labels(`LABEL_CLASSES`=('up','down','expire'))에서 proba-클래스 오정렬 → sklearn(≥1.x) 사전순 정렬 가정 위배로 log_loss 오답(known-answer 0.223 정답 vs 버그 2.303). Phase0 잠복(정렬 라벨 테스트라 미발현), R1 첫 성과 채점서 발각. 첫 R1 DISCARD 원인. | Phase3 R1 | **해소**: labels 정렬 후 proba 정렬(값 순서불변). 회귀(비정렬+known-answer). 규칙19 적용 — blast radius=R1 ledger만(삭제·재생성), 라벨/R0/Phase0-2 무오염(분포전용·uniform ln3 순서불변). 수정 후 재실행 관문1 PASS |

---

## 11. Phase 0 상세계획 (검증 substrate)

**성격**: 모델이 없는 순수 평가 기반. 산출물은 모델과 무관하게 독립 테스트 가능한
유틸. 본질은 코드가 아니라 **인과성·누수를 구조적으로 막는 계약**. 이후 모든 Phase가
이 위에서 채점되므로 load-bearing.

### 11-1. 컴포넌트 A~G

| | 컴포넌트 | 핵심 |
|---|---|---|
| A | **walk-forward splitter** | expanding 주(D-005) + purge/embargo 내장(N 파라미터, F-5). 모델 내부 모름 — fit/predict 콜러블 인터페이스만 |
| B | **국면 태깅**(규칙기반, D-003) | 1d 매크로 추세×변동성 버킷 + 히스테리시스/최소지속. **분석전용·피처 방화벽**(태그가 feature로 새면 즉시 누수) |
| C | **인과성/누수 하네스** | **미래-교란 테스트**(t 이후 값 흔들어 f(t) 불변)·train-only 통계·완성봉 검증. Phase1~4가 자기 함수를 꽂음 |
| D | **정규화**(z-score train-only) | fit=train fold만, transform=val. splitter 계약과 결합(창 방식이 통계 안정성 좌우) |
| E | **지표 + baseline** | 균형정확도·MCC·로그손실(3-클래스)·pinball·semi-dev + random/majority. **폴드·국면별 분포로** 집계(평균 아님), **봉단위 국면 귀속**(D-006) |
| F | **결과 정합성** | trades↔metrics↔equity — **기존 `test_backtest_fees` 재사용**(재구축 금지, DRY). Phase4 배선 |
| G | **experiment harness + 비교개수 원장** | walk-forward 실행 + config·결과 로깅 + **비교 횟수 계수·사전등록**(F-1·F-2 실효장치). **시드 고정**(재현성) |

### 11-2. 핵심 가드레일

- **purge/embargo**(§F-5): 삼중배리어 라벨이 [t, t+N] 걸침 → 경계 겹침 학습표본 제거(purge) + 검증 뒤 완충(embargo). N은 Phase1 파라미터 → splitter가 입력받음.
- **국면 방화벽**: regime 태그는 사후(전구간) 계산 허용(분석전용)이나, **feature 파이프라인 유입 시 즉시 누수** → 회귀 테스트로 강제.
- **비교개수 원장**: 모든 성과기반 비교를 계수·사전등록(홀드아웃 없는 방법론의 이빨).

### 11-3. Step 분할 + 완료 기준 (CLAUDE 9 체크포인트)

| Step | 내용 | 완료 기준 |
|---|---|---|
| **0.1** | 오프라인 데이터 로더(심볼 파라미터화, **연속성·봉정렬 감사**) + 국면 태깅(B) + 방화벽 테스트 | 1d 6.5년 태깅 세그먼테이션 산출·리뷰, 결측봉 감사 통과(발견 시 I-001) |
| **0.2** | walk-forward splitter(A, expanding+purge/embargo) | purge 누수 테스트 통과, 1h 폴드 수 확인 |
| **0.3** | 인과성/누수 하네스(C) + train-only 정규화(D) | 미래-교란 테스트가 합성 누수함수를 잡아냄 |
| **0.4** | 지표 + baseline(E) | 합성 예측/라벨로 지표 정확성 검증 |
| **0.5** | experiment harness(G, 시드고정) + **end-to-end 시연** | baseline 예측기로 1h walk-forward → 폴드·국면별 지표 + 누수 하네스 green |

*시연 타깃*: 실 배리어 라벨(Phase1) 미도착 → **throwaway 타깃**(예: 다음봉 수익률 부호)으로 **배관만** 실증(1회용, 미커밋, 결과만 보고).

### 11-4. 설계상 연기(데이터가 정함, 기둥5) — Phase 0 진행 후 상태

- 국면 태깅 수치 → **확정**(Step 0.1): ma100/slope20/db0.02/vol30/pct0.5/mindur14
- walk-forward T·V·S → **확정(Phase1)**: train_min=8760(1y)·val_size=2160(3mo)·**N=24**·step=val → 1h 22폴드. (F-5 해소)
- purge/embargo = N → **완료**(splitter가 `label_horizon` 단일파라미터로 purge+꼬리예약 구동, embargo 기본 0)

### 11-5. 모듈 배치 (naming 고지)

신규 최상위 **`src/research/`**: `data/`(로더) · `validation/`(splitter·regime·metrics·harness) ·
`causality/`(leakage 유틸) · `normalize.py`. `tests/research/`. 엔진/플러그인 분리(D-002), 엔진 수정 0.

---

## 12. Phase 0 결과·검증 요약 (완료)

**구축된 검증 substrate** (`src/research/`, 엔진 수정 0):
- `data/` — `load_ohlcv`·`load_audited`(표준 진입점: 감사 통과만 하류로) + `audit`(갭·오프그리드·OHLC 정합)
- `validation/` — `regime`(태깅+방화벽+완성봉 ff) · `splitter`(expanding + purge/꼬리예약, N 단일출처) · `metrics`(균형정확도·MCC·log_loss[sklearn]·pinball·semi_dev + 폴드·국면 집계) · `baselines`(Prior/Uniform, 모델 계약) · `harness`(run_walk_forward) · `ledger`(비교개수·예산)
- `causality/leakage` — 절단불변 + 양방향 미래교란 하네스 · `normalize` — z-score train-only(`BaseScaler`)

**검증** (규칙 17): **143 테스트 통과**. full-stack 통짜 테스트(regime+normalize+purge+harness+국면집계+ledger 한 흐름) + end-to-end 시연(1h: 무엣지 baseline이 균형정확도 0.333·log_loss ln(3) — substrate가 무엣지를 정확 측정). fresh-eyes 독립 재스캔 → 핵심 뼈대(purge math·forward_fill·regime_trend 인과·normalize·audit) **clean 판정**, 실결함 3(F1 누수 위음성·F2 log_loss 크래시·F6 vacuous green) 수정+회귀.

---

## 12-1. Phase 1 결과·검증 요약 (완료)

**구축** (`src/research/labeling/`, 엔진 수정 0):
- `volatility` — `atr`·`yang_zhang`(둘 다 **분수 정규화**, 공통 콜러블 `VOL_ESTIMATORS`) + `assert_causal` 회귀(배리어 폭 = 진입시점 정보만)
- `triple_barrier` — `LabelParams`/`BarrierLabels`/`barrier_levels`/`compute_triple_barrier`. 3-class {up,down,expire}(X.index 정렬, 워밍업·꼬리 N·**동시터치 NaN 제외**), `first_touch`(도달시간 원시, expire→N), `label_horizon` 단일출처(N)
- `distribution` — 층1(방향/만료 <0.87 극단회피 + 각 학습창 소수클래스≥100) + 층3(regime ff 슬라이스 극단회피 일관성). **성과 안 봄**(접근 B). 층2 폐기(미구현)

**인과 3중 규율**: ① 배리어 레벨 `assert_causal`(미래교란 불변) ② 라벨 **유한 지평 = N**(t+N 이후 교란해도 labels[:t] 불변) ③ splitter seam(라벨 N이 purge/꼬리예약 구동, train 라벨창 val 무침범).

**R0 결과** (1h 56,894봉, 128 config, splitter 22폴드):
- **51/128 통과**(층1∧층3). N밴드: N=12(28)·24(19)·48(4)·96(0). N=96 전멸=방향 극단(장기간→거의 도달).
- up/down 대칭 견고(|up−down|≈0) — 방향편향 없음 확증(기둥5). ATR>YZ 만료 여유(YZ 타이트).
- **확정 라벨 = atr/w96/x3.0/N24**: 해소율 0.997, 분포 **up 0.31/down 0.31/expire 0.38**, min_minor 2329. (게이트는 통과분 우열 미판정 — 선택은 경제적 1일 지평 + ATR 실행정합 + 3-class 균형의 **원칙** 근거. 예측력은 Phase3.)

**검증** (규칙 17): **188 테스트 통과**. 통짜 full-stack(확정 라벨 층1∧층3 통과 박제 + 층3↔실 regime ff seam) + fresh-eyes 독립 재스캔(규칙14) — **CRITICAL/HIGH 0**, YZ=Yang-Zhang(2000) 정합·누수 하네스 non-vacuous(주입 확인), LOW#4(미래 OHLC NaN 편향)→NaN 가드 수정+회귀.

---

## 12-2. Phase 2 결과·검증 요약 (완료)

**구축** (`src/research/features/`, 엔진 수정 0):
- `simple` — vol_change(ln σ비율)·relative_volume(ln vol/MA_past)·rolling_semi_deviation
  (metrics 재사용)·close_position. `trend_strength` — efficiency_ratio(∈[0,1])·hurst(R/S).
  `kalman` — kf_trend(선형 level+slope, Student-t 1-step 재가중, online 전방필터) → slope+불확실성.
  `build` — build_features → **X 8열** (자기정규화 없음, harness 소유).
- `normalize.RobustScaler` — median/MAD 드롭인(F-6).

**검증 종류 = 소프트웨어 정합성 + 행동(성과 안 봄)**: 각 축 `assert_causal`(인과) + 정상성/유계
+ **KF 행동 3종**(일시스파이크 robust<gaussian damp·지속이동 추종·불확실성 비상수·dof→∞ Gaussian복원).
seam 통합(X↔harness/splitter/normalize)·recursive KF 폴드슬라이싱 인과·정규화후 불확실성 생존.

**F-6 진단**(실 1h): 팻테일 축 z-score 왜곡 유의(semi_dev·kf_uncertainty 초과첨도~73,
kf_uncertainty std/robustScale≈5.0) → RobustScaler 제공. 선택은 Phase3.

**검증**(규칙17): **244 테스트 통과**. 통짜 full-stack(8축 X→harness 두 정규화·splitter·baseline)
+ 실 1h smoke(skip-guard) + fresh-eyes 독립 재스캔(규칙14, KF 수학 독립재구현 대조) **CRITICAL/HIGH 0**.
교차Phase전파(규칙18): X=8열 계약·D-011~019·F-12/F-13.

---

## 12-3. Phase 3 R1 결과·검증 요약 (완료)

**성격**: Phase 3 첫 수직 슬라이스(D-001) — 처음으로 "예측력"을 측정. 1h·확정라벨·기본창·기본하이퍼 **고정, 탐색 없음**(R2+ 소관, 검증셋 과적합 방어).

**구축**:
- `src/research/models/` — `ProbaModel`(계약 base) · `TreeBench`(lightgbm 단일태스크 하한) · `SmallMLP`(torch, 트렁크/헤드 분리 → R3 멀티태스크·3층 웜스타트 이식, R1 단일태스크). 다중 seed 안정성.
- `src/research/experiments/r1_smoke` — **사전등록 판정규칙 박제**(`PREREGISTERED_RULE`, 실행 前 잠금) + 러너(커밋·skip-guard).
- `harness` F-10 NaN 위생 · `loader.resample_ohlcv`(I-001) · `metrics` log_loss 정렬(I-002).

**실행**: 1h(56.9k봉)·삼중배리어 atr/w96/x3.0/N24·22폴드 walk-forward·z-score train-only·regime(파생 1d ff). Prior/Uniform(기준선) + Tree(1) + MLP(5 seed) = **8판, 비교예산 2 사전등록**.

**관문1 = PASS** (사전등록 3조건: ll<Prior ∧ MCC>0 ∧ 일관성>과반. BA는 보고지표):

| model | ll中 | MCC中 | BA中 | 일관성 | flag |
|---|---|---|---|---|---|
| uniform | 1.0986 | 0 | 0.333 | — | - |
| prior | 1.0957 | 0 | 0.333 | — | - |
| tree | 1.0936 | 0.105 | 0.400 | 0.50 | weak |
| **mlp** | **1.0673** | **0.128** | **0.413** | **0.91** | **STRONG** |

- MLP STRONG → **PASS**. 엣지 **실재하나 작음**(정답확률 +~1%p·MCC 약한 양상관·BA 우연+8%p ≈ 무지→완벽의 3~12%), **일관성 0.91**(20/22폴드)·seed 스프레드 타이트 → 노이즈 아님. MLP>Tree(R2 방향, 게이트 아님).
- **해석 규율**: R1 = "엣지 냄새"이지 강엣지·경제성(관문2)·튜닝(R2) 아님. borderline 유보 규율 무발동(명백 STRONG). config 탐색은 R2(사전등록·좌표하강·국면재확인)로만 — ad hoc 금지(F-1).

**발견·해소 (규칙 19 적용)**:
- **I-001**(1d 손상 실범위 14일, audit 미검출): `resample_ohlcv` 파생으로 해소(§10).
- **I-002**(log_loss 오정렬, Phase0 잠복→R1 발각): **첫 실행 DISCARD의 원인**. 수정 후 재실행 PASS. **사전등록 규칙 불변, metric만 교정** — MCC/BA(버그 무관·하드예측)가 처음부터 신호라 PASS를 독립 corroborate(수정 artifact 아님). 통짜 실행이 load-bearing 버그를 포착(규칙17 가치).

**검증**(규칙17): **275 테스트 통과**(harness+5·models+13·r1_smoke+8·loader+4·metrics+1). 인과 self-check(build_features `assert_causal`) green. 러너 통짜 실행이 통합검증 겸함.

---

## 12-4. Phase 3 R2 결과·검증 요약 (완료)

**성격**: R1 엣지가 **창(시간척도) 튜닝으로 강해지나** 규율 평가. 1h·확정라벨·단일태스크 고정. KF·정규화·멀티태스크는 R3(D-025).

**구축**(52a60ac): `features.hurst` 벡터화(F-13, R2.0a) + `experiments/r2_window_search`(coordinate descent 러너).

**방식**(D-024): 초기값=R1 기본창. 피처별 창 스윕(er·hurst·vol_change window/lag/estimator·relative_volume·semi_dev) 1사이클. **각 config = MLP 5seed seed-중앙값 log_loss 최소로 선택**, **트리 참고**(관찰용). config 캐시·**예산 사전등록 19/60판**(F-1). Prior/Uniform config 무관 1회.

**결과**:
- **창 튜닝 개선 미미**: 기본창 MLP ll 1.0673 → 선택창(er96·rv96·lag12) 1.0640, **Δll −0.0034**. MCC 0.128→0.132·BA 0.413→0.414. 개선폭이 **seed 노이즈(±0.002)와 비슷**.
- **R2.4 국면분리 재확인**(F-1): 선택창이 **6국면 중 5·22폴드 중 14 우세**(골고루 약우세, 한 곳 몰림 아님)이나 **미미**(폴드 diff 중앙값 −0.0004, down|low 국면 오히려 +0.0032 악화, 2국면 무승부).
- **트리 vs MLP 순위상관**(가설검증): Spearman 평균 **0.39**(er 0.90·semi_dev 0.80 일치 / **hurst −0.50·vol_change.window −0.20 불일치**). 창 효과가 노이즈 수준이라 부분적 해석이나, **트리 프록시 부실 확인** → MLP 직접 탐색이 옳았음(R3/R4도 트리 프록시 미정당).

**결론**(D-026): 창 튜닝은 약엣지를 **강화하지 못함**. 1h 엣지가 작은 건 튜닝부족이 아닌 **실제 신호 강도** → **기본창 유지**(노이즈 미채택, 기둥5). **282 테스트 통과**.

**부산물**: F-13 해소(hurst 75x) · F-12 해소(D-025) · 트리 프록시 신뢰 불가 확인.

---

## 다음 단계

**Phase 3 R3 착수**: 1h **모델 하이퍼** 튜닝 — **멀티태스크 헤드**(도달시간 co-training, 기둥7 실현 — A-1 스테이징 회수)·**정규화**(z vs robust 최종선택, F-6)·깊이/너비·**KF Q/R/dof**(config 하이퍼, D-025). 비교개수 사전등록·국면분리 재확인(F-1) 지속. **기대는 신중히** — R2 증거상 창은 엣지를 못 키웠으므로, R3(새 지렛대)가 움직일지 미지수. **R3 후 R4**(TF/MTF 확장 — 4h/1d/15m 단독+MTF, 관문1 통과·I-001 해소 충족, best-of-N 금지 F-2). R4 MTF는 `resample_ohlcv` 재사용.
