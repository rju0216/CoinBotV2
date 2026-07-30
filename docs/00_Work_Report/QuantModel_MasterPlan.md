# QuantModel MasterPlan — 개발·검증·결정 관리 문서

> **성격**: "**어떻게 만들고·검증하고·결정하나**". 모델 정체("무엇을")는
> `docs/BTCUSDTperp_QuantModel_Design_vDraft.md` 가 정본이며 이 문서는 그것을
> **참조만** 한다(중복 서술 금지). 이 문서는 개발 계획·학습 스케줄·리뷰어
> 플래그·결정 로그(D-NNN)·잠재 이슈(I-NNN)·진행 기록표를 담는다.
>
> **현재 시점**: **Phase 0~6 완료**. **Phase 6**(최적화3층, §12-9)에서 **용량(단일슬롯)이 1차 지렛대**로 판명 — 단일슬롯이 신호의 85~90%를 버려 판정 자체가 표집에 지배되고 있었다(형제 config 부호 반전 +147.7% vs −128.4%). 엔진을 **N-트랜치**로 확장(D-037, `max_slots`, 기본 1 = 기존 동작)해 **포화(100% 포착)에서 재측정**하니 부호 반전 소멸. **93셀 전수 원장**에서 **ex-2024 양수는 D2/D7 뿐**(= 4h_4d × EV게이트 × 만기게이트) — 층1(연도별 전부 양수) 통과 0. **D2 를 다음 Phase baseline 으로 박제**(D-042). **완전 kill 0 유지**(D-043) — PARK 셀에 정량 재소환 조건 부여(15m 은 방향적중률 ≥60.4%). 발견: **θ knife-edge = 표집 아티팩트**, **알파는 양방향 실재하나 베타 노출(87% 롱)이 상쇄**, **금지가정 "방향편향" 실측**(예측 up 77.7% vs 라벨 32.7%). **다음 = Phase 7(layer 0~2 개선) → layer-3 재측정**(결정트리 제3 경로, §다음 단계) → Phase 8 현실·교차강건·관문3. **Phase 번호 개정(D-044)**: 구 "6.5" → **Phase 7**, 구 Phase 7(현실·교차강건) → **Phase 8**.

---

## 0. 근거 확인 (계획의 전제 — 실측)

| 항목 | 실태 | 함의 |
|---|---|---|
| BTC 퍼프 캔들 | 15m/1h/4h/1d 모두 **2020~2026.6 (~6.5년)** 보유 (1h 56.9k행, 1m도 있음) | 2020폭락·21불·22베어·23회복·24불 포함 → 독립 매크로 국면 확보 |
| 의존성 | **torch 2.11(CUDA)·sklearn·lightgbm·xgboost 이미 설치** | 2층 MLP + 트리 벤치 **신규 의존성 0** |
| 펀딩 데이터 | **없음** | Phase4 = funding 0 baseline + 상수펀딩(~0.01%/8h) 민감도(trades 후처리, D-c). 판정 무영향(비용은 수수료가 지배) → F-3 해소 |
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
                                    [Phase5] 정책→지평 프론티어 (관문2 FAIL 유효, 4h=2024아티팩트)
                                    [Phase6] 최적화3층 (layer-3 소진, 용량이 1차 지렛대, D2 baseline 박제)
                                    [Phase7] layer 0~2 개선 → 3층 재측정 ──────→ [Phase8] 현실·교차강건·관문3
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
| **3 2층 MLP** ✅ | 공유트렁크 멀티태스크 MLP + 트리 벤치 — **R1~R4 완료(관문1 PASS)**. 1h 튜닝저항(창·멀티태스크·정규화·구조·KF 무gain, R1 config 유지) → R4서 **15m+1h 채택**(단독15m>1h·확인CONFIRM·MTF 1h 강화) | **후보2 예측력** | **관문1 통계적엣지 PASS** |
| **4 멍청한 3층** ✅ | 방향·θ진입·고정사이즈·배리어청산 + 비용모델, **기존 엔진 경유** — **완료: 관문2 FAIL-on-cost**(§12-7). 엣지 실재(gross+126%)나 거래당엣지≪비용(1/5) → net 음. plugins/dumb_l3·oos_export·엔진 O(n²)수정(D-031) | **후보3 경제성** | **관문2 FAIL** |
| **5 정책→지평 프론티어** ✅ | 홀딩 5셀 정책 + 지평×배리어×x 프론티어(gate1→관문2) + 실펀딩(Binance) + fresh-eyes — **완료: 관문2 FAIL 유효**(§12-8). 홀딩 무효·gate1프록시 경제성 오판·유일양성 4h는 2024 아티팩트(비정상). oos_export 일반화·dumb_l3 파라미터화 | **후보3 경제성(재확인)** | **관문2 FAIL 유효** |
| **6 최적화3층** ✅ | 국면필터·conviction·청산재설계·EV게이트·만기게이트·**N-트랜치 용량**·지평앙상블 — **완료: layer-3 소진**(§12-9). **93셀** 포화 재측정서 ex-2024 양수 = D2/D7 뿐, 층1 통과 0. **용량이 1차 지렛대**(D-037)·θ knife-edge=표집아티팩트·방향편향 실측. policy_eval·econ_l3·엔진 N-트랜치 | **시간안정성**(ex-2024+최근OOS) | **층1 FAIL / D2 baseline 박제** |
| **7 layer 0~2 개선** | 표본/용량 가설 검정(E-배치)·배치 진단(편향·s/q 분해·피처 방향성)·처방 사다리·**layer-3 재측정**(도구 완비) | **D2 baseline 초과** | — |
| **8 현실·교차강건** | realistic execution(슬리피지·maker fill model)·멀티심볼(ETH/SOL) **교차강건 검증**·재검증 + **트랜치 실집행**(reduce-only 장부)·**베타 중립화** | 배포가능성 | **관문3** |

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
| **R3 하이퍼 소그리드** ✅ | R2 후 | 1h, R3.1 멀티태스크(λ0.1/0.3)·정규화(z/robust) 5판 + R3.2 구조/KF 8판 | 13판 | **완료: 전부 무gain**(멀티태스크·robust·용량↑·KF 다 무개선/악화) → **R1 config 유지**. §12-5 |
| **R4 TF/MTF 확장** ✅ | R1~R3 후 | 4h·1d·15m 단독 + MTF(15m+상위 완성봉) | 단독4 + MTF 3(사전등록) | **완료: 15m 채택**(단독15m>1h·확인CONFIRM 6국면·직교라벨 ROBUST) + **MTF 1h 강화**(+0.0036 material, 4h 중복). §12-6 |
| **R5 경제성** ✅ | Phase4 | 멍청3층, 엔진 경유. 전체 OOS 1m baseline(θ0.40) + 사전등록 θ{.35~.50}×MTF게이트{off,on} 8-config 스윕 | **채택 0** | **관문2 FAIL-on-cost**(§12-7) |

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
| **F-1** | R2 창 순차탐색 = 홀드아웃 없는 검증셋 성과 선별 → **검증셋 과적합 위험**. 방어: 비교개수 사전등록 + 선택 창을 국면분리에서 재확인. | 계획 §5 | **R2~R4 적용**: 예산 사전등록(R2 19/60·R4_15m_confirm 2·R4_mtf 3) + 확인(선별↔확인 분리). R4: 라벨=분포 선정(성과 안 봄)·15m 국면일관+직교라벨 확인·MTF 임계초과 채택(max 고르기 아님). 열림(Phase4 동일 규율) |
| **F-2** | 선별 2단 중첩(라벨=분포/TF=예측력). 다른 TF는 "4개 중 최고 뽑기(부풀림)" 아니라 **확증·강건성**으로. | 계획 §5 | 열림(Phase3 대비) |
| **F-3** | 펀딩 데이터 부재 → 상수 근사가 실비용/엣지 왜곡 가능. Phase4 전 실펀딩 다운로드 옵션. | 계획 §0 | **해소(Phase4, D-c)**: funding 0 baseline + 상수펀딩(~0.01%/8h 항상-비용) 민감도(trades 후처리, 엔진무손). 6h지평서 펀딩 1차영향 작고 **비용은 수수료가 지배**(net 판정 무영향) → 실펀딩 불요. 판정 뒤집는 지점이면 격상(현 미발생) |
| **F-4** | TF별 종료일 불일치 → MTF 결합 시 겹치는 구간 정렬 필수. | 계획 §0 | **해소(R4.2)**: 실측 1h·4h 종료일 ≥ 15m(tail 커버리지 손실 0). `build_mtf_features` 에 **tail 커버리지 가드**(상위TF 완성봉이 결정TF tail 미달 시 하드페일 — stale-fill=무한 forward-fill 방지, fresh-eyes MEDIUM①). 회귀 박제. |
| **F-5** | purge/embargo = 라벨 지평 N에 결합(교차-Phase). splitter가 N을 파라미터로 받아야 하고, walk-forward T·V·S 도 피처/라벨 창에 번인 결합 → 수치는 Phase1~2 후 확정. | 계획 §11 | **해소(Phase1)**: **N=24 확정** → T·V·S = train_min 8760·val_size 2160·N 24·**22폴드**(step=val). embargo 0 유지 |
| **F-6** | 설계 §7이 **z-score** 지정했으나 robust 대비 **논증 없음**(근거는 train-only 인과성뿐). 기둥6(팻테일) 견주면 robust(median/MAD)이 이상치에 강함. **방식**: Phase2에서 ①분포 진단 먼저(비교예산 무관)→②왜곡 유의시만 예측력 비교(F-1 계수). Step0.3 scaler는 `BaseScaler` 교체 인터페이스(robust 드롭인). | Step0.3 | **최종 해소(R3.1)**: 실 1h 예측력 비교 → robust 가 z 보다 **+0.0006 나쁨**(무gain) → **z-score 확정 채택**(팻테일 진단 유의했으나 예측력은 z 우세). Phase2 진단(초과첨도~73)+R3.1 예측력 판정 완결. |
| **F-7** | 겹치는 val 창(step<val_size) → per-regime 봉단위 귀속 **이중계수**. 현재 non-overlap 기본이라 미발현(문서화됨). | fresh-eyes | 열림(Phase3 rolling/overlap 시 dedup·가중) |
| **F-8** | per-regime 귀속 계약: regime_tags는 **모델 TF 정렬** 필수(원시 상위TF 주면 대부분 드롭), n은 "귀속 가능 봉만"(NaN 태그 제외). 문서화됨. | fresh-eyes | 열림(Phase3 커버리지 경고 추가 고려) |
| **F-9** | `forward_fill_completed`·`ZScoreNormalizer.transform`이 unique/complete 컬럼 무가드 전제. | fresh-eyes | **부분해소(R4.2)**: MTF 는 `build_mtf_features` 가 `mtf{tf}_` 접두어로 **unique 컬럼 보장**(충돌 방지). ZScoreNormalizer 의 complete 전제는 harness F-10 NaN 드롭이 선행 보장. 정규화기 자체 가드는 열림. |
| **F-10** | `run_walk_forward`가 fold의 y NaN을 드롭 안 함(harness). 삼중배리어 라벨은 동시터치→NaN을 중간에 낼 수 있어(실측 ~0.08%) Phase3 모델 학습 시 NaN 클래스 유입 가능. R0(Phase1)는 분포만·모델 fit 안 함이라 미발현. | Phase1 seam | **해소(Phase3 R1 Step3.1)**: `run_walk_forward` 가 **정규화 前** train·val 각각 X∪y NaN 드롭 + 커버리지 로깅(`WalkForwardResult.coverage`). val NaN 채점 제외. 회귀 5. 실 R1 val 드롭 50봉 |
| **F-11** | 라벨 참조가=close_i·스캔=후속 high/low(D-007)인데 엔진 진입은 현재봉 open(INFRA §4-3). 라벨-실행 미세 불일치 + 봉내 동시터치(현 NaN 제외)는 1m/15m 인트라바로 복원 가능. | Phase1 | **해소(Phase4)**: 배리어를 **실진입가(open) 앵커**로 재계산(D-b①, barrier_frac=라벨 x·σ 재사용). 동시터치는 **1m master 체결해상**(D-b②)으로 복원 → **I-004 발견**(15m master는 intrabar stop-out 불가시=낙관편향, 1m 필수). seam 통합테스트로 진입타이밍·정합 박제(§12-7) |
| **F-12** | KF Q/R/dof 를 Phase3 에서 어떻게 인과적합하나 — (a)폴드 train-only MLE 재적합(정규화 동형) vs (b)창길이처럼 config 하이퍼 coordinate descent. 설계 §7 문구는 (a) 뉘앙스, R2 파이프라인은 (b) 정합. Phase2 는 고정 기본값이라 무관. | Phase2 Step2.3 | **해소(R2, D-025)**: **(b) config 하이퍼** 채택 — D-019(피처=폴드무관 순수함수·전역X, harness 슬라이스) **보존**. (a)는 폴드내 KF 재계산이라 D-019 붕괴. KF 실 튜닝은 **R3**. |
| **F-13** | hurst(R/S) `rolling.apply(python)` 성능 — 실 56.9k봉 build_features 67.7s(hurst 병목). Phase3 R2 창 순차탐색(다수 config×폴드) 전 벡터화 필요. Phase2 스코프=정합성이라 미해결. | Phase2 Step2.2/2.4 | **해소(R2.0a)**: `sliding_window_view` 벡터화(sub-window R/S 전창 동시). 스칼라와 **수치 동치**(실 1h 56.9k 완전 일치·NaN포함), hurst 48.2s→0.64s(75x), build 67.7s→9.2s. NaN 창=pandas min_periods 매칭, 퇴화행만 폴백. 회귀 박제. |
| **F-14** | (fresh-eyes R4.2) `classify_regime_consistency` 국소붕괴 임계 = **fold-median 전역 ll-margin** vs per_regime **pooled** margin — 집계방식 상이(~2% 스케일차). | fresh-eyes R4 | 열림(LOW). **미발현 지속**. Phase4 MTF 역할2 게이트는 **가격 부호(mtf1h_kf_slope) 게이트**로 도입(ll-margin collapse 게이트 미사용) → F-14 무관. 향후 국면-collapse 게이트 활성화 시 pooled 통일 검토. |
| **F-15** | (fresh-eyes R4.2) `_per_regime_edge` `reversal_seed_frac` = `.ge(prior).mean` — 어떤 seed 가 국면 결여 시 NaN→False 로 반전 과소평가(안전방향 반대). | fresh-eyes R4 | 열림. **미발현**(Prior·전 seed 동일 국면집합, 계약상 결여 없음). 방어적 명시드롭 권장. |
| **F-16** | (fresh-eyes R4.2) MTF X 는 상위TF 워밍업만큼 선두 NaN 이 baseline 보다 많아 fold-0 유효표본 상이 → "오직 피처만 변경" 격리 미세 위반. | fresh-eyes R4 | 열림(LOW). prior 자기정규화(동일 지지)로 대부분 상쇄·선두 국한. |
| **F-17** | (fresh-eyes R4.2) `run_mtf_role1` 이 `baseline_ll_margin` 외부주입 — 동일 라벨/splitter/seed 계산 보장이 코드에 없음(footgun). | fresh-eyes R4 | 열림(LOW). 현 실행은 동일 config baseline(문서 계약). Phase4 재사용 시 주의. |
| **F-18** | (fresh-eyes Phase6.4) **N-트랜치 중첩 시 `n_trades` 가 유효 표본수를 과대** — 동시 보유 거래는 독립 관측이 아니다(대략 √평균동시성 배; D2 avg_conc 2.36 → **약 1.5배**). 연도별 판정·승률 유의성이 실제보다 강해 보인다. **N 이 커질수록 악화.** | fresh-eyes Phase6.4 | 열림(MEDIUM). §12-9·원장 수치에 경고 병기. 다음 Phase 판정 시 유효표본 보정 검토. |
| **F-19** | (fresh-eyes Phase6.4) **`slippage_tolerance_bp_per_fill` 은 1차 근사** — 체결당 s bp 가 정확히 2s bp 노셔널 비용이라 가정하고, 슬리피지가 **SL/TP 체결가**와 **EconL3 자신의 EV 게이트 비용항**(현재 `slippage_pct=0`)까지 바꿔 **거래 선택 자체를 바꾸는** 2차 효과를 무시. **실제 여유는 보고값보다 낮다.** | fresh-eyes Phase6.4 | 열림(MEDIUM). Phase8 realistic execution 에서 실측 대체. |

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
| **D-015** | F-6 해소 = `RobustScaler` 드롭인 제공, 선택은 Phase3 예측력 → **최종 z-score**(R3.1 robust +0.0006 나쁨) | S | 6 | 확정 |
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
| **D-027** | R3 결론 = 멀티태스크·정규화(R3.1)·구조·KF(R3.2) 전부 무gain(용량↑=단조악화, 나머지 노이즈) → **R1 config 유지**(단일태스크·z·d2w32·KF기본). **1h 엣지 튜닝 저항적 확정**(5지렛대 전수). 멀티태스크 헤드 코드는 보존(Phase5 도달시간 청산·웜스타트 경로). 다음 지렛대=R4 다른 TF | S | 5,3,7 | 확정 |
| **D-028** | R4 단독TF 결론 = **15m 채택**(1h 튜닝저항 → 다른 시간척도가 답). 단독 15m이 1h보다 강엣지(+0.045>+0.028) + **확인 CONFIRM**(6국면 전부 양마진·5seed 무반전 / 직교라벨 12h·w96 ROBUST). 4h 약함·1d 데이터부족(2370봉). 라벨=TF별 분포게이트 선정(성과 안 봄) | C | 8,5 | 확정 |
| **D-029** | R4.2 MTF 역할1 = **15m+1h 채택**(상위 1h 완성봉 맥락이 15m 엣지 material·국면일관 강화 +0.0036). **4h 중복**(단독 노이즈·1h위 +0.0005 노이즈) → parsimony 로 제외(규칙 best=1h+4h이나 4h 우위=노이즈, 장식금지·R3.2 정합). 역할2(진입게이트)는 Phase4. MTF 이득은 주라벨서만 측정 | C | 5,3,8 | 확정 |
| **D-030** | `forward_fill_completed` 를 `validation/regime.py` → **`data/loader.py` 승격**(TF정렬 공용 헬퍼, regime·MTF 2사용처 → features→validation 상향의존 회피, docstring 사전약속 이행). 전수스윕 잔존 0 | S | 2,4 | 확정 |
| **D-031** | (Phase4) `_slice_candles` 매봉 부울마스크 O(n) → 전체 O(n²)(실측: 전체OOS 1m ~39h). **`searchsorted+iloc` O(log n) 수정** — 정렬인덱스라 결과 동일(등가성 회귀+330통과+end-to-end 233거래·−3099.72 재현). 전체OOS 1m ~8분. **"engine 수정0"은 플러그인 추상화 규칙이지 성능버그 방치 아님**(규칙19 근본수정) | S | 2 | 확정 |
| **D-032** | (Phase4 배선) **OOS 예측 박제 리플레이**(D-a, gate1 통과 예측 그대로 재생=선별↔확인 분리) · **실진입가 배리어 앵커**(D-b①, F-11) · **1m master 체결해상**(D-b②, I-004) · **funding 상수 민감도**(D-c). 사이징=고정 notional(비복리, 비용분석 해석가능) | S,C | 2,3,8 | 확정 |
| **D-033** | (Phase4 종착) **관문2 판정 = FAIL-on-cost**. 엣지 실재(gross+126%, 전 국면·연도 대부분 양)하나 **거래당 엣지 ~0.02% ≪ 왕복비용 0.1%**(1/5). θ스윕{.35~.50}×MTF게이트{off,on} 채택0(net>0은 θ.50뿐이나 72거래·2/6국면음=winner's curse 탈락). **진입-지렛대(θ·게이트·사이드아웃) 전멸.** churn=비용 절반(진단). **다음=Phase5**(홀딩/persistence 첫실험→비대칭/trailing→maker 시나리오→(조건부)지평 재개). 로드맵 지렛대 한정 금지(새 insight 능동탐색) | C | 5,3,8 | 확정 |
| **D-034** | (Phase5-A 홀딩) **정책은 6h 경제성 못 살림**. 5셀 사다리 최선=no_timeout −315%(sign-flip 불가). **no_tp가 gross를 음으로 붕괴 → "승자 태우기" 반증**(엣지=단기지평, 배리어 넘기면 반납). churn 제거 실재하나 불충분. no_tp·no_timeout·allow_reverse·conviction 파라미터 추가(엔진무손) | C | 5,3,8 | 확정 |
| **D-035** | (Phase5 프론티어 종착) **관문2 FAIL 유효 — 프론티어도 못 살림**. gate1 ll-margin·MCC가 경제성 2회 오판(4h/3·4일 "약함" 오판, 재검증으로 잡음). 경제 홈=4h 넓은배리어. 유일양성 4h/{3,4,5일}/x3이 실펀딩 후에도 양성처럼 보였으나 **fresh-eyes 독립검증 = 2024 아티팩트**(net 80% 2024·ex-2024 노이즈/음·2025-26 음·비정상 트렌드베타, **코드결함 0**). **검증엣지 아님.** fresh-eyes가 과적합 종착전 차단(규칙17) | C | 5,3,8,14,17 | 확정 |
| **D-036** | (Phase5 config 처분) **완전 kill 0, 2단**: CARRY(방향엣지 측정 → Phase6 정책검증, **전부 시간안정성 바** ex-2024+최근OOS held-out) / PARK(전기간 방향신호≈0 → 휴면·소환가능, 정책이 0엣지서 이익 못만듦·과적합기계 방지). **최종 discard는 최적화3층에서**(경제성=정책의존, dumb경제성=스크린 아닌 사형선고 아님). 은밀폐기 0·능동세트 유계(과적합방지 우선순위) | C | 3,5,8,19 | **연장(D-043)** |
| **D-037** | (Phase6 (라)) **엔진 단일슬롯 → N-트랜치 확장**(`backtest.max_slots`, 기본 1). 선형 perp 손익 선형성으로 **넷팅과 손익 동일** → 실거래 재현 가능(봇이 트랜치 장부 보유·reduce-only 부분청산). `max_slots=1` 은 비트동일(**중량 등가성 2회 통과** + 교집합 검정: N=4 거래가 N=24 에 전부 포함·체결가 최대차 0.0). **(B) 제약 ③ 단일슬롯 공식 해제.** "엔진 수정 0"은 플러그인 추상화 규칙이지 **메커니즘 확장 금지가 아님**(D-031 선례 연장) | S | 2,8 | 확정 |
| **D-038** | (Phase6.3) **EV 게이트** = `barrier_frac×(p_up−p_down) ≥ k×실비용`(비용=2×(taker+slip)+보유h/8×0.0001). 도출근거 = **θ가 3-클래스에서 방향이 아닌 "도달"을 쟀다**는 실측(corr(p_dir, 방향마진) = 15m **0.05**·4h 0.55). 자유값은 k 하나·의미 고정(무튜닝). **만기 게이트**(`require_dir_gt_expire`, 무파라미터·Phase4 이후 미사용)는 θ 위에서 산술적 no-op 이라 **EV 위에서만 실효** | S | 3,5 | 확정 |
| **D-039** | (Phase6.4) **트랜치 사이징 규약 = 총노출 고정 N등분**(레버리지 불변, 포착률만 변수). 운용 사이징은 별개 결정 — **N 은 포착률 기준·사이징은 목표 위험 기준으로 원리고정**(성과로 고르면 θ knife-edge 재현) | S | 5,8 | 확정 |
| **D-040** | (Phase6.4) **N 간 비교 규율**: 승률·비용비·**bp/거래**·연도별 **부호**는 N 불변이나 **net% 는 총 거래노셔널이 달라 직접 비교 금지**. 수렴·격차 판정은 `net_af_bp_per_trade` 로 | S | 8 | 확정 |
| **D-041** | (Phase6.4) **지평 앙상블(3+4+5일) 채택 안 함** — MDD 개선(28.6→22.3%)이나 ex-2024 **+10.8→−4.7 음전**(5d·3d 가 4d 희석), 수익/MDD 0.82 < 4d 단독 1.20. fresh-eyes 실측: 슬리브 동시진입 **1862봉 전부 동일 방향** = **방향 분산 0**(시간 분산만). 코드는 보존 | S | 5 | 확정 |
| **D-042** | (Phase6 종착) **layer-3 판정 = 소진**. 93셀 포화 재측정서 **ex-2024 양수는 D2/D7 뿐**, 층1(연도별 전부 양수) 통과 0. **D2 를 다음 Phase baseline 으로 박제** = `4h_4d × EV게이트(k2) × 만기게이트 × N=12` (n1707·gross 61.6bp·net_af 41.3bp·**ex-2024 +13.4bp**·MDD 33.3%·slipTol 20.6/5.4bp). **라이브 보류·"엣지 강화 후 재판정" 제3 경로 채택**(사용자 확정) | C | 3,5,8 | 확정 |
| **D-043** | (Phase6 종착 config) **완전 kill 0 유지 — DISCARD 없음**(사용자 확정). layer-3 기각은 "그 예측 위에서"의 판정이며 **다음 Phase 가 0~2층을 바꾸므로 무효화될 수 있다**. PARK 셀에 **정량 재소환 조건** 부여 — 15m 은 **방향 적중률 ≥60.4~60.8%**(현재 55.3~58.3%)면 손익분기. **2024(편향 최소) 실측 61.1%** 이므로 편향 교정만으로 도달 가능한 크기. 은밀폐기 0 무결 | S | 3,5 | 확정 |
| **D-044** | (Phase7 착수) **Phase 번호 개정** — 구 "6.5(layer 0~2 개선)" → **Phase 7**, 구 Phase 7(현실·교차강건·관문3) → **Phase 8**. 근거: Phase 6 은 layer-3 로 종착·커밋(810e4bb)됐고 layer 0~2 는 층·도구·판정이 다른 별개 작업체다(한 Phase 였다면 Phase 6 에서 layer 0~3 을 함께 했을 것). "6.5" 는 표에서 유일한 소수점·유일한 화살표 표기로 **종착 시 끼워넣은 메모** 성격이었고, §12-1~12-9 ↔ Phase 1~6 대응상 다음 절이 **§12-10** 이어야 정합. 전수 스윕 13곳(MasterPlan 11·INFRA 2·CLAUDE 2) 갱신, 코드·테스트 무영향(phase5/6 산출물은 번호 불변). **naming 변경 고지 완료** | S | — | 확정 |
| **D-045** | (Phase7 착수) **층1 바(연도별 전부 양수) 유지 + F-18 유효표본 보정 신뢰구간 병기**. 2차의견은 "독립 관측 6개에 조건 6개 = 검정력 과소(연 80% 승률 전략도 74% 확률로 탈락), 93셀 미통과는 '바가 표본 대비 과함'의 증거일 수 있다"며 완화(`ex-2024 양수 AND 최악연도 ≥ −X AND 부호 다수결`)를 제안. **완화 기각** — 93셀 실패를 알고 바를 낮추는 것은 사후 합리화이고 `X` 라는 자유값 신설(기둥5 위배)이다. 대신 **F-18 의 올바른 대응은 바 완화가 아니라 측정 정밀화** → 연도별 `net_af_bp` 에 √평균동시성 보정 CI 를 병기해 "0 과 구분되는 음수인가"를 먼저 판정한다. 대부분 연도가 0 과 구분 불가로 나오면 그때 **근거를 갖고** 바를 재정의한다. 함께 **지평 패밀리(3d/4d/5d) 부호 일치**를 승격 요건에 추가(판정 강화 방향) | S | 8,5 | 확정 |

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
| 2026-07-13 | R2.4 국면분리 재확인·종착 | 선택창 국면 5/6·폴드 14/22 우세이나 미미(down\|low 악화)·노이즈수준 → **기본창 유지**(D-026). 문서 종합 갱신·§12-4 | 9774639 |
| 2026-07-13 | R3.0~R3.1 | 멀티태스크 헤드(reach·λ·expire마스킹, reach=None=R1바이트동일) + `r3_multitask` 러너. 실행: 멀티태스크(λ0.1/0.3)·robust 전부 무gain(+0.0006~0.0009) → 단일태스크·z. F-6 최종=z. 신규 models+5·r3+2 | (미커밋) |
| 2026-07-14 | R3.2 구조·KF + 종착 | `r3_arch_kf` 8판 사전등록: 용량↑ 단조악화(d3w64 +0.0025)·KF 노이즈(±0.0004) → **R1 config 유지**(D-027). 1h 튜닝저항 확정. 신규 r3_arch_kf+2. 문서·§12-5 | 5824cdf |
| 2026-07-14 | R4.0~R4.1 단독TF | `tf_expansion`: TF스케일 splitter(시간고정 1yr/3mo 봉환산)·per-TF 라벨 분포게이트·`run_tf_gate1`. **15m PASS 강엣지(1h 초과)**·4h PASS 약함·1d 데이터부족. 신규 tf+2. 세션전환→핸드오프 메모리 | e9e879b |
| 2026-07-15 | R4.1 15m 확인 | `tf_confirm`(사전등록 규칙)·`run_tf_gate1` per_regime 노출. **축① 국면일관 CONFIRM**(6국면 양마진·5seed 무반전)·**축② 라벨강건 ROBUST**(직교 12h·w96). winner's curse 해소 → 15m 확정(D-028). 신규 tf_confirm+단위14·seam2 | (미커밋) |
| 2026-07-15 | R4.2 MTF 역할1 + R4 종착 | `forward_fill_completed`→loader 승격(D-030)·`features/mtf`·`tf_mtf`·`run_tf_gate1` X주입. **mtf_1h +0.0036 material·CONFIRM**·4h 중복 → **15m+1h 채택**(D-029). fresh-eyes(CRIT/HIGH 0, F-4가드 수정·F-14~17 등록). **317 통과**. §12-6·§5·D-028~030 | c9ff96c |
| 2026-07-15 | Phase4 Step4.0 OOS 박제 | `experiments/oos_export`: 채택 15m+1h를 gate1 동일컴포넌트로 5seed walk-forward → OOS proba 박제(D-a). **재현 ll-margin +0.0486=문서 일치**. barrier_frac·mtf1h_kf_slope 파생. 아티팩트 189,734행 | (미커밋) |
| 2026-07-15 | Phase4 Step4.1 플러그인 | `plugins/dumb_l3`(엔진 수정0): 박제조회→θ방향→실진입가 배리어(D-b①)→고정notional→N봉 만기. 단위8+seam4 커밋테스트. 스모크·회귀 | (미커밋) |
| 2026-07-15 | Phase4 Step4.2 seam + D-031 | seam 정합(Σpnl==equity·라벨매칭 97/99%) + **I-004 발견**(15m master 낙관편향, 1m 필수) + **D-031**(엔진 O(n²)→O(log n), 등가성회귀). 전체OOS 1m ~39h→~8분 | (미커밋) |
| 2026-07-16 | Phase4 Step4.3 관문2 채점 | baseline 전체OOS 1m: **gross+126%/net−501%/수수료 5×gross**. 정합성 전수검증(독립재계산 0오차)+독립감사(버그0). 사전등록 8-config 스윕 **채택0**. → **관문2 FAIL-on-cost**(D-033) | (미커밋) |
| 2026-07-16 | Phase4 종착: 보고서 통합 | 교차-Phase 전파(규칙18): §12-7·D-031~033·I-004·F-3/11 해소·Phase5 로드맵. dumb_l3 비활성 섹션(default.yaml). INFRA·CLAUDE 갱신 | (미커밋) |
| 2026-07 | Phase5-A 홀딩 5셀 | dumb_l3 no_tp·no_timeout·allow_reverse 파라미터+테스트. 5셀 사다리(전체OOS 1m): 정책은 6h 경제성 못 살림(best −315%, no_tp가 gross 붕괴). D-034 | (미커밋) |
| 2026-07 | Phase5 지평 프론티어 F0~F4 | oos_export 일반화(export_frontier·build_frontier_xy·등가성 seam테스트)·dumb_l3 decision_tf/conviction. 분포39→gate1 13셀→박제→관문2. **gate1 프록시 경제성 오판**·경제홈=4h. 유일양성 4h/{3,4,5일}/x3 | (미커밋) |
| 2026-07 | Phase5 4h 스캔·펀딩·fresh-eyes | 4h 패밀리·1h 스캔(제외분 재검증)·Binance 실펀딩(OKX 3개월한도)·독립 fresh-eyes. **4h 양성=2024 아티팩트·비정상**(코드결함0) → 관문2 FAIL 유효(D-035). config 원장 CARRY/PARK(D-036, kill0). I-005·006. **338 통과** | (미커밋) |
| 2026-07-25~26 | Phase6 Step 6.0~6.2 | 정책 하네스(`policy_eval`)·**causal 국면**(`causal_tag_regimes`, **I-005 해소**)·`dumb_l3` 게이트·`adaptive_l3` 청산레버. 6.1 국면필터 기각·6.2 청산재설계 전멸 + **I-007 미래누수 적발·수정**(fresh-eyes). 15m 14런 전멸. **357 통과** | 5fbded0 |
| 2026-07-27~28 | Phase6 Step 6.3 EconL3 | **★θ가 방향 아닌 "움직임"을 쟀다★**(corr 0.05) → **EV 게이트**(D-038) + 로버스트 선별. 9셀 채택0. **★만기 인구가 킬러★**(해소 +23.6/거래 vs 만기 −46.1) → 6.3b **만기게이트** 3셀. 4h_4d +147.7% vs 4h_5d −128.4% **부호반전 발견** | (미커밋) |
| 2026-07-28~29 | Phase6 (라) N-트랜치 | **엔진 단일슬롯→N-트랜치**(D-037, `max_slots` 기본1·중량등가성 2회) + `equity_curve_mtm`·adaptive 상태격리·N불변 bp 지표·점유·MDD·슬리피지허용치. **부호반전 소멸**(+39.3/+12.3bp)·**교집합 검정 체결가 최대차 0.0**. I-008~011 적발·수정(fresh-eyes) | (미커밋) |
| 2026-07-29~30 | Phase6 포화 재측정·종착 | 6.1/6.2 기각셀 20 + 앙상블 3 + 15m 청산 6 = **총 93셀**. **ex-2024 양수 = D2/D7 뿐**·**θ knife-edge=표집아티팩트**·**알파 양방향 실재 vs 베타 상쇄**·**방향편향 실측**(예측 up 77.7% vs 라벨 32.7%). 통짜 통합회귀(N>1 6불변식)·델타 fresh-eyes(I-012·013)·**395 통과** | (미커밋) |
| 2026-07-31 | **Phase 7 착수 — 번호 개정·2차의견 반영** | **D-044 Phase 번호 개정**(구 6.5→7, 구 7→8, 전수 스윕 13곳). 2차의견 적대적 검토 수령 → **직접 재현 검증**: §2-2(15m EV게이트 롱 30.0%)·§1(표본/파라미터 4h 1.52) **정확 재현** / §2-3 재현 중 **I-014 발견**(알파/베타 모집단 불일치, 2022 알파 +59.1→0.0). **D-045**(층1 바 유지 + F-18 보정 CI·지평 패밀리 부호 일치). **★E-배치 → 2차의견 핵심가설(4h 편향 = 표본/파라미터 1.5) 양방향 기각★**: **E-a**(데이터·폴드·검증봉 완전고정, 파라미터만 축소) 비 1.5→22.1(=15m 수준)서 편향 77.7→**93.8%** **악화**(저용량일수록 "항상 롱"으로 붕괴 — A3 전 연도 92~95%). **E-b**(학습창 1→3yr) 무통제로는 개선처럼 보이나(77.7→73.4·적중 51.5→53.0) **공통 검증봉 교집합(n=7,500)에선 71.6→73.1%·적중 53.7→52.9 로 불변~미세악화** = 겉보기 개선은 **전부 폴드/구간 교란**. A0 가 얼린 아티팩트 정확 재현(파이프라인 정합). 편향은 라벨 한계분포로 설명 불가(균형 32.7/31.6)한데 저용량서 강해짐 → **피처↔라벨 관계 비대칭**을 가리킴 | (본 커밋) |

---

## 10. 잠재 이슈 트래커 (I-NNN)

| ID | 내용 | 발생 | 상태 |
|---|---|---|---|
| **I-001** | 1d 캔들 손상 — audit 는 **off-grid 5봉**(04-08·09·10·12·14 16:00)만 검출했으나 **실범위는 04-06~05-19 흩어진 14일 값손상**(on-grid 부분봉 = audit **미검출** 클래스, Phase3 R1서 전수스캔으로 확정). 1h clean. | Step 0.1(범위확정 Phase3 R1) | **해소(Phase3 R1 3.3 선행)**: `resample_ohlcv`(loader)로 감사-clean 1h→1d 파생(완전버킷만·00:00그리드), regime·MTF 공용. 파생 1d audit green, clean날 공식 1d와 정확 일치 검증 |
| **I-002** | `multiclass_log_loss` 가 비정렬 labels(`LABEL_CLASSES`=('up','down','expire'))에서 proba-클래스 오정렬 → sklearn(≥1.x) 사전순 정렬 가정 위배로 log_loss 오답(known-answer 0.223 정답 vs 버그 2.303). Phase0 잠복(정렬 라벨 테스트라 미발현), R1 첫 성과 채점서 발각. 첫 R1 DISCARD 원인. | Phase3 R1 | **해소**: labels 정렬 후 proba 정렬(값 순서불변). 회귀(비정렬+known-answer). 규칙19 적용 — blast radius=R1 ledger만(삭제·재생성), 라벨/R0/Phase0-2 무오염(분포전용·uniform ln3 순서불변). 수정 후 재실행 관문1 PASS |
| **I-004** | (Phase4 Step4.2) 백테 엔진은 봉당 **SL/TP검사→진입** 순서라 **15m master는 진입봉 자신의 15분간 청산검사 불가**(intrabar stop-out 불가시) → **낙관 편향**. 실측 spread: 2021H1 15m master +$133 vs 1m master −$3100(부호반전). `sl_first`는 드문 동시터치만 처리, 15m 보수화 못함. | Phase3~4 seam | **해소(Phase4)**: **관문2 판정은 1m 체결해상 필수**(15m 프록시 불가). D-031(엔진 O(n²)수정)으로 전체OOS 1m 실행가능화(~8분) → **전체OOS 1m으로 채점**. 커밋 seam 테스트로 진입타이밍·정합 박제 |
| **I-005** | (Phase5 fresh-eyes) 국면 태그 `tag_regimes` **vol축 = ex-post 전구간 quantile 문턱**(non-causal·분석전용). 트렌드축은 causal(trailing MA slope+forward 히스테리시스, 단 min_duration=14일 지연). **아티팩트 `regime` 열을 causal 매매필터로 그대로 쓰면 lookahead 주입.** | Phase5 프론티어 | **감사 완료**: 선별·경제성·결정은 전부 regime-독립(오염0, `model.fit`·summary 지표 regime무관). **해소(Phase6.0)**: `causal_tag_regimes` 신설 — vol 축을 전구간 quantile → **rolling 1yr(365일봉) 백분위**로 교체(`CAUSAL_VOL_WINDOW_BARS=365`, 사전등록·무튜닝), 트렌드/vol **분리 반환**. `assert_causal`(절단불변+미래교란) 통과·비인과 대조 박제. 아티팩트에 `causal_regime_*` 열로만 증강(예측 불변). |
| **I-006** | (Phase5 fresh-eyes) **엣지 검정 방법론 결함** — 초기 베타/트렌드 반증이 **전체기간 base rate만** 사용해 **시간불안정성을 못 봄**(4h가 2024 아티팩트인데 "진짜 알파"로 오판). | Phase5 프론티어 | **교훈 박제 + Phase6 표준화**: ex-2024 를 **1급 판정축**으로 승격(전 셀 계측·원장 열). Phase6 서 **N=1 측정이 ex-2024 를 체계적 과대**(5셀 중 4셀 부호반전)함이 드러나 → **포화 측정**이 함께 요구됨(§12-9). |
| **I-007** | (Phase6.2) `adaptive_l3.should_force_exit` 가 1m fill 하 **매 1m 호출**되며 `df.index[-1]` 로 **진행 중 결정TF봉**의 예측을 조회 → 최대 3h59m **미래 누수**. E1 signal_decay 결과 전부 무효(5d net 157.5%→23.2%·2025 +2055→−4506). | Phase6.2 fresh-eyes | **해소**: 완성봉 가드(`idx+interval<=now`). 회귀 박제. **교훈: 결정TF 경계 밖에서 예측 조회하는 모든 경로는 완성봉 가드 필수**(generate_signal 은 경계 게이팅으로 우연히 안전했음). |
| **I-008** | (Phase6.4) `_apply_funding` 이 펀딩을 **거래 노셔널이 아닌 초기자본 고정**으로 부과. 고정사이징(notional==init)에선 우연히 일치했으나 **N-트랜치(notional=init/N)에서 정확히 N배 과대**·conviction 사이징에서 과소. | Phase6.4 | **해소**: `size×entry_price` 기준. 회귀 박제. **규칙19 blast radius**: N-트랜치 셀 D1~D6 + B3_conviction **전량 삭제·재생성**(고정사이징 53셀은 무영향 확인). fresh-eyes 독립 재계산으로 post-fix 확인. |
| **I-009** | (Phase6.4) `_occupancy` 가 동시각(청산=진입)에 **진입을 먼저** 세어 `max_concurrent` 가 용량+1 허수 산출(F1 75 > 용량 72). 손익 무영향·진단지표만 오염. | Phase6.4 fresh-eyes | **해소**: **청산 우선 정렬**(엔진의 청산→진입 순서와 일치). 회귀 2 박제(순차거래 동시성 1·실중첩 2). |
| **I-010** | (Phase6.4) bp 지표(`*_bp_per_trade`·슬리피지 허용치·연도별 bp)를 **선언 notional** 로 정규화 → conviction 사이징에서 실노셔널이 최대 3배 커져 **bp 과대**. | Phase6.4 fresh-eyes | **해소**: **실현 노셔널 합**(`Σ size×entry_price`) 기준. S_B3 bp 11% 하향 교정(과대 제거 방향). 회귀 박제. |
| **I-011** | (Phase6.4) reverse 경로가 청산된 포지션 객체로 루프를 계속 → **죽은 포지션 기준 판정·이중 진입** 가능(구버전은 None 참조로 예외). | Phase6.4 fresh-eyes | **해소**: 청산 후 무조건 종료. 전 러너 `allow_reverse=False` 라 **커밋 수치 무영향**(dead code)이며 가드는 순개선. |
| **I-012** | (Phase6.4) `_analyze` 의 gross 역산식 `pnl+fee+funding` 이 `FeeModel`(`net = gross − fees + funding`)의 역이 아님(정답 `pnl+fee−funding`). 엔진이 `funding_fee=0` 만 넘겨 **현재 무영향**(전 trades CSV 확인). | Phase6.4 fresh-eyes | **미해결(잠복)**: **엔진 레벨 펀딩을 배선하는 순간 실화**(방향=deflate). **Phase8 realistic execution 착수 전 수정 필수 — 전제 게이트로 등록.** |
| **I-013** | (Phase6.4) 앙상블 슬리브 spec 에 `notional` 미지정 시 단일전략 규약(=init)이 적용돼 **슬리브당 전액 배정** → `max_slots=N` 이면 최대 노출 N배(net_pct·MDD 그만큼 부풀림). | Phase6.4 fresh-eyes | **해소**: 미지정 시 하드페일. 회귀 박제. 기존 9셀은 전부 명시 지정(`notional_per_trade × max_slots == init` 확인). |
| **I-014** | (Phase7 착수, 2차의견 §2-3 재현 중 발견) **§12-9 알파/베타 분해가 모집단 불일치** — 전략은 **게이트 통과 거래**인데 벤치는 **전체 봉의 무배리어 N봉 표류**다. 벤치가 ① 배리어를 안 쓰고(폭락장 손절이 "타이밍 실력"으로 계상) ② 모집단이 다르다(게이트 선택 효과가 알파로 계상). 2차의견은 ①만 고쳤고 ②는 남겼다. **모집단·배리어 둘 다 정합**(D2 게이트 통과봉 n=2,087, 같은 봉·같은 배리어폭·같은 만기)으로 재계산 시: 전체 알파 **+41.2 → +6.4bp**, **2022 알파 +59.1 → 0.0**(게이트 통과봉에서 **롱 100.0%·숏 0건** — 무조건롱 벤치와 행동이 동일하므로 방향 알파는 정의상 0). 오염 서술 = "**알파는 양방향 실재**"(§12-9)와 그에 근거한 **Phase 8 베타 중립화 배정**. | Phase7 착수 | **미해결**: 규칙19 blast radius = §12-9 알파표·헤더 8행 서술·Phase 8 베타 중립화 근거. **Phase 7 Step B 에서 배리어·모집단 정합 벤치로 전면 재계산** 후 문서 정정. 재계산 전까지 알파 수치 인용 금지. 봉수준 프록시 한계는 있으나 "2022 롱 100% → 방향 알파 0"은 추정이 아닌 산술. |

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

## 12-5. Phase 3 R3 결과·검증 요약 (완료)

**성격**: R1·R2(약엣지·창 무gain) 위에서 **모델**을 튜닝해 엣지가 강해지나. 창 고정(R2 기본창).

**구축**: `models/mlp` 멀티태스크 헤드(reach·λ·**expire 마스킹**·reach=None=R1 바이트동일, R3.0) + `experiments/r3_multitask`(R3.1) + `experiments/r3_arch_kf`(R3.2). 러너 2종 커밋(박제).

**R3.1 멀티태스크·정규화**(5판, 사전등록 예산 12): 단일태스크/z(=R1) 1.0673 기준.
- multitask λ0.1 +0.0008 · λ0.3 +0.0006 · single/robust +0.0006 · multi/robust +0.0009 → **전부 무gain**(기준선이 최선). **F-6 최종 = z-score**(robust 무gain, D-015). 멀티태스크 co-training 배리어 예측 무개선.

**R3.2 구조·KF**(8판 사전등록): base d2w32(=R1) 1.0673 기준.
- 구조 **용량↑ 단조악화**: d2w64 +0.0008 < d3w32 +0.0015 < d3w64 +0.0025 (약신호 과적합, "얕고 좁게" 실증). KF dof·r 4개 **±0.0004 노이즈**. 최소(KF dof2 −0.0002)는 노이즈의 1/10 → 미채택.

**결론**(D-027): **1h 엣지 튜닝 저항적 확정** — 창·멀티태스크·정규화·구조·KF **5지렛대 전수 무gain**. 약함은 실제 신호강도. **R1 config 유지**(단일태스크·z·d2w32·KF기본). 국면재확인 불요(개선 노이즈 미만). 멀티태스크 헤드 코드는 **보존**(Phase5 웜스타트). **291 테스트 통과**.

---

## 12-6. Phase 3 R4 결과 (완료 — 관문1 PASS, 15m+1h 채택)

**인프라**: `experiments/tf_expansion`(시간→봉 `bars`·TF스케일 `splitter_for`·`load_tf`·per-TF 분포게이트 `sweep_labels_for_tf`·`run_tf_gate1` 단독TF 관문1, `df`/`X` 주입·`_per_regime_edge` 국면진단) + `tf_confirm`(15m 확인 사전등록 규칙·`classify_regime_consistency`/`classify_label_robustness`/`combine`) + `features/mtf.build_mtf_features`(결정TF X + 상위TF 완성봉 concat, mtf{tf}_ 접두어, F-4 tail 가드) + `tf_mtf`(`MTF_MATERIAL_RULE`·`classify_mtf`·`run_mtf_role1`). `forward_fill_completed` → `data/loader.py` 승격(D-030).

### R4.1 단독 TF 관문1 (각 TF 라벨은 분포게이트로 1h 균형 근접 선정, 성과 안 봄 F-1)

| TF | 라벨 | 관문1 | ll-margin | MCC | BA | 일관성 |
|---|---|---|---|---|---|---|
| 1h | atr/w96/N24(1일) | PASS | +0.0284 | 0.128 | 0.413 | 0.91 |
| 4h | atr/w24/N24(4일) | PASS | +0.0151 | 0.097 | 0.390 | 0.73 |
| 1d | — | **불가**(데이터부족 2370봉, minMinor<100) | | | | |
| **15m** | atr/w24/N24(6h) | **PASS** | **+0.0450** | **0.149** | **0.424** | **1.00** |

**15m이 1h보다 전 지표 강함**(역할2). 트리 STRONG(226k봉)·MLP 22/22폴드. 4h 약함·1d 데이터한계. → 15m 채택(D-028).

### R4.1 15m 확인 (F-2 pick-and-confirm, 사전등록 규칙 `tf_confirm`)

- **축① 국면일관성 = CONFIRM**: 주라벨 6국면 **전부 양마진**(0.032~0.056)·**5seed 전부 Prior 우세**(reversal 0) — 1h(0.91)보다 깨끗. 킬규칙(D-006, seed과반 반전만 붕괴, 노이즈 허용).
- **축② 라벨강건성 = ROBUST**: 직교 대체라벨 12h/w24(N48) +0.0493·6h/w96 +0.0296 **둘 다 PASS+양마진**. 분포-유효 지평은 6h·12h뿐(1d↑ 방향극단 탈락).
- → winner's curse 대부분 해소, 15m 확정.

### R4.2 MTF 역할1 (상위맥락이 15m 엣지 강화하나, 사전등록 `tf_mtf`, baseline 15m단독 +0.0450, NOISE_FLOOR 0.002)

| config | 관문1 | ll-margin | Δ vs baseline | material | 국면 |
|---|---|---|---|---|---|
| **mtf_1h** | PASS | +0.0486 | **+0.0036** | ✅ | CONFIRM |
| mtf_4h | PASS | +0.0457 | +0.0007 | ❌(노이즈) | — |
| **mtf_1h4h** | PASS | +0.0491 | +0.0041 | ✅ | CONFIRM |

- **1h 상위맥락이 15m 엣지를 material·국면일관 강화**(+0.0036). **4h는 중복**(단독 노이즈·1h위 +0.0005 노이즈). 규칙상 best=1h+4h이나 4h 우위가 노이즈 → **parsimony로 15m+1h 채택**(D-029, 장식금지·R3.2 정합). 역할2(진입게이트)는 Phase4.
- **규모·범위 정직**: MTF 이득 소폭(~8% 상대)·실재·국면일관. **주라벨(6h/w24)에서만 측정**(라벨교차 미확인).

**R4 결론**: 관문1 PASS. **주 엣지 후보 = 15m + 1h 맥락**(16열). 검증 = **317 테스트 통과** + seam 통합(MTF full-stack) + fresh-eyes 재스캔(CRIT/HIGH 0, MEDIUM① F-4가드 수정, ②·LOW F-14~17 등록).

---

## 12-7. Phase 4 결과 (완료 — 관문2 FAIL-on-cost)

**인프라**: `experiments/oos_export`(채택 15m+1h OOS 예측 박제·barrier_frac·mtf1h_kf_slope 파생) + `plugins/dumb_l3`(멍청3층: 박제조회→θ방향→실진입가 배리어→고정notional→N봉 만기, **엔진 수정0**) + 엔진 `_slice_candles` O(n²)→O(log n)(D-031). 결정 D-032(배선)·D-033(판정).

### Step 4.0 OOS 예측 박제 (D-a)
- gate1과 **동일 컴포넌트**(build_mtf_features·splitter_for·ZScoreNormalizer·SmallMLP·5seed)로 walk-forward → 폴드별 OOS proba concat. **재현 검증: ll-margin +0.0486 = 문서 mtf_1h 일치**(오염0). 아티팩트 189,734행(2020-12~2026-05, 22폴드).

### Step 4.2 seam 정합 + I-004 + D-031
- **seam PASS**: 정합성(Σpnl==잔고==equity)·라벨매칭(15m 97.3%·1m 99.1%)·진입타이밍(예측봉 t→t+1 open, lookahead0). 커밋 회귀(seam 4·단위 8·등가성 1).
- **I-004**: 15m master는 intrabar stop-out 불가시=낙관편향(2021H1 +$133 vs 1m −$3100). → **1m 체결해상 필수.**
- **D-031**: 엔진 O(n²)→O(log n). 전체OOS 1m ~39h→~8분(동작보존 회귀).

### Step 4.3 관문2 채점 (baseline + 사전등록 스윕)
- **baseline**(전체OOS 1m·θ0.40·앙상블): 6,274거래 **gross +12,606(+126%) / 수수료 62,742 / net −50,135(−501%)**. **cost/|gross|=4.98**. 거래당 gross +$2.01 vs 고정비용 $10. 국면별 gross 4/6 양·net 6/6 음. 신호포착 16.9%(단일슬롯).
- **정합성 전수검증**: 사이징·수수료공식·**net==독립재계산gross−fees(0오차)**·항등식(gross%−fees%=net%). **독립 감사(fresh-eyes) 정확성버그0** — 모든 단순화(정확체결·슬리피지0·펀딩0)가 낙관편향 → 실제 더 나쁠뿐.
- **사전등록 스윕**(θ{.35,.40,.45,.50}×MTF게이트{off,on}, F-1): **채택 0**. net>0은 θ=0.50뿐(both)이나 **72/68거래·2/6국면 음 = winner's curse**로 탈락. MTF게이트(1h추세 부호) 무효(net/거래 개선 못함). breakeven θ≈0.47~0.50(표본 급감).
- **churn 진단**: 71.8% 같은방향 연속·즉시재진입 3,043 = **총수수료의 49%**($30k). 방향런 평균 3.6. → Phase5 홀딩 동기.

**판정(D-033)**: **관문2 FAIL-on-cost.** 통계엣지는 실재(gross 전 구간 양)하나 **거래당 엣지(0.02%) ≪ 왕복비용(0.1%)** → 유의표본 주는 어떤 빈도서도 net 음. **배율(레버리지)은 비율 불변이라 구제 불가**(fee·gross·net 다 notional 선형). 진입-지렛대(θ·MTF게이트·사이드아웃) 전멸.

---

## 12-8. Phase 5 결과 (완료 — 정책·지평 프론티어로도 관문2 FAIL 유효)

**인프라**: `dumb_l3` 파라미터화(no_tp·no_timeout·decision_tf·conviction 사이징) + `oos_export` 일반화(`export_frontier`·`build_frontier_xy`·repro_target/artifact_name/config_meta, **등가성 seam 테스트**로 채택 15m+1h/6h 재현 박제) + scratchpad 프론티어 러너(미커밋)·Binance 펀딩 다운로더. 결정 D-034~036, 이슈 I-005·006.

### Step A 홀딩 5셀 사다리 (정책은 6h 경제성 못 살림)
- baseline(A 재현 net −29,459=단독15m/6h) 대비 5셀(no_timeout/no_tp/reverse/full holding). **최선 = no_timeout(B) −315%**(gross↑+fee↓)이나 sign-flip 불가. **no_tp가 gross를 음으로 붕괴**(엣지=단기지평, 배리어 넘기면 반납 → "승자 태우기" 반증). churn 제거는 실재하나 불충분. → D-034.

### Step F 지평 프론티어 (지평×배리어×x, gate1→관문2)
- **F1 분포** 39셀 통과 → **(가) 13셀 gate1**(N×x 전부 × window 대표 w24). **gate1 ll-margin·MCC가 경제성을 예측 못 함**(4h/3일·4h/4일을 "약함"으로 오판, 재검증으로 잡음). 경제 홈 = **4h 넓은 배리어**(비용이 배리어의 1%).
- **관문2 baseline**(1m 체결): 15m·1h 전 config net 음(배리어 좁아 breakeven 승률 미달). **유일 양성 = 4h/{3,4,5일}/x3**(net +25/+99/+88%, cost/|gross| 0.18~0.36). 4h x확장(x4.5·6)은 전부 만기(신호 0).
- **실펀딩**(OKX 공개=3개월 한도 → **Binance 전체이력 프록시** 2020-11~, mean 0.010%/8h·85.9% 롱지불): 4h 영역 펀딩 후에도 net +2,106/+7,852/+5,573(펀딩 16~36% 침식).

### fresh-eyes 독립검증 → 4h 양성 = 2024 아티팩트 (D-035)
- **코드결함 0**(splitter purge·피처/배리어σ 인과·정규화 train-only·엔진 fill 낙관편향 없음·펀딩 이중계상 없음).
- **그러나 4h 양성은 표본기간 편향**: **2024 = net의 80%**, **ex-2024 ≈ 노이즈**(4h/4일 +1,948)~**음**(4h/5일 −3,730), **최근 1.5년(2025–26) 두 셀 음**(forward-OOS 소멸), 승률 42.5%↔70.5% **비정상**, 2021–23 ~97–100% 롱 = **롱편향 트렌드베타**.
- 초기 베타/트렌드 반증이 **전체기간 base rate만** 봐서 시간불안정을 놓침(방법론 결함, I-006).

**판정**: **관문2 FAIL-on-cost 유효.** 프론티어(지평·배리어·x·정책)는 **안정·배포가능 엣지를 못 찾음.** 4h 영역은 **2024 과적합·비정상 아티팩트**이지 검증 엣지 아님. **fresh-eyes가 과적합을 종착 전 차단(규칙 17 작동).**

### config 처분 원장 (D-036 — 완전 kill 0)
- **CARRY**(방향엣지 측정): 4h/{3,4,5일}/x3(4h는 비정상 리스크 명시)·15m/6h/x3·15m/12h/{x3,x4.5,x6}·1h/1일/x3 → Phase6서 정책 검증, **전부 시간안정성 바**(ex-2024+최근OOS held-out) 통과해야 승격.
- **PARK**(전기간 방향신호≈0): 15m/6h/x4.5·x6·1h/12h·1h/2일·4h/1일·2일·1h스캔6셀 → 휴면·소환가능(정책이 0엣지서 이익 못 만듦·과적합기계 방지). 원칙적 causal 가설 시 소환.
- **은밀폐기 0.** 능동세트 유계 = 삭제 아닌 과적합방지 우선순위. → **Phase6 에서 D-043 으로 연장**(§12-9).

---

## 12-9. Phase 6 결과 (완료 — layer-3 소진, 용량이 1차 지렛대로 판명)

**인프라**: `experiments/policy_eval`(정책 하네스 — 1m fill·실펀딩·연도별·**N 불변 bp 지표**·점유·MDD·슬리피지 허용치·손익 항등식 하드체크) + `validation/regime.causal_tag_regimes`(**I-005 해소**, vol 축 rolling 1yr) + `plugins/econ_l3`(EV 게이트·로버스트 선별) + `plugins/econ_l3_ensemble`(지평 슬리브 3) + `plugins/adaptive_l3`(청산 레버, trade_id 상태격리) + **엔진 N-트랜치 확장**(D-037). 결정 D-037~043, 이슈 I-007~013, 플래그 F-18·19. **총 93셀 실행**(원장 `data/research/phase6_results/_ledger_phase6.csv`, gitignored).
※ **3층 형태의 정본은 이 절이다** — 설계 초안(`docs/BTCUSDTperp_QuantModel_Design_vDraft.md` L243 "3층 형태 미정")은 2026-07-11 브레인스토밍 보존물로 미갱신.

### 6.1~6.2 레버 사다리 (국면필터·conviction·청산재설계·홀딩) — 채택 0
- 4h 국면필터: V1 트렌드게이트가 2025 를 **악화**(−2818→−3386). V2(+고변동회피)는 엣지 파괴 → **고변동 구간에 엣지가 산다**.
- 15m 계열 14런: conviction θ상향이 churn 극적 삭감(−244%→−43%)이나 여전히 음.
- 청산 재설계 E1/E2/E4 전멸. **I-007 미래누수 적발**(fresh-eyes) — 수정 후 이득 전부 소멸(누수가 이득의 전부였음).

### 6.3 from-scratch 3층 (EconL3) — 진단 2건
- **★θ는 방향이 아니라 "움직임"을 쟀다★**: corr(p_dir, 방향마진) = **15m 0.05·0.06(무관)** / 4h 0.55·0.65. 3-클래스에서 `max(p_up,p_down)` 이 `1−p_expire` 를 추종 → **conv045/050 실패의 원인**. → **EV 게이트** 도출(D-038).
- **★만기 인구가 킬러★**: EV 게이트 거래의 해소분은 **+23.6/거래 흑자**인데 **만기 47.7%·−46.1/거래**가 전부 상쇄(gross/거래 +$0.32). 원인 = `EV=w×m` 이 **넓은 배리어를 보상** → 평균 w 184→553bp 상승 → N봉 내 해소율 하락 → 만기 급증. 공식이 "만기 payoff≈0"을 가정했으나 실제는 체계적 음수. → **만기 게이트** 도입(D-038).
- 로버스트 선별(시드 만장일치+지속성)은 **선별력 부족**(4h 55%·15m 64% 자동통과 — 5시드 상관 높음) → 채택 안 함.

### ★6.4 (라) N-트랜치 — 용량이 1차 지렛대★
- **문제**: 단일슬롯이 게이트 통과봉의 **10~15%만 포착**. 어느 신호를 잡는지가 "그때 슬롯이 비었나"라는 우연으로 결정 → 형제 config(지평 4일 vs 5일)에서 **+147.7% vs −128.4% 부호 반전**.
- **결과**: 완전 포착(N=지평길이)에서 **부호 반전 소멸** — 4h_4d **+39.3bp** / 4h_5d **+12.3bp**(둘 다 양수). MDD 급감(4h_5d 70.6%→29.7%). **gross bp 가 포화에서 평탄화**(45.3→61.6→60.3) → **포화값이 모집단 값**.
- **★교집합 검정(결정적)★**: N=4 의 902거래가 N=24 에 **전부 포함**(A-only 0건), 공통거래 **진입가·청산가·청산사유·gross bp 최대차 0.0000000000**. 용량 확대는 기존 거래를 바꾸지 않고 **버려졌던 거래를 추가**할 뿐 → N 차이는 **100% 선택 효과**(엔진 산술 결함 소거).
- **기전**: 신호는 **뭉쳐서** 오고 그 클러스터가 추세 구간이다. 단일슬롯은 클러스터의 **첫 신호만** 잡고 나머지를 버렸다(TIM 은 44%→46% 로 거의 불변인데 포착률은 43%→82%).

### 6.4 포화 재측정 (6.1/6.2 기각셀 20 + 앙상블 3 + 15m 청산 6)
- 오프라인 스크린(성과 프록시 — **만기비중만큼 편향**: 4h 4~11bp 과소·15m 16~25bp 과대, 실측 4셀로 교정계수 확인)으로 트리아지 → 제외 16(무효/중복 6·15m 진입 8·4h 2) / 실행 20.
- **20셀 전기간 bp 대부분 양전**(용량이 실제로 가리고 있었음: A1 +8.4→+22.2·D1 +13.4→+25.6·E4b_5d −3.6→+6.6) **그러나 ex-2024 전부 음수**(최고 −4.7).
- **★θ knife-edge = 표집 아티팩트 확정★**: N=1 θ0.43/45/47/50 = +13.4/+48.0/+63.6/+59.9(단조 우상향) → 포화 +25.6/+27.0/+13.6/+16.4(**단조성 소멸**). 6.1 재료①("고신뢰 subset 이 시간안정") 은 운이었음 — **무튜닝 원칙이 결과적으로 옳았다.**
- **지평 앙상블 채택 안 함**(D-041). **15m 은 포화(100% 포착)에서도 gross −8.2/+4.4bp < 비용 10bp** — 청산 6셀도 gross≈0·거래 24k~35k 폭증. **용량 핑계 소멸**(단 DISCARD 아님, D-043).

### ★N=1 측정의 체계적 편향 (Phase 5·6 전체를 관통하는 설명)★
93셀 중 ex-2024 양수 8셀인데 **6셀이 N=1**. 포화 대응셀과 대조:

| N=1 셀 | ex-2024 | → 포화 대응셀 | ex-2024 | 변화 |
|---|---|---|---|---|
| C1 (EV+만기게이트) | +24.3 | D7 | **+10.8** | −13.5 |
| D2_5d_θ0.47 | +19.0 | S_D2_5d_theta047 | −26.6 | **−45.7 (반전)** |
| C2_5d_θ0.45 | +13.5 | S_C2_5d_theta045 | −9.6 | **−23.1 (반전)** |
| V1_trend_gate | +1.8 | S_V1_trend_gate | −9.5 | **−11.3 (반전)** |
| V0_baseline | +0.3 | S_V0_baseline | −13.3 | **−13.6 (반전)** |

**예외 없이 하락, 5중 4 부호반전.** Phase 5 부터 반복된 "유망 후보가 검증하면 무너지는" 현상의 **통합 설명** — 매번 다른 원인(과적합·2024 아티팩트·knife-edge)으로 진단했으나 근본은 **용량 제약이 만든 표집 편향** 하나였다.

### 판정 및 유일 생존 후보
- **층1(사전등록 바 = 연도별 전부 양수): 93셀 통과 0.** 2022·2025 가 전 셀 음수.
- **포화 조건 ex-2024 양수 = D2/D7 뿐**(= `4h_4d × EV게이트 k2 × 만기게이트`, N 만 다름).

| 후보 | n | gross | net_af | **ex-2024** | MDD(mtm) | slipTol | win/be |
|---|---|---|---|---|---|---|---|
| **D2 (N=12) — BASELINE** | 1,707 | 61.6bp | +41.3bp | **+13.4bp** | 33.3% | 20.6bp | 56.4/50.1 |
| D7 (N=24, 포화) | 2,087 | 60.3bp | +39.3bp | +10.8bp | 28.6% | 19.7bp | 55.8/49.7 |

**MDD 25% 정규화**(k=25/MDD, 수익·낙폭 모두 k 에 선형): D2 전기간 **연 8.3%** / **ex-2024 연 2.6%**. N=12 가 수익/MDD 최선(1.76 vs N=24 의 1.20 — N=24 는 평균 노출이 계좌의 12%뿐이라 자본 활용도 낮음).
**baseline 경고 4**: ① 형제 지평 부호 미유지(3일 −1.8 / 4일 +10.8 / 5일 −14.5 → **4일 특화 의심**) ② 베타 노출(87% 롱) ③ 유효표본 과대(F-18) ④ 슬리피지 여유 1차근사(F-19, ex-2024 기준 5.4bp).

### ★알파 vs 베타 분해★
후보가 87% 롱이라 "롱 베타 아니냐"는 의심 → **데이터가 반박**. 전략 롱 vs **같은 기간·같은 보유시간 시장 표류**:

| 연도 | 전략 롱 gross | 시장 표류 | **알파** |
|---|---|---|---|
| 2021 | +75.7bp | +53.2 | +22.5 |
| **2022(하락장)** | **−2.1bp** | **−61.3** | **+59.1** |
| 2023 | +107.4 | +79.4 | +28.0 |
| 2024 | +161.0 | +66.0 | +95.1 |
| 2025 | +29.2 | +0.5 | +28.7 |
| 전체 | +67.6 | +26.4 | **+41.2** |

숏도 +21.0bp(역풍 −26.4 기대 대비 알파 ≈+47, n=219). **알파는 양방향 실재.** **다만 베타 노출이 알파를 상쇄한다** — 2022 는 알파 +59 + 베타 −61 = −2 로 본전, 수수료로 적자. → **베타 중립화**가 후속 후보 지렛대(순노출 헤지, 수수료 2배 비용 검증 필요).

### ★금지 가정 "방향편향" 실측 발견 (규칙 15 고정 5문 ②)★

| | 실제 라벨 | 모델 예측 | 방향 적중률 |
|---|---|---|---|
| 전체 | up 32.7% / down 31.6% (**균형**) | up **77.7%** / down 22.3% | 51.5% |
| 2022(하락장) | up 27.2% / **down 34.7%** | up **91.5%** / down 8.5% | 46.2% |
| 2024 | up 36.8 / down 30.0 | up **54.6%**(균형) | **61.1%** |

- **학습창 base rate 로는 설명 불가** — 상관 **−0.241(부호 반대)**. 하락이 많았던 해에 **오히려 더** 롱으로 누움.
- **시드 5개 전부 롱 편향**(63.9~79.9%) → 초기화 우연 아닌 **학습 절차 전반의 성질**.
- **|예측편향| vs 방향적중률 상관 = −0.739** → **편향은 정보가 아니라 정보 부재의 증상**.
- 원인 후보(미검증): ① **피처의 상승 비대칭**(상승=완만·지속 / 하락=급격·단기인데 1층은 시간압축기라 지속만 포착) ② **정규화 분포 이동**(train-only ZScore, 레짐 급변 시 test 피처가 학습 분포 밖).

### config 처분 원장 갱신 (D-042·D-043)
**완전 kill 0 유지 — DISCARD 없음.** layer-3 기각은 "그 예측 위에서"의 판정이며 다음 Phase 가 0~2층을 바꾸므로 무효화될 수 있다. **재소환에 필요한 크기가 정량 확인된다**:

| config | 비용 | 엔진 gross | 현 적중률 | 방향항 | **만기항** | **BE 적중률** | 2배마진 |
|---|---|---|---|---|---|---|---|
| 15m_6h_x3 | 10.8bp | +4.4 | 58.3% | +26.2 | **−21.7** | **60.4%** | 63.8% |
| 15m_12h_x4.5 | 11.5bp | −8.2 | 55.3% | +19.3 | **−27.5** | **60.8%** | 64.0% |
| 4h_4d_x3 (BASELINE) | 22.0bp | +60.3 | 58.5% | +61.5 | **−1.2** | 53.2% | 56.3% |

(엔진 gross = 방향항 `w̄·해소율·(2a−1)` + 만기항, 만기항 상수 가정)
- **15m 재소환 조건 = 방향 적중률 ≥60.4~60.8%**(현재 55.3~58.3%, 필요 +2.1~5.5%p). **2024(편향 최소) 실측 61.1%** → **편향 교정만으로 도달 가능한 크기.**
- **4h 가 구조적으로 유리한 이유**: 만기항이 **−1.2bp**(15m 은 −21.7/−27.5). 배리어가 넓어 N봉 내 해소율이 높다(4h 65.4% vs 15m 41.7~48.0%).
- **BASELINE**: `4h_4d_x3 × EV게이트(k=2) × 만기게이트 × N=12`. **PARK**: 15m 전 계열·4h_3d·4h_5d·1h/1일·기존 PARK. 앙상블은 채택 안 함이나 코드 보존.

**Phase 6 결론**: layer-3 전 레버를 **포화 조건**에서 소진. 유일 생존 후보 D2 는 사전등록 바 미통과이나 **2024 비의존 양수로는 프로젝트 최초**. 검증 = **395 테스트 통과** + **통짜 통합 회귀**(N>1 경로 6불변식 박제: 손익항등식·포함관계·용량·인과·게이트실효·bp정규화) + **fresh-eyes 2회**(트랜치 엔진 clean / 델타 clean·잠재 LOW 4건 → I-012·013 등록).

---

## 다음 단계

**Phase 6 종착 — layer-3 소진, 용량이 1차 지렛대**(§12-9). 93셀 포화 재측정 결과 유일 생존 후보는 **D2**(`4h_4d × EV게이트 × 만기게이트 × N=12`, **ex-2024 +13.4bp**)이나 사전등록 바(연도별 전부 양수)는 미통과.

**라이브 시도는 보류**(사용자 확정) — ex-2024 기준 **연 2.6%·슬리피지 여유 5.4bp** 로 얇고 그마저 1차 근사(F-19)이며, 라이브 인프라가 현재 제거 상태라 **재구축 비용을 엣지를 두껍게 만든 뒤 쓰는 편이 낫다**. Phase 6 산출은 **Phase 7 의 baseline 으로 이월**한다.

**결정 트리 (갱신)**: Phase6 정책이 시간안정 net 을 내나?
- **예** → Phase 8 현실검증 → 관문3
- **아니오** → **(가) 엣지 강화 후 재판정** ← **현재 선택 경로**
  0~2층 개선(아래) → **layer-3 재측정**(도구 완비·한계비용 낮음) → **D2 baseline 대비** 판정
  → 그래도 안 되면 **(나) 새 엣지 방향**(다른 패밀리)

※ 원 결정트리의 "아니오 → 패밀리 사망 확정"은 **(가)를 신설해 대체**한다. 0~2층을 바꾸면 3층 판정이 무효화될 수 있으므로 패밀리 사망은 (가) 실패 후에만 판단한다(D-043 정합).
※ **중단 규칙은 사전등록하지 않는다**(사용자 확정) — 새 insight 발생 시 그때 판단. 다만 수익이 얇아지거나 반복이 소득 없이 길어지면 **관찰로 보고**한다(게이트 아닌 정보).

**Phase 7 = layer 0~2 개선** (Phase 6 진단 기반 우선순위. 착수 시 2차의견·재현검증으로 순위 조정 — §12-10 예정)
1. **★폴드별 예측 편향 모니터★** — 예측 up 비율 vs 라벨 base rate 대조. **현재 아예 계측하지 않아 5년간 못 봤다.** 비용 최저·정보량 최대.
2. **★피처 대칭성 감사★** — 각 피처를 상승/하락 국면별 분포 비교("상승 비대칭" 가설 직접 검증).
3. **헤드 분리** `P(도달) × P(위|도달)` + 클래스 가중 균등화 — **만기가 확률질량 35~39%를 먹어 방향 확률을 구조적으로 압축**(15m 에선 p_dir 이 방향과 무상관 0.05).
4. **캘리브레이션 보정**(walk-forward isotonic/Platt) — 예측 마진이 실현 엣지를 **1.4~9배 과대**(Q5 예측 110.9bp vs 실현 25.2bp).
5. **레짐 강건 정규화**(rolling rank 등) — train-only ZScore 의 분포 이동.
6. **파생 데이터 피처**(펀딩률·OI·베이시스·청산) — 현재 입력이 **OHLCV 뿐**. 2층은 벽(D-027)이므로 병목은 **정보량**. ※ 펀딩은 **Phase8=비용 1급 / 여기=피처 후보**로 층위가 다르다.
7. **타깃을 실현 손익 회귀로** — 3층의 EV 근사(만기 payoff=0 가정)가 실제로 틀렸음(§12-9).
8. **멀티심볼 pooled 학습** — 표본 확대(Phase8 의 교차강건 **검증**과 구분). 전제 충족: ETH/SOL/XRP/DOGE 4h 캔들 보유(§0). **Phase 7 E-배치의 표본 가설 처방(E-c)과 동일 축.**

**판정 기준**: 개선판이 **D2 baseline**(ex-2024 +13.4bp·gross 61.6bp·MDD 33.3%)을 넘는가. PARK 셀 재소환은 §12-9 의 정량 조건(15m: 방향적중률 ≥60.4%)으로.
**강건성 추가 요건(D-045)**: D2 는 형제 지평 부호 미유지(3d −1.8 / 4d +10.8 / 5d −14.5) = 4d 특화 의심이 경고로 달려 있다 → 개선판은 **3d/4d/5d 부호 일치**를 함께 요구한다(판정 강화 방향).
**층1 바(연도별 전부 양수)는 유지**하되, **F-18 유효표본 보정 신뢰구간을 병기**한다(D-045) — 연도별 부호가 0 과 구분 가능한지를 먼저 재고 판단은 데이터가 하게 한다(기둥5).

**Phase 8 (현실·교차강건·관문3)** — 전제 갱신:
- realistic execution(슬리피지·**maker fill model**: SL=taker·비체결·adverse selection)·**멀티심볼(ETH/SOL) 교차강건**·실펀딩 1급·walk-forward 재검증 → **관문3 = 배포가능 net?**
- **[신규] 트랜치 실집행 설계** — 거래소는 순포지션 1개만 주므로 봇이 **트랜치 장부 보유·reduce-only 부분청산**(부분체결·재시작 복구·증거금 포함). D-037 의 하류 요구사항.
- **[신규] 슬리피지 검증이 1순위 게이트로 격상** — ex-2024 여유 5.4bp 이고 F-19 상 실제는 더 낮다. **판정을 뒤집을 수 있는 위치.**
- **[신규] 베타 중립화** 후보 지렛대 — 알파는 양방향 실재하나 순노출 87% 롱(§12-9). 수수료 2배 비용 검증 필요.
- **[전제] I-012 수정 선행 필수** — 엔진 레벨 펀딩 배선 시 gross 역산식 오류가 실화.

**방법론 교훈(박제)**: ① **스크린(gate1·dumb경제성)은 사형선고 아님** — 경제성은 정책의존, 최종 discard는 최적화3층. ② **엣지 검정은 전체기간 base rate 아닌 시간안정성·최근OOS·연도별 승률vs breakeven**(I-006). ③ **놀라운 양성일수록 fresh-eyes 독립검증**(규칙 17이 4h 과적합 차단·Phase6 서 I-007 누수와 "잡음감소" 오해석 2회 차단). ④ **용량 제약 하 측정은 판정을 왜곡한다** — 포착률이 낮으면 결과가 표집 운에 지배된다. **포화(100% 포착)에서 재측정**해야 모집단 값을 얻는다(N=1 ex-2024 는 체계적 과대, 5셀 중 4셀 부호반전). ⑤ **N 간 비교는 bp/거래로** — net% 는 총 거래노셔널이 달라 비교 불가(D-040). ⑥ **파라미터 knife-edge 는 표집 아티팩트일 수 있다** — θ 사다리의 단조 우상향이 포화에서 소멸. 문턱에 민감한 결과는 채택 전 **포화·재표집으로 재확인**.
