# PATH_E_TREND_FOLLOWING — 룰베이스 추세추종 전략 검증

작성 시점: 2026-06-23 (PATH_D Phase R 종착 → C6 분기 = 룰베이스 추세추종 결정)
선행: `PATH_D_PREDICTABILITY_SCAN_260623.md` (단기 방향 ML NO-GO 확정)
작업 브랜치: `path-c-lookahead-fix` (causal features 기반)

---

## 0. 문서 목적

PATH_D 에서 **단기 방향 ML(모멘텀·평균회귀 포함)은 비용 넘는 edge 없음**이 측정으로 확정됐다. C6 분기로 **룰베이스 추세추종**을 검증한다 — 방향을 *예측*하지 않고 추세에 *반응*하며, trailing 으로 평균 R 을 키워 낮은 승률을 비대칭 보상으로 극복하는 구조. Phase R 의 IC 측정으로 부정되지 않은(꼬리 수익이라 IC 가 못 보는) 유일 영역. BTC 에서 실제로 비용 넘는지 정식 백테로 검증한다.

---

## 1. 배경 (PATH_D 계승)

- 단기 방향 *예측* edge 없음 확정 (PATH_D §4.2/§4.4). 추세추종은 예측이 아니라 반응 → 별도 검증 가치.
- **추세추종이 다른 점**: ① 더 긴 TF(4h) — 단기 노이즈 회피 ② 예측 없이 돌파 반응 ③ trailing 으로 평균 R 확대 → 승률 30~40%로도 양의 기대값(꼬리 수익).
- 단 BTC 에서의 실증은 미검증 → 추측 금지(PATH_C/R 교훈), 정식 walkforward 로 검증.

---

## 2. 설계 결정 (확정, 2026-06-23)

| 항목 | 결정 | 근거 |
|---|---|---|
| 진입 룰 | Donchian 돌파 + ATR trailing | 전형적·단순·파라미터 적음(과최적화 저항) |
| entry_tf | 4h | 중기 추세, 노이즈/표본/fold 균형 |
| 검증 방식 | plugin + BacktestEngine walkforward | 라이브-백테 일관(설계원칙 4), FeeModel·RiskManager 그대로. 룰베이스라 구현 가벼움 |

## 3. 의존성 확인 (2026-06-23)
- ✅ `update_stop_loss` 훅 엔진 호출 확인(`engine_base.py:754`, BacktestEngine 동일 경로) → **ATR/Donchian trailing 을 plugin 훅으로 구현, 엔진 수정 0**.
- ✅ `should_force_exit` 훅도 호출됨(보조 청산 옵션).
- ⚠️ Donchian 지표 부재 → `compute_donchian`(rolling high/low) indicators.py 추가 필요(DRY).

## 4. 단계 (TF-1 ~ TF-4)

```
TF-1  compute_donchian + trend_donchian plugin 구현      ✅ 완료 (단위 13 + 회귀 527 pass)
TF-2  백테 (전체 + 연도별 성과 분해)                      ✅ 완료 — 1차 GO (PF 1.31, +120%, R 2.42)
TF-3  파라미터 민감도 스윕 (과최적화 배제)                ✅ 완료 — robust 확정 (14/14 조합 PF>1)
TF-4  go/no-go (PF>1·양수 연도·Calmar·robust)            ✅ GO 확정 (funding 최악 상한도 PF 1.18)
TF-5a edge 강화 코어 (exit·long-only·레짐필터)           ✅ 종착 — 채택 long_only+chop50 (PF 1.31→1.86)
TF-5b cross-TF 추세 정렬 (1d)                            ⏸ 보류 (chop 중복 위험 → 라이브 후 재고)
TF-5c 고급 요소 (피라미딩/부분청산/chandelier/풀백)       ⬜ 선택적 (생략 가능)
LIVE  정식화 → paper 정합성 → 소액 라이브(risk 1%)        🔄 진행 — 남은 작업 §8.4 참조
```

> **새 세션은 §8(현재 상태 + 남은 작업)부터 읽으면 진입점 파악 가능.**

### 4.1 TF-2 백테 결과 (2026-06-23, 사용자 수행)

기간 2020-01-01~2026-06-01, 4h, 비용=taker 0.05%(백테 funding=0·slippage=0).

**전체**: +120.93%($12,093), 441거래, **승률 35.15%, PF 1.31, 평균 R 2.42**(avgW 328/avgL 136), MDD 14.89%. 청산 거의 전부 trailing SL(sl_hit 440/441, TP 청산 0 → TP 비활성 설계대로).

**연도별 분해**:
| 연도 | 거래 | 승률% | pnl($) | PF | 평균R | 비고 |
|---|---|---|---|---|---|---|
| 2020 | 62 | 43.5 | +3940 | 2.33 | 3.01 | 강세장 |
| 2021 | 74 | 32.4 | −347 | 0.93 | 1.94 | 거의 본전 |
| 2022 | 66 | 37.9 | +251 | 1.06 | 1.73 | **약세장인데 양수**(short +1074) |
| 2023 | 69 | 27.5 | +5277 | 1.73 | 4.56 | 큰 추세 |
| 2024 | 63 | 46.0 | +4685 | 1.81 | 2.12 | |
| 2025 | 76 | 27.6 | −1323 | 0.86 | 2.25 | 최대 손실년 |
| 2026 | 31 | 32.3 | −390 | 0.89 | 1.88 | 부분년 |

- **양수 4/7년**. 음수년도 대부분 소폭(2021·2026), 2025 −13% 부담.
- **방향**: long PF 1.61(+12,796) 주도, short PF 0.96(−703). BTC 상승편향 → short 거의 본전. **단 2022 약세장엔 short +1074로 방어**(양방향 작동 증거).
- **보유기간**: 평균 2.9일, **승리 5.2일 / 손실 1.7일** = 추세추종 정석(이익 길게·손실 짧게).
- **수익 집중**: 2023+2024 가 전체 pnl 의 82% → 추세추종 전형(소수 큰 추세 의존), 큰 추세 없는 해 부진.

**1차 판정 = GO(조건부)**: 비용 차감 후 PF>1·양수 기대값·견고한 비대칭 보상 구조 + 약세장 방어 → ML(NO-GO)과 명확 대비. 단 ⚠️ **백테 funding=0·slippage=0** — 추세추종은 보유 길어(평균 2.9일=funding ~9회) funding 영향이 ML(1.6h)보다 큼 → 실제 성과는 다소 낮을 수 있음(I-PE001). TF-3 민감도 + funding 점검으로 최종 확정.

### 4.2 TF-3 파라미터 민감도 결과 (2026-06-23)

`_tmp_tf3_param_sweep.py`(config 오버라이드 → BacktestEngine 14조합). 사이징·TP 고정, 타이밍만 스윕.

**핵심: 14/14 조합 전부 PF>1 (1.14~1.41), 전부 양수연도 ≥4/7, R 2.28~2.90.** → 과최적화 아님, robust 확정.

**1D 민감도** (나머지 baseline):
| 축 | 값 → PF | 해석 |
|---|---|---|
| entry_period | 10→1.19, 20→1.31, 30→1.28, 55→1.30 | 둔감·평탄 (진입 타이밍 robust) |
| exit_period | 5→1.25, 10→1.31, 20→1.35 | **길수록 PF·수익↑** (추세 더 오래 탐 = 비대칭 보상) |
| atr_sl_mult | 1.5→1.26, 2.0→1.31, 3.0→1.32 | 위험-수익 trade-off (작으면 수익↑MDD↑) |

**2D grid entry×exit PF** (atr=2.0): 전 셀 1.14~1.41. exit_period 최대 영향(길수록↑), entry 둔감. baseline 20/10 = 평탄 영역 중간점(특이점 아님).

→ **TF-3 결론: 추세추종 edge 는 파라미터에 robust = 실재(과최적화 아님).** baseline 20/10/2.0 유지하되, exit_period 확대(15~20)가 일관 우수 — 운영 시 검토 여지(과최적화 경계 주의). **남은 관문 = I-PE001 funding 점검**(낙관 편향 정량) → TF-4 최종 판정.

### 4.3 TF-4 funding 점검 + 최종 판정 (2026-06-23)

I-PE001 보수적 가정 근사(trades.csv 집계). 보유 평균 69.8h(8.4 funding회), 명목 평균 $6,426.

| rate/8h | funding합 | A 보수상한 PF(모두 비용) | B 방향반영 PF(long지불/short수취) |
|---|---|---|---|
| 0.01%(역사평균) | $2,347 | 1.25 | 1.30 |
| 0.015% | $3,520 | 1.21 | 1.30 |
| 0.02%(보수상한) | $4,693 | **1.18** | 1.29 |

- **최악 상한(모든 거래 funding 비용)에서도 PF 1.18 > 1.** B(현실)는 1.29~1.30 ≈ 원본(1.31) — **추세추종 양방향이라 short funding 수취가 long 비용 자연 헤지**.
- slippage(미반영, 추정 ~$567=수익 4.7%)도 작아 합산해도 PF>1 유지.

**TF-4 최종 = GO 확정**: PF>1(원본 1.31 / funding 최악 1.18) + robust(TF-3 14/14) + 양수 4/7년 + MDD 14.89% + funding·slippage 차감 후 edge 유지.

→ **PATH_E 결론: BTC 룰베이스 추세추종은 비용·funding 차감 후에도 양의 기대값 + 파라미터 robust. ML 단기 방향예측(PATH_D NO-GO)과 명확히 대비 — 실거래 후보 전략 확보.**

### 4.5 TF-5 edge 강화 — 로드맵·검증 프레임 (확정, 2026-06-23)
- **단계적·조건부**: TF-5a 코어 → (순개선 시) TF-5b → (선택) TF-5c. 각 단계가 walkforward 순개선 입증해야 다음.
- **멈춤 기준**: OOS 개선 미미 / robust·단순함 훼손 / 복잡도>가치 → 중단하고 라이브·자금관리로.
- **검증 = 연도별 walkforward** (단일 in-sample/OOS 2분할 기각): 추세추종은 소수 큰 추세 의존이라 단일 OOS(2.4년) 변동 큼 + 단계 반복 시 OOS 신선도 상실. 각 edge 요소를 baseline 에 더해 전기간 연도별 baseline 대비 개선 측정, **다수 연도 일관 개선 시 채택**(TF-3 robust 논리). 레짐 임계는 robust 스윕(평탄성). 최종 OOS=라이브.
- **격리**: baseline freeze, TF-5 는 새 plugin `trend_donchian_exp.py` + config 새 섹션 + (필요시) 새 indicator. paper 와 코드 독립.

### 4.6 TF-5a 1차 ablation 결과 (2026-06-23)

`_tmp_tf5_ablation.py` 17변형(exit/long_only/regime 6지표×임계). 결과: `data/tf5_ablation/`.
- **정합 ✅**: `baseline`(옵션 off) = baseline trend_donchian **441거래/PF 1.31/pos 4/7/mdd 14.9%** 정확 일치 (exp plugin 정확성 검증).

| 변형 | PF(dPF) | pos(dPos) | MDD(dMDD) | 판정 |
|---|---|---|---|---|
| long_only | **1.61**(+0.30) | 4/7(0) | **10.8**(−4.1) | ⭐ 채택 — short(PF0.96) 제거 구조적 개선 |
| chop50 | 1.49(+0.18) | **6/7**(+2) | 11.8(−3.1) | ⭐ 채택 — 횡보 회피, ret 171.8% |
| chop38 | **1.68**(+0.37) | 5/7(+1) | **8.0**(−6.9) | ⭐ (거래 197 적음) |
| ma200 | 1.46(+0.15) | 5/7(+1) | 11.9(−3.0) | ○ 후보 |
| er0.4 | 1.43(+0.12) | 5/7(+1) | 10.5(−4.4) | △ 임계 비단조(0.3 pos−1) = cherry-pick 의심 |
| exit20 | 1.35(+0.04) | 4/7 | 16.9(+2.0) | △ 약함, MDD↑ |
| adx/vol/di | ~0 | −1~0 | — | ✗ 기각(수익급감/해로움/무효) |

- **robust 판정**: chop 38/50/62 **단조·평탄**(낮을수록 강한 필터·강한 개선) → cherry-pick 아님. long_only 구조적. er 은 0.4만 좋아 비단조 → 신중.
- **결론**: whipsaw 회피(chop) + short 제거(long_only)가 추세추종 edge 실증 강화 — PF 1.31→1.5~1.6, MDD 15%→8~11%. → 조합 검증으로.

**조합 검증 (8변형, 2026-06-23)**:
| 변형 | PF(dPF) | pos | MDD | 거래 |
|---|---|---|---|---|
| **LO+chop50** | **1.86**(+0.55) | **6/7** | 10.8 | 205 |
| LO+chop38 | **2.22**(+0.91) | 5/7 | **7.9** | 111 |
| LO+ma200 | 1.82(+0.51) | 5/7 | 9.8 | 184 |

- **시너지 확인 ✅**: 조합 PF가 각 단독보다 높음(LO+chop50 1.86 > long_only 1.61·chop50 1.49) — short 제거+횡보 회피 독립 메커니즘 곱셈 효과.
- **채택 = `long_only + chop50`**: baseline 대비 **PF 1.31→1.86, 양수연도 4→6/7, MDD 14.9→10.8%**, 거래 205(연 32건 적정). LO+chop38은 PF 2.22지만 거래 111(연 17건 빈도 낮음·표본 약) → chop50 안전.
- → **TF-5a 종착: edge 강화 확정**. 다음 TF-5b(cross-TF) 조건부 진행.

### 4.7 BTC 국면 분석 — 라이브 타이밍 보조 (2026-06-24)

ZigZag 20% 추세 전환 분할(32 세그먼트, 1d 2026-06-23 갱신) + LO+chop50 국면별 성과. **탐색적·예측 아님**(확증편향 경계).

- **현재 국면(최근 60일)**: **하락 추세** — slope −0.33%/d, ATR 3.0%, **ADX 34**, ATH 대비 −50%.
- **핵심 발견 — LO+chop50 = 상승장 전략** (long_only 추세추종 본질):

| 국면 방향 | 국면 수 | 총거래 | 수익>0 국면 | 대표 |
|---|---|---|---|---|
| 상승(up) | 16 | **153** | **12/16** | 2022-11~2024-03 PF 3.75/+103% |
| 하락(down) | 16 | 52 | **3/16** | 대부분 PF<1, 손실 |

- **시사**: 현재 하락 국면 → **라이브 초기 부진 각오**. 보수 사이징(risk 1%)+인내+상승 전환 대기 정당화 — 자금관리 결론과 정합. (예외: 2025-10~2026-02 하락 국면은 반등 LONG 으로 PF 1.8/+9%, 현재와 최유사 #1).
- 1회용 `_tmp_regime_analysis.py`(commit 제외). 평균 PF 는 소수 거래 국면 극단값 영향 — 방향성(up 강/down 약)이 핵심.

---

## 5. 진행 기록

| 시점 | 단계 | 상태 | 커밋 | 비고 |
|---|---|---|---|---|
| 2026-06-23 | PATH_E 신설 + 설계 결정 | ✅ | §8.6 | Donchian 돌파+ATR trailing / 4h / plugin+BacktestEngine. 의존성 확인(trailing 훅 OK, donchian 추가 필요) |
| 2026-06-23 | TF-1 구현 | ✅ 완료 | §8.6 | `compute_donchian`(causal shift) + `trend_donchian` plugin(진입/초기SL/TP비활성/trailing 단조) + config 섹션. 함정 반영(trailing 단조 plugin 보장, PositionSide≠SignalSide). 단위 13건 신규, 회귀 514→**527 pass**. 엔진 수정 0 |
| 2026-06-23 | TF-2 백테 (사용자 수행) | ✅ 1차 GO | §8.6 | 2020~2026 +120.93%, 441거래, 승률 35%, **PF 1.31, 평균 R 2.42**, MDD 14.89%. 양수 4/7년, 약세장(2022) short 방어. 결과 §4.1. ML(NO-GO)과 대비 — 추세추종 비대칭 보상 작동. funding/slippage 미반영 낙관 편향(I-PE001) → TF-3 + funding 점검으로 확정 |
| 2026-06-23 | paper 검증 → I-PE002/003 fix | ✅ 완료 (e2e 검증) | §8.6 | paper 첫 기동에서 **시작 직후 진입** 포착 → I-PE002(진행중 봉 신호, 라이브-백테 불일치) + I-PE003(TP 음수→거래소 등록 위험) 발견·수정. `_build_ctx` 진행중봉 제외 + TP None 지원 + ENTRY 로그 TP None 처리. 단위 4 신규, 회귀 527→**529**. 백테 불변(441/PF1.31/TP None). **paper 재기동 e2e ✅**: 에러 없음, 마감 봉(62846.30 실제 Donchian 돌파) SHORT 정상 진입, `TP=None` 표기 — 라이브-백테 신호 동등성 확보 |
| 2026-06-23 | TF-5a 1차 ablation | ✅ 1차 결과 | §8.6 | `trend_donchian_exp` plugin(long_only+regime 6지표) + 단위 8 + 회귀 529→**537**. 17변형 ablation: 정합✓(baseline_exp=441/PF1.31). **채택유력: long_only(PF→1.61,MDD−4.1), chop(38/50 robust, chop50 pos 6/7·ret171.8%)**. 후보 ma200. 기각 adx/vol/di/er(cherry-pick). §4.6. 다음=조합 검증. 1회용 `_tmp_tf5_ablation.py` |
| 2026-06-23 | TF-5a 조합 종착 | ✅ 채택 | §8.6 | 조합 8변형: **시너지 확인**(LO+chop50 PF1.86 > long_only 1.61·chop50 1.49 단독). **채택 = long_only+chop50** (PF 1.31→1.86, pos 4→6/7, MDD 14.9→10.8). LO+chop38 PF2.22지만 거래 111(빈도 낮음·표본 약) → chop50 안전. TF-5a 종착. 다음 TF-5b cross-TF |
| 2026-06-23 | TF-5b cross-TF 방향 점검 | ⏸ 보류 | — | cross-TF(1d 정렬)는 chop 과 **중복 위험**(둘 다 추세 필터) + LO+chop50 이미 강함 → 멈춤 기준(복잡도>가치) 적용. **LO+chop50 정식 검증→라이브 우선**, cross-TF 는 라이브 누적 후 재고 |
| 2026-06-23 | LO+chop50 백테 재검증 | ✅ 라이브 후보 확정 | §8.6 | 비용·funding 보수 차감: 최악(2bp slip + funding 0.02% 전부 비용) **PF 1.64**, 연도별 **6/7 양수**(2022만 본전 −866). baseline 최악 PF 1.15 대비 전면 우수. 보유 평균 3.1일. → **라이브 후보 확정**. 다음: 정식화(paper 후)·자금관리 1차·정합성 도구 |
| 2026-06-24 | 자금관리 1차 + MDD 재확인 | ✅ 완료 | §8.6 | **MDD 함정 영향 작음**: 미실현/실현 MDD 배율 baseline 1.04x·LO+chop50 1.14x(보유 3.1일+trailing이 미실현 손실 제한). 기존 MDD 거의 유효, LO+chop50 실제 MDD **12.4%**(실현 10.8%). 연속 켈리 f*=14.5%→**1/4 Kelly 3.6%**. MDD-risk: ~25% 포화(risk 2.5%+), risk 3%가 효율 정점(MDD 23.6%/수익 415%), 4%는 과베팅. **권장: 라이브 초기 1~1.7%(보수) → 검증 후 점진 risk 3%(30% 감수 여력)**. 잠정값(라이브 후 재계산). I-PE004 등록. 1회용 `_tmp_tf5_sizing.py` |
| 2026-06-24 | BTC 국면 분석 (라이브 타이밍) | ✅ 완료 | §8.6 | ZigZag 20% 32국면(1d 06-23) + LO+chop50 성과. **현재=하락 국면**(ADX 34, ATH −50%). **LO+chop50 상승장 전략**: up 12/16 양수(거래 153) vs down 3/16(거래 52). 현재 라이브 초기 불리 → 보수+인내+상승전환 대기 정당화. 결과 §4.7. 1회용 `_tmp_regime_analysis.py` |
| 2026-06-23 | TF-3 파라미터 민감도 | ✅ robust 확정 | §8.6 | 14조합 스윕 **전부 PF>1(1.14~1.41), 양수연도≥4/7, R 2.28~2.90**. 20/10/2.0=평탄영역 중간점(cherry-pick 아님). exit_period 길수록 우수. 결과 §4.2. 1회용 `_tmp_tf3_param_sweep.py`. 디버그: initialize() 누락 fix. 남은 관문=I-PE001 funding |
| 2026-06-23 | TF-4 funding 점검 + 최종 | ✅ **GO 확정** | §8.6 | funding 보수 점검: 최악 상한(0.02%/8h 모두 비용)도 PF 1.18>1, 현실(방향반영) 1.29~1.30≈원본(양방향 헤지). 결과 §4.3. **PATH_E 결론: 추세추종 실거래 후보 확보 — ML(NO-GO)과 대비** |
| 2026-06-24 | MS 멀티 종목 검증 계획 수립 | 🔄 착수 | — | LO+chop50 종목 보편성 검증 (ETH/SOL/XRP/DOGE). 2단계 게이트(검증→탐색)+BTC robust 통제 그대로. 사전 함정 점검: **I-PE005(max_position_size_btc 수량 cap → 알트 백테 클램프) 발견**, ATR 변동성 흡수 ✅, 종목 동조성(0.7~0.9)으로 "보편" 신호 강등. 상세 §9. 다음=MS-1 데이터+cap 처리 |
| 2026-06-24 | MS-1 데이터+cap / MS-2 검증 | ✅ 완료 | (미커밋) | `download_history.py --symbol` 추가. 5종목 4h 정합 통과(SOL 2021-01-25/DOGE 2020-07-11 상장보정). **검증: cap 무력화 불변(BTC 441/1.31 재현), DOGE 레버리지 실측 0.205x(부풀리기 없음·5x cap 0% 바인딩), funding=0 확인.** MS-2: **LO+chop50 5/5 PF>1·baseline 개선** → 추세추종 보편 입증(동조성 한정). ma200 계열 강함(ret 보존)→MS-3 후보. 결과 §9.5. 1회용 `_tmp_ms_multisymbol.py` |
| 2026-06-25 | MS-3a/b 필터 robust 탐색 | ✅ 완료 | `f236de9` | chop 임계 평탄성 **5/5 단조**(cherry-pick 아님, c30 표본 과적합·c44~50 균형). 연도 walkforward: 공통 약점 2022/2026, **ma 계열 연도 일관 우수**(ETH/XRP 6/7). `trend_donchian_exp` ma_period 파라미터화(회귀 BTC LO+ma200 1.82 일치). ma 기간 스윕: **ma 가 chop 보다 임계 robust+ret 보존 우수**(chop 강필터 ret 붕괴, ma 평탄). 결과 §9.6. 다음=MS-4 chop50 vs ma 채택 재검토. 1회용 `_tmp_ms3_explore.py`/`_tmp_ms3b_filter_table.py` |
| 2026-06-26 | MS-4 종합 — 채택 결정 | ✅ 완료 | `1c979b2` | LO+chop50 vs LO+ma200 정면 비교: **ma 위험조정(MDD 5/5)·연도 일관·효율 우위, BTC 동급**. **결정 (가) LO+ma200 으로 정식화 후보 전환**(§7/§9.7). §8 라이브 로드맵 갱신(정식화·자금관리 ma 기준). 한계(동조성·DOGE 폭등의존·funding=0) 유지. chop50 대안 후보 보존. **MS Phase 종착** |
| 2026-06-28 | A1 paper 청산 e2e + 개선점 | ✅ 완료 | (미커밋) | **청산 e2e 완성**: trailing 단조하향→sl_hit @60759.90→net +$105.29, DB 정합(§8.3). 부수 발견: I-PE006(heartbeat 부재)/007(trailing SL DB 미기록)/008(paper restore dead code)/009(거래소 trailing SL 미갱신). **I-PE006 fix**(feed heartbeat 10분, e2e ✅ 10분 throttle 확인) + **I-PE007 fix**(store.update_trade_sl+live 감지) + **I-PE008 fix**(paper 재기동 balance/포지션 복원 — balance 리셋·dd 왜곡 해소). 단위 3 + 회귀 540. I-PE009(거래소 trailing SL 미갱신) 등록(라이브 전). 다음=paper 재기동 e2e(balance 복원·trailing DB)→A2 정합성 |

---

## 6. 잠재 이슈 트래커

| ID | 이슈 | 상태 |
|---|---|---|
| I-PE001 | 백테 funding=0·slippage=0 → 추세추종(평균 보유 2.9일=funding ~9회)은 funding 영향이 커 성과 낙관 편향 가능. 라이브에선 `_close_with_funding`로 반영되나 백테 미반영(I-BP001 carry) | **✅ 점검 완료(TF-4)** — 보수 가정 근사: 최악 상한(0.02%/8h 모두 비용) PF 1.18>1, 현실(방향반영) 1.29~1.30(양방향 헤지). GO 유지. 단 정밀 백테 통합(BLE-5: ccxt funding history)은 **라이브 전 권장**(carry) |
| I-PE002 | **paper 검증 중 발견 (2026-06-23)** — 라이브/paper 에서 trend_donchian 이 `ctx.candles.iloc[-1]`(진행 중 봉, ccxt single tick)으로 신호·trailing 판정 → 백테(마감 봉, `_slice_candles`)와 불일치. 시작 직후 SHORT 진입이 증거. candles 직접 쓰는 plugin 공통(example 포함). ML plugin 은 `get_features_for_ctx(ts<now)`로 보호받아 무영향 | **✅ fix** — `_build_ctx`(engine_base)에서 `LAST_CLOSED_BAR_IDX` 기준 진행 중 봉 제외(`n_drop = -1 - IDX`; 백테 0 무변경, 라이브 1 제외). 추상화 교정(모드 차이를 엔진이 흡수). 백테 결과 불변(441/PF 1.31). 단위 2건 |
| I-PE003 | I-PE002 분석 중 발견 — trend_donchian TP=`현재가±risk×RR(100)` 가 SHORT 에서 **음수 가격**(-97946). 라이브는 `place_take_profit`로 거래소(`takeProfitPrice`)에 실제 등록 → 음수 등록 시 거부·진입 실패 위험 | **✅ fix** — `compute_take_profit` 시그니처 `float\|None` 확장 + trend_donchian `→None`. 엔진: TP None 이면 거래소 등록 skip + check_candle_sl_tp skip. ENTRY 로그(`engine_base.py:557`) `TP=%.2f`→None 안전 처리(paper e2e 에서 발견). 추세추종 정석(TP 미설정, trailing SL 청산). 단위 2건 |
| I-PE004 | 자금관리 1차 중 발견 — 백테 `equity_curve` 가 실현 잔고만 추적(미실현 drawdown 미반영, `engine.py:445` "unrealized 미추적") → 백테 MDD 가 실제 경험 MDD 보다 과소 | 미해결(영향 작음) — **영향 정량(post-hoc 재구성)**: 미실현/실현 MDD 배율 baseline 1.04x·LO+chop50 1.14x. 추세추종 짧은 보유(3.1일)+trailing 으로 차이 작아 기존 MDD 거의 유효. 근본 fix(엔진 매 봉 미실현 equity 추적)는 2차, 현재는 post-hoc 보정으로 충분 |
| I-PE005 | **MS 계획 중 발견 (2026-06-24)** — `RiskManager.calculate_position_size`(`manager.py:214`)의 `size = min(raw_size, max_size_by_leverage, max_position_size_btc=1.0)` 가 **절대 수량 1.0 cap**. BTC(1.0≈$60k)는 raw_size 가 거의 안 걸려 무영향이나, 알트는 수량 의미가 달라(SOL 1.0≈$150, XRP 1.0≈$2, DOGE 1.0≈$0.2) **알트 백테 전 거래가 1.0 으로 클램프 → 결과 왜곡/무효**. cap 이 *수량* 기준이라 종목 간 비교 자체 불가 | **회피(백테)** — MS 백테 시 config `risk.max_position_size_btc` 무력화(1e12) 오버라이드 + BTC 재현으로 cap 미작동(결과 불변) 정합 확인. **근본(명목가 $ 기준 cap or 종목별 설정)은 멀티 종목 라이브 시 재설계 필요(carry)** — 단일 BTC 라이브엔 무영향 |
| I-PE006 | **paper 운영 점검 중 발견 (2026-06-26)** — `[POSITION]`/`[ACCOUNT]` 로그가 master_tf(4h) 봉 마감마다만 출력(`engine.py:1082`) → 사이 4시간 무로그, **구동 상태(살아있는지) 확인 불가**. WebSocket tick 은 수신하나 추세추종은 마감봉만 써 tick 로그 없음 | **✅ 구현 (2026-06-28)** — feed watch loop heartbeat(`feed.py`, `HEARTBEAT_INTERVAL_SEC=600` 10분 throttle, `time.monotonic`). tick 수신 시 `[HEARTBEAT] {tf} feed alive, last_price=…` 1줄. 연결 끊기면 heartbeat 도 멈춰 이상 신호. live 전용(백테 무관), 회귀 538. **paper 재기동 e2e 검증 대기** |
| I-PE007 | **paper 점검 중 발견 (2026-06-26)** — `update_stop_loss` 훅이 `position.stop_loss`(메모리)+거래소(`place_stop_loss`)만 갱신, **`trades.stop_loss` DB UPDATE 없음** → 청산 trade SL=초기값(궤적 유실). 라이브 재기동 시 엔진 SL 이 DB 초기값으로 롤백(거래소 conditional trailing 과 불일치). A2 라이브-백테 정합성·사후분석에서 trailing 추적 불가 | **✅ 구현 (2026-06-28)** — `store.update_trade_sl(trade_id, sl)` + live engine `check_strategy_exits` 전후 `position.stop_loss` 변화 감지 시 DB UPDATE(라이브 전용). 단위 `test_update_trade_sl_persists` + 회귀 538. **paper 재기동 e2e 대기**. (b) `sl_history` 이력·백테 trades.csv 최종 SL 은 A2 시 보강. **거래소 conditional 미갱신=I-PE009 별도** |
| I-PE008 | **복원 분석 중 발견 (2026-06-26)** — `PaperExecutor.restore_state`(`paper_executor.py:36`)가 **호출처 0 dead code** → paper 재기동 시 포지션 복원 안 됨. `_restore_state` 가 "거래소 없음+DB open" 분기(`engine.py:538`)로 **DB open trade 를 청산 처리**(fallback SL 추정가). 라이브는 거래소 `get_position` 복원되나 **paper 만 불일치** | **✅ 구현 (2026-06-28)** — `_restore_state` 에 paper 복원 블록: `not broker.is_live` 시 `get_last_balance`(equity 마지막)+open 포지션 → `executor.restore_state`(dead code 활성화). balance 리셋·dd 왜곡 해소 + 보유 포지션 오청산 방지. last_balance=None(첫 기동) 시 skip. live 무영향. 단위 2 + 회귀 540. **paper 재기동 e2e(balance 복원) 대기** |
| I-PE009 | **I-PE007 구현 중 발견 (2026-06-28)** — trailing SL 갱신 시 거래소 conditional order(`place_stop_loss`)는 **재등록(`engine.py:733`)에만** 호출, **trailing 갱신 후 거래소 SL 미갱신** → 거래소엔 초기 SL 잔존. 엔진 정상 시 메모리 SL 로 봉마감 청산(정상)이나, **엔진 다운 시 거래소 안전망이 초기 SL** → trailing 이익 손실 위험 | 미해결 — **라이브 전 필수**(paper 무관, 거래소 없음). trailing 갱신 시 거래소 SL amend(또는 cancel+재등록) 추가. 라이브 정식화 전 처리 |

신규 이슈는 I-PE001~ 형태로 등록.

---

## 7. 결정 사항 로그

| 시점 | 사안 | 결정 | 근거 |
|---|---|---|---|
| 2026-06-23 | 진입 룰/TF/검증 | Donchian 돌파+ATR trailing / 4h / plugin+BacktestEngine | 전형·단순·라이브백테 일관 |
| 2026-06-23 | 사이징 최적화 (향후) | edge·라이브 검증 후 fractional Kelly + 목표 MDD 제약 기반. `max_leverage` cap 유지(저변동 size 폭증 안전장치). 변동성타깃은 TF-5② 통합 | risk%는 edge 무관·**위험선호 결정**(백테 수익최대화=파산위험 금지). 현재 1% 룰(실배율~0.4배)은 라이브 전 보수 유지. max_lev 5는 baseline 상속값(trend_donchian 특화 아님) |
| 2026-06-24 | 라이브 시작 방향 | **(가) 지금 소액(risk 1%) 시작 + 사이징 국면연동** | 라이브 목적=정합성·인프라·실체결 검증(edge 실현 아님) + 추세추종은 마켓타이밍 안 함("언제든 가동, 추세 오면 탄다"). 현재 하락 국면 부진=예상(국면분석), risk 1%+하락장 진입 적어 손실 작음. 가동 유지로 상승 전환 포착. **상승 전환+라이브 edge 확인 후 점진 증액(→3%)**. (나) 대기는 전환 타이밍 포착 불가+검증 지연 |
| 2026-06-26 | 정식화 전략 (MS-4) | **(가) LO+chop50 → LO+ma200 전환** | 5종목 비교: ma 가 위험조정(MDD 5/5 낮음)·연도 일관(ETH/XRP 6/7)·거래효율(거래 적음) 우위, 절대 PF 약우위. BTC 동급(1.82 vs 1.86, MDD 9.8 vs 10.8)이라 전환 손해 없고 위험조정 이득. ma100~250 robust. 한계(동조성·DOGE 폭등의존·funding=0) 인지. chop50 은 대안 후보 보존. 결과 §9.7 |

---

## 8. 현재 상태 + 남은 작업 (★ 새 세션 진입점)

### 8.1 한 줄 요약
**추세추종 룰베이스 전략 라이브 후보 확정. MS 멀티 종목 검증(5종목, §9) 완료 → LO+필터 보편 입증 + MS-4 에서 정식화 후보를 LO+chop50 → `LO+ma200`(long_only+ma200 추세필터)로 전환(위험조정·연도일관·효율 우위, BTC 동급). 현재 baseline `trend_donchian` paper 가동 중(SHORT 보유, 청산 대기). 다음 = paper 청산 → 라이브-백테 정합성 → LO+ma200 정식화·자금관리 ma 재확인 → 소액 라이브(risk 1%).**

### 8.2 채택 전략·사이징 (MS-4 갱신 2026-06-26)
- **전략 = LO+ma200** (MS-4 채택, §9.7): `trend_donchian_exp` + `long_only=True` + `regime_filter_type=ma` + `ma_period=200`.
  - 5종목 위험조정(MDD 5/5 chop50 대비 낮음)·연도 일관·거래효율 우위, BTC 동급(PF 1.82 vs chop50 1.86, MDD 9.8 vs 10.8).
  - ma100~250 robust(평탄, cherry-pick 아님). ma_period 파라미터화 완료(회귀 537 pass).
- **이전 후보 LO+chop50** (PF 1.86/MDD 10.8): BTC 동급이라 대안 후보로 보존. 전환 손해 없음.
  - ⚠️ **아직 정식 아님** — 둘 다 실험 plugin `trend_donchian_exp` *옵션*. config 섹션은 ablation 기본값(옵션 off=baseline).
- **baseline `trend_donchian`** (TF-1~4 GO, PF 1.31): 정식 plugin/config 완비, paper 운영 중. 인프라(엔진·trailing·TP None·진행중봉 fix·cap)는 공유.
- **사이징**: 라이브 초기 `risk_per_trade_pct` **1%** → 검증 후 점진 **3%**(자금관리 1차, §170). `max_leverage 5` cap 유지. ⚠️ 자금관리는 LO+chop50 기준 — LO+ma200(MDD 더 낮음) 기준 재확인 필요(라이브 정식화 시). 단일 BTC 라이브엔 I-PE005(수량 cap) 무영향.
- **라이브 방향**: (가) 지금 소액 시작 + 국면연동(§7). 현재 BTC **하락 국면**이라 초기 부진 예상(정상).

### 8.3 paper 운영 현황 — A1 청산 e2e 완성 (2026-06-28)
- **baseline `trend_donchian`** paper: trade_id 1 SHORT **청산 완료**.
- **청산 e2e 검증 ✅**: 진입 62471.3 → trailing SL 단조 하향(64038→63221→…→60759.90) → **sl_hit @ 60759.90** (104h 보유), net_pnl **+$105.29**(+2.64%, fee $3.93). DB trade record 정합(규칙 10) + 청산 후 HOLD(재진입 조건 미충족). **진입→trailing→sl_hit 청산→PnL 전 사이클 라이브 경로 검증.**
- ⚠️ I-PE007 실증: DB `stop_loss`=초기값 64038(청산은 60759.90) → trailing DB 미기록 확인 → **I-PE007 fix 적용**(재기동 e2e 대기).
- 개선점 적용본(I-PE006 heartbeat / I-PE007 trailing SL DB) 반영 후 **재기동 시 e2e 재검증 예정**(포지션 없어 안전).
- config `active: ["trend_donchian"]` (paper용 임시 — 뼈대는 `[]`). paper DB: `data/coinbot_paper.db`.

### 8.4 남은 작업 로드맵
```
1. [✅ 완료] paper 청산 e2e — trailing→sl_hit 청산→PnL 전 사이클 검증(§8.3). 부수 발견 I-PE006/007/008/009, I-PE006/007 fix 적용
2. [다음] 라이브-백테 정합성(A2) — paper 거래(trade 1) vs 같은 기간 백테 1:1 비교
     (compare_live_backtest.py 를 paper용 단일구간 변형). trailing 궤적 비교는 I-PE007 적용 후 재기동 e2e 데이터로
3. ★ LO+ma200 정식화 (MS-4 채택) — 방법 *미결정*:
     (a) baseline trend_donchian 에 long_only/regime 옵션 통합 (paper 청산 후, baseline freeze 해제)
     (b) trend_donchian_exp 을 정식 채택 (config LO+ma200 고정값) — exp plugin 이미 ma_period 지원
4. LO+ma200 백테 재검증 (slippage/funding 보수 차감 — chop50 기준 §169 완료, ma 기준 재확인) + 자금관리 ma 재확인
5. 소액 라이브 시작 (risk 1%, 하락장 부진 각오, 검증 목적). 서버 운영 시 AWS Lightsail 1GB(~$7/월) or EC2 t4g.small(무료) + systemd + 텔레그램 모니터링
6. 상승 전환 + 라이브 edge 확인 후 사이징 점진 증액 (→3%)

[완료] MS 멀티 종목 보편성 검증 (§9) — LO+필터 5종목 robust 입증 + 정식화 후보 LO+ma200 전환(MS-4).
  멀티 종목 동시 운영 확장 시 PATH_F 분기 (I-PE005 수량 cap 명목가 재설계 필요).
```

### 8.5 후순위·향후 (carry)
- **TF-5b cross-TF**: 보류(chop 중복 위험). 라이브 누적 후 재고.
- **TF-5c 고급 요소**(피라미딩/부분청산 등): 선택적, 생략 가능.
- **BLE-5 funding 백테 정밀 통합**(I-PE001 carry): 라이브 전 권장.
- **I-PE004 엔진 미실현 equity 추적**: 영향 작아 2차.
- **자금관리 켈리 재계산**: 라이브 R 분포 누적 후.

### 8.6 커밋 이력 (브랜치 `path-c-lookahead-fix`)
| 커밋 | 내용 |
|---|---|
| `3f1d2e4` | PATH_E 신설 + GO (TF-1~4): compute_donchian, trend_donchian plugin, 백테/스윕/funding |
| `f924af7` | I-PE002/003 fix (진행중봉 신호 불일치, TP None) |
| `70a62d7` | TF-5a edge 강화 (trend_donchian_exp + ablation → LO+chop50 채택 PF 1.86) |
| `57502d1` | LO+chop50 재검증 + 자금관리 1차(MDD 재확인) + cross-TF 보류 |
| `17d46f6` | BTC 국면 분석 + 라이브 시작 방향 (가) |

### 8.7 1회용 탐사 스크립트 (commit 제외=`.gitignore`, 결과는 본 문서 보존)
- `_tmp_tf3_param_sweep.py` — TF-3 파라미터 민감도(14조합)
- `_tmp_tf5_ablation.py` — TF-5a ablation (VARIANTS 수정해 1차 17변형/조합 8변형)
- `_tmp_tf5_sizing.py` — 자금관리 1차(R-multiple·연속켈리·post-hoc 미실현 MDD·risk 스윕)
- `_tmp_regime_analysis.py` — BTC 국면 분석(ZigZag 20%, 1d)
- `_tmp_ms_multisymbol.py` — MS-2 멀티 종목 검증(5종목×8변형, cap 무력화)
- `_tmp_ms3_explore.py` — MS-3a chop 임계 평탄성 + 연도 walkforward
- `_tmp_ms3b_filter_table.py` — MS-3b chop·ma 임계 PF/ret 테이블(ma_period 스윕)
- (PATH_D: `_tmp_r1_ic_scan.py`, `_tmp_r2_meanrev_sim.py`)
- 재현: `config/default.yaml` active/params 오버라이드 + `BacktestEngine` (각 스크립트 상단 주석 참조)

---

## 9. MS — 멀티 종목 보편성 검증 (★ 2026-06-24 착수)

### 9.0 목적
LO+chop50 는 **BTC in-sample 으로 선택된** 설정(TF-5a ablation). 이를 ETH/SOL/XRP/DOGE 에
**손대지 않고 그대로 던져** 추세추종 edge 가 종목 보편적인지(=과최적화 아닌지) 검증한다.
부차로 종목별 통하는 로직을 탐색한다. 다른 종목 = (불완전한) out-of-sample 표본.

### 9.1 설계 결정 (확정, 2026-06-24)
| 항목 | 결정 | 근거 |
|---|---|---|
| 검증 구조 | **2단계 게이트**: ① 검증(BTC 확정 8변형 고정 파라미터) → ② 탐색(부진 종목, robust 통제) | 종목별 베스트 1개 선택 = 다중비교 함정. 검증을 탐색의 문지기로 두어 cherry-pick 억제 |
| 종목 | ETH / SOL / XRP / DOGE | 대형·고변동강추세·횡보박스·밈 = 성격 분산(추세추종 우호/불리 모두 표본화) |
| 방법론 | BTC robust 통제 **그대로**: 고정 변형 풀(즉흥 지표 금지)·임계 평탄성·연도 walkforward·표본 게이트 | TF-3/TF-5a 와 동일 잣대로 실재 edge vs 노이즈 구분 |
| 평가 | 종목별 PF/양수연도/MDD/거래수 + **메타: 4종 중 몇 종에서 baseline 개선·PF>1·LO+chop50 재현** | "종목별 베스트 나열"이 아니라 "동일 설정의 종목 간 일관성"이 판정 포인트 |
| 격리 | plugin/엔진/RiskManager 수정 0. config 오버라이드(symbol·cap)로만 전환. 1회용 스크립트 | 설계원칙·규칙 8(캡슐화). download_history `--symbol`만 영구 커밋(멀티 종목 영구 지원) |

### 9.2 사전 함정 점검 (구현 전 코드 확인 완료)
- **I-PE005 (치명, §6)**: `max_position_size_btc=1.0` 수량 cap → 알트 백테 클램프. **백테 무력화(1e12) 필수**. 미처리 시 MS-2 결과 전부 무효.
- **ATR 변동성 흡수 ✅ 확인**: 사이징 `raw_size = (balance×risk%) / (ATR×mult)` (`manager.py:204`) — risk 는 $ 고정, 분모가 ATR 이라 변동성 큰 종목 size 자동 축소 → **같은 파라미터 던지기가 기술적으로 정당**(cap 만 제거하면).
- **종목 동조성 한계**: ETH/SOL/XRP/DOGE ↔ BTC 0.7~0.9 상관(같은 매크로 사이클). 4종이 "4 독립 표본"이 아니라 사실상 1.x → **"보편 edge" 신호 강도 한 단계 낮춰** 해석.
- **funding 낙관 편향**: 백테 funding=0(I-PE001). 알트는 funding 변동성·역프 커 BTC 보다 편향 큼 → 보수적 해석, 사후 점검.
- **DOGE 데이터 제약**: OKX 무기한 상장 늦음(미확인, MS-1 실측). 데이터 짧으면 MS-3 탐색·연도 walkforward 신뢰 강등.

### 9.3 단계 (MS-1 ~ MS-4)
```
MS-1  데이터 + cap 처리                                          ✅ 완료
        a) download_history.py --symbol 인자 추가(영구 커밋 대기)
        b) ETH/SOL/XRP/DOGE 4h 다운로드 — SOL/DOGE 상장 늦어 start 보정
        c) 정합 검증 통과(gap·중복 0, 첫봉 거래량·가격 정상)
        d) cap 무력화 + BTC 재현(441/PF1.31 불변) 정합 ✅ + DOGE 레버리지 실측
MS-2  검증 — _tmp_ms_multisymbol.py: 5종목 × 8변형, 파라미터 고정   ✅ 완료 (§9.5)
        → LO+chop50 5/5 PF>1·baseline 개선. 추세추종 보편 입증(동조성 한정)
MS-3  탐색 — chop 임계 평탄성 + 연도 walkforward + ma 기간 robust    ✅ 완료 (§9.6)
        → LO+필터 보편. chop 단조(PF↑/ret↓ trade-off), ma robust+ret 보존 우수
MS-4  종합 — chop50 vs ma 채택 재검토 + 라이브 함의                 ✅ 완료 (§9.7)
        → (가) LO+ma200 정식화 후보 전환 (위험조정·연도·효율 우위, BTC 동급)
```

### 9.4 코드 변경 매트릭스 (구현 계획)
| 파일 | 변경 | 종류 |
|---|---|---|
| `scripts/download_history.py` | `--symbol` optional 인자 → config symbol 오버라이드 | **영구(commit)** |
| `src/strategy/plugins/trend_donchian_exp.py` | ma 분기 `compute_ema(df,200)` → `ma_period` 파라미터화 (default 200 불변, MS-3b ma 스윕용) | **영구(commit)** |
| `scripts/_tmp_ms_multisymbol.py` / `_tmp_ms3_explore.py` / `_tmp_ms3b_filter_table.py` | 신규 1회용 — 종목×변형/임계 스윕 + cap 무력화 | commit 제외 |
| config / 엔진 / risk | **무변경** (스크립트 config 오버라이드로 흡수) | — |

> 결과 텍스트는 본 §9 에 누적 보존. 1회용 스크립트는 `.gitignore`(`scripts/_tmp_*.py`).

### 9.5 MS-1~2 결과 (2026-06-24)

**MS-1 데이터** (4h, ccxt probe 로 상장 시점 확정):
| 종목 | 봉수 | 시작 | 종료 | gap/중복 |
|---|---|---|---|---|
| BTC | 14193 | 2020-01-01 | 2026-06-23 | 0/0 |
| ETH | 14059 | 2020-01-01 | 2026-06-01 | 0/0 |
| XRP | 14059 | 2020-01-01 | 2026-06-01 | 0/0 |
| SOL | 11719 | **2021-01-25** | 2026-06-01 | 0/0 |
| DOGE | 12907 | **2020-07-11** | 2026-06-01 | 0/0 |
- SOL/DOGE 는 OKX 무기한 상장이 2020-01-01 이후 → `--start` 보정. 첫 봉 거래량·가격 정상(실데이터).

**검증 (사용자 "정상 결과인지" 점검 — 모두 통과)**:
- **MS-1d cap 무력화 불변 ✅**: BTC baseline = 441거래/PF1.31/pos4-7/mdd14.9 정확 재현(기존 cap=1.0 대비). 8변형 전부 §4.6 일치.
- **사이징 부풀리기 없음 ✅ (실측, DOGE baseline)**: 거래별 명목 레버리지 **평균 0.205x**(95% 0.38x, 최대 0.66x), **5x cap·수량 cap 0% 바인딩**. cap 무력화가 사이징을 안 부풀림 — risk 1% 사이징 100% 지배. ret 은 종목 추세를 risk 1%로 정직히 탄 결과.
- **funding=0 확인 ✅**: DOGE funding_fee 합=$0(I-PE001 carry, 알트 낙관 편향 BTC 보다 큼). trading_fee=$2386(taker 0.05% 정상).
- **ret_pct 정의**: `Σpnl/INIT = (최종-초기 balance)/INIT` = **복리 반영 equity 총수익률**(balance 비례 risk% 사이징). cap 무력화로 BTC ret 120.9→123.4(2020 저가구간 소량 클램프 해제, edge 구조 PF/pos/mdd 불변).

**MS-2 종목 × 변형 (PF / pos_years / MDD%)** — `data/ms_multisymbol/results.csv`:
| 종목 | baseline | long_only | chop50 | **LO+chop50** | ma200 | LO+ma200 |
|---|---|---|---|---|---|---|
| BTC | 1.31/4-7/14.9 | 1.61/4/10.8 | 1.49/6/11.8 | **1.86/6/10.8** | 1.46/5/11.9 | 1.82/5/9.8 |
| ETH | 1.28/6-7/13.1 | 1.45/5/13.3 | 1.30/6/15.0 | **1.45/5/11.3** | 1.45/6/11.7 | 1.61/6/9.2 |
| XRP | 1.32/4-7/30.2 | 1.69/5/20.5 | 1.36/5/27.0 | **1.79/5/18.6** | 1.45/5/24.6 | 1.88/6/16.9 |
| SOL | 1.24/4-6/15.6 | 1.37/3/14.4 | 1.38/6/9.9 | **1.65/4/11.7** | 1.33/4/13.0 | 1.53/4/9.0 |
| DOGE | 1.56/5-7/17.0 | 1.93/5/13.2 | 1.53/5/17.5 | **1.83/5/14.7** | 1.75/5/11.0 | 2.26/5/11.5 |

**메타 (변형별 PF>1 종목 / baseline개선 종목, 전체 5)**:
- baseline 5/5 PF>1 · long_only 5/5개선 · chop50 4/5개선 · ma200 5/5개선 · **LO+chop50 5/5개선** · LO+chop38 5/5 · LO+ma200 5/5.

**판정**:
- **추세추종 + LO+chop50 edge 는 5종목 PF>1·구조 보편 → 과최적화(BTC 특이) 아님 강하게 입증.**
- 단 **동조성(BTC 0.7~0.9 상관)** 으로 독립 5표본 아님 → "BTC 결론 강화" 수준으로 해석(신호 강도 강등).
- LO+chop50 개선폭 **ETH 가 최약(+0.17)**, chop50 단독은 ETH 거의 무효(+0.02)·chop38 강함(PF1.61/7-7). **ma200 계열은 5/5 개선 + ret 보존**(LO+chop50 과 다른 프로파일) → MS-3 1순위.
- 종목 특이: XRP short 손실 큼(long_only MDD 30→20), SOL long_only 약화(pos 3/6, short 일부 기여), DOGE ret 321%는 역사적 폭등 의존(PF 1.56 신뢰).

### 9.6 MS-3 결과 — 필터 robust 탐색 (2026-06-25)

**MS-3a Part A — chop 임계 평탄성** (LO+chop, PF(거래수)): **5/5 단조(낮을수록 강필터·PF↑) = cherry-pick 아님.**
| 종목 | c30 | c38 | c44 | c50 | c56 | c62 |
|---|---|---|---|---|---|---|
| BTC | 2.61(51) | 2.22(111) | 1.82(161) | 1.86(205) | 1.73(228) | 1.64(236) |
| ETH | 2.62(40) | 1.76(115) | 1.62(174) | 1.45(227) | 1.45(252) | 1.45(258) |
| XRP | 3.84(33) | 1.85(100) | 1.97(148) | 1.79(190) | 1.67(210) | 1.66(216) |
| SOL | 2.33(26) | 1.80(80) | 1.68(133) | 1.65(165) | 1.41(197) | 1.41(201) |
| DOGE | 3.27(38) | 2.36(89) | 2.26(129) | 1.83(163) | 1.74(187) | 1.89(188) |
- ⚠️ c30 은 거래 26~51(연 4~10)로 **표본 과적합**(PF 화려해도 신뢰 못 함). **c44~50 이 표본·PF 균형**(chop50 채택 유효). ETH 특이성(앞 의심) 해소 — ETH 도 단조, c50~62 포화일 뿐.

**MS-3a Part B — 연도 walkforward** (pos_years): 공통 약점 **2022(베어)·2026(하락 부분년)**, 나머지 다수 양수.
- LO+chop50: BTC6/ETH5/XRP5/SOL4/DOGE5. ma200: BTC5/**ETH6**/XRP5/SOL4/DOGE5(BTC 2022 +937 방어). LO+ma200: **ETH6/XRP6**/BTC5/SOL4/DOGE5.
- → **ma 계열이 연도 일관성 우수**(ETH/XRP 6/7). DOGE 는 모든 변형 2021 폭등 의존.

**MS-3b — ma 기간 robust + PF/ret 동반** (plugin `ma_period` 파라미터화, 회귀 BTC LO+ma200 PF1.82=MS-2 일치 ✅). 전부 long_only.

PF:
| 종목 | chop30 | chop44 | **chop50** | chop62 | ma100 | ma150 | **ma200** | ma250 |
|---|---|---|---|---|---|---|---|---|
| BTC | 2.61 | 1.82 | **1.86** | 1.64 | 1.68 | 1.78 | **1.82** | 1.95 |
| ETH | 2.62 | 1.62 | **1.45** | 1.45 | 1.51 | 1.60 | **1.61** | 1.52 |
| XRP | 3.84 | 1.97 | **1.79** | 1.66 | 1.84 | 1.77 | **1.88** | 1.76 |
| SOL | 2.33 | 1.68 | **1.65** | 1.41 | 1.46 | 1.50 | **1.53** | 1.71 |
| DOGE | 3.27 | 2.26 | **1.83** | 1.89 | 2.22 | 2.32 | **2.26** | 2.44 |

ret%:
| 종목 | chop30 | chop44 | **chop50** | chop62 | ma100 | ma150 | **ma200** | ma250 |
|---|---|---|---|---|---|---|---|---|
| BTC | 42.1 | 109.6 | **149.3** | 129.8 | 130.1 | 135.7 | **126.6** | 135.8 |
| ETH | 35.2 | 88.6 | **91.9** | 105.9 | 107.2 | 116.4 | **103.8** | 84.3 |
| XRP | 62.8 | 155.8 | **163.0** | 152.2 | 168.3 | 137.7 | **140.8** | 113.6 |
| SOL | 18.6 | 53.5 | **62.2** | 54.3 | 49.8 | 52.0 | **52.7** | 64.1 |
| DOGE | 69.0 | 201.3 | **185.8** | 261.2 | 310.6 | 310.1 | **286.7** | 288.9 |

**판정 (PF+ret 동반 해석)**:
- **chop = PF↑ vs ret↓ trade-off**: 강필터(낮은 임계)일수록 PF↑지만 거래 급감 → ret 급락(BTC c30 ret 42 vs c50 149). chop30 고PF 는 표본 착시. 균형점 c44~50.
- **ma = 임계 robust(평탄) + ret 보존**: ma100~250 PF 평탄(ma200 cherry-pick 아님), ret 도 안정. chop 같은 강필터 ret 붕괴 없음. **DOGE 는 ma 가 chop 압도**(ret 287~311 vs chop50 186).
- → **ma 필터가 chop 보다 robust·ret 동시 우수.** MS-4 에서 **LO+chop50(현 채택) vs LO+ma 채택 재검토** 필요. 단 동조성·funding=0·DOGE 폭등의존 한계는 유지.

### 9.7 MS-4 종합 — LO+chop50 vs LO+ma200 채택 결정 (2026-06-26)

**정면 비교 (5종목, MS-2/3 데이터) — PF / pos_years / MDD% / 거래수**:
| 종목 | LO+chop50 | LO+ma200 | 우위 |
|---|---|---|---|
| BTC | 1.86/6/10.8/205 | 1.82/5/**9.8**/184 | 동급(PF chop·MDD ma) |
| ETH | 1.45/5/11.3/227 | **1.61/6/9.2**/204 | **ma 완승** |
| XRP | 1.79/5/18.6/190 | **1.88/6/16.9**/156 | **ma 완승** |
| SOL | **1.65**/4/11.7/165 | 1.53/4/**9.0**/153 | PF chop·MDD ma |
| DOGE | 1.83/5/14.7/163 | **2.26/5/11.5**/137 | **ma 완승** |

- **PF**: ma 3종 우위(ETH/XRP/DOGE), chop 2종(SOL·BTC 미세) → ma 약우위.
- **MDD**: **ma 5/5 전종목 낮음**(위험조정 우위, 결정적).
- **연도 일관**: ma 2종 우위(ETH·XRP 6/7), BTC만 chop.
- **거래수**: ma 전종목 적음 → 수수료·funding 노출 적어 효율 유리.
- **robust**: ma100~250 평탄(cherry-pick 아님) vs chop 강필터 ret 붕괴.

**판정**: ma 가 위험조정·비용효율·연도 일관에서 전반 우위, 절대 PF 약우위. chop 의 BTC PF 미세 우위(1.86)는 MDD 열위(10.8 vs 9.8)로 상쇄.

**한계 (결정적 우위는 아님)**: ① 동조성(독립 5표본 아님) ② ma 큰 ret 우위는 DOGE 폭등 의존(라이브 BTC 무관) ③ BTC 단독은 chop50≈ma200(위험조정만 ma 미세 우위) ④ funding=0 미정밀(단 ma 거래 적어 유리 방향).

**결정 = (가) LO+ma200 으로 정식화 후보 전환** (2026-06-26, §7):
- 위험조정(MDD 5/5)·연도 일관·효율 전반 우위 + BTC 동급 → 전환 손해 없고 위험조정 이득.
- **신전략 = `trend_donchian_exp` + long_only=True + regime_filter_type=ma + ma_period=200**.
- 후속(라이브 정식화 시): ① 자금관리(켈리·MDD) ma 기준 재확인 — BTC ma200 MDD 9.8%로 chop50(10.8) 보다 낮아 더 보수적, 기존 결론 대체로 유효 ② §4.7 국면분석(LO+chop50=상승장 전략) ma 도 long_only 기반이라 동일 적용 ③ paper 재검증.
