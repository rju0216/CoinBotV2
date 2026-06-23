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
```

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

---

## 5. 진행 기록

| 시점 | 단계 | 상태 | 커밋 | 비고 |
|---|---|---|---|---|
| 2026-06-23 | PATH_E 신설 + 설계 결정 | ✅ | (대기) | Donchian 돌파+ATR trailing / 4h / plugin+BacktestEngine. 의존성 확인(trailing 훅 OK, donchian 추가 필요) |
| 2026-06-23 | TF-1 구현 | ✅ 완료 | (대기) | `compute_donchian`(causal shift) + `trend_donchian` plugin(진입/초기SL/TP비활성/trailing 단조) + config 섹션. 함정 반영(trailing 단조 plugin 보장, PositionSide≠SignalSide). 단위 13건 신규, 회귀 514→**527 pass**. 엔진 수정 0 |
| 2026-06-23 | TF-2 백테 (사용자 수행) | ✅ 1차 GO | (대기) | 2020~2026 +120.93%, 441거래, 승률 35%, **PF 1.31, 평균 R 2.42**, MDD 14.89%. 양수 4/7년, 약세장(2022) short 방어. 결과 §4.1. ML(NO-GO)과 대비 — 추세추종 비대칭 보상 작동. funding/slippage 미반영 낙관 편향(I-PE001) → TF-3 + funding 점검으로 확정 |
| 2026-06-23 | paper 검증 → I-PE002/003 fix | ✅ 완료 (e2e 검증) | (대기) | paper 첫 기동에서 **시작 직후 진입** 포착 → I-PE002(진행중 봉 신호, 라이브-백테 불일치) + I-PE003(TP 음수→거래소 등록 위험) 발견·수정. `_build_ctx` 진행중봉 제외 + TP None 지원 + ENTRY 로그 TP None 처리. 단위 4 신규, 회귀 527→**529**. 백테 불변(441/PF1.31/TP None). **paper 재기동 e2e ✅**: 에러 없음, 마감 봉(62846.30 실제 Donchian 돌파) SHORT 정상 진입, `TP=None` 표기 — 라이브-백테 신호 동등성 확보 |
| 2026-06-23 | TF-3 파라미터 민감도 | ✅ robust 확정 | (대기) | 14조합 스윕 **전부 PF>1(1.14~1.41), 양수연도≥4/7, R 2.28~2.90**. 20/10/2.0=평탄영역 중간점(cherry-pick 아님). exit_period 길수록 우수. 결과 §4.2. 1회용 `_tmp_tf3_param_sweep.py`. 디버그: initialize() 누락 fix. 남은 관문=I-PE001 funding |
| 2026-06-23 | TF-4 funding 점검 + 최종 | ✅ **GO 확정** | (대기) | funding 보수 점검: 최악 상한(0.02%/8h 모두 비용)도 PF 1.18>1, 현실(방향반영) 1.29~1.30≈원본(양방향 헤지). 결과 §4.3. **PATH_E 결론: 추세추종 실거래 후보 확보 — ML(NO-GO)과 대비** |

---

## 6. 잠재 이슈 트래커

| ID | 이슈 | 상태 |
|---|---|---|
| I-PE001 | 백테 funding=0·slippage=0 → 추세추종(평균 보유 2.9일=funding ~9회)은 funding 영향이 커 성과 낙관 편향 가능. 라이브에선 `_close_with_funding`로 반영되나 백테 미반영(I-BP001 carry) | **✅ 점검 완료(TF-4)** — 보수 가정 근사: 최악 상한(0.02%/8h 모두 비용) PF 1.18>1, 현실(방향반영) 1.29~1.30(양방향 헤지). GO 유지. 단 정밀 백테 통합(BLE-5: ccxt funding history)은 **라이브 전 권장**(carry) |
| I-PE002 | **paper 검증 중 발견 (2026-06-23)** — 라이브/paper 에서 trend_donchian 이 `ctx.candles.iloc[-1]`(진행 중 봉, ccxt single tick)으로 신호·trailing 판정 → 백테(마감 봉, `_slice_candles`)와 불일치. 시작 직후 SHORT 진입이 증거. candles 직접 쓰는 plugin 공통(example 포함). ML plugin 은 `get_features_for_ctx(ts<now)`로 보호받아 무영향 | **✅ fix** — `_build_ctx`(engine_base)에서 `LAST_CLOSED_BAR_IDX` 기준 진행 중 봉 제외(`n_drop = -1 - IDX`; 백테 0 무변경, 라이브 1 제외). 추상화 교정(모드 차이를 엔진이 흡수). 백테 결과 불변(441/PF 1.31). 단위 2건 |
| I-PE003 | I-PE002 분석 중 발견 — trend_donchian TP=`현재가±risk×RR(100)` 가 SHORT 에서 **음수 가격**(-97946). 라이브는 `place_take_profit`로 거래소(`takeProfitPrice`)에 실제 등록 → 음수 등록 시 거부·진입 실패 위험 | **✅ fix** — `compute_take_profit` 시그니처 `float\|None` 확장 + trend_donchian `→None`. 엔진: TP None 이면 거래소 등록 skip + check_candle_sl_tp skip. ENTRY 로그(`engine_base.py:557`) `TP=%.2f`→None 안전 처리(paper e2e 에서 발견). 추세추종 정석(TP 미설정, trailing SL 청산). 단위 2건 |

신규 이슈는 I-PE001~ 형태로 등록.

---

## 7. 결정 사항 로그

| 시점 | 사안 | 결정 | 근거 |
|---|---|---|---|
| 2026-06-23 | 진입 룰/TF/검증 | Donchian 돌파+ATR trailing / 4h / plugin+BacktestEngine | 전형·단순·라이브백테 일관 |
