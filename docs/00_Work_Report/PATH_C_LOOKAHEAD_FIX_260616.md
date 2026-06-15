# PATH_C_LOOKAHEAD_FIX — 멀티TF 피처 lookahead 제거 + 재학습

작성 시점: 2026-06-16 (BLE-6-2 정합성 검증 중 I-BLE012 발견 → 별도 PATH 승격)
선행: `PATH_B_LIVE_EXTENSION_260505.md` (BLE-6-2 보류 — 본 PATH 종착 후 재개)
작업 브랜치: `path-c-lookahead-fix` (라이브 운영 중인 main 보호)

---

## 0. 문서 목적

BLE-6-2(라이브-백테 정합성 검증)에서 라이브 14건 vs 백테 24건이 **매칭 0건**(완전 신호 불일치)으로 드러나, 원인 규명 결과 **멀티TF 피처 병합의 lookahead(I-BLE012)** 를 확정. 이는 백테 신뢰성 + 운영 모델(v010) 전반에 영향을 주는 프로젝트 최대 근본 버그라, 별도 PATH 로 승격해 fix → 재학습 → 재평가 → 라이브 교체를 한 흐름으로 관리한다.

---

## 1. I-BLE012 분석 (확정)

### 메커니즘
`compute_multi_tf_features` 가 상위TF 피처를 **봉 시작 시각 index** 에 둔 채 완성값을 `reindex+ffill` 병합 → 봉이 진행 중인 시점의 entry 봉에 그 봉의 **미래 완성값**이 병합됨. `now_ms=datetime.now()` 기준 "마지막 봉만" 진행중 판정이라, 전체 데이터를 한 번에 계산하는 **백테/학습은 과거 봉이 모두 마감 간주 → lookahead 발동**. 라이브만 마지막 봉을 부분 회피 → **라이브-백테 신호 불일치**.

### 실증
- 05-13 12:00 15m행의 `1h_macd`: 백테(전체계산) **-52.13** vs 라이브(causal) **+5.95** (부호 반전)
- BLE-6-2: 라이브 14건 vs 백테 24건 매칭 0건, 백테 승률 79%/+31%(lookahead 부풀림) vs 라이브 28.6%/-$176(현실)

### 영향 범위
- 81 피처 중 **54개(1h_*/4h_*)** 오염, 27개(15m entry_tf)는 causal(영향 없음)
- **학습 features 도 동일 lookahead** → v010 모델이 미래 상위TF 정보로 학습 → 라이브 저조의 직접 원인
- 과거 모든 백테/walkforward 평가 수치(OOS 1118% 등)가 lookahead 부풀림 → 재평가 필요

---

## 2. fix (causal 멀티TF 병합)

`src/strategy/features.py::compute_multi_tf_features`:
- 상위TF `tf_feat.index += pd.Timedelta(milliseconds=TF_MS[tf])` (봉 시작 → 마감 시각) 후 `reindex+ffill`
- → 각 entry 시점에 **이미 마감된 상위봉만** 병합 (causal). 전체/부분 계산 무관 동일
- `now_ms`/`_is_in_progress_bar`/`datetime` import 제거 (불필요)
- entry_tf base(27 causal)는 불변

검증: causal 실증(전체=window 일치) + 단위 테스트 `TestMultiTfCausalMerge` 4건(lookahead 회귀 방지 / causal 일치 / ffill 시각 정합 / 컬럼 수). 회귀 514 pass.

---

## 3. 단계 (P-C-1 ~ P-C-4)

```
P-C-1 features causal fix          ✅ 완료 (c99222e)
P-C-2 모델 4종 재학습 (causal)      ✅ 완료 (v011 — 학습 F1 0.69→0.45 부풀림 제거)
P-C-3 causal 백테 재평가            ⚠️ 거래 희소·edge 증거 없음 (ensemble/DL 0건)
P-C-3.5 walkforward (ML 개별)       ✅ ML edge 없음 확정 (PF 0.83/0.79, 33 folds 857/1583건)
P-C-4 라이브 모델 교체              ❌ 무의미 확정 (causal 환경 edge 없음 → 교체 불가)
```

→ **PATH_C 종착 결론: 전략 edge 가 lookahead 의존이었음이 확정됨 (§7 참조). 전략 paradigm 재검토 필요.**

### ★ 라이브 보호 (실수 방지)
현 라이브 v010 은 *lookahead features 로 학습*되어 학습-추론이 (잘못된 채로) 일관된 상태. **features fix 를 main 에 머지하면 라이브가 causal features 를 v010 에 줘서 불일치 악화**. fix+재학습+검증 일괄 완성 후 라이브 교체. 그때까지 라이브는 **현 main 코드 유지** (브랜치에서만 작업).

**재학습 기간 중 라이브 재시작 정책 = (가) 재시작 금지** (결정 2026-06-16): P-C-2 재학습은 `latest.json` 을 v011 로 갱신하므로, 라이브를 재시작하면 v011(causal 학습본)이 main 의 lookahead features 와 만나 불일치가 악화된다. 따라서 **P-C-2 ~ P-C-4(main 머지) 전까지 라이브를 재시작하지 않는다** (메모리의 v010 유지 → 영향 없음). 안전장치로 재학습 전 latest.json 4종을 백업해, 부득이한 재시작 시 v010 으로 복원 후 기동한다.

---

## 4. P-C-2 재학습 가이드 (사용자 GPU)

**전제**: 반드시 `path-c-lookahead-fix` 브랜치 체크아웃 상태에서 실행 (causal features.py). main 에서 실행하면 lookahead features 로 학습되어 무의미.

### 4.1 출력 경로 & 버전 명명 (v010 안전)
- 경로: `models/{model}/v{NNN}_{entry_tf}_{start}_{end}` (예 `v011_15m_2020-01-01_2026-04-01`)
- `next_model_version` = 기존 최대 번호 +1 → 현재 v010 → 재학습 시 **v011 신규 생성**
- ✅ **v010 디렉토리 보존** (덮어쓰기 없음) → 비교·롤백 가능
- ⚠️ 스크립트는 단순 `v011` 부여 (causal 표식 없음) → v010(lookahead)과 구분 위해 train_meta/본 문서에 "v011=causal fix 후" 명기

### 4.2 latest.json 메커니즘
- 각 `train_*.py` 가 학습 끝에 `models/{model}/latest.json {"path": v011}` 자동 갱신
- ensemble.yaml `sub_params.*.model_path: "models/{model}/latest"` → plugin/calibrate 가 latest.json 으로 실제 버전 해석 → **재학습 시 자동 v011 참조** (ensemble.yaml 수정 불필요)

### 4.3 calibration 흐름 (train 후 별도)
- `scripts/calibrate_models.py` 가 latest.json 이 가리키는 model_dir 에 `calibrator_isotonic.joblib` 저장
- → **train(latest→v011) 직후, latest=v011 상태에서 calibrate 실행해야 v011 에 calibrator 저장**

### 4.4 라이브 보호 = (가) 재시작 금지
재학습으로 latest=v011 이 되면 라이브 재시작 시 v011 + main(lookahead) 불일치. **P-C-2~P-C-4 전까지 라이브 재시작 금지** (메모리 v010 유지). 안전장치: 0단계에서 latest.json 4종 백업 → 부득이한 재시작 시 v010 복원.

### 4.5 실행 순서
⚠️ **피처 캐시 공유 주의**: `features_{tf}_{entry_tf}_{start}_{end}.parquet` 캐시는 모델명이 없어 **4 모델이 공유**(모두 15m·동일 기간). 4개 모두 `--force-features` 동시 실행 시 같은 캐시 파일 쓰기 충돌(corruption). → **첫 1개만 `--force-features`** 로 causal 캐시 재생성, **나머지 3개는 force 없이** 재사용.

```
# 0. (안전) latest.json 4종 백업 → 부득이한 라이브 재시작 시 v010 복원용
#    models/{lightgbm,xgboost,lstm,transformer}/latest.json → latest.v010.bak

# ※ 각 train_*.py 는 개별 모델 config 사용 (config[ "ml_lightgbm"/"ml_xgboost"/"dl_lstm"/
#   "dl_transformer" ] top-level 키). ensemble.yaml 은 sub_params 중첩이라 KeyError.
#   개별 config 의 required_timeframes/model_path 가 ensemble 과 동일 → 학습 결과 정합.

# 1. 캐시 causal 재생성 (첫 1개 단독, --force-features) — lightgbm 이 CPU+빠름
python scripts/train_lightgbm.py    --config config/ml_lightgbm.yaml    --start 2020-01-01 --end 2026-04-01 --force-features

# 2. 나머지 3종 (force 없이 = 캐시 재사용, 쓰기 없음 → 병렬 안전)
python scripts/train_xgboost.py     --config config/ml_xgboost.yaml     --start 2020-01-01 --end 2026-04-01
python scripts/train_lstm.py        --config config/dl_lstm.yaml        --start 2020-01-01 --end 2026-04-01
python scripts/train_transformer.py --config config/dl_transformer.yaml --start 2020-01-01 --end 2026-04-01

# 3. calibration (latest=v011 상태에서 → v011 에 calibrator 저장)
python scripts/calibrate_models.py --strategy all --start 2020-01-01 --end 2026-04-01
```
- 기간은 v010 과 동일 (train·calibration 모두 2020-01-01 ~ 2026-04-01 — v010 train_meta/calibration_meta 확인값)
- 완료 시 v011 4종 + calibrator. ensemble.yaml(latest 참조)은 자동 v011

### 4.5.1 병렬 실행 가능 여부
- **첫 lightgbm(`--force-features`) 은 단독 실행** (캐시 생성 중 다른 train 이 같은 캐시 접근 금지)
- lightgbm 완료 후, 나머지 3종은 캐시 **읽기만** → 충돌 없음:
  - **xgboost(CPU)** 는 GPU 모델과 자원이 달라 병렬 가능
  - **lstm/transformer(GPU)** 는 GPU 메모리를 공유 → 동시 실행 시 OOM 위험. **순차 권장** (GPU 메모리 여유가 확실하면 병렬 가능)
- 권장: `lightgbm(force)` → 그 후 `xgboost` + `lstm` 병렬(CPU/GPU 분리) → `transformer` (lstm 완료 후)

### 4.6 예상
causal 재학습본은 기존 백테 수치(OOS 1118%, 백테 79% 승률 등 lookahead 부풀림)보다 **성능이 낮을 것** = 진짜 edge. lookahead 부풀림 제거 후 실제 성능으로 운영 판단.

---

## 5. 진행 기록

| 시점 | 단계 | 상태 | 커밋 | 비고 |
|---|---|---|---|---|
| 2026-06-16 | P-C-1 features causal fix | ✅ 완료 | c99222e | compute_multi_tf_features 마감시각 shift + ffill. now_ms/_is_in_progress_bar/datetime 제거. 테스트 TestMultiTfCausalMerge 4건. 회귀 514 pass. causal 실증(전체=window) |
| 2026-06-16 | P-C-2 모델 4종 재학습 (v011) | ✅ 완료 | (모델 git 외) | v011_15m_2020-01-01_2026-04-01 4종 + calibrator(isotonic). lightgbm만 --force-features(캐시 공유) + 개별 config. **lookahead 부풀림 정량: OOS 학습 F1 macro v010~0.69 → v011~0.45, Acc ~0.77→~0.65**. causal SHORT/LONG precision~0.40 recall 0.16~0.24 (HOLD 편중) |
| 2026-06-16 | P-C-3 causal 백테 재평가 | ⚠️ 결과 | — | OOS 2026-04-01~06-16 threshold 0.55: **ensemble 0 / dl_lstm 0 / dl_transformer 0 / ml_lightgbm 2(+0.81%) / ml_xgboost 4(-1.38%)**. (참고 ensemble threshold 0.40: 7건 -1.58%). → causal 거래 희소 + edge 증거 없음. 표본 부족(6건)으로 단정 불가. **라이브 v010(lookahead 학습)은 동일 구간 활발히 거래·수익 → v010 수익은 학습-추론 불일치 상의 운/bias 가능성 ↑** |
| 2026-06-16 | P-C-3.5 walkforward (ML 개별) | ✅ 결과 | — | ml_lightgbm/ml_xgboost `--save-all-folds` 재학습 + walkforward 평가(2020~2026, 33 folds). **결과: ml_lightgbm 857건 승률 31.7% PF 0.83 누적 -$9,811 (양수 8/33), ml_xgboost 1583건 승률 30.8% PF 0.79 누적 -$22,710 (양수 9/33). 큰 표본서 profit_factor<1 명확 손실 → causal ML edge 없음 확정** (calibration none 기준). max_dd 35%(DD락 도달 fold 다수) |
| (확정) | P-C-4 라이브 교체 | ❌ 무의미 | — | causal 환경 edge 없음 확정 → v011/v012 교체 불가. 전략 paradigm 재검토 필요 |

---

## 6. 잠재 이슈 트래커

| ID | 이슈 | 상태 |
|---|---|---|
| I-BLE012 | 멀티TF 병합 lookahead (백테/학습 상위TF 완성봉을 진행중 시점 병합) | **✅ fix 완료 (P-C-1)**. 영향 규명 완료: causal 재학습(P-C-2) + walkforward(P-C-3.5)로 **전략 edge 가 lookahead 의존이었음 확정** → §7 |

---

## 7. PATH_C 최종 결론 (2026-06-16)

**전략 edge 가 lookahead 에 의존했음이 확정되었다.** lookahead(I-BLE012) 제거 후 causal 환경에서:
- **ensemble / DL(lstm·transformer)**: 진입 신호 전무 (threshold 0.55 거래 0)
- **ML 개별 walkforward** (33 folds, lightgbm 857건 / xgboost 1583건 = 통계적 충분): **profit_factor 0.83 / 0.79, 양수 fold 24~27%, 누적 -$9.8k / -$22.7k 명확한 손실**

→ **현 전략(15m ensemble ML 단기 방향 예측)은 진짜 거래 edge 가 없다.** 과거 모든 백테 성과(OOS 1118%, 백테 79% 승률 등)는 lookahead 부풀림이었다. 라이브 v010 수익(+$594)은 edge 가 아니라 운/bias 일 가능성이 매우 높다 (lookahead 학습 모델 + 라이브 causal 입력의 우발적 결과).

### 의의
- 가짜 성과로 자금을 확대하기 전에 lookahead 를 발견·차단 → 큰 잠재 손실 회피
- causal 검증 인프라 확보: features causal fix(P-C-1), `compare_live_backtest.py`, walkforward 평가

### 미결정 (사용자 결정 대기)
1. **전략 방향** (paradigm 재검토 불가피): (가) 더 긴 TF(1h/4h/1d 추세) / (나) 룰베이스 추세추종 / (다) 타깃·feature 재설계 / (라) 별도 리서치
2. **라이브 v010 운영**: edge 없음 확정 → 유지 / 축소 / 중단. **자금 확대는 보류 권고**

### 후속
- features causal fix(P-C-1)는 정확하므로 향후 전략 재설계의 베이스로 유지 (main 머지는 전략 재정립 시 함께 판단)
- BLE-6-2(라이브-백테 정합성): causal 모델 edge 없어 재비교 의미 약화 → 전략 재설계 후로 보류
- 과거 백테/평가 문서 수치에 "lookahead 부풀림" 주석 필요
- PATH_B 잔여(BLE-1/2/5/4/3, I-BLE011): 전략 재정립 후 우선순위 재검토
