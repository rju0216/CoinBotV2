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
P-C-1 features causal fix          ✅ 완료 (이 커밋)
P-C-2 모델 4종 재학습 (causal)      ⬜ 사용자 GPU
P-C-3 재평가 (walkforward/백테)     ⬜ causal 진짜 성능 측정
P-C-4 라이브 모델 교체              ⬜ 재학습본 검증 후 main 머지 + 교체
```

### ★ 라이브 보호 (실수 방지)
현 라이브 v010 은 *lookahead features 로 학습*되어 학습-추론이 (잘못된 채로) 일관된 상태. **features fix 를 main 에 머지하면 라이브가 causal features 를 v010 에 줘서 불일치 악화**. fix+재학습+검증 일괄 완성 후 라이브 교체. 그때까지 라이브는 **현 main 코드 유지** (브랜치에서만 작업).

---

## 4. P-C-2 재학습 가이드 (사용자 GPU)

각 모델을 v010 과 동일 기간(2020-01-01 ~ 2026-04-01 cutoff)으로, fix 된 features 로 재학습. **`--force-features` 필수** (기존 lookahead 피처 캐시 무효화 + causal 재계산):

```
python scripts/train_lightgbm.py   --config config/ensemble.yaml --start 2020-01-01 --end 2026-04-01 --force-features
python scripts/train_xgboost.py    --config config/ensemble.yaml --start 2020-01-01 --end 2026-04-01 --force-features
python scripts/train_lstm.py        --config config/ensemble.yaml --start 2020-01-01 --end 2026-04-01 --force-features
python scripts/train_transformer.py --config config/ensemble.yaml --start 2020-01-01 --end 2026-04-01 --force-features
```
- 신규 버전(예 v011_causal)으로 저장 → v010(lookahead) 보존(비교용)
- 학습 후 calibration(isotonic) + ensemble.yaml `sub_params.*.model_path` 갱신
- 상세 경로/버전 규칙은 재학습 착수 시 확정

### 예상
causal 재학습본은 기존 백테 수치보다 **성능이 낮을 것** = 진짜 edge. lookahead 부풀림 제거 후 실제 성능으로 운영 판단.

---

## 5. 진행 기록

| 시점 | 단계 | 상태 | 커밋 | 비고 |
|---|---|---|---|---|
| 2026-06-16 | P-C-1 features causal fix | ✅ 완료 | (이 커밋) | compute_multi_tf_features 마감시각 shift + ffill. now_ms/_is_in_progress_bar/datetime 제거. 테스트 TestMultiTfCausalMerge 4건(_is_in_progress_bar 3건 + InProgressExclusion 2건 대체). 회귀 514 pass. causal 실증(전체=window) |
| (대기) | P-C-2 모델 4종 재학습 | ⬜ | — | 사용자 GPU. --force-features 필수 |
| (대기) | P-C-3 재평가 | ⬜ | — | causal walkforward/백테 |
| (대기) | P-C-4 라이브 교체 | ⬜ | — | 검증 후 main 머지 |

---

## 6. 잠재 이슈 트래커

| ID | 이슈 | 상태 |
|---|---|---|
| I-BLE012 | 멀티TF 병합 lookahead (백테/학습 상위TF 완성봉을 진행중 시점 병합) | **P-C-1 fix 완료** / 재학습(P-C-2)으로 모델 영향 해소 진행 |

### 후속 (본 PATH 종착 후)
- BLE-6-2 재개 (causal 재학습본으로 라이브-백테 재비교 — 진짜 정합성)
- 과거 백테/평가 문서 수치 재검토 (lookahead 부풀림 표기)
- PATH_B 잔여: BLE-1/2/5/4/3, I-BLE011
