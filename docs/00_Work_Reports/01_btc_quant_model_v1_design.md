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
| D1 | 메커니즘 A 독립 검증 + 스펙↔인프라 갭 분석 | ✅ 완료 | 2026-06-29 | (미커밋) |
| D2 | 정책 계약 표면 B 재설계 + O-1~O-5 결정 | 예정 | | |
| D3 | 학습 파이프라인 설계 (t-HMM·feature·walk-forward) | | | |
| D4 | 구현 + 백테 검증 + 회귀 테스트 추가 | | | |

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
- **D-4 HMM filtering = 전략 incremental 상태로 관리** — 매 봉 전체 재계산은 O(T²)
  (1h·수년=~5만 봉). filtered α_t를 인스턴스에 보유, 새 봉마다 1스텝 전진. 엔진 수정 불필요.

### 인지사항 (메커니즘 수정 불필요, 해석/구현 시 감안)
- **백테 낙관 편향**: 백테는 봉 full high/low로 SL/TP 시뮬, 라이브는 시초 한 점 +
  거래소 conditional order 의존 → 백테가 SL/TP를 더 정확/자주 체결. + 백테 funding=0
  stub. 트레일링·장기보유(추세) 모델이라 백테-라이브 갭 요인.
- **회귀 테스트 부재 4건** (코드는 맞으나 미보장) → I-002, D4 백로그.

---

## 잠재 이슈 트래커

| ID | 이슈 | 발생 | 상태 | 해결 경로 |
|---|---|---|---|---|
| I-001 | 백테 funding=0 → 추세 장기보유 비용 과소평가 (라이브 갭) | D1 | OPEN | O-5 결정 대기 |
| I-002 | 회귀 테스트 부재 4건: ①엔진측 lookahead 절단(`_build_ctx`/`_slice_candles`) ②진입가 백테=라이브 동일성 ③트레일링 SL end-to-end ④레짐 스왑 시퀀스 | D1 | OPEN | D4 테스트 추가 |
| I-003 | 2-플러그인 시 HMM 레짐 봉당 중복 계산 | D1 | 설계반영 | D2 공유 regime 모듈(D-3) |

---

## 미결정 항목 (D2에서 결정 — 협업규칙 6 형식으로 제시)

| ID | 항목 | 현재 가닥 |
|---|---|---|
| O-1 | type 경계 임계값 (손 고정+스윕 / 데이터 자동) | 손 고정+스윕 추천 |
| O-2 | Size ∝ confidence 구체 형태 (비례식·상하한·레버리지 매핑) | 선형+클램프 추천 |
| O-3 | walk-forward 재추정 방식 (오프라인 사전학습 / 런타임 재학습 / 단일모델) | 오프라인 사전학습 추천 |
| O-4 | 적대청산 매핑 | D1에서 force_exit+빈슬롯으로 가닥 (D-2) |
| O-5 | gen1 펀딩 처리 (백테 0 수용 / estimate_funding 구현) | I-001 연계 |

---

## 변경 기록 (문서·코드)

- **2026-06-29**: "종가 진입" → "다음 봉 시초가 진입" 정정 (결정 D-1).
  - `spec_v1.md` L131·136·175 인라인 갱신.
  - `transcript.md` 상단 정정 배너 추가 (verbatim 대화 텍스트는 보존).
  - 코드(src/): 해당 기록 없음 (플러그인 비어 있음).
