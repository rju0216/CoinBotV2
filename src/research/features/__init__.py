"""1층 신호처리 피처 (Phase 2) — OHLCV → X DataFrame (설계 §7·§8, MasterPlan §2 계약).

원시 캔들을 **인과적으로 정제**해 2층(MLP)이 소비할 피처 벡터를 만든다. 6축(설계 §8):

| 계열 | 피처 | 모듈 |
|---|---|---|
| 1 추세 | robust KF slope + 불확실성 | ``kalman`` (Step 2.3) |
| 2 레짐 | ER + Hurst | ``regime`` (Step 2.2) |
| 3 변동성 | 변화(수축팽창) | ``simple.vol_change`` |
| 4 거래량 | 상대거래량 | ``simple.relative_volume`` |
| 5 분포 | semi-deviation | ``simple.rolling_semi_deviation`` |
| 6 형태 | 종가위치 | ``simple.close_position`` |

**substrate 계약 (MasterPlan §2, 하류 규칙 18 준수)**:
- 산출 = **X DataFrame**(라벨 ``BarrierLabels`` 의 index 와 정렬 가능한 OHLCV.index).
- **정규화는 harness 소유**(폴드 train-only) → 피처는 **자기정규화 불필요**. 대신 정상성을
  *구성으로* 확보한다(로그비율·유계값). F-6(z-score vs robust)은 Step 2.4 분포진단.
- 입력 = **OHLCV 만**. regime 태그가 피처로 새면 즉시 누수 → 방화벽(기둥 2).
- **기본창만**(Phase 2). 창 순차탐색(coordinate descent)은 Phase 3 R2 로 이연.
- 검증 = **소프트웨어 정합성**(인과 ``assert_causal`` + 정상성/스케일). 성과는 안 봄.
"""
