"""라벨 생성 (Phase 1) — 삼중배리어 라벨·변동성 추정·분포 게이트.

MasterPlan §2 substrate 계약: 라벨 = 클래스 Series(X.index 정렬), N=label_horizon 이
splitter purge/꼬리예약 구동. 엔진/플러그인과 분리(엔진 수정 0), 순수 오프라인.

- ``volatility`` — ATR·YZ 변동성 추정(공통 콜러블, 분수 정규화, 인과)
- ``triple_barrier`` — 삼중배리어 스캔 → ``BarrierLabels``
- ``distribution`` — 층1 학습가능성 + 층3 국면일관성 분포 게이트 (성과 안 봄)
"""
