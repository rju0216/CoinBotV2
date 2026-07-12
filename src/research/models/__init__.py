"""2층 학습 모델 (Phase 3, MasterPlan §2 모델 콜러블 계약).

harness(``run_walk_forward``)에 baseline 처럼 ``model_factory`` 로 꽂히는 학습 모델들:
- ``TreeBench`` (lightgbm) — 배리어결과 단일태스크 하한선(설계 §6 트리 벤치).
- ``SmallMLP`` (torch) — 얕고 좁은 공유트렁크(R1 단일헤드, R3 멀티태스크 이식).

전부 ``ProbaModel`` 계약(fit/predict_proba→DataFrame[classes]/predict/classes_)을 구현 —
baselines.BaselineModel 과 동일한 duck-typed 인터페이스(harness 는 둘 다 받음).
"""

from src.research.models.base import ProbaModel
from src.research.models.mlp import SmallMLP
from src.research.models.tree_bench import TreeBench

__all__ = ["ProbaModel", "SmallMLP", "TreeBench"]
