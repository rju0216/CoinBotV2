"""모델 artifact — walk-forward 한 윈도우의 완성 모델 묶음 save/load (D4-2d).

단일 JSON. 모델이 작아(K≤6 → 수백 float) npz 불필요하고, 검사가능 + pickle
회피(보안·이식). 배열은 list 로 직렬화.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.strategy.regime.features import FeatureConfig, ZScoreParams
from src.strategy.regime.hmm import HMM
from src.strategy.regime.mapping import StateMapping


@dataclass(frozen=True)
class RegimeModel:
    """walk-forward 한 윈도우의 완성 모델 (생산↔소비 계약 단위).

    valid_period [start, end): 이 모델을 적용할 OOS 구간 (ISO ts).
    meta: train_window 범위·emission_type·fit_ts 등 (참조용).
    """

    hmm: HMM
    mapping: StateMapping
    zscore: ZScoreParams
    feature_config: FeatureConfig
    valid_period: tuple[str, str]
    meta: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "hmm": self.hmm.to_dict(),
            "mapping": self.mapping.to_dict(),
            "zscore": self.zscore.to_dict(),
            "feature_config": self.feature_config.to_dict(),
            "valid_period": list(self.valid_period),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegimeModel":
        return cls(
            hmm=HMM.from_dict(d["hmm"]),
            mapping=StateMapping.from_dict(d["mapping"]),
            zscore=ZScoreParams.from_dict(d["zscore"]),
            feature_config=FeatureConfig.from_dict(d["feature_config"]),
            valid_period=(d["valid_period"][0], d["valid_period"][1]),
            meta=d["meta"],
        )


def save_model(model: RegimeModel, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(model.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_model(path: str | Path) -> RegimeModel:
    return RegimeModel.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
