"""3-class confidence calibration (I-B009 해결용 — Phase E-2-3 Step 2).

Transformer가 학습 OOS Acc 1위인데 백테 최저인 모순 해소 시도. confidence_threshold
필터링 후 통과 신호의 정답률 정합성을 calibration으로 재조정.

알고리즘:
- Platt scaling (sigmoid): 각 클래스에 LogisticRegression OvR 학습
- Isotonic regression (비모수, 단조): 각 클래스에 IsotonicRegression OvR

3-class (SHORT=0, HOLD=1, LONG=2) 멀티클래스라 OvR 방식.
calibrator.transform 후 합 1로 softmax-normalize.

joblib pickle 호환 — sklearn classifier만 보유하므로 표준 직렬화 가능.
plugin은 `from src.ml.calibration import MulticlassCalibrator` 후 joblib.load 사용.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["platt", "isotonic"]


class MulticlassCalibrator:
    """3-class probability calibration with OvR."""

    def __init__(self, method: CalibrationMethod = "platt") -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown method: {method}. Use 'platt' or 'isotonic'.")
        self.method = method
        self.n_classes: int = 0
        # 각 클래스별 1D calibrator (LogisticRegression for Platt, IsotonicRegression for Isotonic)
        self.calibrators: list = []

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "MulticlassCalibrator":
        """raw_probs: (N, C), y_true: (N,) — OvR fit per class.

        Args:
            raw_probs: 모델의 raw probability 출력 (각 행 합 ~1)
            y_true: 정답 클래스 라벨 (0..C-1)
        """
        if raw_probs.ndim != 2:
            raise ValueError(f"raw_probs must be 2D, got shape {raw_probs.shape}")
        if len(raw_probs) != len(y_true):
            raise ValueError("raw_probs and y_true length mismatch")

        self.n_classes = raw_probs.shape[1]
        self.calibrators = []

        for c in range(self.n_classes):
            y_binary = (y_true == c).astype(int)
            # 학습 데이터에 단일 클래스만 있으면 calibrator 학습 skip — identity 처리
            if len(np.unique(y_binary)) < 2:
                self.calibrators.append(None)
                continue

            if self.method == "platt":
                # raw_probs[:, c]를 1D feature로 LogisticRegression 학습
                cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                cal.fit(raw_probs[:, c:c + 1], y_binary)
            else:  # isotonic
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                cal.fit(raw_probs[:, c], y_binary)

            self.calibrators.append(cal)

        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """raw_probs: (N, C) → calibrated_probs: (N, C). 각 행 합 1로 정규화."""
        if not self.calibrators:
            raise RuntimeError("MulticlassCalibrator not fitted. Call fit() first.")
        if raw_probs.ndim != 2:
            raise ValueError(f"raw_probs must be 2D, got shape {raw_probs.shape}")
        if raw_probs.shape[1] != self.n_classes:
            raise ValueError(
                f"raw_probs has {raw_probs.shape[1]} classes, expected {self.n_classes}"
            )

        out = np.zeros_like(raw_probs, dtype=np.float64)
        for c in range(self.n_classes):
            cal = self.calibrators[c]
            if cal is None:
                # 학습 시 단일 클래스만 있던 경우 — raw 그대로 사용
                out[:, c] = raw_probs[:, c]
                continue

            if self.method == "platt":
                # predict_proba returns (N, 2) — class 1 prob
                out[:, c] = cal.predict_proba(raw_probs[:, c:c + 1])[:, 1]
            else:  # isotonic
                out[:, c] = cal.predict(raw_probs[:, c])

        # softmax-normalize: 각 행 합 1로 (모든 calibrator 결과가 0 미만 또는 양일 수도)
        out = np.clip(out, 1e-12, None)
        row_sum = out.sum(axis=1, keepdims=True)
        return out / row_sum
