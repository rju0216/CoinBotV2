"""MulticlassCalibrator 단위 테스트 (Phase E-2-3 Step 2).

회귀 보호: I-B009 해결 시도의 정확성 보증.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest

from src.ml.calibration import MulticlassCalibrator


@pytest.fixture
def synthetic_3class():
    """확률이 라벨과 적당히 일치하는 3-class 합성 데이터.

    - 클래스 0: raw_probs[:, 0] 평균 높음
    - 클래스 1: raw_probs[:, 1] 평균 높음
    - 클래스 2: raw_probs[:, 2] 평균 높음
    """
    rng = np.random.default_rng(42)
    n_per = 100
    probs_list = []
    labels_list = []
    for c in range(3):
        # 각 클래스 c에 대해 (n_per, 3) 행렬 — c번 열이 평균 0.6, 나머지 0.2
        probs = rng.dirichlet([2.0 if i == c else 1.0 for i in range(3)], size=n_per)
        probs_list.append(probs)
        labels_list.append(np.full(n_per, c))
    raw_probs = np.vstack(probs_list)
    y = np.concatenate(labels_list)
    # 셔플
    idx = rng.permutation(len(y))
    return raw_probs[idx], y[idx]


class TestPlattScaling:
    def test_fit_then_transform_shape(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt").fit(raw, y)
        out = cal.transform(raw)
        assert out.shape == raw.shape
        assert cal.n_classes == 3

    def test_transform_rows_sum_to_one(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt").fit(raw, y)
        out = cal.transform(raw)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)

    def test_transform_in_range(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt").fit(raw, y)
        out = cal.transform(raw)
        assert (out >= 0).all() and (out <= 1).all()


class TestIsotonicRegression:
    def test_fit_then_transform_shape(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="isotonic").fit(raw, y)
        out = cal.transform(raw)
        assert out.shape == raw.shape

    def test_transform_rows_sum_to_one(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="isotonic").fit(raw, y)
        out = cal.transform(raw)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)

    def test_isotonic_monotonic(self, synthetic_3class):
        """Isotonic은 단조 → raw_prob[:, c] 정렬 시 calibrated[:, c]도 비감소."""
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="isotonic").fit(raw, y)
        out = cal.transform(raw)

        for c in range(3):
            order = np.argsort(raw[:, c])
            # 정규화 전 값을 직접 확인하기 위해 isotonic 학습 결과만
            calibrator = cal.calibrators[c]
            if calibrator is None:
                continue
            sorted_pred = calibrator.predict(raw[order, c])
            # 비감소 (단조 증가 또는 같음)
            assert np.all(np.diff(sorted_pred) >= -1e-9)


class TestErrors:
    def test_unknown_method(self):
        with pytest.raises(ValueError):
            MulticlassCalibrator(method="unknown")  # type: ignore

    def test_transform_before_fit(self, synthetic_3class):
        raw, _ = synthetic_3class
        cal = MulticlassCalibrator(method="platt")
        with pytest.raises(RuntimeError):
            cal.transform(raw)

    def test_fit_dimension_mismatch(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt")
        with pytest.raises(ValueError):
            cal.fit(raw, y[:50])

    def test_transform_class_count_mismatch(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt").fit(raw, y)
        bad_raw = raw[:, :2]  # 2-class 입력
        with pytest.raises(ValueError):
            cal.transform(bad_raw)


class TestPickleCompatibility:
    """plugin이 joblib.load로 deserialize할 수 있는지 (model_dir/calibrator_*.joblib 패턴)."""

    def test_pickle_then_unpickle(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="platt").fit(raw, y)
        buf = io.BytesIO()
        pickle.dump(cal, buf)
        buf.seek(0)
        cal2 = pickle.load(buf)
        out1 = cal.transform(raw)
        out2 = cal2.transform(raw)
        np.testing.assert_array_equal(out1, out2)

    def test_pickle_isotonic(self, synthetic_3class):
        raw, y = synthetic_3class
        cal = MulticlassCalibrator(method="isotonic").fit(raw, y)
        buf = io.BytesIO()
        pickle.dump(cal, buf)
        buf.seek(0)
        cal2 = pickle.load(buf)
        out1 = cal.transform(raw)
        out2 = cal2.transform(raw)
        np.testing.assert_array_equal(out1, out2)


class TestSingleClassEdgeCase:
    """학습 데이터에 단일 클래스만 있으면 해당 클래스 calibrator는 None — raw 그대로 사용."""

    def test_single_class_in_one_dim(self):
        # 모든 샘플이 클래스 1이라 클래스 0/2의 binary indicator 단일값 (0)
        rng = np.random.default_rng(0)
        raw_probs = rng.dirichlet([1, 5, 1], size=50)
        y = np.full(50, 1)  # 모두 클래스 1

        cal = MulticlassCalibrator(method="platt").fit(raw_probs, y)
        # 클래스 0과 2는 단일 클래스 (모두 0) → calibrator None
        assert cal.calibrators[0] is None
        assert cal.calibrators[2] is None
        # 클래스 1은 단일 (모두 1) → calibrator None
        assert cal.calibrators[1] is None

        # transform은 정상 동작 (raw 그대로 후 정규화)
        out = cal.transform(raw_probs)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9)
