"""XGBoost 워크포워드 학습 파이프라인.

LightGBM 학습 스크립트(train_lightgbm.py)와 구조 동일 — XGBoost API만 차이:
- xgb.DMatrix (lgb.Dataset 대응)
- objective="multi:softprob", eval_metric="mlogloss"
- 모델 저장: model.json (UBJSON/JSON, LightGBM의 .txt 대응)

Usage:
    python scripts/train_xgboost.py --config config/ml_xgboost.yaml \
        --start 2020-01-01 --end 2024-12-31

출력: models/xgboost/v{NNN}_{tf}_{start}_{end}/
  - model.json         XGBoost 모델 파일
  - feature_names.json 피처 컬럼명 목록
  - train_meta.json    학습 메타데이터 (날짜, 하이퍼파라미터, OOS 성능)
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402

from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.ml.feature_pipeline import build_features  # noqa: E402
from src.ml.label_generator import build_labels_from_config  # noqa: E402
from src.ml.walk_forward import generate_walk_forward_splits  # noqa: E402
from src.strategy.features import get_feature_names  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost 워크포워드 학습")
    parser.add_argument("--config", required=True, help="config YAML 경로")
    parser.add_argument("--start", required=True, help="학습 데이터 시작 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="학습 데이터 종료 (YYYY-MM-DD)")
    parser.add_argument("--force-features", action="store_true", help="피처 캐시 무시, 재생성")
    args = parser.parse_args()

    config = load_config(args.config)
    strategy_cfg = config["ml_xgboost"]
    train_cfg = strategy_cfg.get("train", {})

    entry_tf = strategy_cfg.get("entry_timeframe", "15m")
    timeframes = strategy_cfg.get("required_timeframes", [entry_tf])
    if entry_tf not in timeframes:
        timeframes = [entry_tf] + timeframes

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)

    # ── 1. 피처 생성 ──
    logger.info("피처 생성 중... (TF: %s)", timeframes)
    features = await build_features(
        config, timeframes, entry_tf, start_ms, end_ms,
        force=args.force_features,
    )

    # ── 2. 캔들 로드 (레이블 생성용) ──
    loader = HistoricalDataLoader(config)
    df = await loader.download_range_merged(entry_tf, start_ms, end_ms)

    # ── 3. 레이블 생성 (BP-3-3: label_method 분기) ──
    labels, label_params, effective_horizon = build_labels_from_config(df, train_cfg)
    logger.info(
        "Label method=%s, effective_horizon=%d",
        label_params["method"], effective_horizon,
    )

    # ── 4. 유효 행만 추출 ──
    feature_names = get_feature_names(
        entry_tf, [t for t in timeframes if t != entry_tf]
    )
    valid_cols = [c for c in feature_names if c in features.columns]
    if len(valid_cols) < len(feature_names):
        logger.warning(
            "피처 컬럼 누락: 기대 %d, 가용 %d", len(feature_names), len(valid_cols)
        )

    merged = features[valid_cols].copy()
    merged["label"] = labels
    merged = merged.dropna()
    merged = merged[merged["label"] >= 0]

    X = merged[valid_cols]
    y = merged["label"].astype(int)
    logger.info("학습 데이터: %d행, %d피처, 레이블 분포: %s", len(X), len(valid_cols), dict(y.value_counts().sort_index()))

    # ── 5. Walk-forward 분할 ──
    folds = generate_walk_forward_splits(
        X.index,
        train_months=int(train_cfg.get("train_months", 6)),
        test_months=int(train_cfg.get("test_months", 2)),
        step_months=int(train_cfg.get("step_months", 2)),
        embargo_bars=effective_horizon,
    )
    if not folds:
        logger.error("데이터 기간이 너무 짧아 walk-forward 분할 불가")
        return

    logger.info("Walk-forward: %d folds", len(folds))

    # ── 6. XGBoost 하이퍼파라미터 ──
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 7,
        "verbosity": 0,
        "seed": 42,
        **(train_cfg.get("xgb_params", {})),
    }
    n_estimators = int(train_cfg.get("n_estimators", 500))
    early_stopping = int(train_cfg.get("early_stopping_rounds", 50))

    # ── 7. Walk-forward 학습 루프 ──
    oos_predictions: list[int] = []
    oos_labels: list[int] = []
    best_model: xgb.Booster | None = None

    for fold in folds:
        train_mask = (X.index >= fold.train_start) & (X.index <= fold.train_end)
        test_mask = (X.index >= fold.test_start) & (X.index <= fold.test_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if effective_horizon > 0 and len(X_train) > effective_horizon:
            X_train = X_train.iloc[:-effective_horizon]
            y_train = y_train.iloc[:-effective_horizon]

        if len(X_train) < 100 or len(X_test) < 10:
            logger.warning("Fold %d: 데이터 부족 (train=%d, test=%d), 스킵",
                           fold.fold_id, len(X_train), len(X_test))
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=valid_cols)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=valid_cols)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "valid")],
            early_stopping_rounds=early_stopping,
            verbose_eval=100,
        )

        probs = model.predict(dval)
        preds = np.argmax(probs, axis=1).tolist()
        oos_predictions.extend(preds)
        oos_labels.extend(y_test.values.tolist())

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        logger.info(
            "Fold %d: train=%s~%s, test=%s~%s, acc=%.4f, f1=%.4f",
            fold.fold_id,
            fold.train_start.strftime("%Y-%m-%d"),
            fold.train_end.strftime("%Y-%m-%d"),
            fold.test_start.strftime("%Y-%m-%d"),
            fold.test_end.strftime("%Y-%m-%d"),
            acc, f1,
        )

        best_model = model

    if not oos_predictions:
        logger.error("유효한 fold가 없음. 학습 실패.")
        return

    # ── 8. 전체 OOS 성능 ──
    oos_acc = accuracy_score(oos_labels, oos_predictions)
    oos_f1 = f1_score(oos_labels, oos_predictions, average="macro")
    print(f"\n{'='*50}")
    print(f"전체 OOS 성능 (Walk-forward {len(folds)} folds)")
    print(f"{'='*50}")
    print(f"Accuracy: {oos_acc:.4f}")
    print(f"F1 (macro): {oos_f1:.4f}")
    print(classification_report(
        oos_labels, oos_predictions,
        target_names=["SHORT(0)", "HOLD(1)", "LONG(2)"],
    ))

    # ── 9. 모델 저장 ──
    models_root = Path("models/xgboost")
    models_root.mkdir(parents=True, exist_ok=True)

    existing = list(models_root.glob("v*"))
    version = f"v{len(existing) + 1:03d}"
    model_dir = models_root / f"{version}_{entry_tf}_{args.start}_{args.end}"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model.save_model(str(model_dir / "model.json"))

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_cols, f, indent=2)

    train_meta = {
        "version": version,
        "model_type": "xgboost",
        "created": datetime.now().isoformat(),
        "entry_timeframe": entry_tf,
        "timeframes": timeframes,
        "train_period": f"{args.start} ~ {args.end}",
        "walk_forward_folds": len(folds),
        "oos_accuracy": round(oos_acc, 4),
        "oos_f1_macro": round(oos_f1, 4),
        "feature_count": len(valid_cols),
        "label_params": label_params,
        "xgb_params": xgb_params,
    }
    with open(model_dir / "train_meta.json", "w") as f:
        json.dump(train_meta, f, indent=2, default=str)

    # latest 포인터
    with open(models_root / "latest.json", "w") as f:
        json.dump({"path": str(model_dir)}, f)

    print(f"\n모델 저장: {model_dir}")
    print(f"latest → {model_dir}")


if __name__ == "__main__":
    asyncio.run(main())
