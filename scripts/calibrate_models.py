"""Phase E-2-3 Step 2: 4 분류 모델 v001에 calibrator 학습/저장 (I-B009 해결 시도).

walk-forward 26 folds 재실행 → 모든 fold OOS probabilities 수집 →
MulticlassCalibrator (Platt + Isotonic) 학습 → 기존 v001 모델 디렉토리에
calibrator_<method>.joblib + calibration_meta.json 추가 저장.

기존 모델 파일(model.txt/model.json/model.pth)은 변경 0. calibrator만 부가.
plugin이 config의 calibration_method에 따라 자동 로드/적용.

Usage:
    python scripts/calibrate_models.py --strategy ml_lightgbm \\
        --start 2020-01-01 --end 2024-12-31
    python scripts/calibrate_models.py --strategy all \\
        --start 2020-01-01 --end 2024-12-31

PPO 제외 — 정책 모델이라 confidence calibration 무의미.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.ml.calibration import MulticlassCalibrator  # noqa: E402
from src.ml.feature_pipeline import build_features  # noqa: E402
from src.ml.label_generator import build_labels_from_config  # noqa: E402  # I-BL001 fix
from src.ml.walk_forward import generate_walk_forward_splits  # noqa: E402
from src.strategy.features import get_feature_names  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STRATEGY_CONFIGS = {
    "ml_lightgbm": "config/ml_lightgbm.yaml",
    "ml_xgboost": "config/ml_xgboost.yaml",
    "dl_lstm": "config/dl_lstm.yaml",
    "dl_transformer": "config/dl_transformer.yaml",
}

STRATEGY_DIR_PREFIX = {
    "ml_lightgbm": "lightgbm",
    "ml_xgboost": "xgboost",
    "dl_lstm": "lstm",
    "dl_transformer": "transformer",
}


# ─── GBDT (LightGBM/XGBoost) walk-forward 재실행 ───

def _collect_oos_probs_lightgbm(
    train_cfg: dict, X: pd.DataFrame, y: pd.Series, folds, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    lgb_params = {
        "objective": "multiclass", "num_class": 3,
        "metric": "multi_logloss", "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": 7, "verbose": -1, "seed": 42,
        **(train_cfg.get("lgb_params", {})),
    }
    n_estimators = int(train_cfg.get("n_estimators", 500))
    early_stopping = int(train_cfg.get("early_stopping_rounds", 50))

    oos_probs_list, oos_labels_list = [], []
    for fold in folds:
        train_mask = (X.index >= fold.train_start) & (X.index <= fold.train_end)
        test_mask = (X.index >= fold.test_start) & (X.index <= fold.test_end)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if horizon > 0 and len(X_train) > horizon:
            X_train = X_train.iloc[:-horizon]
            y_train = y_train.iloc[:-horizon]
        if len(X_train) < 100 or len(X_test) < 10:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
        model = lgb.train(
            lgb_params, dtrain, num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
        )
        probs = model.predict(X_test)  # (N, 3) softmax
        oos_probs_list.append(probs)
        oos_labels_list.append(y_test.values)
        logger.info("  Fold %d: %d test samples", fold.fold_id, len(X_test))

    return np.vstack(oos_probs_list), np.concatenate(oos_labels_list)


def _collect_oos_probs_xgboost(
    train_cfg: dict, X: pd.DataFrame, y: pd.Series, folds, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb

    xgb_params = {
        "objective": "multi:softprob", "num_class": 3,
        "eval_metric": "mlogloss", "learning_rate": 0.05,
        "max_depth": 7, "verbosity": 0, "seed": 42,
        **(train_cfg.get("xgb_params", {})),
    }
    n_estimators = int(train_cfg.get("n_estimators", 500))
    early_stopping = int(train_cfg.get("early_stopping_rounds", 50))

    oos_probs_list, oos_labels_list = [], []
    for fold in folds:
        train_mask = (X.index >= fold.train_start) & (X.index <= fold.train_end)
        test_mask = (X.index >= fold.test_start) & (X.index <= fold.test_end)
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        if horizon > 0 and len(X_train) > horizon:
            X_train = X_train.iloc[:-horizon]
            y_train = y_train.iloc[:-horizon]
        if len(X_train) < 100 or len(X_test) < 10:
            continue

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(
            xgb_params, dtrain, num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        probs = model.predict(dval)  # (N, 3) softmax
        oos_probs_list.append(probs)
        oos_labels_list.append(y_test.values)
        logger.info("  Fold %d: %d test samples", fold.fold_id, len(X_test))

    return np.vstack(oos_probs_list), np.concatenate(oos_labels_list)


# ─── DL (LSTM/Transformer) walk-forward 재실행 ───

def _collect_oos_probs_dl(
    strategy: str, strategy_cfg: dict, train_cfg: dict,
    X_scaled: pd.DataFrame, y_full: pd.Series, folds,
    n_features: int, horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    import copy
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score

    from src.ml.models import LSTMClassifier, TransformerClassifier
    from src.ml.sequence_utils import make_sequences

    lookback = int(strategy_cfg.get("lookback", 60))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 256))
    epochs = int(train_cfg.get("epochs", 50))
    patience = int(train_cfg.get("early_stopping_patience", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("DL device: %s", device)

    def _make_model() -> torch.nn.Module:
        if strategy == "dl_lstm":
            params = dict(train_cfg.get("lstm_params", {}))
            return LSTMClassifier(
                n_features=n_features,
                hidden_size=int(params.get("hidden_size", 64)),
                num_layers=int(params.get("num_layers", 1)),
                dropout=float(params.get("dropout", 0.3)),
            )
        else:  # dl_transformer
            params = dict(train_cfg.get("transformer_params", {}))
            return TransformerClassifier(
                n_features=n_features,
                d_model=int(params.get("d_model", 64)),
                nhead=int(params.get("nhead", 4)),
                num_layers=int(params.get("num_layers", 2)),
                dim_ff=int(params.get("dim_ff", 128)),
                dropout=float(params.get("dropout", 0.3)),
            )

    def _train_fold(X_tr, y_tr, X_te, y_te) -> torch.nn.Module:
        model = _make_model().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        val_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        for _epoch in range(1, epochs + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss_sum += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
            val_loss = val_loss_sum / len(val_ds)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _predict(model, X) -> np.ndarray:
        model.eval()
        out_chunks = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = torch.from_numpy(X[i:i + batch_size]).float().to(device)
                probs = torch.softmax(model(X_batch), dim=1).cpu().numpy()
                out_chunks.append(probs)
        return np.concatenate(out_chunks, axis=0)

    oos_probs_list, oos_labels_list = [], []
    for fold in folds:
        train_mask = (X_scaled.index >= fold.train_start) & (X_scaled.index <= fold.train_end)
        test_mask = (X_scaled.index >= fold.test_start) & (X_scaled.index <= fold.test_end)
        X_tr_df, y_tr_s = X_scaled[train_mask], y_full[train_mask]
        X_te_df, y_te_s = X_scaled[test_mask], y_full[test_mask]
        if horizon > 0 and len(X_tr_df) > horizon:
            X_tr_df = X_tr_df.iloc[:-horizon]
            y_tr_s = y_tr_s.iloc[:-horizon]

        X_tr, y_tr, _ = make_sequences(X_tr_df, y_tr_s, lookback=lookback)
        X_te, y_te, _ = make_sequences(X_te_df, y_te_s, lookback=lookback)
        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        model = _train_fold(X_tr, y_tr, X_te, y_te)
        probs = _predict(model, X_te)
        oos_probs_list.append(probs)
        oos_labels_list.append(y_te)
        logger.info("  Fold %d: %d test sequences", fold.fold_id, len(X_te))

    return np.vstack(oos_probs_list), np.concatenate(oos_labels_list)


# ─── 데이터 준비 helper ───

async def _prepare_data(strategy: str, config: dict, start: str, end: str):
    """4 모델 공통 features/labels/folds 준비 (DL은 X_scaled 추가)."""
    from sklearn.preprocessing import StandardScaler

    strategy_cfg = config[strategy]
    train_cfg = strategy_cfg.get("train", {})
    entry_tf = strategy_cfg.get("entry_timeframe", "15m")
    timeframes = strategy_cfg.get("required_timeframes", [entry_tf])
    if entry_tf not in timeframes:
        timeframes = [entry_tf] + timeframes

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    features = await build_features(config, timeframes, entry_tf, start_ms, end_ms, force=False)
    loader = HistoricalDataLoader(config)
    df = await loader.download_range_merged(entry_tf, start_ms, end_ms)
    await loader.close()
    # I-BL001 fix: train.label_method 분기 (direction / triple_barrier).
    # train_*.py와 동일 helper 사용 → 모델 학습과 calibrator 학습이 같은 라벨 정의.
    # effective_horizon은 embargo + train tail 제거에 사용 (direction=horizon, triple_barrier=time_barrier_bars).
    labels, label_params, effective_horizon = build_labels_from_config(df, train_cfg)
    logger.info(
        "[%s] label method=%s, effective_horizon=%d",
        strategy, label_params["method"], effective_horizon,
    )

    feature_names = get_feature_names(entry_tf, [t for t in timeframes if t != entry_tf])
    valid_cols = [c for c in feature_names if c in features.columns]

    is_dl = strategy in ("dl_lstm", "dl_transformer")
    if is_dl:
        X_full = features[valid_cols]
        y_full = labels.reindex(X_full.index).fillna(-1)
        X_clean = X_full.dropna()
        scaler = StandardScaler()
        scaler.fit(X_clean.values)
        X_scaled = X_full.copy()
        valid_mask = X_full.notna().all(axis=1)
        X_scaled.loc[valid_mask, :] = scaler.transform(X_full.loc[valid_mask].values)
        X_for_folds = X_scaled
        y_for_calibrator = y_full
    else:
        merged = features[valid_cols].copy()
        merged["label"] = labels
        merged = merged.dropna()
        merged = merged[merged["label"] >= 0]
        X_for_folds = merged[valid_cols]
        y_for_calibrator = merged["label"].astype(int)

    folds = generate_walk_forward_splits(
        X_for_folds.index,
        train_months=int(train_cfg.get("train_months", 6)),
        test_months=int(train_cfg.get("test_months", 2)),
        step_months=int(train_cfg.get("step_months", 2)),
        embargo_bars=effective_horizon,
    )
    return (
        strategy_cfg, train_cfg, X_for_folds, y_for_calibrator,
        folds, effective_horizon, len(valid_cols), label_params,
    )


# ─── 단일 strategy 처리 ───

async def calibrate_strategy(strategy: str, start: str, end: str) -> None:
    config_path = STRATEGY_CONFIGS[strategy]
    config = load_config(config_path)
    (
        strategy_cfg, train_cfg, X, y, folds,
        effective_horizon, n_features, label_params,
    ) = await _prepare_data(strategy, config, start, end)
    horizon = effective_horizon  # _collect_oos_probs_* 함수가 horizon 인자 받음 (이름 호환)
    logger.info(
        "[%s] walk-forward %d folds, %d features",
        strategy, len(folds), n_features,
    )

    if strategy == "ml_lightgbm":
        oos_probs, oos_labels = _collect_oos_probs_lightgbm(train_cfg, X, y, folds, horizon)
    elif strategy == "ml_xgboost":
        oos_probs, oos_labels = _collect_oos_probs_xgboost(train_cfg, X, y, folds, horizon)
    elif strategy in ("dl_lstm", "dl_transformer"):
        oos_probs, oos_labels = _collect_oos_probs_dl(
            strategy, strategy_cfg, train_cfg, X, y, folds, n_features, horizon,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.info(
        "[%s] OOS predictions: %d samples, label dist=%s",
        strategy, len(oos_labels),
        dict(pd.Series(oos_labels).value_counts().sort_index()),
    )

    # ── Calibrator 학습 (Platt + Isotonic) ──
    cal_platt = MulticlassCalibrator(method="platt").fit(oos_probs, oos_labels)
    cal_isotonic = MulticlassCalibrator(method="isotonic").fit(oos_probs, oos_labels)

    # ── 모델 디렉토리 (latest.json 통해 v001 해석) ──
    dir_prefix = STRATEGY_DIR_PREFIX[strategy]
    latest_json = Path(f"models/{dir_prefix}/latest.json")
    with open(latest_json) as f:
        model_dir = Path(json.load(f)["path"])
    logger.info("[%s] saving calibrators to %s", strategy, model_dir)

    joblib.dump(cal_platt, model_dir / "calibrator_platt.joblib")
    joblib.dump(cal_isotonic, model_dir / "calibrator_isotonic.joblib")

    # 메타 (I-BL001 fix: label_params 기록 — 모델 학습과 라벨 정합성 추적)
    meta = {
        "strategy": strategy,
        "model_dir": str(model_dir),
        "created": datetime.now().isoformat(),
        "calibration_period": f"{start} ~ {end}",
        "walk_forward_folds": len(folds),
        "oos_samples": int(len(oos_labels)),
        "label_params": label_params,
        "label_distribution": {
            int(k): int(v) for k, v in
            pd.Series(oos_labels).value_counts().sort_index().items()
        },
        "methods_saved": ["platt", "isotonic"],
    }
    with open(model_dir / "calibration_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("[%s] calibration_meta.json saved", strategy)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrator 학습 (Phase E-2-3 Step 2)")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGY_CONFIGS.keys()) + ["all"],
        required=True,
        help="대상 strategy 또는 'all' (4 분류 모델 모두)",
    )
    parser.add_argument("--start", default="2020-01-01", help="학습 데이터 시작")
    parser.add_argument("--end", default="2024-12-31", help="학습 데이터 종료")
    args = parser.parse_args()

    targets = list(STRATEGY_CONFIGS.keys()) if args.strategy == "all" else [args.strategy]
    for strat in targets:
        logger.info("=" * 60)
        logger.info("[%s] calibration 시작", strat)
        await calibrate_strategy(strat, args.start, args.end)
        logger.info("[%s] calibration 완료", strat)
    logger.info("=" * 60)
    logger.info("All done: %s", targets)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
