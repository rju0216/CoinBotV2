"""Transformer 워크포워드 학습 파이프라인.

B-2a(LSTM) 학습 스크립트와 구조 동일 — 모델 클래스만 교체:
- LSTMClassifier → TransformerClassifier
- lstm_params(hidden_size 등) → transformer_params(d_model, nhead, num_layers, dim_ff, dropout)

Usage:
    python scripts/train_transformer.py --config config/dl_transformer.yaml \
        --start 2020-01-01 --end 2024-12-31

출력: models/transformer/v{NNN}_{tf}_{start}_{end}/
  - model.pth          PyTorch state_dict (CPU)
  - scaler.joblib      StandardScaler 직렬화
  - feature_names.json 피처 컬럼명 목록
  - train_meta.json    학습 메타 (model_arch, OOS 성능 등)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.ml.feature_pipeline import build_features  # noqa: E402
from src.ml.label_generator import build_labels_from_config  # noqa: E402
from src.ml.models import TransformerClassifier  # noqa: E402
from src.ml.sequence_utils import make_sequences  # noqa: E402
from src.ml.walk_forward import generate_walk_forward_splits  # noqa: E402
from src.strategy.features import get_feature_names  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    transformer_params: dict,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[TransformerClassifier, dict]:
    """단일 fold PyTorch 학습. (best_model, info) 반환."""
    model = TransformerClassifier(n_features=n_features, **transformer_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
        val_loss = val_loss_sum / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # I-B003 / §11-#3: 얕은 복사 금지 — deepcopy 필수
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "epochs_run": epoch,
    }
    return model, info


def _predict(model: TransformerClassifier, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """배치 추론 → (N, num_classes) softmax 확률."""
    model.eval()
    out_chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[i : i + batch_size]).float().to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            out_chunks.append(probs)
    return np.concatenate(out_chunks, axis=0)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer 워크포워드 학습")
    parser.add_argument("--config", required=True, help="config YAML 경로")
    parser.add_argument("--start", required=True, help="학습 데이터 시작 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="학습 데이터 종료 (YYYY-MM-DD)")
    parser.add_argument("--force-features", action="store_true", help="피처 캐시 무시, 재생성")
    args = parser.parse_args()

    config = load_config(args.config)
    strategy_cfg = config["dl_transformer"]
    train_cfg = strategy_cfg.get("train", {})

    entry_tf = strategy_cfg.get("entry_timeframe", "15m")
    timeframes = strategy_cfg.get("required_timeframes", [entry_tf])
    if entry_tf not in timeframes:
        timeframes = [entry_tf] + timeframes

    lookback = int(strategy_cfg.get("lookback", 60))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    batch_size = int(train_cfg.get("batch_size", 256))
    epochs = int(train_cfg.get("epochs", 50))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    transformer_params = dict(train_cfg.get("transformer_params", {}))

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 1. 피처 생성 ──
    logger.info("피처 생성 중... (TF: %s, lookback=%d)", timeframes, lookback)
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

    X_full = features[valid_cols]
    y_full = labels.reindex(X_full.index).fillna(-1)
    n_features = len(valid_cols)

    # ── 5. Scaler fit (전체 train 데이터 1회 fit — B-2a와 동일) ──
    X_clean = X_full.dropna()
    scaler = StandardScaler()
    scaler.fit(X_clean.values)
    logger.info(
        "StandardScaler fit 완료: %d행 / %d피처", len(X_clean), n_features
    )

    # 전체 X에 scaler 적용 (NaN 행은 보존 — 시퀀스 변환 시 자동 제외됨)
    X_scaled = X_full.copy()
    valid_mask = X_full.notna().all(axis=1)
    X_scaled.loc[valid_mask, :] = scaler.transform(X_full.loc[valid_mask].values)

    # ── 6. Walk-forward 분할 ──
    folds = generate_walk_forward_splits(
        X_scaled.index,
        train_months=int(train_cfg.get("train_months", 6)),
        test_months=int(train_cfg.get("test_months", 2)),
        step_months=int(train_cfg.get("step_months", 2)),
        embargo_bars=effective_horizon,
    )
    if not folds:
        logger.error("데이터 기간이 너무 짧아 walk-forward 분할 불가")
        return

    logger.info("Walk-forward: %d folds", len(folds))

    # ── 7. Walk-forward 학습 루프 ──
    oos_predictions: list[int] = []
    oos_labels: list[int] = []
    best_model: TransformerClassifier | None = None

    for fold in folds:
        train_mask = (X_scaled.index >= fold.train_start) & (X_scaled.index <= fold.train_end)
        test_mask = (X_scaled.index >= fold.test_start) & (X_scaled.index <= fold.test_end)

        X_tr_df, y_tr_s = X_scaled[train_mask], y_full[train_mask]
        X_te_df, y_te_s = X_scaled[test_mask], y_full[test_mask]

        # Embargo: train 끝 effective_horizon개 행 제거
        if effective_horizon > 0 and len(X_tr_df) > effective_horizon:
            X_tr_df = X_tr_df.iloc[:-effective_horizon]
            y_tr_s = y_tr_s.iloc[:-effective_horizon]

        X_tr, y_tr, _ = make_sequences(X_tr_df, y_tr_s, lookback=lookback)
        X_te, y_te, _ = make_sequences(X_te_df, y_te_s, lookback=lookback)

        if len(X_tr) < 100 or len(X_te) < 10:
            logger.warning(
                "Fold %d: 시퀀스 부족 (train=%d, test=%d), 스킵",
                fold.fold_id, len(X_tr), len(X_te),
            )
            continue

        model, info = _train_one_fold(
            X_tr, y_tr, X_te, y_te,
            n_features=n_features,
            transformer_params=transformer_params,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            device=device,
        )

        probs = _predict(model, X_te, batch_size, device)
        preds = np.argmax(probs, axis=1).tolist()
        oos_predictions.extend(preds)
        oos_labels.extend(y_te.tolist())

        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="macro")
        logger.info(
            "Fold %d: train=%s~%s, test=%s~%s, "
            "best_epoch=%d/%d, val_loss=%.4f, acc=%.4f, f1=%.4f",
            fold.fold_id,
            fold.train_start.strftime("%Y-%m-%d"),
            fold.train_end.strftime("%Y-%m-%d"),
            fold.test_start.strftime("%Y-%m-%d"),
            fold.test_end.strftime("%Y-%m-%d"),
            info["best_epoch"], info["epochs_run"],
            info["best_val_loss"], acc, f1,
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
    models_root = Path("models/transformer")
    models_root.mkdir(parents=True, exist_ok=True)

    existing = list(models_root.glob("v*"))
    version = f"v{len(existing) + 1:03d}"
    model_dir = models_root / f"{version}_{entry_tf}_{args.start}_{args.end}"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_model.cpu().state_dict(), str(model_dir / "model.pth"))
    joblib.dump(scaler, str(model_dir / "scaler.joblib"))

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_cols, f, indent=2)

    train_meta = {
        "version": version,
        "model_type": "transformer",
        "created": datetime.now().isoformat(),
        "entry_timeframe": entry_tf,
        "timeframes": timeframes,
        "train_period": f"{args.start} ~ {args.end}",
        "walk_forward_folds": len(folds),
        "lookback": lookback,
        "oos_accuracy": round(oos_acc, 4),
        "oos_f1_macro": round(oos_f1, 4),
        "feature_count": len(valid_cols),
        "label_params": label_params,
        "model_arch": {
            "n_features": n_features,
            "d_model": int(transformer_params.get("d_model", 64)),
            "nhead": int(transformer_params.get("nhead", 4)),
            "num_layers": int(transformer_params.get("num_layers", 2)),
            "dim_ff": int(transformer_params.get("dim_ff", 128)),
            "dropout": float(transformer_params.get("dropout", 0.3)),
        },
        "train_hyperparams": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "early_stopping_patience": patience,
        },
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
