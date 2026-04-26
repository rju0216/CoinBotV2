"""PPO 학습 파이프라인.

ML/DL 학습과 다른 점:
- walk-forward 분할 안 함 (RL은 단일 학습 + episode 랜덤 샘플링)
- TradingEnv (Gym) 환경에서 stable-baselines3 PPO 학습
- 평가는 백테 엔진에서 별도 OOS 기간으로 수행

Usage:
    python scripts/train_ppo.py --config config/rl_ppo.yaml \
        --start 2020-01-01 --end 2024-12-31

출력: models/ppo/v{NNN}_{tf}_{start}_{end}/
  - model.zip          SB3 PPO 정책 (자동 .zip)
  - scaler.joblib      StandardScaler 직렬화
  - feature_names.json 피처 컬럼명 목록
  - train_meta.json    학습 메타
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.ml.env_trading import TradingEnv  # noqa: E402
from src.ml.feature_pipeline import build_features  # noqa: E402
from src.strategy.features import get_feature_names  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="PPO 학습 (RL)")
    parser.add_argument("--config", required=True, help="config YAML 경로")
    parser.add_argument("--start", required=True, help="학습 데이터 시작 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="학습 데이터 종료 (YYYY-MM-DD)")
    parser.add_argument("--force-features", action="store_true", help="피처 캐시 무시, 재생성")
    args = parser.parse_args()

    config = load_config(args.config)
    strategy_cfg = config["rl_ppo"]
    train_cfg = strategy_cfg.get("train", {})

    entry_tf = strategy_cfg.get("entry_timeframe", "15m")
    timeframes = strategy_cfg.get("required_timeframes", [entry_tf])
    if entry_tf not in timeframes:
        timeframes = [entry_tf] + timeframes

    lookback = int(strategy_cfg.get("lookback", 60))
    episode_length = int(train_cfg.get("episode_length", 2000))
    total_timesteps = int(train_cfg.get("total_timesteps", 200_000))
    fee_pct = float(train_cfg.get("fee_pct", 0.0005))
    ppo_params = dict(train_cfg.get("ppo_params", {}))

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── 1. 피처 생성 ──
    logger.info("피처 생성 중... (TF: %s, lookback=%d)", timeframes, lookback)
    features = await build_features(
        config, timeframes, entry_tf, start_ms, end_ms,
        force=args.force_features,
    )

    # ── 2. 캔들 로드 (가격 추출용) ──
    loader = HistoricalDataLoader(config)
    df = await loader.download_range_merged(entry_tf, start_ms, end_ms)

    # ── 3. 피처 컬럼 정렬 + 유효 행 추출 ──
    feature_names = get_feature_names(
        entry_tf, [t for t in timeframes if t != entry_tf]
    )
    valid_cols = [c for c in feature_names if c in features.columns]
    if len(valid_cols) < len(feature_names):
        logger.warning(
            "피처 컬럼 누락: 기대 %d, 가용 %d", len(feature_names), len(valid_cols)
        )

    X_full = features[valid_cols]
    # NaN 행 제거 + 가격 정렬
    valid_mask = X_full.notna().all(axis=1) & X_full.index.isin(df.index)
    X_clean = X_full.loc[valid_mask]
    prices = df.loc[X_clean.index, "close"].values
    n_features = len(valid_cols)
    logger.info("학습 데이터: %d행, %d피처", len(X_clean), n_features)

    # ── 4. Scaler fit (전체 데이터 1회 fit — B-2와 동일) ──
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_clean.values).astype(np.float32)
    logger.info("StandardScaler fit 완료")

    # ── 5. Gym 환경 생성 ──
    def make_env():
        return TradingEnv(
            features_scaled=features_scaled,
            prices=prices,
            lookback=lookback,
            episode_length=episode_length,
            fee_pct=fee_pct,
        )

    env = DummyVecEnv([make_env])
    logger.info(
        "TradingEnv: lookback=%d, episode_length=%d, fee_pct=%.4f",
        lookback, episode_length, fee_pct,
    )

    # ── 6. PPO 학습 ──
    logger.info("PPO 학습 시작 (total_timesteps=%d)", total_timesteps)
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        verbose=1,
        seed=42,
        **ppo_params,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    logger.info("PPO 학습 완료")

    # ── 7. 모델 저장 ──
    models_root = Path("models/ppo")
    models_root.mkdir(parents=True, exist_ok=True)

    existing = list(models_root.glob("v*"))
    version = f"v{len(existing) + 1:03d}"
    model_dir = models_root / f"{version}_{entry_tf}_{args.start}_{args.end}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # SB3는 .zip 확장자 자동 추가
    model.save(str(model_dir / "model"))
    joblib.dump(scaler, str(model_dir / "scaler.joblib"))

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(valid_cols, f, indent=2)

    train_meta = {
        "version": version,
        "model_type": "ppo",
        "created": datetime.now().isoformat(),
        "entry_timeframe": entry_tf,
        "timeframes": timeframes,
        "train_period": f"{args.start} ~ {args.end}",
        "lookback": lookback,
        "feature_count": len(valid_cols),
        "train_hyperparams": {
            "episode_length": episode_length,
            "total_timesteps": total_timesteps,
            "fee_pct": fee_pct,
            **ppo_params,
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
