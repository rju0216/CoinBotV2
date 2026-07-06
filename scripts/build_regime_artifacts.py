"""레짐 모델 artifact 사전학습 (offline, O-3) — walk-forward HMM(Gaussian/Student-t) 빌드.

[valid-start, valid-end) 를 retrain-months 마다 잘라 anchored walk-forward 로 학습하고,
각 윈도우 모델을 data/regime_models/window_NN.json 으로 저장한다. RegimeQuantStrategy
(plugins/regime_quant.py)가 이 artifact 를 로드해 백테/페이퍼/라이브에서 사용한다.

실 K 선택 스윕·최종 계수 확정은 D4-5. 여기는 고정 K/τ baseline 빌드 도구(재사용).
emission 은 --emission 으로 선택 (gaussian 기본 / student_t = fat tail, D4-4 O-6).

사용 예 (cmd):
  REM 1) 1h 캔들 먼저 다운로드
  python scripts\\download_history.py --config config\\default.yaml --timeframe 1h ^
    --start 2020-01-01 --end 2023-01-01
  REM 2) Dev walk-forward artifact 빌드 (train anchored 2020-01~, valid 2020-07~2022-12, 3개월 재추정)
  python scripts\\build_regime_artifacts.py --config config\\default.yaml ^
    --candles-start 2020-01-01 --valid-start 2020-07-01 --valid-end 2023-01-01 ^
    --k 3 --tau 0.5 --retrain-months 3
  REM 3) config 의 strategies.active 에 "regime_quant" 추가 후 Dev 백테
  REM    (--end 는 --valid-end 와 일치시킨다 → valid 구간 전체 평가, OOS 끝 누락 방지)
  python -m src.main backtest --config config\\default.yaml --start 2020-07-01 --end 2023-01-01
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.historical import HistoricalDataLoader
from src.strategy.regime.artifact import save_model
from src.strategy.regime.training import make_anchored_windows, walk_forward
from src.utils.config_loader import load_config


async def _load_1h(config: dict, start: str, end: str) -> pd.DataFrame:
    """1h 캔들 [start, end] 로드 (캐시 CSV + API 병합). train anchored 전체 범위."""
    loader = HistoricalDataLoader(config)
    try:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        return await loader.download_range_merged("1h", start_ms, end_ms)
    finally:
        await loader.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="레짐 모델 walk-forward artifact 사전학습 (offline)"
    )
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--candles-start", required=True,
                   help="1h 캔들 로드·train anchored 시작 (예 2020-01-01)")
    p.add_argument("--valid-start", required=True, help="walk-forward valid(=백테) 시작")
    p.add_argument("--valid-end", required=True,
                   help="walk-forward valid(=백테) 끝 (exclusive)")
    p.add_argument("--k", type=int, default=3, help="HMM 상태 수 (실 선택은 D4-5)")
    p.add_argument("--tau", type=float, default=0.5, help="type 경계 임계 |μ|/σ (O-1)")
    p.add_argument("--retrain-months", type=int, default=3, help="재추정 주기(개월)")
    p.add_argument("--out-dir", default="data/regime_models")
    p.add_argument("--n-init", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-range", action="store_true",
        help="비trend 상태를 RANGE 대신 NONE(무매매)로 매핑 → 추세-단독 gen1 (O-7 (가))",
    )
    p.add_argument(
        "--emission", choices=["gaussian", "student_t"], default="gaussian",
        help="HMM emission (gaussian 기본 / student_t = fat tail, D4-4 O-6)",
    )
    p.add_argument(
        "--share-nu", action="store_true",
        help="student_t: 전 상태 공통 ν (기본 = 상태별 ν_k)",
    )
    p.add_argument(
        "--nu-init", type=float, default=10.0,
        help="student_t: ν 초기값 (fit 재현성)",
    )
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    print(f"[1/3] 1h 캔들 로드: {args.candles_start} ~ {args.valid_end}")
    candles = await _load_1h(config, args.candles_start, args.valid_end)
    print(f"      {len(candles)} 봉 로드됨")
    if candles.empty:
        raise SystemExit("캔들이 비었다. download_history.py 로 1h 캔들을 먼저 받아라.")
    if pd.Timestamp(args.valid_start, tz="UTC") <= candles.index[0]:
        raise SystemExit(
            "valid-start 가 캔들 시작 이하 → train 구간(anchored)이 빈다. "
            "candles-start 를 valid-start 보다 충분히 앞당겨라(train warmup)."
        )

    windows = make_anchored_windows(
        args.valid_start, args.valid_end, args.retrain_months
    )
    print(
        f"[2/3] walk-forward {len(windows)} 윈도우 학습 "
        f"(재추정 {args.retrain_months}개월, K={args.k}, τ={args.tau}, "
        f"emission={args.emission})"
    )
    models = walk_forward(
        candles, windows, args.k, args.tau,
        n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
        enable_range=not args.no_range,
        emission_kind=args.emission,
        emission_params={"share_nu": args.share_nu, "nu_init": args.nu_init},
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 항상 이전 walk-forward 세트 정리 → 윈도우 수 감소 시 stale window_NN 잔존 차단
    # (소비자 _load_models 는 window_*.json 전체를 로드하므로 잔존은 stale 모델 오염).
    for f in out.glob("window_*.json"):
        f.unlink()
    print(f"[3/3] 저장 → {out}")
    for i, m in enumerate(models):
        save_model(m, out / f"window_{i:02d}.json")
        print(f"      window_{i:02d}: valid {m.valid_period[0]} ~ {m.valid_period[1]}")

    print(
        f"완료: {len(models)} artifact. 백테: config 의 strategies.active 에 "
        f"'regime_quant' 추가 후 → python -m src.main backtest "
        f"--start {args.valid_start} --end {args.valid_end}"
    )


if __name__ == "__main__":
    asyncio.run(_main())
