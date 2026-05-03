"""모델 통합 평가 스크립트 (Phase E-2).

5개 모델 × 6개 OOS 분할 자동 백테 + 베이스라인(Buy & Hold + example_macross) + 결과 수집.

분할 매트릭스:
  - 분할 1 (v001, 5년 학습): OOS 2025-01-01 ~ 2025-12-31  (1년)
  - Anchored A (v003, 3년): OOS 2023-01-01 ~ 2023-12-31    (1년)
  - Anchored B (v004, 4년): OOS 2024-01-01 ~ 2024-12-31    (1년)
  - Expanding 2 (v004, 4년): OOS 2024-01-01 ~ 2025-12-31   (2년)
  - Expanding 3 (v003, 3년): OOS 2023-01-01 ~ 2025-12-31   (3년)
  - Expanding 4 (v002, 2년): OOS 2022-01-01 ~ 2025-12-31   (4년)

출력: data/backtest_reports/00_Working/eval_{YYMMDD}/
  - {strategy}_{split}/    각 모델 백테 결과 (5×6 = 30개)
  - macross_{split}/       example_macross 백테 결과 (6개)
  - buy_and_hold.json      Buy & Hold 결과 (메모리에 모아 한 파일에)
  - comparison.csv         종합 비교 표

Usage:
    # E-2-2 (전체 자동 실행):
    python scripts/evaluate_models.py --mode full

    # 특정 모델만:
    python scripts/evaluate_models.py --mode single --strategy ml_lightgbm --split A

    # 결과 수집만 (이미 백테 끝났을 때):
    python scripts/evaluate_models.py --mode collect

E-2-3/E-2-4에서 슬리피지·calibration·통계 분석 추가 예정.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

# Phase E-2-2-OPT Step 3: PyTorch/MKL thread thrashing 방지.
# 4 워커 × N thread = 같은 코어 경합 → 오히려 느려짐. spawn 자식이 상속하도록
# import 전에 설정.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.backtest.engine import BacktestEngine  # noqa: E402
from src.data.historical import HistoricalDataLoader  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

DEFAULT_CONFIG = "config/default.yaml"
BUY_AND_HOLD_FILE = "buy_and_hold.json"
_NUM_WORKERS = 4  # Phase E-2-2-OPT Step 3 multiprocessing.Pool 워커 수 (사안 D)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Phase E-2-2-OPT Step 3: worker가 eval_root을 알아야 하지만 BacktestSpec dataclass를
# 변경하지 않기 위해 Pool initializer로 모듈 글로벌에 주입.
_WORKER_EVAL_ROOT: Path | None = None

REPORT_BASE = Path("data/backtest_reports/00_Working")


@dataclass
class BacktestSpec:
    """단일 백테 명세."""
    strategy: str            # "ml_lightgbm", "ml_xgboost", ...
    config_path: str         # config YAML 경로
    model_dir: str           # 모델 디렉토리 (model_path 오버라이드용)
    split_id: str            # "1", "A", "B", "Exp2", "Exp3", "Exp4"
    oos_start: str           # "2025-01-01"
    oos_end: str             # "2025-12-31"
    # Phase E-2-3 Step 1 (슬리피지 sensitivity, I-B010): None이면 config 기본값 사용,
    # 값이면 accounting.slippage_pct 오버라이드. 미래 다른 sensitivity 차원 추가 시
    # 같은 패턴으로 fee/leverage 필드 추가 가능.
    slippage_pct: float | None = None
    # Phase E-2-3 Step 2 (calibration, I-B009): None이면 config 기본값 사용("none"),
    # "platt" 또는 "isotonic"이면 plugin이 자동 calibrator 로드/적용.
    calibration_method: str | None = None

    @property
    def label(self) -> str:
        """결과 디렉토리 이름. sensitivity/calibration 차원 지정 시 suffix 추가."""
        base = f"{self.strategy}_{self.split_id}"
        if self.slippage_pct is not None:
            base += f"_slip{self.slippage_pct:.4f}"
        if self.calibration_method is not None:
            base += f"_cal{self.calibration_method}"
        return base


# ─── 모델 × 분할 매트릭스 ───

STRATEGIES = {
    "ml_lightgbm":   "config/ml_lightgbm.yaml",
    "ml_xgboost":    "config/ml_xgboost.yaml",
    "dl_lstm":       "config/dl_lstm.yaml",
    "dl_transformer": "config/dl_transformer.yaml",
    "rl_ppo":        "config/rl_ppo.yaml",
}

# 모델 디렉토리 폴더명 패턴: v{NNN}_15m_2020-01-01_{end}
MODEL_VERSIONS = {
    "v001": "v001_15m_2020-01-01_2024-12-31",  # 5년
    "v002": "v002_15m_2020-01-01_2021-12-31",  # 2년
    "v003": "v003_15m_2020-01-01_2022-12-31",  # 3년
    "v004": "v004_15m_2020-01-01_2023-12-31",  # 4년
}

# 모델 종류 → 디렉토리 prefix (xgboost는 ml_xgboost가 아닌 xgboost)
STRATEGY_DIR_PREFIX = {
    "ml_lightgbm": "lightgbm",
    "ml_xgboost":  "xgboost",
    "dl_lstm":     "lstm",
    "dl_transformer": "transformer",
    "rl_ppo":      "ppo",
}

# (split_id, model_version, oos_start, oos_end)
SPLIT_DEFINITIONS = [
    ("1",    "v001", "2025-01-01", "2025-12-31"),  # Anchored C / Expanding 1
    ("A",    "v003", "2023-01-01", "2023-12-31"),  # Anchored A
    ("B",    "v004", "2024-01-01", "2024-12-31"),  # Anchored B
    ("Exp2", "v004", "2024-01-01", "2025-12-31"),  # Expanding 2 (2년 OOS)
    ("Exp3", "v003", "2023-01-01", "2025-12-31"),  # Expanding 3 (3년 OOS)
    ("Exp4", "v002", "2022-01-01", "2025-12-31"),  # Expanding 4 (4년 OOS)
]


def build_specs(strategies: list[str] | None = None) -> list[BacktestSpec]:
    """모델 × 분할 매트릭스 → BacktestSpec 리스트.

    strategies가 None이면 전체 5개. 일부만 받으면 그 모델만.
    """
    target_strategies = strategies or list(STRATEGIES.keys())
    specs: list[BacktestSpec] = []
    for strat in target_strategies:
        if strat not in STRATEGIES:
            logger.warning("알 수 없는 strategy: %s — 스킵", strat)
            continue
        config_path = STRATEGIES[strat]
        dir_prefix = STRATEGY_DIR_PREFIX[strat]
        for split_id, version, oos_start, oos_end in SPLIT_DEFINITIONS:
            model_dir = f"models/{dir_prefix}/{MODEL_VERSIONS[version]}"
            specs.append(BacktestSpec(
                strategy=strat,
                config_path=config_path,
                model_dir=model_dir,
                split_id=split_id,
                oos_start=oos_start,
                oos_end=oos_end,
            ))
    return specs


# Phase E-2-3 Step 1: 슬리피지 sensitivity 매트릭스 (I-B010 — 사안 G/H 결정 반영)
SENSITIVITY_TARGET_SPLITS = ["1", "Exp4"]  # 양 끝점: 1년 OOS + 4년 OOS
SENSITIVITY_SLIPPAGES = [0.0, 0.0002, 0.0005, 0.001]  # 0% / 0.02% / 0.05% / 0.1%


# Phase E-2-3 Step 3: Calibration 백테 매트릭스 (사안 H — 분할 1만)
CALIBRATION_TARGET_SPLIT = "1"  # v001 5년 학습, OOS 2025-01-01 ~ 2025-12-31
CALIBRATION_METHODS = ["platt", "isotonic"]
# PPO 제외 — 정책 모델이라 confidence calibration 무의미
CALIBRATION_TARGET_STRATEGIES = ["ml_lightgbm", "ml_xgboost", "dl_lstm", "dl_transformer"]


def build_calibration_specs() -> list[BacktestSpec]:
    """Calibration 백테 매트릭스: 4 분류 모델 × 분할 1 × 2 알고리즘 = 8 specs.

    raw baseline은 별도 (calibration_method=None) — eval_260503_sensitivity의
    *_slip0.0000 결과 또는 별도 단일 백테 사용.
    """
    specs: list[BacktestSpec] = []
    for strat in CALIBRATION_TARGET_STRATEGIES:
        config_path = STRATEGIES[strat]
        dir_prefix = STRATEGY_DIR_PREFIX[strat]
        for split_id, version, oos_start, oos_end in SPLIT_DEFINITIONS:
            if split_id != CALIBRATION_TARGET_SPLIT:
                continue
            model_dir = f"models/{dir_prefix}/{MODEL_VERSIONS[version]}"
            for method in CALIBRATION_METHODS:
                specs.append(BacktestSpec(
                    strategy=strat,
                    config_path=config_path,
                    model_dir=model_dir,
                    split_id=split_id,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    calibration_method=method,
                ))
    return specs


def build_sensitivity_specs() -> list[BacktestSpec]:
    """슬리피지 sensitivity 백테 매트릭스: 5 모델 × 2 분할(끝점) × 4 슬리피지 = 40 specs."""
    specs: list[BacktestSpec] = []
    for strat in STRATEGIES:
        config_path = STRATEGIES[strat]
        dir_prefix = STRATEGY_DIR_PREFIX[strat]
        for split_id, version, oos_start, oos_end in SPLIT_DEFINITIONS:
            if split_id not in SENSITIVITY_TARGET_SPLITS:
                continue
            model_dir = f"models/{dir_prefix}/{MODEL_VERSIONS[version]}"
            for slip in SENSITIVITY_SLIPPAGES:
                specs.append(BacktestSpec(
                    strategy=strat,
                    config_path=config_path,
                    model_dir=model_dir,
                    split_id=split_id,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    slippage_pct=slip,
                ))
    return specs


def _override_model_path(config: dict[str, Any], strategy: str, model_dir: str) -> dict[str, Any]:
    """config의 strategy 섹션에서 model_path를 명시적 디렉토리로 오버라이드."""
    if strategy in config and isinstance(config[strategy], dict):
        config[strategy]["model_path"] = model_dir
    return config


def _override_slippage(config: dict[str, Any], slippage_pct: float | None) -> dict[str, Any]:
    """Phase E-2-3 Step 1: spec.slippage_pct가 None이 아니면 accounting.slippage_pct 오버라이드.

    슬리피지는 FeeModel의 per_side_rate에 합산되어 진입/청산 비용으로 차감됨.
    BacktestEngine은 이 오버라이드를 인식할 필요 없음 — config.from_config에서 자동 적용.
    """
    if slippage_pct is not None:
        config.setdefault("accounting", {})["slippage_pct"] = float(slippage_pct)
    return config


def _override_calibration(
    config: dict[str, Any], strategy: str, calibration_method: str | None,
) -> dict[str, Any]:
    """Phase E-2-3 Step 2: spec.calibration_method가 None이 아니면 strategy 섹션 오버라이드.

    plugin이 _ensure_model에서 자동으로 calibrator_<method>.joblib 로드/적용.
    """
    if calibration_method is not None and strategy in config and isinstance(config[strategy], dict):
        config[strategy]["calibration_method"] = str(calibration_method)
    return config


async def run_one_backtest(spec: BacktestSpec, eval_root: Path) -> tuple[Path, dict] | None:
    """단일 백테 실행 → (결과 디렉토리, metrics dict) 반환.

    실패 시 None.
    """
    logger.info("[BACKTEST] %s | %s | OOS %s ~ %s",
                spec.strategy, spec.split_id, spec.oos_start, spec.oos_end)

    config = load_config(spec.config_path)
    config = _override_model_path(config, spec.strategy, spec.model_dir)
    config = _override_slippage(config, spec.slippage_pct)
    config = _override_calibration(config, spec.strategy, spec.calibration_method)

    engine = BacktestEngine(config, start=spec.oos_start, end=spec.oos_end)
    out_dir = None
    try:
        await engine.initialize()
        await engine.run()
        result = await engine.get_result()
        # eval_root 하위에 spec.label 디렉토리로 저장
        out_dir = engine.write_reports(
            config_path=spec.config_path,
            out_root=eval_root / spec.label,
        )
    except Exception:
        logger.exception("백테 실패: %s", spec.label)
        return None
    finally:
        await engine.shutdown()

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error("metrics.json 없음: %s", metrics_path)
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    logger.info("[OK] %s → %s (trades=%d, return=%.2f%%)",
                spec.label, out_dir,
                metrics["integrated"]["total_trades"],
                metrics["integrated"]["total_return_pct"])
    return out_dir, metrics


def collect_metrics(eval_root: Path, specs: list[BacktestSpec]) -> pd.DataFrame:
    """eval_root 하위의 모든 metrics.json 수집 → 비교 DataFrame."""
    rows: list[dict] = []
    for spec in specs:
        # write_reports 내부 구조: {eval_root/spec.label}/{config_name}/metrics.json
        config_name = Path(spec.config_path).stem
        metrics_path = eval_root / spec.label / config_name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("metrics 없음: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        integ = m.get("integrated", {})
        rows.append({
            "strategy": spec.strategy,
            "split": spec.split_id,
            "slippage_pct": spec.slippage_pct,
            "calibration_method": spec.calibration_method,
            "oos_start": spec.oos_start,
            "oos_end": spec.oos_end,
            "total_trades": integ.get("total_trades"),
            "win_rate_pct": integ.get("win_rate_pct"),
            "total_return_pct": integ.get("total_return_pct"),
            "max_drawdown_pct": integ.get("max_drawdown_pct"),
            "profit_factor": integ.get("profit_factor"),
            "avg_win": integ.get("avg_win"),
            "avg_loss": integ.get("avg_loss"),
        })
    return pd.DataFrame(rows)


async def run_all(specs: list[BacktestSpec], eval_root: Path) -> None:
    """specs 전체를 순차 실행 (single-process 경로 — debugging용 유지)."""
    eval_root.mkdir(parents=True, exist_ok=True)
    logger.info("총 %d개 백테 시작 (출력: %s)", len(specs), eval_root)
    n_success = 0
    for i, spec in enumerate(specs, 1):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i, len(specs), spec.label)
        result = await run_one_backtest(spec, eval_root)
        if result is not None:
            n_success += 1
    logger.info("=" * 60)
    logger.info("백테 완료: 성공 %d / 전체 %d", n_success, len(specs))


# ─── Phase E-2-2-OPT Step 3: 캔들 캐시 워밍업 + multiprocessing.Pool ───

async def _warmup_candle_cache(specs: list[BacktestSpec]) -> None:
    """unique OOS 기간 × required TF에 대해 캔들 다운로드/CSV 캐시 1회 보장 (사안 F).

    이게 없으면 4 워커가 동시에 같은 CSV에 write → race condition 가능.
    캐시 hit이면 수 초, miss면 다운로드 — 비용은 크지 않음.
    """
    splits = _unique_splits(specs)
    base_config = load_config(DEFAULT_CONFIG)
    timeframes = ["15m", "1h", "4h"]
    loader = HistoricalDataLoader(base_config)
    try:
        for split_id, (oos_start, oos_end) in splits.items():
            start_ms = int(pd.Timestamp(oos_start, tz="UTC").timestamp() * 1000)
            end_ms = int(pd.Timestamp(oos_end, tz="UTC").timestamp() * 1000)
            for tf in timeframes:
                df = await loader.download_range_merged(tf, start_ms, end_ms)
                logger.info(
                    "[Warmup] split=%s tf=%s rows=%d", split_id, tf, len(df)
                )
    finally:
        await loader.close()


def _init_worker(eval_root_str: str) -> None:
    """Pool worker initializer — eval_root 주입 + thread 환경 재확인."""
    global _WORKER_EVAL_ROOT
    _WORKER_EVAL_ROOT = Path(eval_root_str)
    # main에서 spawn 전 환경변수 설정했으나 자식에서도 명시적 보장
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_one_sync(spec: BacktestSpec) -> str | None:
    """worker가 단일 spec 백테 실행 (asyncio.run으로 비동기 wrapping).

    pickle 호환을 위해 모듈 top level에 정의 (Windows spawn 호환).
    eval_root는 _init_worker에서 주입된 모듈 글로벌 사용.
    """
    if _WORKER_EVAL_ROOT is None:
        logger.error("worker eval_root 미초기화 — Pool initializer 누락")
        return None
    try:
        result = asyncio.run(run_one_backtest(spec, _WORKER_EVAL_ROOT))
    except Exception:
        logger.exception("worker 백테 실패: %s", spec.label)
        return None
    return spec.label if result is not None else None


def run_all_parallel(specs: list[BacktestSpec], eval_root: Path) -> None:
    """multiprocessing.Pool(N=_NUM_WORKERS)로 specs 병렬 실행.

    chunksize=1 — spec마다 OOS 길이 차이 (1년~4년) 큼 → imap_unordered가
    부하 분산. 베이스라인은 별도 main process 순차 (run_baselines).
    """
    eval_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "총 %d개 백테 병렬 시작 (워커 %d, 출력: %s)",
        len(specs), _NUM_WORKERS, eval_root,
    )
    n_success = 0
    with Pool(
        processes=_NUM_WORKERS,
        initializer=_init_worker,
        initargs=(str(eval_root),),
    ) as pool:
        for i, label in enumerate(
            pool.imap_unordered(_run_one_sync, specs, chunksize=1), 1
        ):
            if label is not None:
                n_success += 1
                logger.info("[%d/%d] [OK] %s", i, len(specs), label)
            else:
                logger.warning("[%d/%d] [FAIL] (worker 반환 None)", i, len(specs))
    logger.info("=" * 60)
    logger.info("병렬 백테 완료: 성공 %d / 전체 %d", n_success, len(specs))


# ─── 베이스라인: Buy & Hold ───

async def compute_buy_and_hold(
    config: dict[str, Any],
    split_id: str,
    oos_start: str,
    oos_end: str,
) -> dict | None:
    """OOS 기간 첫 봉 close에 매수, 마지막 봉 close에 청산.

    equity curve 시뮬레이션으로 MDD 계산. 수수료 0 가정 (1회 거래라 영향 미미).
    """
    loader = HistoricalDataLoader(config)
    start_ms = int(pd.Timestamp(oos_start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(oos_end, tz="UTC").timestamp() * 1000)

    try:
        df = await loader.download_range_merged("15m", start_ms, end_ms)
    finally:
        await loader.close()

    if df.empty:
        logger.warning("Buy & Hold: %s 캔들 없음", split_id)
        return None

    closes = df["close"].astype(float)
    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])

    # equity curve: 첫 close 매수 후 가격 변화 추적
    initial = float(config.get("paper", {}).get("initial_balance", 10000.0))
    equity = closes / first_close * initial
    peak = equity.cummax()
    drawdown = (peak - equity) / peak * 100
    max_dd = float(drawdown.max())

    total_return_pct = (last_close / first_close - 1) * 100

    return {
        "strategy": "buy_and_hold",
        "split": split_id,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "total_trades": 1,
        "win_rate_pct": 100.0 if total_return_pct > 0 else 0.0,
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": None,
        "avg_win": None,
        "avg_loss": None,
    }


# ─── 베이스라인: example_macross ───

async def run_baseline_macross(
    split_id: str,
    oos_start: str,
    oos_end: str,
    eval_root: Path,
) -> tuple[Path, dict] | None:
    """example_macross 백테. config 임시 생성 → 백테 → 임시 파일 삭제."""
    logger.info("[BASELINE macross] %s | OOS %s ~ %s", split_id, oos_start, oos_end)

    base_config = load_config(DEFAULT_CONFIG)
    # active를 example_macross로 임시 변경
    base_config.setdefault("strategies", {})["active"] = ["example_macross"]

    # 임시 config 파일 (config/_eval_macross_{split_id}.yaml)
    temp_config_path = Path(f"config/_eval_macross_{split_id}.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base_config, f, allow_unicode=True, sort_keys=False)

    out_dir = None
    try:
        engine = BacktestEngine(base_config, start=oos_start, end=oos_end)
        try:
            await engine.initialize()
            await engine.run()
            await engine.get_result()
            out_dir = engine.write_reports(
                config_path=str(temp_config_path),
                out_root=eval_root / f"macross_{split_id}",
            )
        except Exception:
            logger.exception("example_macross 백테 실패: %s", split_id)
        finally:
            await engine.shutdown()
    finally:
        # 임시 config 정리
        if temp_config_path.exists():
            temp_config_path.unlink()

    if out_dir is None:
        return None

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error("metrics.json 없음: %s", metrics_path)
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    logger.info("[OK] macross_%s → %s (trades=%d, return=%.2f%%)",
                split_id, out_dir,
                metrics["integrated"]["total_trades"],
                metrics["integrated"]["total_return_pct"])
    return out_dir, metrics


# ─── 베이스라인 통합 실행 ───

def _unique_splits(specs: list[BacktestSpec]) -> dict[str, tuple[str, str]]:
    """모델 spec에서 unique (split_id → (oos_start, oos_end)) 추출.

    같은 split_id는 동일 OOS 기간이라 가정 (build_specs가 보장).
    """
    seen: dict[str, tuple[str, str]] = {}
    for s in specs:
        seen.setdefault(s.split_id, (s.oos_start, s.oos_end))
    return seen


async def run_baselines(
    specs: list[BacktestSpec], eval_root: Path
) -> list[dict]:
    """베이스라인 전체 실행. Buy & Hold 결과는 메모리 → 파일 저장."""
    splits = _unique_splits(specs)
    logger.info("=" * 60)
    logger.info("베이스라인: Buy & Hold + example_macross × %d 분할", len(splits))

    base_config = load_config(DEFAULT_CONFIG)

    # 1) Buy & Hold (단순 계산, 6개)
    bh_results: list[dict] = []
    for split_id, (oos_start, oos_end) in splits.items():
        bh = await compute_buy_and_hold(base_config, split_id, oos_start, oos_end)
        if bh:
            bh_results.append(bh)
            logger.info("[OK] buy_and_hold_%s: return=%.2f%%, MDD=%.2f%%",
                        split_id, bh["total_return_pct"], bh["max_drawdown_pct"])

    # Buy & Hold 결과 파일 저장 (mode=collect에서 재사용)
    eval_root.mkdir(parents=True, exist_ok=True)
    with open(eval_root / BUY_AND_HOLD_FILE, "w") as f:
        json.dump(bh_results, f, indent=2)

    # 2) example_macross 백테 (6개)
    n_macross = 0
    for split_id, (oos_start, oos_end) in splits.items():
        result = await run_baseline_macross(split_id, oos_start, oos_end, eval_root)
        if result is not None:
            n_macross += 1

    logger.info("베이스라인 완료: B&H %d, macross %d / %d",
                len(bh_results), n_macross, len(splits))
    return bh_results


def collect_macross_metrics(
    eval_root: Path, splits: dict[str, tuple[str, str]]
) -> list[dict]:
    """example_macross 결과 metrics.json 수집."""
    rows: list[dict] = []
    for split_id, (oos_start, oos_end) in splits.items():
        # write_reports 내부 구조: {macross_{split_id}}/{config_name}/metrics.json
        # config_name = "_eval_macross_{split_id}" (Path stem)
        config_name = f"_eval_macross_{split_id}"
        metrics_path = eval_root / f"macross_{split_id}" / config_name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("macross metrics 없음: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        integ = m.get("integrated", {})
        rows.append({
            "strategy": "example_macross",
            "split": split_id,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "total_trades": integ.get("total_trades"),
            "win_rate_pct": integ.get("win_rate_pct"),
            "total_return_pct": integ.get("total_return_pct"),
            "max_drawdown_pct": integ.get("max_drawdown_pct"),
            "profit_factor": integ.get("profit_factor"),
            "avg_win": integ.get("avg_win"),
            "avg_loss": integ.get("avg_loss"),
        })
    return rows


def load_buy_and_hold(eval_root: Path) -> list[dict]:
    """저장된 Buy & Hold 결과 로드. 없으면 빈 리스트."""
    bh_path = eval_root / BUY_AND_HOLD_FILE
    if not bh_path.exists():
        return []
    with open(bh_path) as f:
        return json.load(f)


def save_comparison(df: pd.DataFrame, eval_root: Path) -> Path:
    """비교 표 CSV 저장."""
    eval_root.mkdir(parents=True, exist_ok=True)
    csv_path = eval_root / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("비교 표 저장: %s (%d rows)", csv_path, len(df))
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="모델 통합 평가 (Phase E-2)")
    parser.add_argument(
        "--mode",
        choices=["full", "single", "collect", "sensitivity", "calibration"],
        default="full",
        help=(
            "full: 모든 모델×분할 백테 / single: 특정 조합 / collect: 결과 수집만 / "
            "sensitivity: 슬리피지 sensitivity (Phase E-2-3 Step 1) / "
            "calibration: Calibration 백테 (Phase E-2-3 Step 3)"
        ),
    )
    parser.add_argument("--strategy", help="--mode single 시 특정 strategy 이름")
    parser.add_argument("--split", help="--mode single 시 특정 split id")
    parser.add_argument(
        "--eval-date",
        default=datetime.now().strftime("%y%m%d"),
        help="결과 디렉토리 날짜 prefix (기본: 오늘)",
    )
    args = parser.parse_args()

    eval_root = REPORT_BASE / f"eval_{args.eval_date}"

    if args.mode == "full":
        import time as _time
        _t0 = _time.perf_counter()
        specs = build_specs()

        # Phase E-2-2-OPT Step 3: 모델 백테는 multiprocessing.Pool 병렬,
        # 베이스라인(B&H + macross)은 main process 순차 (간단/빠름).
        # 사전 캔들 캐시 워밍업으로 워커 race condition 방지 (사안 F).
        asyncio.run(_warmup_candle_cache(specs))
        run_all_parallel(specs, eval_root)
        asyncio.run(run_baselines(specs, eval_root))

        # 종합 비교 표
        splits = _unique_splits(specs)
        df_models = collect_metrics(eval_root, specs)
        df_macross = pd.DataFrame(collect_macross_metrics(eval_root, splits))
        df_bh = pd.DataFrame(load_buy_and_hold(eval_root))
        df = pd.concat([df_models, df_macross, df_bh], ignore_index=True)
        save_comparison(df, eval_root)
        _elapsed = _time.perf_counter() - _t0
        logger.info("=" * 60)
        logger.info(
            "[Phase E-2-2-OPT Step 3] 전체 wall time: %.1f sec (%.2f h)",
            _elapsed, _elapsed / 3600,
        )
    elif args.mode == "single":
        if not args.strategy or not args.split:
            logger.error("--mode single은 --strategy와 --split 필요")
            return 2
        all_specs = build_specs([args.strategy])
        specs = [s for s in all_specs if s.split_id == args.split]
        if not specs:
            logger.error("매칭되는 spec 없음: strategy=%s split=%s",
                         args.strategy, args.split)
            return 2
        asyncio.run(run_all(specs, eval_root))
    elif args.mode == "sensitivity":
        # Phase E-2-3 Step 1: 슬리피지 sensitivity (I-B010).
        # 5 모델 × 분할{1, Exp4} × 슬리피지{0%, 0.02%, 0.05%, 0.1%} = 40 백테.
        # 출력 폴더는 baseline 보존을 위해 별도 권장 (예: --eval-date 260503_sensitivity).
        import time as _time
        _t0 = _time.perf_counter()
        specs = build_sensitivity_specs()
        asyncio.run(_warmup_candle_cache(specs))
        run_all_parallel(specs, eval_root)
        df = collect_metrics(eval_root, specs)
        save_comparison(df, eval_root)
        _elapsed = _time.perf_counter() - _t0
        logger.info("=" * 60)
        logger.info(
            "[Phase E-2-3 Step 1] sensitivity wall time: %.1f sec (%.2f h)",
            _elapsed, _elapsed / 3600,
        )
    elif args.mode == "calibration":
        # Phase E-2-3 Step 3: Calibration 백테 (I-B009).
        # 4 분류 모델 × 분할 1 × {Platt, Isotonic} = 8 백테. PPO 제외.
        # raw baseline은 eval_260503_sensitivity/*_slip0.0000/ 사용.
        import time as _time
        _t0 = _time.perf_counter()
        specs = build_calibration_specs()
        asyncio.run(_warmup_candle_cache(specs))
        run_all_parallel(specs, eval_root)
        df = collect_metrics(eval_root, specs)
        save_comparison(df, eval_root)
        _elapsed = _time.perf_counter() - _t0
        logger.info("=" * 60)
        logger.info(
            "[Phase E-2-3 Step 3] calibration wall time: %.1f sec (%.2f h)",
            _elapsed, _elapsed / 3600,
        )
    elif args.mode == "collect":
        specs = build_specs()
        splits = _unique_splits(specs)
        df_models = collect_metrics(eval_root, specs)
        df_macross = pd.DataFrame(collect_macross_metrics(eval_root, splits))
        df_bh = pd.DataFrame(load_buy_and_hold(eval_root))
        df = pd.concat([df_models, df_macross, df_bh], ignore_index=True)
        save_comparison(df, eval_root)
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
