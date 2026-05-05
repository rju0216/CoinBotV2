"""디렉토리 충돌 방지 helper (BL-1 Step A).

두 가지 패턴:
1. `next_model_version(models_root)` — `v001`/`v002`/... 모델 디렉토리에서 max + 1.
   v005처럼 의도적으로 비워둔 번호 자동 skip. train_*.py 4개가 사용.
2. `resolve_unique_dir(base_dir)` — 임의 디렉토리 충돌 시 `_1`/`_2`/... postfix.
   백테 결과(BacktestEngine.write_reports), 평가 결과(evaluate_models.py)가 사용.

설계 원칙:
- 파일시스템에 부수효과 없음 (mkdir 안 함). 호출자가 결과 path로 mkdir.
- 단일 사용자 환경 가정 (race condition 미고려).
"""

from __future__ import annotations

import re
from pathlib import Path

def next_model_version(models_root: Path, prefix: str = "v") -> str:
    """`models_root` 안의 `v001`, `v002`, ... 디렉토리들에서 가장 큰 번호 + 1을 반환.

    예: models/lightgbm/v001_..., v002_..., v004_..., v006_..., v007_... → "v008"
    v005, v003 비어있어도 max+1로 자동 skip.

    Args:
        models_root: 모델 루트 디렉토리 (예: Path("models/lightgbm"))
        prefix: 버전 prefix (default "v"). 다른 prefix도 동일 패턴으로 처리.

    Returns:
        "v{NNN}" 형식 (3자리 zero-pad). 기존 디렉토리 0개면 "v001".
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(?:_.*)?$")
    if not models_root.exists():
        return f"{prefix}001"
    max_n = 0
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m is None:
            continue
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if n > max_n:
            max_n = n
    return f"{prefix}{max_n + 1:03d}"


def resolve_unique_dir(base_dir: Path) -> Path:
    """`base_dir`이 존재하지 않으면 그대로 반환. 존재하면 `_1`/`_2`/... postfix.

    예: base_dir = "data/.../260505_backtest_...".
    이미 존재 → "..._1". 그것도 존재 → "..._2".

    Args:
        base_dir: 후보 디렉토리 path

    Returns:
        존재하지 않는 디렉토리 path (호출자가 mkdir).
    """
    if not base_dir.exists():
        return base_dir
    parent = base_dir.parent
    name = base_dir.name
    i = 1
    while True:
        candidate = parent / f"{name}_{i}"
        if not candidate.exists():
            return candidate
        i += 1
