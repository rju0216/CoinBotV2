"""MTF 피처 조립 — 결정TF X + 상위TF 완성봉 피처 concat (R4.2 역할1, 설계 §9).

1h 튜닝저항·15m 강엣지(R4.1) 위에서, **상위맥락이 결정TF 통계엣지를 더 강화하나**를
본다(역할1=관문1). 상위TF 피처는 `forward_fill_completed` 로 **완성봉만** 결정TF index 에
인과 매핑(진행중 봉 배제 = 미래누수 차단, 기둥2). 상위TF 열은 접두어 **mtf{tf}_**
(결정TF 기본열과 충돌 방지 + 출처 명시). 정규화는 harness 소유(여기 안 함, F-6).

시작부 상위TF 워밍업·미완성 구간은 NaN → harness 가 폴드 슬라이스서 드롭(F-10).
상위TF 종료일이 결정TF ≥ 이면 tail 커버리지 손실 없음(F-4, 호출부가 보장·seam 검증).
"""

from __future__ import annotations

import pandas as pd

from src.data.historical import TF_MS
from src.research.data.loader import forward_fill_completed
from src.research.features.build import build_features


def build_mtf_features(
    decision_df: pd.DataFrame,
    higher: list[tuple[str, pd.DataFrame]],
    params: dict[str, dict] | None = None,
    higher_params: dict[str, dict[str, dict]] | None = None,
) -> pd.DataFrame:
    """결정TF OHLCV → 결정TF X(8열) + 각 상위TF 완성봉 피처(8열, mtf{tf}_ 접두어) concat.

    Args:
        decision_df: 결정TF OHLCV(감사 clean).
        higher: [(tf, higher_df), ...] 상위TF OHLCV(각 감사 clean, 종료일 ≥ 결정TF 권장).
        params: 결정TF build_features 창 오버라이드.
        higher_params: {tf: build_features 창 오버라이드} 상위TF별.

    반환 X (index = decision_df.index): 결정 8열 + 상위TF·8열들. NaN 은 harness 드롭.
    """
    base = build_features(decision_df, params)
    dec_end = decision_df.index[-1]
    frames: list[pd.DataFrame] = [base]
    for tf, hdf in higher:
        # F-4 가드(fresh-eyes MEDIUM①): 상위TF 데이터가 결정TF tail 을 덮지 못하면
        # forward_fill_completed 가 마지막 완성봉 값을 tail 전체에 무한 fill(=stale, NaN 아님
        # → harness NaN 위생 우회 → tail val 이 stale 피처로 채점). 완성봉이 tail 미달이면 하드페일.
        last_completion = hdf.index[-1] + pd.Timedelta(milliseconds=TF_MS[tf])
        if last_completion < dec_end:
            raise ValueError(
                f"{tf} 마지막 완성봉 {last_completion} < 결정TF 종료 {dec_end} — "
                "tail stale-fill 위험(F-4). 상위TF 데이터가 결정TF tail 을 덮어야 함")
        hp = higher_params.get(tf) if higher_params else None
        hx = build_features(hdf, hp).add_prefix(f"mtf{tf}_")
        # 완성봉만 결정TF index 로 인과 ff (DataFrame → DataFrame, 열 접두어 보존)
        frames.append(forward_fill_completed(hx, base.index, tf))
    return pd.concat(frames, axis=1)
