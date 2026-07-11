"""실행 원장 — 비교개수 계수·사전등록 (MasterPlan §11 컴포넌트 G, F-1/F-2 실효장치).

설계는 홀드아웃 대신 "비교 개수 제한"으로 부풀림을 막는다(기둥 8). 이게 말로만
있으면 R2~R5 에서 무너지므로, **모든 성과기반 비교를 계수·사전등록**하는 장치를 둔다.

경량(장식금지): JSONL append + 캠페인별 비교 카운트 + 예산 경고. 풀 실험추적기 아님.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunRecord:
    run_id: str
    campaign: str | None
    config: dict
    summary: dict
    is_comparison: bool
    count_so_far: int


class RunLedger:
    """JSONL 원장. 레코드 kind = "run" | "budget"."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def _lines(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [
            json.loads(ln)
            for ln in self.path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    def _append(self, obj: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    def register_budget(self, campaign: str, budget: int) -> None:
        """캠페인의 비교 예산 사전등록."""
        self._append({"kind": "budget", "campaign": campaign, "budget": int(budget)})

    def record(
        self,
        config: dict,
        summary: dict,
        *,
        campaign: str | None = None,
        is_comparison: bool = True,
        run_id: str | None = None,
    ) -> RunRecord:
        n_runs = sum(1 for r in self._lines() if r.get("kind") == "run")
        rid = run_id or f"run{n_runs}"
        count = self.comparison_count(campaign) + (1 if is_comparison else 0)
        self._append({
            "kind": "run", "run_id": rid, "campaign": campaign,
            "config": config, "summary": summary,
            "is_comparison": is_comparison, "count_so_far": count,
        })
        return RunRecord(rid, campaign, config, summary, is_comparison, count)

    def comparison_count(self, campaign: str | None = None) -> int:
        return sum(
            1 for r in self._lines()
            if r.get("kind") == "run" and r.get("is_comparison")
            and (campaign is None or r.get("campaign") == campaign)
        )

    def budget(self, campaign: str) -> int | None:
        bs = [
            r["budget"] for r in self._lines()
            if r.get("kind") == "budget" and r.get("campaign") == campaign
        ]
        return bs[-1] if bs else None

    def remaining(self, campaign: str) -> int | None:
        b = self.budget(campaign)
        return None if b is None else b - self.comparison_count(campaign)

    def over_budget(self, campaign: str) -> bool:
        r = self.remaining(campaign)
        return r is not None and r < 0
