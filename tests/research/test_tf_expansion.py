"""R4 TF 확장 인프라 — 시간→봉 환산·splitter 스케일 (결정3 박제)."""

from __future__ import annotations

from src.research.experiments.tf_expansion import bars, splitter_for


def test_bars_time_to_count():
    # 갭0 전제: 1yr = 정수 봉 (TF별)
    assert bars("1h", 365) == 8760
    assert bars("4h", 365) == 2190
    assert bars("1d", 365) == 365
    assert bars("15m", 365) == 35040
    assert bars("1h", 90) == 2160          # 3mo = 현재 1h val_size


def test_splitter_for_scales_by_tf():
    # splitter 는 봉 기반 그대로, 시간(1yr/3mo)을 TF 봉수로 환산해 전달
    sp1h = splitter_for("1h", 24)
    assert sp1h.train_min == 8760 and sp1h.val_size == 2160   # = 현재 1h 설정
    sp4h = splitter_for("4h", 24)
    assert sp4h.train_min == 2190 and sp4h.val_size == 540
    sp1d = splitter_for("1d", 24)
    assert sp1d.train_min == 365 and sp1d.val_size == 90
