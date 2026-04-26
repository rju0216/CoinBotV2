"""src/ml/env_trading.py 단위 테스트.

TradingEnv의 observation/action space, reset/step, reward 부호, 수수료 차감 검증.
SB3 호환성 (1D obs, gymnasium API).
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from src.ml.env_trading import TradingEnv
    _GYMNASIUM_AVAILABLE = True
except ImportError:
    _GYMNASIUM_AVAILABLE = False


def _make_data(n: int = 3000, n_features: int = 10):
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (n, n_features)).astype(np.float32)
    prices = (100.0 + np.cumsum(rng.normal(0, 1, n))).astype(np.float32)
    return features, prices


@pytest.mark.skipif(not _GYMNASIUM_AVAILABLE, reason="gymnasium 미설치")
class TestTradingEnv:
    def test_observation_space_is_1d(self):
        """§11-#1: SB3 MlpPolicy 호환 위해 obs는 1D."""
        f, p = _make_data()
        env = TradingEnv(f, p, lookback=60, episode_length=500)
        assert env.observation_space.shape == (60 * 10,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self):
        f, p = _make_data()
        env = TradingEnv(f, p, lookback=60, episode_length=500)
        assert env.action_space.n == 3

    def test_reset_returns_obs_and_info(self):
        f, p = _make_data()
        env = TradingEnv(f, p, lookback=60, episode_length=500)
        obs, info = env.reset(seed=42)
        assert obs.shape == (60 * 10,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_5tuple(self):
        """gymnasium API: (obs, reward, terminated, truncated, info)."""
        f, p = _make_data()
        env = TradingEnv(f, p, lookback=60, episode_length=500)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (60 * 10,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_truncates_at_length(self):
        f, p = _make_data()
        env = TradingEnv(f, p, lookback=60, episode_length=100)
        env.reset(seed=42)
        truncated = False
        for _ in range(100):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break
        assert truncated

    def test_long_profits_on_uptrend(self):
        """가격 단조 증가 + LONG 보유 → 양수 reward."""
        n = 1000
        features = np.zeros((n, 5), dtype=np.float32)
        prices = np.linspace(100.0, 200.0, n).astype(np.float32)
        env = TradingEnv(features, prices, lookback=10, episode_length=100, fee_pct=0.0)
        env.reset(seed=42)
        env.step(1)  # LONG 진입 (수수료=0)
        _, reward, _, _, _ = env.step(1)  # LONG 유지 — 가격 상승분 반영
        assert reward > 0

    def test_short_profits_on_downtrend(self):
        n = 1000
        features = np.zeros((n, 5), dtype=np.float32)
        prices = np.linspace(200.0, 100.0, n).astype(np.float32)
        env = TradingEnv(features, prices, lookback=10, episode_length=100, fee_pct=0.0)
        env.reset(seed=42)
        env.step(2)  # SHORT
        _, reward, _, _, _ = env.step(2)
        assert reward > 0

    def test_fee_charged_on_position_change(self):
        n = 1000
        features = np.zeros((n, 5), dtype=np.float32)
        prices = np.full(n, 100.0, dtype=np.float32)  # 가격 고정 → PnL 0
        env = TradingEnv(features, prices, lookback=10, episode_length=100, fee_pct=0.001)
        env.reset(seed=42)
        # 신규 진입 → 수수료 한 번
        _, r1, _, _, _ = env.step(1)
        assert r1 < 0
        # 같은 방향 유지 → 가격 고정이라 reward 0
        _, r2, _, _, _ = env.step(1)
        assert abs(r2) < 1e-5
        # 청산 → 수수료 한 번
        _, r3, _, _, _ = env.step(0)
        assert r3 < 0

    def test_data_too_short_raises(self):
        f = np.zeros((50, 5), dtype=np.float32)
        p = np.zeros(50, dtype=np.float32)
        with pytest.raises(ValueError):
            TradingEnv(f, p, lookback=60, episode_length=500)

    def test_features_prices_length_mismatch_raises(self):
        f = np.zeros((1000, 5), dtype=np.float32)
        p = np.zeros(900, dtype=np.float32)
        with pytest.raises(ValueError):
            TradingEnv(f, p, lookback=60, episode_length=500)
