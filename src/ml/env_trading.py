"""트레이딩 시뮬레이터 Gym 환경 — PPO 학습용.

§11-#1 적용: `observation_space`와 `_get_obs()`는 1D flatten ((lookback × n_features,)).
SB3 MlpPolicy가 1D 벡터 기대.

설계 (§4.6):
- Observation: (lookback × n_features,) — 최근 lookback개 봉의 scaled 피처
- Action: Discrete(3) — 0=HOLD, 1=LONG, 2=SHORT
- Reward: 스텝별 PnL(%) - 수수료 (mark-to-market)
- 에피소드: 연속 episode_length봉, 랜덤 시작점, 종료 시 강제 청산
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """OHLCV 시뮬레이터. 단일 포지션, 즉시 체결, 수수료 차감."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features_scaled: np.ndarray,  # (N, F) — 이미 StandardScaler.transform 적용됨
        prices: np.ndarray,           # (N,) — close prices
        lookback: int = 60,
        episode_length: int = 2000,
        fee_pct: float = 0.0005,
    ):
        super().__init__()
        if features_scaled.shape[0] != prices.shape[0]:
            raise ValueError("features와 prices 길이 불일치")
        if features_scaled.shape[0] < lookback + episode_length + 1:
            raise ValueError(
                f"데이터 부족: {features_scaled.shape[0]}행 < {lookback + episode_length + 1} 필요"
            )

        self.features = features_scaled.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.lookback = lookback
        self.episode_length = episode_length
        self.fee_pct = float(fee_pct)
        self.n_features = features_scaled.shape[1]

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback * self.n_features,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=HOLD, 1=LONG, 2=SHORT

        self._t = 0
        self._steps = 0
        self._position = 0  # 0=none, 1=long, 2=short

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        max_start = len(self.features) - self.episode_length - 1
        self._t = int(self.np_random.integers(self.lookback, max_start))
        self._steps = 0
        self._position = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev_price = float(self.prices[self._t - 1])
        price = float(self.prices[self._t])

        # mark-to-market PnL: 보유 중이면 직전 봉 대비 가격 변화율(%)
        reward = 0.0
        if self._position == 1:  # long
            reward = (price - prev_price) / prev_price * 100.0
        elif self._position == 2:  # short
            reward = (prev_price - price) / prev_price * 100.0

        # 액션이 다르면 포지션 변경 (수수료 차감)
        action = int(action)
        if action != self._position:
            if self._position != 0:  # 청산 수수료
                reward -= self.fee_pct * 100.0
            if action != 0:  # 신규 진입 수수료
                reward -= self.fee_pct * 100.0
            self._position = action

        self._t += 1
        self._steps += 1

        terminated = False
        truncated = self._steps >= self.episode_length

        # 강제 청산: 에피소드 종료 시 포지션 보유 중이면 청산 수수료
        if truncated and self._position != 0:
            reward -= self.fee_pct * 100.0
            self._position = 0

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            {"position": self._position, "t": self._t},
        )

    def _get_obs(self) -> np.ndarray:
        # 최근 lookback개 봉의 피처 → (L, F) → flatten (L*F,)
        seq = self.features[self._t - self.lookback + 1 : self._t + 1]
        return seq.flatten()
