import numpy as np
from collections import deque
from .base_agent import BaseAgent


class SlidingWindowUCB(BaseAgent):
    def __init__(
        self,
        num_arms: int,
        window_size: int = 200,
        c: float = 0.1,
        name: str = "SW_UCB",
    ):
        super().__init__(num_arms, name=name)
        self.window_size = window_size
        self.c = c
        self.window: list[deque] = [
            deque(maxlen=window_size) for _ in range(num_arms)
        ]
    def _window_stats(self, arm: int):
        if len(self.window[arm]) == 0:
            return 0.0, 0
        rewards = [r for _, r in self.window[arm]]
        return np.mean(rewards), len(rewards)
    def choice(self) -> int:
        unexplored = [a for a in range(self.num_arms) if len(self.window[a]) == 0]
        if unexplored:
            return np.random.choice(unexplored)
        t_w = min(self.t, self.window_size)
        sw_ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            q_w, n_w = self._window_stats(arm)
            sw_ucb_values[arm] = q_w + self.c * np.sqrt(np.log(t_w) / n_w)
        max_val = np.max(sw_ucb_values)
        best_arms = np.where(sw_ucb_values == max_val)[0]
        return int(np.random.choice(best_arms))
    def getReward(self, arm: int, reward: float):
        super().getReward(arm, reward)
        self.window[arm].append((self.t, reward))