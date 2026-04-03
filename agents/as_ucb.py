import numpy as np
from .base_agent import BaseAgent

class AS_UCB(BaseAgent):
    """
    적응형 주기 가산 밴딧 v2 (Smoothed Adaptive Seasonal UCB)
    - 초기 신뢰도 보정 아이디어를 베이지안 스무딩으로 구현했습니다.
    - 초기에는 일반 UCB처럼 글로벌 평균에 의존하며, 데이터가 쌓일수록 주기성을 강하게 반영합니다.
    """
    def __init__(self, nbArms, period=7, c=0.05, smoothing=5.0, name="AS_UCB_v2"):
        super().__init__(nbArms, name=name)
        self.nbArms = nbArms
        self.period = period
        self.c = c
        self.smoothing = smoothing
        self.clear()

    def clear(self):
        self.counts = np.zeros(self.nbArms)
        self.global_sum = np.zeros(self.nbArms)
        
        self.cycle_counts = np.zeros((self.nbArms, self.period))
        self.cycle_offsets = np.zeros((self.nbArms, self.period))
        self.t = 0

    def choice(self):
        self.t += 1
        current_phase = self.t % self.period

        if 0 in self.counts:
            return np.where(self.counts == 0)[0][0]

        q_values = []
        for a in range(self.nbArms):
            # 1. 베이스라인: 전체 평균
            mu_global = self.global_sum[a] / self.counts[a]

            # 2. 베이지안 스무딩이 적용된 가감값 계산
            raw_sum_offset = self.cycle_offsets[a, current_phase]
            n_phase = self.cycle_counts[a, current_phase]
            
            # n_phase가 작을 때는 smoothing 값이 분모를 키워 offset을 0(무시)에 가깝게 만듭니다.
            offset = raw_sum_offset / (n_phase + self.smoothing)

            # 3. 예측 가치 = 베이스라인 + 보정된 가감값 + 불확실성 보너스
            expected_reward = mu_global + offset
            bonus = self.c * np.sqrt(np.log(self.t) / self.counts[a])
            
            q_values.append(expected_reward + bonus)

        return np.argmax(q_values)

    def getReward(self, arm, reward):
        current_phase = self.t % self.period
        
        # 1. 베이스라인 업데이트
        self.counts[arm] += 1
        self.global_sum[arm] += reward
        mu_global_after = self.global_sum[arm] / self.counts[arm]

        # 2. 순수 오차(가감치) 누적
        pure_cycle_effect = reward - mu_global_after
        self.cycle_counts[arm, current_phase] += 1
        self.cycle_offsets[arm, current_phase] += pure_cycle_effect