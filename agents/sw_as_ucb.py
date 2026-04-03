# agents/sw_as_ucb.py
import numpy as np
from .base_agent import BaseAgent

class SW_AS_UCB(BaseAgent):
    """
    Sliding Window Adaptive Seasonal UCB (SW-AS-UCB)
    - 트렌드 추종(SW): 최근 W일치의 이동평균을 베이스라인으로 사용하여 장기 변화에 적응합니다.
    - 데이터 희소성 극복: UCB 보너스는 매장의 전체 방문 횟수(Global)를 공유하여 탐색 지연을 막습니다.
    """
    def __init__(self, nbArms, window_size=100, period=7, c=0.05, smoothing=10.0, name="SW_AS_UCB"):
        super().__init__(nbArms, name=name)
        self.nbArms = nbArms
        self.window_size = window_size
        self.period = period
        self.c = c
        self.smoothing = smoothing
        self.clear()

    def clear(self):
        # 1. 글로벌 방문 횟수 (탐색 보너스 계산용 - 데이터 희소성 극복)
        self.global_counts = np.zeros(self.nbArms)
        
        # 2. 단기 트렌드 기억 (SW - 100일치 이동평균용)
        self.window_memories = [[] for _ in range(self.nbArms)]
        
        # 3. 주기적 오차 기억 (AS)
        self.cycle_counts = np.zeros((self.nbArms, self.period))
        self.cycle_offsets = np.zeros((self.nbArms, self.period))
        self.t = 0

    def choice(self):
        self.t += 1
        current_phase = self.t % self.period

        if 0 in self.global_counts:
            return np.where(self.global_counts == 0)[0][0]

        q_values = []
        for a in range(self.nbArms):
            # 1. 베이스라인: 최근 100일의 평균 (없으면 0) -> 트렌드 완벽 추종
            if len(self.window_memories[a]) > 0:
                mu_window = np.mean(self.window_memories[a])
            else:
                mu_window = 0.0

            # 2. 주기별 가감치: 베이지안 스무딩 적용
            n_phase = self.cycle_counts[a, current_phase]
            raw_sum_offset = self.cycle_offsets[a, current_phase]
            
            # 초기에는 노이즈 억제 (smoothing), 후반에는 주기성 반영
            offset = raw_sum_offset / (n_phase + self.smoothing)

            # 3. 불확실성 보너스: 요일별(N_phase)이 아닌 글로벌 횟수(N_global) 사용!
            # -> 1/7 데이터 토막으로 인한 과대 탐색(오버피팅) 방지
            bonus = self.c * np.sqrt(np.log(self.t) / self.global_counts[a])
            
            expected_reward = mu_window + offset
            q_values.append(expected_reward + bonus)

        return np.argmax(q_values)

    def getReward(self, arm, reward):
        current_phase = self.t % self.period
        
        # 1. 글로벌 방문 횟수 갱신
        self.global_counts[arm] += 1

        # 2. 현재 상태의 베이스라인(업데이트 전의 이동평균)을 먼저 구함
        if len(self.window_memories[arm]) > 0:
            mu_window_before = np.mean(self.window_memories[arm])
        else:
            mu_window_before = reward # 첫 데이터면 자기 자신
            
        # 3. 슬라이딩 윈도우(SW) 메모리 업데이트
        self.window_memories[arm].append(reward)
        if len(self.window_memories[arm]) > self.window_size:
            self.window_memories[arm].pop(0)

        # 4. 순수 주기적 오차 누적 (실제 보상 - 당시의 단기 트렌드)
        pure_cycle_effect = reward - mu_window_before
        
        self.cycle_counts[arm, current_phase] += 1
        self.cycle_offsets[arm, current_phase] += pure_cycle_effect