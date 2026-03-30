# envs/custom_evaluator.py
import numpy as np
from tqdm import tqdm

class CustomEvaluator:
    def __init__(self, env, agents, horizon=10000):
        self.env = env
        self.agents = agents
        self.horizon = horizon
        self.num_agents = len(agents)
        self.num_arms = env.nbArms

        self.rewards_log = np.zeros((self.num_agents, self.horizon))
        self.actions_log = np.zeros((self.num_agents, self.horizon))

    def _calculate_slippage(self, base_reward, num_collisions):
        """
        (고도화됨) 출혈 경쟁을 모사하는 비선형 슬리피지 로직
        """
        if num_collisions <= 1:
            return base_reward 
        
        # 🌟 파이가 단순히 N분의 1이 되는 것을 넘어, 
        # 마케팅 비용 증가/출혈 경쟁 등으로 인해 수익이 지수함수적으로 박살남 (알파 = 1.5)
        PENALTY_FACTOR = 1.5
        return base_reward / (num_collisions ** PENALTY_FACTOR)

    def run_simulation(self):
        for t in tqdm(range(self.horizon), desc="🏃 시뮬레이션 진행 중", ncols=100, ascii=False):
            # 1. 모든 에이전트가 오늘 어디 갈지 결정 (선택)
            choices = [agent.choice() for agent in self.agents]
            self.actions_log[:, t] = choices

            # 🌟 [버그 픽스] 2. 오늘 시장(Arm)의 전체 파이를 딱 한 번만 결정!
            unique_chosen_arms = np.unique(choices)
            market_rewards = {}
            for arm_idx in unique_chosen_arms:
                # 선택된 매장에 대해서만 오늘자 진짜 수요를 계산해 둠
                market_rewards[arm_idx] = self.env.draw(arm_idx, t)

            # 3. 각 매장별로 몇 명이나 몰렸는지(충돌) 계산
            pulls_count = np.bincount(choices, minlength=self.num_arms)

            # 4. 결과 판정 및 분배
            for i, agent in enumerate(self.agents):
                chosen_arm = choices[i]
                num_collisions = pulls_count[chosen_arm]
                
                # 아까 구워둔 '정확히 동일한 파이(market_rewards)'를 가져와서 슬리피지 적용
                base_reward = market_rewards[chosen_arm]
                final_reward = self._calculate_slippage(base_reward, num_collisions)
                
                agent.getReward(chosen_arm, final_reward)
                self.rewards_log[i, t] = final_reward

        return self.rewards_log, self.actions_log