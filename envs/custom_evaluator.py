import numpy as np
from tqdm import tqdm

class CustomEvaluator:
    def __init__(self, env, agents, horizon=10000, global_scaler=0.0001, context_builder=None):
        self.env = env
        self.agents = agents
        self.horizon = horizon
        self.num_agents = len(agents)
        self.num_arms = env.nbArms
        self.global_scaler = global_scaler
        self.context_builder = context_builder

        self.rewards_log = np.zeros((self.num_agents, self.horizon))
        self.actions_log = np.zeros((self.num_agents, self.horizon))

    def _calculate_slippage(self, base_reward, num_collisions):
        if num_collisions <= 1:
            return base_reward 
        PENALTY_FACTOR = 1.5
        return base_reward / (num_collisions ** PENALTY_FACTOR)

    def run_simulation(self):
        # for t in tqdm(range(self.horizon), desc="🏃 시뮬레이션 진행 중", ncols=100, ascii=False, leave=False):
        for t in range(self.horizon):
            # Context가 필요한 agent에게 context 제공
            if self.context_builder is not None:
                context = self.context_builder.get_context(t)
                for agent in self.agents:
                    if hasattr(agent, 'set_context'):
                        agent.set_context(context)
            
            choices = [agent.choice() for agent in self.agents]
            self.actions_log[:, t] = choices

            # 1. 오늘 시장의 거대한 파이를 딱 한 번만 캐싱
            unique_chosen_arms = np.unique(choices)
            market_rewards = {}
            for arm_idx in unique_chosen_arms:
                raw_reward = self.env.draw(arm_idx, t)
                
                scaled_reward = raw_reward * self.global_scaler 
                market_rewards[arm_idx] = scaled_reward

            # 2. 분배 및 슬리피지 계산
            pulls_count = np.bincount(choices, minlength=self.num_arms)
            for i, agent in enumerate(self.agents):
                chosen_arm = choices[i]
                num_collisions = pulls_count[chosen_arm]
                
                base_reward = market_rewards[chosen_arm]
                final_reward = self._calculate_slippage(base_reward, num_collisions)
                
                agent.getReward(chosen_arm, final_reward)
                self.rewards_log[i, t] = final_reward

        return self.rewards_log, self.actions_log