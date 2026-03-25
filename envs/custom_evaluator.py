import numpy as np
from tqdm import tqdm

class CustomEvaluator:
    def __init__(self, env, agents, horizon=10000):
        """
        다중 에이전트 시뮬레이션을 관장하는 메인 심판(Evaluator) 클래스
        
        :param env: SMPyBandits의 MAB 객체 (여러 개의 자산(Arm)이 묶인 시장)
        :param agents: 시뮬레이션에 참여할 에이전트 객체들의 리스트 (AI, 개미, 분탕충)
        :param horizon: 총 시뮬레이션 스텝 수 (기본값 10,000)
        """
        self.env = env
        self.agents = agents
        self.horizon = horizon
        self.num_agents = len(agents)
        self.num_arms = env.nbArms # 시장에 존재하는 자산의 총 개수

        # 결과 데이터를 저장할 2차원 배열 초기화 (형태: [에이전트 수, 스텝 수])
        self.rewards_log = np.zeros((self.num_agents, self.horizon))
        self.actions_log = np.zeros((self.num_agents, self.horizon))

    def _calculate_slippage(self, base_reward, num_collisions):
        """
        (핵심) 여러 명이 같은 자산에 몰렸을 때 보상을 깎는 슬리피지 로직
        * 산업공학 팀원들이 이 부분을 고도화할 예정입니다.
        """
        if num_collisions <= 1:
            return base_reward # 혼자 선택했으면 온전한 보상 획득
        
        # 임시 로직: 몰린 인원수만큼 1/N 로 나눠서 가짐 (수정 가능)
        return base_reward / num_collisions

    def run_simulation(self):
        """
        정해진 Horizon만큼 시뮬레이션을 진행하는 메인 루프
        """
        for t in tqdm(range(self.horizon), desc="🏃 시뮬레이션 진행 중", ncols=100, ascii=False):
            # 1. 모든 에이전트 실행
            choices = [agent.choice() for agent in self.agents]
            self.actions_log[:, t] = choices

            # 2. 충돌 감지
            pulls_count = np.bincount(choices, minlength=self.num_arms)

            # 3. 결과 판정 및 보상 분배
            for i, agent in enumerate(self.agents):
                chosen_arm = choices[i]
                num_collisions = pulls_count[chosen_arm]
                base_reward = self.env.draw(chosen_arm, t)
                final_reward = self._calculate_slippage(base_reward, num_collisions)
                agent.getReward(chosen_arm, final_reward)
                self.rewards_log[i, t] = final_reward

        return self.rewards_log, self.actions_log