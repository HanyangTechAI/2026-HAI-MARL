import numpy as np

# 1. 방금 적출해 온 SMPyBandits 핵심 모듈들을 불러옵니다.
# (경로는 폴더 구조에 맞게 envs.SMPyBandits... 로 설정)
from envs.SMPyBandits.Environment.MAB import MAB
from envs.SMPyBandits.Arms.Gaussian import Gaussian
from envs.custom_evaluator import CustomEvaluator

# =====================================================================
# [테스트용 임시 에이전트] 
# 나중에 DS 팀원들이 이 자리를 UCB나 Epsilon-Greedy로 채울 것입니다.
# =====================================================================
class DummyRandomAgent:
    def __init__(self, num_arms, name):
        self.num_arms = num_arms
        self.name = name

    def choice(self):
        # 아무 생각 없이 무작위 자산(Arm) 선택
        return np.random.randint(self.num_arms)

    def getReward(self, arm, reward):
        # 나중에는 여기서 Q-value 수식을 업데이트하지만, 지금은 그냥 무시!
        pass 

# =====================================================================
# [메인 실행부]
# =====================================================================
if __name__ == "__main__":
    # 시드 고정 (재현성 확인용)
    np.random.seed(42)

    # 1단계: SMPyBandits의 Arm들을 생성하여 환경(주식 시장) 구축
    # 자산 0: 평균 수익률 10%, 변동성(분산) 5%
    # 자산 1: 평균 수익률  2%, 변동성 1%
    # 자산 2: 평균 수익률 -5%, 변동성 10%
    arm_configuration = [
        Gaussian(0.10, 0.05),
        Gaussian(0.02, 0.01),
        Gaussian(-0.05, 0.10)
    ]
    
    # SMPyBandits의 MAB 객체로 시장 묶기
    env = MAB(arm_configuration)

    # 2단계: 테스트용 에이전트 3명 생성
    agents = [DummyRandomAgent(env.nbArms, f"Player_{i}") for i in range(3)]

    # 3단계: 우리가 만든 심판(CustomEvaluator) 등판! (테스트니까 10 스텝만)
    evaluator = CustomEvaluator(env, agents, horizon=10)

    # 4단계: 시뮬레이션 슛!
    rewards_log, actions_log = evaluator.run_simulation()

    print("\n[테스트 결과 확인]")
    print(f"시장(Env)에 생성된 자산 개수: {env.nbArms}개")
    print(f"첫 번째 에이전트가 10 스텝 동안 고른 자산 번호:\n{actions_log[0]}")
    print(f"첫 번째 에이전트가 10 스텝 동안 받은 실제 보상(슬리피지 적용됨):\n{np.round(rewards_log[0], 3)}")