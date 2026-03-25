import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# 커스텀 모듈 임포트
from envs.SMPyBandits.Environment.MAB import MAB
from envs.SMPyBandits.Arms.Gaussian import Gaussian
from envs.custom_evaluator import CustomEvaluator
from agents.epsilon_greedy import EpsilonGreedy

def setup_logger(output_dir):
    """실시간 출력과 파일 저장을 동시에 해주는 로거 설정"""
    logger = logging.getLogger("MARL_Logger")
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 초기화 (중복 출력 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    # 1. 터미널 출력용 (StreamHandler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 파일 저장용 (FileHandler)
    log_file_path = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main():
    # ==========================================
    # 0. 타임스탬프 폴더 생성 및 로거 셋팅 (먼저 해야 로그를 저장할 수 있음!)
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"🚀 새로운 시뮬레이션 세션을 시작합니다. (저장 폴더: {output_dir})")

    # ==========================================
    # 1. 시뮬레이션 설정
    # ==========================================
    HORIZON = 10000
    np.random.seed(42)
    logger.info(f"⚙️ 설정: Horizon={HORIZON}, Seed=42")

    arm_configuration = [
        Gaussian(0.10, 0.05),
        Gaussian(0.02, 0.01),
        Gaussian(-0.05, 0.10)
    ]
    env = MAB(arm_configuration)
    logger.info(f"📈 환경 세팅 완료: 총 {env.nbArms}개의 자산(Arm) 생성됨")

    # ==========================================
    # 2. 에이전트 라인업 구성
    # ==========================================
    agents = [
        EpsilonGreedy(env.nbArms, epsilon=0.1, name="AI_Eps_0.1"),
        EpsilonGreedy(env.nbArms, epsilon=0.2, name="AI_Eps_0.2"),
        EpsilonGreedy(env.nbArms, epsilon=0.3, name="AI_Eps_0.3")
    ]
    agent_names = [agent.name for agent in agents]
    logger.info(f"🤖 참여 에이전트 명단: {agent_names}")

    # ==========================================
    # 3. 심판 배정 및 시뮬레이션 실행 (tqdm 바가 터미널에 그려짐)
    # ==========================================
    logger.info("⚔️ 시뮬레이션을 가동합니다...")
    evaluator = CustomEvaluator(env, agents, horizon=HORIZON)
    rewards_log, actions_log = evaluator.run_simulation()
    logger.info("🏁 시뮬레이션 루프 완료!")

    # ==========================================
    # 4. 결과 데이터를 CSV로 저장
    # ==========================================
    logger.info("💾 결과를 CSV 파일로 변환 및 저장 중...")
    df_rewards = pd.DataFrame(rewards_log.T, columns=agent_names)
    df_actions = pd.DataFrame(actions_log.T, columns=agent_names)

    rewards_path = os.path.join(output_dir, "rewards_log.csv")
    actions_path = os.path.join(output_dir, "actions_log.csv")
    
    df_rewards.to_csv(rewards_path, index_label="Step")
    df_actions.to_csv(actions_path, index_label="Step")

    logger.info("✅ 모든 작업이 성공적으로 끝났습니다!")

if __name__ == "__main__":
    main()