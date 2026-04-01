import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# 커스텀 모듈 임포트
from envs.SMPyBandits.Environment.MAB import MAB
from envs.custom_evaluator import CustomEvaluator
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCBAgent
from agents.softmax import SoftmaxAgent
from agents.wsls import WSLS
from agents.world_model import WorldModelAgent

# 🌟 새롭게 설계한 다이내믹 환경 모듈 임포트
from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from arms.arm_registry import STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY

# 시각화 모듈
from utils.plot_results import plot_experiment_results

def setup_logger(output_dir):
    """실시간 출력과 파일 저장을 동시에 해주는 로거 설정"""
    logger = logging.getLogger("MARL_Logger")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file_path = os.path.join(output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main():
    # ==========================================
    # 0. 타임스탬프 폴더 생성 및 로거 셋팅
    # ==========================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"🚀 새로운 월마트 시뮬레이션 세션을 시작합니다. (저장 폴더: {output_dir})")

    # ==========================================
    # 1. 시뮬레이션 설정 (월마트 다이내믹 마켓)
    # ==========================================
    HORIZON = 1941 # 월마트 데이터의 총 스텝(일수)
    np.random.seed(66)

    # 🌟 공통 환경 설정 (경로 및 스케일러)
    # 현재 실행 위치(root)를 기준으로 data 폴더 내의 csv 경로 지정
    SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
    SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
    SWITCH_FILE = os.path.join("data", "walmart", "switched_data", "regime_switches.csv")
    GLOBAL_SCALER = 0.0001 # 보상을 [0, 1] 수준으로 정규화

    logger.info("📦 시뮬레이션 환경(Arm)을 조립합니다...")
    arm_configuration = []

    # 1-A. 안정적인 캐시카우 라인업 선택 (STATIONARY_REGISTRY 활용)
    selected_stationary = ["CA_HOBBIES_1", "CA_FOODS_1"]
    for name in selected_stationary:
        params = STATIONARY_REGISTRY[name]
        arm = StationaryArm(
            arm_name=name,
            mean=params["mean"],
            variance=params["variance"]
            )
        arm_configuration.append(arm)
        logger.info(f"  - [우량주] {name} 조립 완료")

    # 1-B. 다이내믹 폭발 라인업 선택 (EVENT_SHOCK_REGISTRY 활용)
    selected_shocks = ["CA_FOODS_2", "TX_FOODS_2"]
    for name in selected_shocks:
        params = EVENT_SHOCK_REGISTRY[name]
        arm = EventShockArm(
            arm_name=name,
            base_mean=params["base_mean"],
            base_variance=params["base_variance"],
            shocks_csv=SHOCKS_FILE,
            season_csv=SEASON_FILE
        )
        arm_configuration.append(arm)
        logger.info(f"  - [다이내믹주] {name} 조립 완료")

    # 1-C. 장기 트렌드 성장형 라인업 (Trend)
    selected_trends = ["CA_HOUSEHOLD_1", "WI_FOODS_2"]
    for name in selected_trends:
        params = TREND_REGISTRY[name]
        arm = TrendArm(
            arm_name=name,
            start_mean=params["start_mean"],
            slope=params["slope"],
            variance=params["variance"]
        )
        arm_configuration.append(arm)
        logger.info(f"  - [트렌드주] {name} 조립 완료")

    # 1-D. 국면 전환형 라인업 (Switch)
    selected_switches = ["TX_HOUSEHOLD_1", "WI_HOUSEHOLD_1"]
    for name in selected_switches:
        params = SWITCH_REGISTRY[name]
        arm = SwitchArm(
            arm_name=name,
            base_mean=params["base_mean"],
            base_variance=params["base_variance"],
            switch_csv=SWITCH_FILE
        )
        arm_configuration.append(arm)
        logger.info(f"  - [전환주] {name} 조립 완료")

    env = MAB(arm_configuration)
    
    logger.info(f"📊 총 {env.nbArms}개의 물류 라인(Arm)이 전장에 배치되었습니다.")

    # ==========================================
    # 2. 에이전트 라인업 구성
    # ==========================================
    agents = [
        EpsilonGreedy(env.nbArms, epsilon=0.05, name="AI_Eps_0.05"),
        EpsilonGreedy(env.nbArms, epsilon=0.1, name="AI_Eps_0.1"),
        EpsilonGreedy(env.nbArms, epsilon=0.2, name="AI_Eps_0.2"),
        # EpsilonGreedy(env.nbArms, epsilon=0.3, name="AI_Eps_0.3"),
        # EpsilonGreedy(env.nbArms, epsilon=0.5, name="AI_Eps_0.5"),
        UCBAgent(env.nbArms, c=0.1, name="AI_UCB_0.1"),
        UCBAgent(env.nbArms, c=0.15, name="AI_UCB_0.15"),
        SoftmaxAgent(env.nbArms, temperature=0.05, name="AI_Softmax_0.05"),
        SoftmaxAgent(env.nbArms, temperature=0.1, name="AI_Softmax_0.1"),
        WSLS(env.nbArms, initial_aspiration=0.12, aspiration_lr=0.05, name="WSLS_I0.12_LR0.05"),
        WSLS(env.nbArms, initial_aspiration=0.15, aspiration_lr=0.1, name="WSLS_I0.15_LR0.1"),
        WorldModelAgent(num_arms=env.nbArms, name='WorldModel_Dreamer")
    ]
    agent_names = [agent.name for agent in agents]
    logger.info(f"🤖 참여 에이전트 명단: {agent_names}")

    # ==========================================
    # 3. 심판 배정 및 시뮬레이션 실행
    # ==========================================
    logger.info("⚔️ 시뮬레이션을 가동합니다...")
    evaluator = CustomEvaluator(env, agents, horizon=HORIZON, global_scaler=GLOBAL_SCALER)
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

    # 🌟 추가된 자동화 시각화 로직 🌟
    logger.info("📊 시각화(Plotting) 모듈을 호출하여 분석 그래프를 생성합니다...")
    # SCALER의 역수인 10000을 넘겨주어 그래프 Y축을 원래 단위로 복구합니다.
    plot_experiment_results(output_dir, scaler=int(1/GLOBAL_SCALER))

    logger.info("✅ 모든 작업 및 그래프 생성이 성공적으로 끝났습니다!")

if __name__ == "__main__":
    main()
