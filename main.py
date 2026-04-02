# main.py
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import logging

# 환경 및 에이전트 임포트
from envs.SMPyBandits.Environment.MAB import MAB
from envs.custom_evaluator import CustomEvaluator
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCBAgent
from agents.softmax import SoftmaxAgent
from agents.wsls import WSLS
from agents.sliding_window_ucb import SlidingWindowUCB
# from agents.world_model import WorldModelAgent # 나중에 추가 시 주석 해제

from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from arms.arm_registry import STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY

from utils.plot_batch_results import plot_monte_carlo_results

# 데이터 경로 상수
SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
SWITCH_FILE = os.path.join("data", "walmart", "extracted_data", "regime_switches.csv")

def parse_args():
    """터미널에서 실행할 때 다양한 변수를 주입받는 파서입니다."""
    parser = argparse.ArgumentParser(description="MARL Monte Carlo Evaluation Master")
    parser.add_argument("-n", "--iters", type=int, default=10, help="실행할 몬테카를로 시드(평행우주)의 개수 (기본값: 10)")
    parser.add_argument("-t", "--horizon", type=int, default=1941, help="1회 시뮬레이션의 스텝 수 (기본값: 1941)")
    parser.add_argument("-s", "--strategy", type=str, default="balanced", choices=["balanced", "random"], help="Arm 추출 전략 (balanced: 각 2개씩, random: 16개 중 아무거나 8개)")
    # 필요하다면 나중에 --epsilon 0.1 등 에이전트 하이퍼파라미터도 여기서 받을 수 있습니다.
    return parser.parse_args()

def select_arms(strategy="balanced"):
    """전략에 따라 16개의 레지스트리에서 8개의 시장을 차출합니다."""
    sampled_arms = {"stationary": [], "shocks": [], "trends": [], "switches": []}
    
    if strategy == "balanced":
        # 제가 추천하는 방식: 시장 구조 유지 (각 2개씩)
        sampled_arms["stationary"] = random.sample(list(STATIONARY_REGISTRY.keys()), 2)
        sampled_arms["shocks"] = random.sample(list(EVENT_SHOCK_REGISTRY.keys()), 2)
        sampled_arms["trends"] = random.sample(list(TREND_REGISTRY.keys()), 2)
        sampled_arms["switches"] = random.sample(list(SWITCH_REGISTRY.keys()), 2)
    else:
        # 완전 무작위 방식 (극단적 환경 테스트용)
        all_keys = (
            [("stationary", k) for k in STATIONARY_REGISTRY.keys()] +
            [("shocks", k) for k in EVENT_SHOCK_REGISTRY.keys()] +
            [("trends", k) for k in TREND_REGISTRY.keys()] +
            [("switches", k) for k in SWITCH_REGISTRY.keys()]
        )
        chosen = random.sample(all_keys, 8)
        for cat, name in chosen:
            sampled_arms[cat].append(name)
            
    return sampled_arms

def setup_mab_environment(sampled_arms):
    """차출된 명단으로 MAB 환경 객체를 조립합니다."""
    arm_configuration = []
    
    for name in sampled_arms["stationary"]:
        p = STATIONARY_REGISTRY[name]
        arm_configuration.append(StationaryArm(name, p["mean"], p["variance"]))
        
    for name in sampled_arms["shocks"]:
        p = EVENT_SHOCK_REGISTRY[name]
        arm_configuration.append(EventShockArm(name, p["base_mean"], p["base_variance"], SHOCKS_FILE, SEASON_FILE))
        
    for name in sampled_arms["trends"]:
        p = TREND_REGISTRY[name]
        arm_configuration.append(TrendArm(name, p["start_mean"], p["slope"], p["variance"]))
        
    for name in sampled_arms["switches"]:
        p = SWITCH_REGISTRY[name]
        arm_configuration.append(SwitchArm(name, p["base_mean"], p["base_variance"], SWITCH_FILE))
        
    return MAB(arm_configuration)

def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join("output", f"MC_Batch_{timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    
    print("="*60)
    print(f"🚀 MARL 몬테카를로 대규모 검증을 시작합니다!")
    print(f" - 반복 횟수(Seeds): {args.iters} 회")
    print(f" - 추출 전략: {args.strategy.upper()}")
    print(f" - 호라이즌: {args.horizon} Days")
    print("="*60)
    
    all_final_rewards = []
    
    for i in range(1, args.iters + 1):
        # 1. 🌟 매 반복마다 완전히 새로운 무작위 시드 생성 (1 ~ 999999)
        current_seed = random.randint(1, 999999)
        np.random.seed(current_seed)
        
        # 2. 시장 구성 (통제된 무작위성)
        sampled_arms_dict = select_arms(args.strategy)
        env = setup_mab_environment(sampled_arms_dict)
        
        # 3. 에이전트 라인업 (매 시드마다 뇌를 초기화해야 하므로 여기서 새로 생성)
        agents = [
            EpsilonGreedy(env.nbArms, epsilon=0.05, name="Eps_0.05"),
            EpsilonGreedy(env.nbArms, epsilon=0.1, name="Eps_0.1"),
            UCBAgent(env.nbArms, c=0.05, name="UCB_0.05"),
            SoftmaxAgent(env.nbArms, temperature=0.1, name="Softmax_0.1"),
            WSLS(env.nbArms, initial_aspiration=0.15, aspiration_lr=0.1, name="WSLS_Trend"),
            SlidingWindowUCB(env.nbArms, window_size=100, c=0.05, name="SW_UCB_100")
        ]
        
        print(f"▶️ [Run {i}/{args.iters} | Seed: {current_seed}] 시뮬레이션 가동 중...")
        
        # 4. 심판 배정 및 실행
        evaluator = CustomEvaluator(env, agents, horizon=args.horizon, global_scaler=0.0001)
        # tqdm 출력을 끄고 싶다면 CustomEvaluator 내부의 tqdm을 조절하거나, 빠른 실행을 위해 유지
        rewards_log, _ = evaluator.run_simulation()
        
        # 5. 이번 시드의 최종 성적 기록
        final_rewards = {'Seed': current_seed}
        for idx, agent in enumerate(agents):
            final_rewards[agent.name] = np.sum(rewards_log[idx, :])
        all_final_rewards.append(final_rewards)

    # ==========================================
    # 🏆 종합 통계 도출 및 출력
    # ==========================================
    df_results = pd.DataFrame(all_final_rewards).set_index('Seed')
    
    # 지표 계산
    mean_rewards = df_results.mean()
    std_rewards = df_results.std()
    win_counts = df_results.idxmax(axis=1).value_counts()
    win_rates = (win_counts / args.iters) * 100
    
    summary_df = pd.DataFrame({
        "Avg Reward": mean_rewards,
        "Risk (Std)": std_rewards,
        "Win Count": win_counts,
        "Win Rate (%)": win_rates
    }).fillna(0).sort_values(by="Avg Reward", ascending=False)
    
    print("\n" + "="*60)
    print("🎯 [FINAL SCORECARD] 몬테카를로 검증 완료")
    print("="*60)
    print(summary_df.to_string(float_format="{:,.0f}".format))
    
    # CSV 저장
    summary_df.to_csv(os.path.join(master_output_dir, "master_scorecard.csv"))
    df_results.to_csv(os.path.join(master_output_dir, "all_seeds_raw_data.csv"))
    
    # plotting
    plot_monte_carlo_results(master_output_dir)
    
    print(f"\n✅ 상세 결과가 {master_output_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()