# main.py
import os
import random
import argparse
import numpy as np
from datetime import datetime

# 에이전트 및 평가기
from envs.custom_evaluator import CustomEvaluator
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCBAgent
from agents.sliding_window_ucb import SlidingWindowUCB
from agents.as_ucb import AS_UCB
from agents.sw_as_ucb import SW_AS_UCB
from agents.wsls import WSLS
from agents.softmax import SoftmaxAgent

# 분리된 유틸리티 모듈
from utils.env_builder import build_dynamic_market
from utils.mc_logger import MCLogger
from utils.plot_batch_results import plot_monte_carlo_results

def parse_args():
    parser = argparse.ArgumentParser(description="MARL Monte Carlo Evaluation Master")
    parser.add_argument("-n", "--iters", type=int, default=10)
    parser.add_argument("-t", "--horizon", type=int, default=1941)
    parser.add_argument("-s", "--strategy", type=str, default="balanced", choices=["balanced", "random"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join("output", f"MC_Batch_{timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    
    print("="*60)
    print(f"🚀 MARL 몬테카를로 대규모 검증 시작 (Iters: {args.iters}, Strategy: {args.strategy.upper()})")
    print("="*60)
    
    # # 더미 에이전트 생성하여 이름 리스트 확보
    # dummy_agents = [
    #     EpsilonGreedy(8, epsilon=0.05, name="Eps_0.05"),
    #     UCBAgent(8, c=0.05, name="UCB_0.05"),
    #     SlidingWindowUCB(8, window_size=100, c=0.05, name="SW_UCB_100"),
    #     AS_UCB(8, period=7, c=0.05, smoothing=10.0, name="AS_UCB_7s10"),
    # ]
    # agent_names = [a.name for a in dummy_agents]
    
    # 🌟 3-Tier 로거 초기화
    logger = None
    
    
    for i in range(args.iters):
        current_seed = random.randint(1, 999999)
        np.random.seed(current_seed)
        
        # 1. 공장에서 환경 조립해 오기
        env, arm_names = build_dynamic_market(args.strategy)
        
        # 2. 에이전트 라인업 세팅 (매 시드 초기화)
        agents = [
            EpsilonGreedy(env.nbArms, epsilon=0.05, name="Eps_0.05"),
            UCBAgent(env.nbArms, c=0.05, name="UCB_0.05"),
            SlidingWindowUCB(env.nbArms, window_size=100, c=0.05, name="SW_UCB_100_c0.05"),
            SlidingWindowUCB(env.nbArms, window_size=200, c=0.05, name="SW_UCB_200_c0.05"),
            AS_UCB(env.nbArms, period=7, c=0.05, smoothing=5.0, name="AS_UCB_7s5"),
            AS_UCB(env.nbArms, period=7, c=0.05, smoothing=10.0, name="AS_UCB_7s10"),
            AS_UCB(env.nbArms, period=7, c=0.05, smoothing=20.0, name="AS_UCB_7s20"),
            AS_UCB(env.nbArms, period=7, c=0.05, smoothing=50.0, name="AS_UCB_7s50"),
            SW_AS_UCB(env.nbArms, window_size=100, period=7, c=0.05, smoothing=10.0, name="SW_AS_UCB_7s10"),
            SoftmaxAgent(env.nbArms, temperature=0.05, name="Softmax_0.05"),
            WSLS(env.nbArms, name="WSLS"),
            EpsilonGreedy(env.nbArms, epsilon=0.1, name="Eps_0.1"),
            UCBAgent(env.nbArms, c=0.1, name="UCB_0.1"),
            SlidingWindowUCB(env.nbArms, window_size=100, c=0.1, name="SW_UCB_100_c0.1"),
            SlidingWindowUCB(env.nbArms, window_size=200, c=0.1, name="SW_UCB_200_c0.1"),
            AS_UCB(env.nbArms, period=7, c=0.1, smoothing=5.0, name="AS_UCB_7s5_c0.1"),
            AS_UCB(env.nbArms, period=7, c=0.1, smoothing=10.0, name="AS_UCB_7s10_c0.1"),
            SW_AS_UCB(env.nbArms, window_size=100, period=7, c=0.1, smoothing=10.0, name="SW_AS_UCB_7s10_c0.1"),
            SoftmaxAgent(env.nbArms, temperature=0.1, name="Softmax_0.1"),
        ]
        
        if logger is None:
            agent_names = [a.name for a in agents]
            logger = MCLogger(master_output_dir, agent_names, args.horizon, args.iters)
        
        print(f"▶️ [Run {i+1}/{args.iters} | Seed: {current_seed}] 시뮬레이션 가동 중...")
        evaluator = CustomEvaluator(env, agents, horizon=args.horizon, global_scaler=0.0001)
        rewards_log, actions_log = evaluator.run_simulation()
        
        # 3. 이번 시드의 요약 보상 계산 및 로거에 던져주기
        final_rewards = {'Seed': current_seed}
        for idx, agent in enumerate(agents):
            final_rewards[agent.name] = np.sum(rewards_log[idx, :])
            
        logger.record_seed(i, current_seed, rewards_log, actions_log, arm_names, final_rewards)

    # 모든 시드 종료 후 일괄 저장 및 통계 출력
    summary_df = logger.save_all_and_summarize()
    
    print("\n" + "="*60)
    print("🎯 [FINAL SCORECARD] 몬테카를로 검증 완료")
    print("="*60)
    print(summary_df.to_string(float_format="{:,.0f}".format))
    
    # 시각화 호출
    plot_monte_carlo_results(master_output_dir)
    print(f"\n✅ 완료! 모든 데이터와 그래프가 {master_output_dir} 에 저장되었습니다.")

if __name__ == "__main__":
    main()