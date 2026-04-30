#!/usr/bin/env python
# docs/individuals/양민주/test_21arms.py
"""
21개 Arm 환경에서 Thompson Collision Aware vs Epsilon Greedy 성능 테스트

실행 방법:
python docs/individuals/양민주/test_21arms.py -n 20
"""

import os
import sys
import random
import argparse
import numpy as np
from datetime import datetime

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from envs.custom_evaluator import CustomEvaluator
from docs.individuals.양민주.env_from_mj.env_builder_21 import build_dynamic_market_21
from utils.mc_logger import MCLogger
from utils.plot_batch_results import plot_monte_carlo_results
from agents.epsilon_greedy import EpsilonGreedy
from agents.thompson_collision_aware import ThompsonCollisionAware
from agents.ucb import UCBAgent

def parse_args():
    parser = argparse.ArgumentParser(description="21-Arm 환경 성능 테스트")
    parser.add_argument("-n", "--iters", type=int, default=20, 
                        help="몬테카를로 반복 횟수 (기본: 20)")
    parser.add_argument("-t", "--horizon", type=int, default=1941, 
                        help="시뮬레이션 스텝 수 (기본: 1941)")
    parser.add_argument("-a", "--num_arms", type=int, default=21,
                        help="Arm 개수 (기본: 21, 최대 21)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="난수 시드 (재현성)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔒 시드 고정: {args.seed}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("docs", "individuals", "양민주", f"results_21arms_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print(f"🚀 21-Arm 환경 성능 테스트")
    print(f"   Arm 개수: {args.num_arms}")
    print(f"   반복 횟수: {args.iters}")
    print(f"   Horizon: {args.horizon}")
    print("="*70)
    
    # 테스트할 에이전트들
    agent_names = [
        "Eps_0.05",
        "Eps_0.1", 
        "UCB_0.05",
        "Thompson_Collision_Aware"
    ]
    
    logger = None
    
    for i in range(args.iters):
        current_seed = random.randint(1, 999999) if args.seed is None else args.seed + i
        np.random.seed(current_seed)
        
        env, arm_names = build_dynamic_market_21("balanced", args.num_arms)
        
        agents = [
            EpsilonGreedy(env.nbArms, epsilon=0.05, name="Eps_0.05"),
            EpsilonGreedy(env.nbArms, epsilon=0.1, name="Eps_0.1"),
            UCBAgent(env.nbArms, c=0.05, name="UCB_0.05"),
            ThompsonCollisionAware(env.nbArms, name="Thompson_Collision_Aware")
        ]
        
        if logger is None:
            logger = MCLogger(output_dir, agent_names, args.horizon, args.iters)
        
        print(f"▶️  [Run {i+1}/{args.iters} | Seed: {current_seed}] 시뮬레이션 진행 중...")
        evaluator = CustomEvaluator(env, agents, horizon=args.horizon, global_scaler=0.0001)
        rewards_log, actions_log = evaluator.run_simulation()
        
        final_rewards = {'Seed': current_seed}
        for idx, agent_name in enumerate(agent_names):
            final_rewards[agent_name] = np.sum(rewards_log[idx, :])
            
        logger.record_seed(i, current_seed, rewards_log, actions_log, arm_names, final_rewards)

    summary_df = logger.save_all_and_summarize()
    print("\n" + "="*70)
    print("🏆 [결과 요약]")
    print("="*70)
    print(summary_df.to_string(float_format="{:,.0f}".format))
    
    plot_monte_carlo_results(output_dir)
    print(f"\n✅ 완료! 결과: {output_dir}")

if __name__ == "__main__":
    main()
