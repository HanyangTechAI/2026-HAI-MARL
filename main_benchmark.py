# main_benchmark.py

"""
예시 실행 명령어:
python main_benchmark.py -m density -k 32 -c "SW_AS_UCB_w100_p7_s10.0_c0.05" -n 10
python main_benchmark.py -m baseline -c "TS_Collision_Aware" -b "Eps_e0.01,Eps_e0.05,Eps_e0.1,WSLS,WSLS,Softmax_t0.05,DecayEps_e0.5" --seed 777 -n 100
python main_benchmark.py -m density -k 32 -c "Thompson_Collision_Aware" -n 20
python main_benchmark.py -m baseline -s shocks_only -c "AS_UCB_p7_s5.0_c0.05" -n 10
"""

import os
import random
import argparse
import numpy as np
from datetime import datetime

# 평가기 및 유틸리티 임포트
from envs.custom_evaluator import CustomEvaluator
from utils.env_builder import build_dynamic_market
from utils.mc_logger import MCLogger
from utils.plot_batch_results import plot_monte_carlo_results
from utils.agent_factory import get_agent

def parse_args():
    """터미널에서 주입받는 하이퍼파라미터 및 환경 통제 변수들을 정의합니다."""
    parser = argparse.ArgumentParser(
        description="MARL Benchmark Arena - 다중 에이전트 강화학습 논문용 평가 시뮬레이터",
        formatter_class=argparse.RawTextHelpFormatter # 터미널 도움말에서 줄바꿈을 유지하기 위해 추가
    )
    
    # ---------------------------------------------------------
    # 1. 시뮬레이션 기본 설정 (Core Settings)
    # ---------------------------------------------------------
    parser.add_argument("-n", "--iters", type=int, default=10, 
                        help="[기본: 10] 몬테카를로 평행우주(Seed) 반복 횟수. 논문용은 100 이상 권장.")
    parser.add_argument("-t", "--horizon", type=int, default=1941, 
                        help="[기본: 1941] 1회 시뮬레이션의 스텝 수 (월마트 데이터 기준 1941일).")
    parser.add_argument("--seed", type=int, default=None, 
                        help="[기본: None] 난수 고정용 시드값. 완벽히 동일한 환경에서의 재현(Reproducibility)이 필요할 때 사용.")

    # ---------------------------------------------------------
    # 2. 시장 환경 설정 (Market Environment)
    # ---------------------------------------------------------
    parser.add_argument("-s", "--market_strategy", type=str, default="balanced", 
                        choices=["balanced", "random", "stationary_only", "shocks_only"],
                        help="[기본: balanced] 8개의 매장(Arm)을 차출하는 전략.\n"
                             " - balanced: 4가지 타입(국밥, 파동, 트렌드, 스위치)을 2개씩 공평하게 차출\n"
                             " - random: 타입 상관없이 무작위 8개 차출\n"
                             " - stationary_only: 극한의 안정성 테스트용\n"
                             " - shocks_only: 극한의 주기성 테스트용")

    # ---------------------------------------------------------
    # 3. 에이전트 및 리그 모드 설정 (Agent & League)
    # ---------------------------------------------------------
    parser.add_argument("-m", "--mode", type=str, default="baseline", 
                        choices=["baseline", "density", "custom"], 
                        help="[기본: baseline] 벤치마크 평가 모드.\n"
                             " - baseline: 도전자 1명 vs 고정 기준군(적폐) 7명\n"
                             " - density: 인원수 과밀 스트레스 테스트\n"
                             " - custom: 팀원 통합 자유 난투극")
    parser.add_argument("-c", "--challenger", type=str, default="SW_AS_UCB_w100_p7_s10.0_c0.05", 
                        help="[기본: SW_AS_UCB...] 평가하고자 하는 핵심 도전자 모델의 팩토리 명칭.")
    parser.add_argument("-k", "--num_agents", type=int, default=8, 
                        help="[기본: 8] density 모드에서 투입할 총 에이전트 수 (경쟁 밀도 조절).")
    parser.add_argument("-b", "--baselines", type=str, default="Eps_e0.05,UCB_c0.05,SW_UCB_w100_c0.05,WSLS,Softmax_t0.1,DecayEps_e0.2,Eps_e0.1", 
                        help="[기본: 7종 콤보] baseline 모드에서 샌드백/경쟁자 역할을 할 7명의 에이전트 리스트 (쉼표로 구분).")

    return parser.parse_args()

def get_lineup(args):
    """설정된 모드와 파라미터에 따라 출전 명단을 구성합니다."""
    lineup = []
    
    if args.mode == "baseline":
        # 쉼표로 구분된 문자열을 리스트로 변환하고 도전자를 추가합니다.
        baseline_agents = [agent.strip() for agent in args.baselines.split(',')]
        lineup = baseline_agents + [args.challenger]
        
    elif args.mode == "density":
        lineup.append(args.challenger)
        # 시장을 교란할 무작위 UCB 봇들 대거 투입
        for i in range(args.num_agents - 1):
            lineup.append(f"UCB_c0.0{random.randint(1, 9)}") 
            
    elif args.mode == "custom":
        lineup = [
            "Eps_e0.05", "UCB_c0.05", "SW_UCB_w100_c0.05", "AS_UCB_p7_s5.0_c0.05", 
            "SW_AS_UCB_w100_p7_s10.0_c0.05", "TS", "TS_Collision_Aware"
        ]
        
    return lineup

def main():
    args = parse_args()
    
    # 시드 고정
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🔒 마스터 시드가 고정되었습니다: {args.seed}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = os.path.join("output", f"MC_{args.mode.upper()}_{timestamp}")
    os.makedirs(master_output_dir, exist_ok=True)
    
    print("="*60)
    print(f"🚀 MARL 벤치마크 아레나 가동 (Mode: {args.mode.upper()}, Iters: {args.iters})")
    print(f"🎯 도전자(Challenger): {args.challenger}")
    if args.mode == 'density':
        print(f"💀 경쟁 밀도(Density): 총 {args.num_agents} 명 투입")
    print("="*60)
    
    logger = None
    
    for i in range(args.iters):
        # 마스터 시드가 없으면 매번 완전히 새로운 무작위 우주 생성
        current_seed = random.randint(1, 999999) if args.seed is None else args.seed + i
        np.random.seed(current_seed)
        
        env, arm_names = build_dynamic_market(args.market_strategy)
        agent_names_list = get_lineup(args)
        
        agents = [get_agent(name, env.nbArms) for name in agent_names_list]
        
        if logger is None:
            unique_names = []
            name_counts = {}
            for name in agent_names_list:
                name_counts[name] = name_counts.get(name, 0) + 1
                unique_names.append(f"{name} ({name_counts[name]})" if name_counts[name] > 1 else name)
            logger = MCLogger(master_output_dir, unique_names, args.horizon, args.iters)
        
        print(f"▶️ [Run {i+1}/{args.iters} | Seed: {current_seed}] 시뮬레이션 진행 중...")
        evaluator = CustomEvaluator(env, agents, horizon=args.horizon, global_scaler=0.0001)
        rewards_log, actions_log = evaluator.run_simulation()
        
        final_rewards = {'Seed': current_seed}
        for idx, agent_name in enumerate(unique_names):
            final_rewards[agent_name] = np.sum(rewards_log[idx, :])
            
        logger.record_seed(i, current_seed, rewards_log, actions_log, arm_names, final_rewards)

    summary_df = logger.save_all_and_summarize()
    print("\n" + "="*60)
    print("🏆 [ARENA SCORECARD] 시뮬레이션 완료")
    print("="*60)
    print(summary_df.to_string(float_format="{:,.0f}".format))
    
    plot_monte_carlo_results(master_output_dir)
    print(f"\n✅ 완료! 결과 파일이 {master_output_dir} 에 저장되었습니다.")

if __name__ == "__main__":
    main()