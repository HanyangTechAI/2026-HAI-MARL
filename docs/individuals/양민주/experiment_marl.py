"""
MARL 환경에서 Thompson Sampling 개선 실험

목표: 다중 에이전트 경쟁 환경을 유지하면서 Thompson의 강점 보이기
전략:
1. 주말/평일 분리 학습 (타이밍 차별화로 충돌 회피)
2. 충돌 인식 Thompson (명시적 충돌 회피)
3. 다양한 reward_scale (선택 분산)
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import numpy as np

from envs.SMPyBandits.Environment.MAB import MAB
from envs.custom_evaluator import CustomEvaluator
from agents.epsilon_greedy import EpsilonGreedy
from agents.ucb import UCBAgent

from docs.individuals.양민주.thompson_sampling import ThompsonSampling
from docs.individuals.양민주.thompson_weekly import ThompsonWeekendWeekday
from docs.individuals.양민주.thompson_collision_aware import ThompsonCollisionAware

from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from arms.arm_registry import STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY


def run_experiment(experiment_name, agents, seed=66):
    """실험 실행"""
    print(f"\n{'='*70}")
    print(f"실험: {experiment_name}")
    print(f"에이전트 수: {len(agents)}명")
    print(f"{'='*70}")
    
    SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
    SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
    SWITCH_FILE = os.path.join("data", "walmart", "switched_data", "regime_switches.csv")
    GLOBAL_SCALER = 0.0001
    HORIZON = 1941
    np.random.seed(seed)
    
    # Arm 구성
    arm_configuration = []
    for name in ["CA_HOBBIES_1", "WI_HOBBIES_1"]:
        p = STATIONARY_REGISTRY[name]
        arm_configuration.append(StationaryArm(arm_name=name, mean=p["mean"], variance=p["variance"]))
    for name in ["CA_FOODS_2", "TX_FOODS_2"]:
        p = EVENT_SHOCK_REGISTRY[name]
        arm_configuration.append(EventShockArm(arm_name=name, base_mean=p["base_mean"], 
                                               base_variance=p["base_variance"], 
                                               shocks_csv=SHOCKS_FILE, season_csv=SEASON_FILE))
    for name in ["CA_HOUSEHOLD_1", "WI_FOODS_2"]:
        p = TREND_REGISTRY[name]
        arm_configuration.append(TrendArm(arm_name=name, start_mean=p["start_mean"], 
                                         slope=p["slope"], variance=p["variance"]))
    for name in ["TX_HOBBIES_1", "WI_FOODS_1"]:
        p = SWITCH_REGISTRY[name]
        arm_configuration.append(SwitchArm(arm_name=name, base_mean=p["base_mean"], 
                                          base_variance=p["base_variance"], switch_csv=SWITCH_FILE))
    
    env = MAB(arm_configuration)
    evaluator = CustomEvaluator(env, agents, horizon=HORIZON, global_scaler=GLOBAL_SCALER)
    rewards_log, actions_log = evaluator.run_simulation()
    
    # 결과 집계
    agent_names = [a.name for a in agents]
    results = {}
    for i, name in enumerate(agent_names):
        total = rewards_log[i].sum()
        results[name] = total
    
    # 정렬 출력
    print("\n결과:")
    for rank, (name, score) in enumerate(sorted(results.items(), key=lambda x: -x[1]), 1):
        marker = ""
        if "Thompson" in name:
            marker = " ★"
        print(f"  {rank:2d}. {name:<35s} {score:>10.4f}{marker}")
    
    return results


def main():
    print("\n" + "="*70)
    print("MARL 환경에서 Thompson Sampling 개선 실험")
    print("="*70)
    
    # ========================================
    # 실험 1: 기존 에이전트들 + 기본 Thompson
    # ========================================
    exp1_agents = [
        EpsilonGreedy(8, epsilon=0.05, name="Eps_0.05"),
        EpsilonGreedy(8, epsilon=0.1, name="Eps_0.1"),
        EpsilonGreedy(8, epsilon=0.2, name="Eps_0.2"),
        UCBAgent(8, c=0.1, name="UCB_0.1"),
        UCBAgent(8, c=0.15, name="UCB_0.15"),
        ThompsonSampling(8, reward_scale=7, name="Thompson_x7"),
    ]
    results1 = run_experiment("실험 1: 기본 Thompson (6명 경쟁)", exp1_agents)
    
    # ========================================
    # 실험 2: 주말/평일 Thompson 추가
    # ========================================
    exp2_agents = [
        EpsilonGreedy(8, epsilon=0.05, name="Eps_0.05"),
        EpsilonGreedy(8, epsilon=0.1, name="Eps_0.1"),
        EpsilonGreedy(8, epsilon=0.2, name="Eps_0.2"),
        UCBAgent(8, c=0.1, name="UCB_0.1"),
        UCBAgent(8, c=0.15, name="UCB_0.15"),
        ThompsonSampling(8, reward_scale=7, name="Thompson_x7"),
        ThompsonWeekendWeekday(8, reward_scale=7, name="Thompson_Weekend"),
    ]
    results2 = run_experiment("실험 2: 주말/평일 Thompson (7명 경쟁)", exp2_agents)
    
    # ========================================
    # 실험 3: 충돌 인식 Thompson 추가
    # ========================================
    exp3_agents = [
        EpsilonGreedy(8, epsilon=0.05, name="Eps_0.05"),
        EpsilonGreedy(8, epsilon=0.1, name="Eps_0.1"),
        EpsilonGreedy(8, epsilon=0.2, name="Eps_0.2"),
        UCBAgent(8, c=0.1, name="UCB_0.1"),
        UCBAgent(8, c=0.15, name="UCB_0.15"),
        ThompsonSampling(8, reward_scale=7, name="Thompson_x7"),
        ThompsonWeekendWeekday(8, reward_scale=7, name="Thompson_Weekend"),
        ThompsonCollisionAware(8, reward_scale=7, 
                              collision_penalty_rate=0.5, 
                              penalty_decay=0.95,
                              name="Thompson_CollisionAware"),
    ]
    results3 = run_experiment("실험 3: 충돌 인식 Thompson (8명 경쟁)", exp3_agents)
    
    # ========================================
    # 실험 4: 다양한 파라미터 Thompson들 (선택 분산)
    # ========================================
    exp4_agents = [
        EpsilonGreedy(8, epsilon=0.05, name="Eps_0.05"),
        EpsilonGreedy(8, epsilon=0.1, name="Eps_0.1"),
        EpsilonGreedy(8, epsilon=0.2, name="Eps_0.2"),
        UCBAgent(8, c=0.1, name="UCB_0.1"),
        UCBAgent(8, c=0.15, name="UCB_0.15"),
        ThompsonSampling(8, reward_scale=5, name="Thompson_x5"),
        ThompsonSampling(8, reward_scale=7, name="Thompson_x7"),
        ThompsonSampling(8, reward_scale=10, name="Thompson_x10"),
        ThompsonWeekendWeekday(8, reward_scale=7, name="Thompson_Weekend"),
        ThompsonCollisionAware(8, reward_scale=7, name="Thompson_CollisionAware"),
    ]
    results4 = run_experiment("실험 4: 다양한 Thompson (10명 경쟁)", exp4_agents)
    
    # ========================================
    # 최종 요약
    # ========================================
    print("\n" + "="*70)
    print("최종 요약")
    print("="*70)
    
    def print_top3(results, exp_name):
        print(f"\n[{exp_name}]")
        top3 = sorted(results.items(), key=lambda x: -x[1])[:3]
        for rank, (name, score) in enumerate(top3, 1):
            marker = " ★" if "Thompson" in name else ""
            print(f"  {rank}위. {name:<35s} {score:>10.4f}{marker}")
    
    print_top3(results1, "실험 1: 기본 Thompson")
    print_top3(results2, "실험 2: 주말/평일 Thompson")
    print_top3(results3, "실험 3: 충돌 인식 Thompson")
    print_top3(results4, "실험 4: 다양한 Thompson")
    
    print("\n" + "="*70)
    print("결론:")
    print("="*70)
    
    # Thompson이 상위 3위 안에 들었는지 확인
    for exp_name, results in [
        ("실험 1", results1), ("실험 2", results2), 
        ("실험 3", results3), ("실험 4", results4)
    ]:
        top3 = [name for name, _ in sorted(results.items(), key=lambda x: -x[1])[:3]]
        thompson_in_top3 = any("Thompson" in name for name in top3)
        if thompson_in_top3:
            thompson_names = [n for n in top3 if "Thompson" in n]
            print(f"✓ {exp_name}: {', '.join(thompson_names)} 상위 3위 진입")
        else:
            print(f"✗ {exp_name}: Thompson 계열 상위 3위 진입 실패")


if __name__ == "__main__":
    main()
