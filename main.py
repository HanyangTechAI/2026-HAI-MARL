# trueskill_qualifier.py
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import trueskill
import re
import warnings
from contextlib import redirect_stdout
import concurrent.futures  # 🌟 멀티프로세싱을 위한 모듈

from envs.custom_evaluator import CustomEvaluator
from utils.env_builder import build_dynamic_market
from utils.agent_factory import get_agent

warnings.filterwarnings("ignore")

def generate_candidate_pool():
    pool = []
    cs = [0.01, 0.05, 0.1]
    for c in cs:
        pool.extend([f"UCB_c{c}", f"Periodic_UCB_p7_c{c}"])
        for w in [50, 100, 200]: pool.append(f"SW_UCB_w{w}_c{c}")
        for s in [5.0, 10.0, 20.0]: pool.append(f"AS_UCB_p7_s{s}_c{c}")
        for w in [50, 100]:
            for s in [5.0, 10.0]: pool.append(f"SW_AS_UCB_w{w}_p7_s{s}_c{c}")
    for w in [50, 100]:
        for s in [0.5, 1.0, 2.0]: pool.append(f"FFT_UCB_w{w}_s{s}_c0.05")
    for e in [0.01, 0.05, 0.1, 0.2]:
        pool.extend([f"Eps_e{e}", f"DecayEps_e{e}"])
    pool.extend(["Softmax_t0.05", "Softmax_t0.1", "Softmax_t0.2", "WSLS"])
    pool.extend(["TS", "TS_Weekly", "TS_Collision_Aware"])
    return list(set(pool))

def extract_family_and_params(agent_name):
    family = re.split(r'_[ecwpst][0-9]', agent_name)[0]
    c_val = float(re.search(r'_c([0-9.]+)', agent_name).group(1)) if '_c' in agent_name else None
    return family, c_val

# 🌟 [NEW] 독립된 프로세스(CPU 코어)에서 병렬로 실행될 단일 시뮬레이션 함수
# (Windows 멀티프로세싱 규칙상 최상단 레벨에 정의되어야 합니다.)
def run_single_match(match_size, selected_players, horizon):
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        market_env, _ = build_dynamic_market("random")
        agents = [get_agent(name, market_env.nbArms) for name in selected_players]
        evaluator = CustomEvaluator(market_env, agents, horizon=horizon, global_scaler=0.0001)
        rewards_log, _ = evaluator.run_simulation()
        
    final_rewards = [np.sum(rewards_log[i, :]) for i in range(match_size)]
    sorted_indices = np.argsort(final_rewards)[::-1]
    
    ranks = [0] * match_size
    for rank, idx in enumerate(sorted_indices): 
        ranks[idx] = rank 
        
    top30_cutoff = max(1, int(match_size * 0.3))
    top30_indices = sorted_indices[:top30_cutoff]
    winner_idx = sorted_indices[0]
    
    return selected_players, ranks, winner_idx, top30_indices, match_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--matches", type=int, default=1500)
    parser.add_argument("-t", "--horizon", type=int, default=1000)
    args = parser.parse_args()
    
    candidates = generate_candidate_pool()
    env_ts = trueskill.TrueSkill(draw_probability=0.0)
    ratings = {name: env_ts.create_rating() for name in candidates}
    
    match_sizes = [8, 10, 12, 16, 20, 24, 28, 32]
    stats = {name: {"matches": 0, "wins": 0, "top30": 0} for name in candidates}
    stats_by_size = {size: {name: {"matches": 0, "top30": 0} for name in candidates} for size in match_sizes}

    # 매치업 세팅 사전 준비
    match_configs = []
    for _ in range(args.matches):
        size = random.choice(match_sizes)
        players = random.sample(candidates, size)
        match_configs.append((size, players, args.horizon))

    # 🌟 CPU 코어 개수 확인 (1개는 시스템용으로 남겨둠)
    max_workers = max(1, os.cpu_count() - 1)
    print(f"⚡ CPU 코어 {max_workers}개를 풀가동하여 병렬 처리를 시작합니다...")

    # 🌟 멀티프로세싱 풀(Pool) 가동
    with tqdm(total=args.matches, desc="⚡ 병렬 매치메이킹", file=sys.stdout, position=0, leave=True, dynamic_ncols=True, ascii=False) as pbar:
        
        # 멀티프로세싱 풀 가동
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_match, cfg[0], cfg[1], cfg[2]) for cfg in match_configs]
            
            for future in concurrent.futures.as_completed(futures):
                players, ranks, winner_idx, top30_indices, match_size = future.result()
                
                # 승리 통계 업데이트
                stats[players[winner_idx]]["wins"] += 1
                for i, p in enumerate(players):
                    stats[p]["matches"] += 1
                    stats_by_size[match_size][p]["matches"] += 1
                    if i in top30_indices:
                        stats[p]["top30"] += 1
                        stats_by_size[match_size][p]["top30"] += 1
                
                # TrueSkill 점수 업데이트
                teams = [(ratings[name],) for name in players]
                new_ratings = env_ts.rate(teams, ranks=ranks)
                for i, name in enumerate(players): 
                    ratings[name] = new_ratings[i][0]
                    
                # 🌟 [핵심] 한 판이 끝날 때마다 바를 1씩 전진시킵니다.
                pbar.update(1)

    # 정규화 및 결과 집계
    min_bound = min([ratings[n].mu - 3 * ratings[n].sigma for n in candidates])
    max_bound = max([ratings[n].mu + 3 * ratings[n].sigma for n in candidates])
    range_bound = max_bound - min_bound + 1e-9

    results = []
    for name in candidates:
        r = ratings[name]
        family, c_val = extract_family_and_params(name)
        
        norm_mu = ((r.mu - min_bound) / range_bound) * 100
        norm_sigma = (r.sigma / range_bound) * 100
        norm_conservative = ((r.mu - 3 * r.sigma - min_bound) / range_bound) * 100
        
        row = {
            "Agent": name,
            "Family": family,
            "Param (c)": c_val,
            "Total Matches": stats[name]["matches"],
            "Win Rate (%)": (stats[name]["wins"] / max(1, stats[name]["matches"])) * 100,
            "Total Top 30% (%)": (stats[name]["top30"] / max(1, stats[name]["matches"])) * 100,
            "Norm Mu (0-100)": norm_mu,
            "Norm Sigma": norm_sigma,
            "Score (Conservative 0-100)": norm_conservative
        }
        for size in match_sizes:
            m_cnt = stats_by_size[size][name]["matches"]
            row[f"Top 30% (N={size})"] = (stats_by_size[size][name]["top30"] / max(1, m_cnt)) * 100
            
        results.append(row)
        
    df_leaderboard = pd.DataFrame(results).sort_values(by="Score (Conservative 0-100)", ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"TS_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "Leaderboard_Detailed.csv")
    df_leaderboard.to_csv(save_path, index_label="Rank")
    print(f"\n✅ 완료! 결과 파일이 {output_dir} 에 저장되었습니다.")
    
    try:
        from utils.plot_trueskill import plot_trueskill_results
        plot_trueskill_results(save_path, output_dir)
    except Exception as e:
        print(f"⚠️ 시각화 모듈 에러: {e}")

if __name__ == "__main__":
    main()