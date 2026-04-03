# utils/mc_logger.py
import os
import numpy as np
import pandas as pd

class MCLogger:
    def __init__(self, master_dir, agent_names, horizon, iters):
        self.master_dir = master_dir
        self.agent_names = agent_names
        
        # 에이전트별 시계열 데이터 저장소 (3D Tensor 형태를 Dict로 관리)
        self.all_rewards = {name: np.zeros((iters, horizon)) for name in agent_names}
        self.all_actions = {name: np.zeros((iters, horizon)) for name in agent_names}
        
        self.market_envs = []
        self.scorecard = []

    def record_seed(self, seed_idx, current_seed, rewards_log, actions_log, arm_names, final_rewards):
        """1회 시뮬레이션(시드)이 끝날 때마다 데이터를 RAM에 기록합니다."""
        for i, name in enumerate(self.agent_names):
            self.all_rewards[name][seed_idx, :] = rewards_log[i, :]
            self.all_actions[name][seed_idx, :] = actions_log[i, :]
            
        self.market_envs.append({"Seed": current_seed, "Arms_Config": ", ".join(arm_names)})
        self.scorecard.append(final_rewards)

    def save_all_and_summarize(self):
        """모든 몬테카를로 반복이 끝나면 CSV로 구워내고 요약표를 반환합니다."""
        print(f"💾 {self.master_dir} 에 세부 로그 데이터를 저장 중입니다...")
        
        # Tier 2: 에이전트별 상세 시계열 로그 저장
        agent_dir = os.path.join(self.master_dir, "agent_logs")
        os.makedirs(agent_dir, exist_ok=True)
        
        for name in self.agent_names:
            pd.DataFrame(self.all_rewards[name]).to_csv(os.path.join(agent_dir, f"{name}_rewards.csv"), index=False)
            pd.DataFrame(self.all_actions[name]).to_csv(os.path.join(agent_dir, f"{name}_actions.csv"), index=False)

        # Tier 3: 시장 환경 기록 저장
        pd.DataFrame(self.market_envs).to_csv(os.path.join(self.master_dir, "market_environments.csv"), index=False)

        # Tier 1: 마스터 스코어카드 저장 및 통계 계산
        df_results = pd.DataFrame(self.scorecard).set_index('Seed')
        df_results.to_csv(os.path.join(self.master_dir, "all_seeds_raw_data.csv"))

        mean_rewards = df_results.mean()
        std_rewards = df_results.std()
        win_counts = df_results.idxmax(axis=1).value_counts()
        win_rates = (win_counts / len(df_results)) * 100
        
        summary_df = pd.DataFrame({
            "Avg Reward": mean_rewards,
            "Risk (Std)": std_rewards,
            "Win Count": win_counts,
            "Win Rate (%)": win_rates
        }).fillna(0).sort_values(by="Avg Reward", ascending=False)
        
        summary_df.to_csv(os.path.join(self.master_dir, "master_scorecard.csv"))
        return summary_df