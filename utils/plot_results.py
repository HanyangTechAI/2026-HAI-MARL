import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 💡 분석하고 싶은 실험의 타임스탬프 폴더 경로를 여기에 넣으세요!
CSV_DIR = "output/20260325_181221"

def plot_and_save_individual_metrics(csv_dir):
    rewards_path = os.path.join(csv_dir, "rewards_log.csv")
    actions_path = os.path.join(csv_dir, "actions_log.csv")
    
    df_rewards = pd.read_csv(rewards_path, index_col="Step")
    df_actions = pd.read_csv(actions_path, index_col="Step")
    
    OPTIMAL_ARM = 0
    OPTIMAL_REWARD = 0.10
    
    # 공통 스타일 설정
    plt.style.use('default')
    
    # ---------------------------------------------------
    # [Figure 1] Cumulative Rewards
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    cumulative_rewards = df_rewards.cumsum()
    for col in cumulative_rewards.columns:
        plt.plot(cumulative_rewards.index, cumulative_rewards[col], label=col, linewidth=2)
    plt.title("Cumulative Rewards (Account Balance)", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig1_cumulative_rewards.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------
    # [Figure 2] Cumulative Regret
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    ideal_rewards = OPTIMAL_REWARD * (df_rewards.index + 1)
    regret = df_rewards.apply(lambda actual: ideal_rewards - actual.cumsum())
    for col in regret.columns:
        plt.plot(regret.index, regret[col], label=col, linewidth=2)
    plt.title("Cumulative Regret (Lower is Better)", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Regret", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig2_cumulative_regret.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------
    # [Figure 3] Learning Curve (% Optimal Action)
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    is_optimal = (df_actions == OPTIMAL_ARM).astype(int)
    rolling_accuracy = is_optimal.rolling(window=200, min_periods=1).mean()
    for col in rolling_accuracy.columns:
        plt.plot(rolling_accuracy.index, rolling_accuracy[col], label=col, linewidth=2)
    plt.title("Optimal Action Selection % (Learning Curve)", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Optimal Choice Probability", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig3_learning_curve.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------
    # [Figure 4] Market Collision Trend
    # ---------------------------------------------------
    plt.figure(figsize=(8, 6))
    num_agents = len(df_actions.columns)
    unique_choices = df_actions.apply(lambda row: len(np.unique(row)), axis=1)
    collision_intensity = num_agents - unique_choices
    collision_trend = collision_intensity.rolling(window=200, min_periods=1).mean()
    
    plt.plot(collision_trend.index, collision_trend.values, color='firebrick', linewidth=2)
    plt.title("Market Collision Trend (Slippage Intensity)", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Avg Collisions per Step", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(collision_trend.index, collision_trend.values, color='red', alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig4_collision_trend.png"), dpi=300)
    plt.close()

    print(f"✅ 4개의 개별 그래프가 성공적으로 저장되었습니다: {csv_dir}")

if __name__ == "__main__":
    plot_and_save_individual_metrics(CSV_DIR)