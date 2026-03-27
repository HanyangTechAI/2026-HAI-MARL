import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_experiment_results(csv_dir, scaler=10000):
    """
    저장된 CSV 파일을 읽어 전체 결과 및 에이전트별 심층 분석 그래프를 자동 생성합니다.
    (main_time.py 에서 시뮬레이션 종료 직후 자동으로 호출됩니다.)
    """
    print(f"📊 [시각화 모듈] 데이터 분석 및 그래프 생성을 시작합니다... ({csv_dir})")
    
    rewards_path = os.path.join(csv_dir, "rewards_log.csv")
    actions_path = os.path.join(csv_dir, "actions_log.csv")
    
    if not os.path.exists(rewards_path) or not os.path.exists(actions_path):
        print(f"🚨 에러: CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return
    
    # 데이터 로드 및 역스케일링 (스케일러를 곱해 실제 단위로 복구)
    df_rewards = pd.read_csv(rewards_path, index_col="Step") * scaler
    df_actions = pd.read_csv(actions_path, index_col="Step")
    
    # 공통 스타일 설정
    plt.style.use('default')
    
    # ===================================================
    # [Phase 1] 거시적 지표 (Macro Metrics) - 기존 + 3중 이동평균
    # ===================================================
    
    # 1. 누적 보상 (Cumulative Rewards)
    plt.figure(figsize=(10, 6))
    cumulative_rewards = df_rewards.cumsum()
    for col in cumulative_rewards.columns:
        plt.plot(cumulative_rewards.index, cumulative_rewards[col], label=col, linewidth=2)
    plt.title("Cumulative Rewards (Total Profit)", fontsize=16, fontweight='bold')
    plt.xlabel("Steps (Days)", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig1_cumulative_rewards.png"), dpi=200)
    plt.close()

    # 2. 10일 / 50일 / 100일 이동평균 (Moving Averages) 🌟 추가됨
    # Subplot을 사용하여 하나의 이미지에 3개의 그래프를 예쁘게 배치합니다.
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    windows = [10, 50, 100]
    titles = ["10-Day MA (Short-term Shock Response)", 
              "50-Day MA (Mid-term Trend)", 
              "100-Day MA (Long-term Stability)"]

    for ax, window, title in zip(axes, windows, titles):
        rolling_rewards = df_rewards.rolling(window=window, min_periods=1).mean()
        for col in rolling_rewards.columns:
            ax.plot(rolling_rewards.index, rolling_rewards[col], label=col, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Reward")
        ax.grid(True, linestyle='--', alpha=0.6)
        if window == 10: # 첫 번째 그래프에만 범례 표시
            ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            
    plt.xlabel("Steps (Days)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig2_moving_averages_10_50_100.png"), dpi=200)
    plt.close()

    # 3. 시장 충돌 트렌드 (Market Collision Trend)
    plt.figure(figsize=(10, 5))
    num_agents = len(df_actions.columns)
    unique_choices = df_actions.apply(lambda row: len(np.unique(row)), axis=1)
    collision_intensity = num_agents - unique_choices
    collision_trend = collision_intensity.rolling(window=100, min_periods=1).mean()
    
    plt.plot(collision_trend.index, collision_trend.values, color='firebrick', linewidth=2)
    plt.title("Market Collision Trend (Slippage Intensity)", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Avg Collisions per Step", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(collision_trend.index, collision_trend.values, color='red', alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(csv_dir, "fig3_collision_trend.png"), dpi=200)
    plt.close()

    # ===================================================
    # [Phase 2] 에이전트별 미시적 행동 분석 (Micro Analysis) 🌟 추가됨
    # ===================================================
    print("📂 에이전트별 심층 분석 폴더 및 행동 변화 그래프를 생성합니다...")
    
    # Arm의 총 개수 파악 (컬러맵 매칭을 위해)
    unique_arms = np.sort(pd.unique(df_actions.values.ravel()))
    
    for agent_name in df_actions.columns:
        # 1. 에이전트 전용 폴더 생성
        agent_dir = os.path.join(csv_dir, agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        
        agent_actions = df_actions[agent_name]
        
        # 2. 파이 차트 (전체 기간 동안 각 Arm을 선택한 총 빈도)
        action_counts = agent_actions.value_counts().sort_index()
        plt.figure(figsize=(6, 6))
        plt.pie(action_counts, labels=[f"Arm {int(arm)}" for arm in action_counts.index], 
                autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10(action_counts.index / max(unique_arms)))
        plt.title(f"Total Arm Selection Distribution\n({agent_name})", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(agent_dir, f"{agent_name}_action_pie_chart.png"), dpi=150)
        plt.close()

        # 3. 행동 변화 추이 (100일 이동평균 Stacked Area Chart) - 논문 핵심 그래프!
        # 행동을 One-hot 인코딩 후 이동평균을 구해서 확률 변화를 시각화
        action_dummies = pd.get_dummies(agent_actions)
        rolling_prob = action_dummies.rolling(window=100, min_periods=1).mean()
        
        plt.figure(figsize=(10, 5))
        plt.stackplot(rolling_prob.index, rolling_prob.T, 
                      labels=[f"Arm {int(col)}" for col in rolling_prob.columns],
                      colors=plt.cm.tab10(rolling_prob.columns / max(unique_arms)), alpha=0.8)
        
        plt.title(f"Arm Selection Probability Over Time (100-Day Rolling)\n({agent_name})", fontsize=14, fontweight='bold')
        plt.xlabel("Steps (Days)")
        plt.ylabel("Selection Probability")
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.margins(x=0, y=0) # 여백 제거
        plt.tight_layout()
        plt.savefig(os.path.join(agent_dir, f"{agent_name}_action_transition.png"), dpi=200)
        plt.close()

    print(f"✅ 모든 시각화 작업이 완료되었습니다! 폴더를 확인하세요: {csv_dir}")

# 단독으로 실행할 때를 대비한 코드
if __name__ == "__main__":
    # 테스트용 기본 폴더 (실제 사용 시 경로 수정)
    TEST_DIR = "output/20260327_132358" 
    plot_experiment_results(TEST_DIR)