import pandas as pd
import matplotlib.pyplot as plt
import os

# 💡 여기에 방금 실행해서 생성된 가장 최근 폴더 이름을 넣으세요!
CSV_DIR = "output/20260325_174835" 

def plot_cumulative_rewards():
    rewards_path = os.path.join(CSV_DIR, "rewards_log.csv")
    
    # CSV 읽기 (Step 컬럼을 인덱스로 사용)
    df = pd.read_csv(rewards_path, index_col="Step")
    
    # 매 스텝 받은 보상을 계속 누적해서 더함 (계좌 잔고)
    cumulative_rewards = df.cumsum()
    
    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    for column in cumulative_rewards.columns:
        plt.plot(cumulative_rewards.index, cumulative_rewards[column], label=column, linewidth=2)
    
    plt.title("Cumulative Rewards Comparison (Epsilon-Greedy in MARL)", fontsize=14)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 그래프를 화면에 띄우고 파일로도 저장
    save_path = os.path.join(CSV_DIR, "cumulative_rewards.png")
    plt.savefig(save_path)
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_cumulative_rewards()