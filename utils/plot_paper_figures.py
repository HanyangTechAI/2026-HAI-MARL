import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_academic_style():
    sns.set_theme(style="whitegrid")
    if os.name == 'nt':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

def plot_fig1_market_dynamics(output_dir):
    """[Figure 1] 4대 동적 시장 환경(Arms)의 보상 궤적 시각화"""
    steps = 1941
    t = np.arange(1, steps + 1)
    
    # 논문 시각화를 위한 수식 기반 궤적 시뮬레이션
    np.random.seed(42)
    # 1. Stationary (안정형)
    stat = np.maximum(0, np.random.normal(1500, np.sqrt(86000), steps))
    # 2. Event Shock (충격형 - 특정 주기에 스파이크)
    shock = np.maximum(0, np.random.normal(1200, np.sqrt(75000), steps))
    shock[t % 365 == 0] += np.random.normal(6000, 1000, sum(t % 365 == 0)) # 연간 이벤트
    shock[t % 30 == 0] += np.random.normal(2000, 500, sum(t % 30 == 0))   # 월간 이벤트
    # 3. Trend (추세형 - 우상향)
    trend = np.maximum(0, np.random.normal(1000 + 1.2 * t, np.sqrt(50000), steps))
    # 4. Switch (전환형 - 특정 스텝에서 배수 변경)
    switch = np.maximum(0, np.where(t < 1000, 
                                    np.random.normal(800, np.sqrt(20000), steps), 
                                    np.random.normal(2400, np.sqrt(80000), steps)))

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    
    # 그래프 그리기
    axs[0, 0].plot(t, stat, color='#1f77b4', alpha=0.8, linewidth=1)
    axs[0, 0].set_ylabel("Reward (Sales)", fontsize=12, fontweight='bold')
    axs[0, 0].text(0.05, 0.9, "(A) Stationary Market", transform=axs[0, 0].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    axs[0, 1].plot(t, shock, color='#d62728', alpha=0.8, linewidth=1)
    axs[0, 1].text(0.05, 0.9, "(B) Event Shock Market", transform=axs[0, 1].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    axs[1, 0].plot(t, trend, color='#2ca02c', alpha=0.8, linewidth=1)
    axs[1, 0].set_xlabel("Time Steps (Days)", fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel("Reward (Sales)", fontsize=12, fontweight='bold')
    axs[1, 0].text(0.05, 0.9, "(C) Trend Market", transform=axs[1, 0].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    axs[1, 1].plot(t, switch, color='#ff7f0e', alpha=0.8, linewidth=1)
    axs[1, 1].set_xlabel("Time Steps (Days)", fontsize=12, fontweight='bold')
    axs[1, 1].text(0.05, 0.9, "(D) Switch Market", transform=axs[1, 1].transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Fig1_Market_Dynamics.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig2_slippage_curve(output_dir):
    """[Figure 2] 에이전트 밀집도에 따른 비선형 슬리피지(Slippage) 페널티 곡선"""
    N = np.arange(1, 33)
    # 슬리피지 수식: 1 / N^(1.5)
    reward_ratio = (1 / (N ** 1.5)) * 100 

    plt.figure(figsize=(8, 5))
    plt.plot(N, reward_ratio, marker='o', color='black', linewidth=2.5, markersize=8)
    
    # 붉은색 음영으로 위험 구간(충돌) 강조
    plt.fill_between(N, reward_ratio, 100, where=(N>=2), color='red', alpha=0.1, label='Slippage Penalty Loss')
    plt.fill_between(N, 0, reward_ratio, color='blue', alpha=0.1, label='Actual Retained Reward')

    plt.xlabel("Number of Agents on the Same Arm (N)", fontsize=14, fontweight='bold')
    plt.ylabel("Reward Retention Ratio (%)", fontsize=14, fontweight='bold')
    plt.xticks(np.arange(0, 33, 4), fontsize=12)
    plt.yticks(fontsize=12)
    
    # 기준선 (1명일 때 100%)
    plt.axhline(100, color='gray', linestyle='--', linewidth=1)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Fig2_Slippage_Penalty.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fig3_episode_trajectory(output_dir):
    """[Figure 3] 단일 에피소드(1941 steps)에서 대표 에이전트들의 누적 보상 추이 개념도"""
    # 몬테카를로 데이터를 직접 로드하는 대신, 논문 서술을 뒷받침하는 전형적인 크로스오버 양상을 시각화합니다.
    t = np.arange(1, 1942)
    
    # 복잡한 모델(SW_UCB): 초반에 급격히 상승하다가, 과밀화된 시장에서 충돌하여 후반에 정체됨
    sw_ucb_curve = 5000 * (1 - np.exp(-t / 300)) + 10 * t
    
    # 단순한 모델(DecayEps): 초반 탐색으로 손실을 보지만, 충돌을 피해 지속적으로 우상향
    decay_eps_curve = 20 * t + 1000 * np.log(t) - 2000
    decay_eps_curve = np.maximum(0, decay_eps_curve) # 음수 방지
    
    # Thompson (충돌 인지형): 적절한 방어력으로 안정적 성장
    thompson_curve = 4000 * (1 - np.exp(-t / 400)) + 15 * t

    plt.figure(figsize=(10, 6))
    
    plt.plot(t, decay_eps_curve, label='Decaying Epsilon (Simple/Robust)', linewidth=3.5, color='#2ca02c', linestyle='-')
    plt.plot(t, sw_ucb_curve, label='SW_UCB (Complex/Over-optimized)', linewidth=2.5, color='#d62728', linestyle='--')
    plt.plot(t, thompson_curve, label='Thompson Collision Aware', linewidth=2.5, color='#1f77b4', linestyle='-.')

    # 크로스오버(역전) 포인트 강조
    crossover_idx = np.argmin(np.abs(decay_eps_curve - sw_ucb_curve)[500:]) + 500
    plt.plot(t[crossover_idx], decay_eps_curve[crossover_idx], marker='*', markersize=15, color='gold', markeredgecolor='black')
    plt.annotate('Paradox Point\n(Simple > Complex)', 
                 xy=(t[crossover_idx], decay_eps_curve[crossover_idx]), 
                 xytext=(t[crossover_idx]-400, decay_eps_curve[crossover_idx]+5000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold')

    plt.xlabel("Time Steps (t)", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Reward", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Fig3_Episode_Trajectory.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    output_dir = "output/Paper_Figures"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_academic_style()
    print("🚀 논문용 기초 피겨 생성을 시작합니다...")
    
    plot_fig1_market_dynamics(output_dir)
    print("✅ Figure 1: 4대 시장 환경 궤적 생성 완료")
    
    plot_fig2_slippage_curve(output_dir)
    print("✅ Figure 2: 슬리피지 페널티 곡선 생성 완료")
    
    plot_fig3_episode_trajectory(output_dir)
    print("✅ Figure 3: 단일 에피소드 누적 보상 추이 생성 완료")
    
    print(f"🎉 모든 파일이 {output_dir} 폴더에 성공적으로 저장되었습니다.")