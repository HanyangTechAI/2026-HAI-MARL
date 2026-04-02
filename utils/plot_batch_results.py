# utils/plot_batch_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_monte_carlo_results(master_output_dir):
    print(f"📊 시각화 모듈 가동: {master_output_dir} 의 데이터를 바탕으로 그래프를 생성합니다...")
    
    # 데이터 불러오기
    raw_data_path = os.path.join(master_output_dir, "all_seeds_raw_data.csv")
    scorecard_path = os.path.join(master_output_dir, "master_scorecard.csv")
    
    if not os.path.exists(raw_data_path) or not os.path.exists(scorecard_path):
        print("🚨 에러: 시각화할 CSV 데이터 파일이 없습니다.")
        return

    df_raw = pd.read_csv(raw_data_path).set_index('Seed')
    df_score = pd.read_csv(scorecard_path, index_col=0)

    # 한글 폰트 깨짐 방지 및 스타일 설정
    plt.rc('font', family='Malgun Gothic') # 윈도우 맑은고딕 (맥은 'AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic" if os.name == 'nt' else "AppleGothic")

    # ==========================================
    # 📈 Figure 1: 에이전트별 누적 수익 분포 (Boxplot)
    # ==========================================
    plt.figure(figsize=(14, 8))
    
    # 박스플롯: 각 에이전트의 안정성(Risk)과 아웃라이어를 한눈에 보여줌
    ax = sns.boxplot(data=df_raw, orient="h", palette="Set2", showmeans=True, 
                     meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"8"})
    
    plt.title("Distribution of Total Cumulative Rewards Across Monte Carlo Seeds", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Total Units Sold (Reward)", fontsize=14)
    plt.ylabel("Agent Models", fontsize=14)
    
    # X축 숫자가 너무 크면 보기 좋게 포맷팅
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_01_Reward_Distribution_Boxplot.png"), dpi=300)
    plt.close()

    # ==========================================
    # 🏆 Figure 2: 에이전트별 승률 (Bar Chart)
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # 승률 높은 순으로 정렬하여 바 차트 생성
    win_rates = df_score['Win Rate (%)'].sort_values(ascending=False)
    
    ax2 = sns.barplot(x=win_rates.index, y=win_rates.values, palette="rocket")
    
    plt.title("Agent Win Rate (%) in Dynamic Markets", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Agent Models", fontsize=14)
    plt.ylabel("Win Rate (%)", fontsize=14)
    plt.ylim(0, 105) # 승률이므로 0~100 고정
    
    # 바 위에 정확한 퍼센티지 텍스트 추가
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.1f}%", 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_02_Win_Rate_BarChart.png"), dpi=300)
    plt.close()

    print(f"✅ 시각화 완료! 2장의 분석 그래프가 저장되었습니다.")