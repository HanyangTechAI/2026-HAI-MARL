# utils/plot_batch_results.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def plot_monte_carlo_results(master_output_dir):
    print(f"📊 시각화 모듈 가동: {master_output_dir} 의 데이터를 바탕으로 6대 그래프를 생성합니다...")
    
    raw_data_path = os.path.join(master_output_dir, "all_seeds_raw_data.csv")
    scorecard_path = os.path.join(master_output_dir, "master_scorecard.csv")
    agent_logs_dir = os.path.join(master_output_dir, "agent_logs")
    
    if not os.path.exists(raw_data_path) or not os.path.exists(scorecard_path):
        print("🚨 에러: 시각화할 CSV 데이터 파일이 없습니다.")
        return

    df_raw = pd.read_csv(raw_data_path).set_index('Seed')
    df_score = pd.read_csv(scorecard_path, index_col=0)

    # 한글 폰트 설정
    plt.rc('font', family='Malgun Gothic') 
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic" if os.name == 'nt' else "AppleGothic")

    # ==========================================
    # 📈 1. 누적 수익 분포 (Boxplot)
    # ==========================================
    plt.figure(figsize=(12, 8))
    ax1 = sns.boxplot(data=df_raw, orient="h", palette="Set2", showmeans=True, 
                     meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"})
    plt.title("MC_01: 에이전트별 누적 수익 분포 (Risk & Return)", fontsize=16, fontweight='bold')
    plt.xlabel("Total Cumulative Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_01_Reward_Distribution.png"), dpi=300)
    plt.close()

    # ==========================================
    # 🏆 2. 승률 (Bar Chart)
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    # 1등 승률과 상위 30% 진입률 데이터만 추출하여 상위 진입률 기준으로 정렬
    df_rates = df_score[['Win Rate (%)', 'Top 30% Rate (%)']].sort_values(by='Top 30% Rate (%)', ascending=False)
    
    # Pandas 내장 plot 함수를 사용하면 다중 바 차트를 쉽게 그릴 수 있습니다.
    ax2 = df_rates.plot(kind='bar', figsize=(14, 7), color=['#e74c3c', '#3498db'], edgecolor='black', alpha=0.8)
    
    plt.title("MC_02: 에이전트별 1등 승률 및 상위 30% 진입률 (안정성 평가)", fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Agent Models", fontsize=14)
    plt.ylabel("Percentage (%)", fontsize=14)
    plt.ylim(0, 105)
    plt.legend(["1st Place Win Rate (%)", "Top 30% Entry Rate (%)"], fontsize=12)
    
    # 바 위에 퍼센티지 텍스트 추가
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f"{height:.1f}%", 
                         (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_02_Win_and_TopN_Rates.png"), dpi=300)
    plt.close()

    # ==========================================
    # 🎯 3. Risk-Return 마코위츠 전선 (Scatter Plot)
    # ==========================================
    plt.figure(figsize=(10, 8))
    ax3 = sns.scatterplot(x=df_score['Risk (Std)'], y=df_score['Avg Reward'], 
                          hue=df_score.index, s=200, palette="tab10", edgecolor="black")
    plt.title("MC_03: Risk-Return 효율적 전선 (좌상단일수록 우수)", fontsize=16, fontweight='bold')
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return (Average Reward)")
    # 주석 달기
    for i, txt in enumerate(df_score.index):
        ax3.annotate(txt, (df_score['Risk (Std)'][i]+1, df_score['Avg Reward'][i]), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_03_Risk_Return_Scatter.png"), dpi=300)
    plt.close()

    # 시계열 로그가 있는지 확인 후 고도화 그래프 진행
    if not os.path.exists(agent_logs_dir):
        print("✅ 기본 시각화 완료 (시계열 로그가 없어 고도화 그래프는 스킵합니다).")
        return

    print("🔍 시계열 데이터(Action/Reward Logs)를 분석하여 심층 시각화를 진행합니다...")
    agent_names = df_score.index.tolist()
    
    # ==========================================
    # 🗺️ 4. 행동 선호도 히트맵 (Action Preference)
    # ==========================================
    # Arm 인덱스 맵핑 (0,1: Stationary, 2,3: Shocks, 4,5: Trends, 6,7: Switches)
    category_map = {0: '1_Stationary', 1: '1_Stationary', 2: '2_Shocks', 3: '2_Shocks', 
                    4: '3_Trends', 5: '3_Trends', 6: '4_Switches', 7: '4_Switches'}
    
    heatmap_data = []
    for agent in agent_names:
        action_file = os.path.join(agent_logs_dir, f"{agent}_actions.csv")
        if os.path.exists(action_file):
            df_act = pd.read_csv(action_file)
            # 모든 시드, 모든 스텝의 행동을 1차원 배열로 펼침
            all_actions = df_act.values.flatten()
            mapped_actions = pd.Series(all_actions).map(category_map)
            counts = mapped_actions.value_counts(normalize=True) * 100
            counts.name = agent
            heatmap_data.append(counts)
            
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data).fillna(0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Selection %'})
        plt.title("MC_04: 에이전트별 카테고리 선호도 히트맵", fontsize=16, fontweight='bold')
        plt.ylabel("Agents")
        plt.xlabel("Market Categories")
        plt.tight_layout()
        plt.savefig(os.path.join(master_output_dir, "MC_04_Action_Preference_Heatmap.png"), dpi=300)
        plt.close()

    # ==========================================
    # 📉 5. Learning Curve (이동평균 시계열)
    # ==========================================
    plt.figure(figsize=(14, 8))
    for agent in agent_names[:6]: # 너무 많으면 복잡하므로 Top 6만 그림
        reward_file = os.path.join(agent_logs_dir, f"{agent}_rewards.csv")
        if os.path.exists(reward_file):
            df_rew = pd.read_csv(reward_file)
            # 시드(행)별 평균을 구하고, 가독성을 위해 30일 이동평균 적용
            mean_rewards = df_rew.mean(axis=0).rolling(window=30, min_periods=1).mean()
            plt.plot(mean_rewards.index, mean_rewards.values, label=agent, linewidth=2)
            
    plt.title("MC_05: Learning Curve (일일 보상 30일 이동평균)", fontsize=16, fontweight='bold')
    plt.xlabel("Steps (Days)")
    plt.ylabel("Daily Expected Reward (Avg over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(master_output_dir, "MC_05_Learning_Curve.png"), dpi=300)
    plt.close()

    # ==========================================
    # 🕸️ 6. 스탯 방사형 차트 (Radar Chart - Top 4)
    # ==========================================
    top_4_agents = df_score.sort_values(by='Top 30% Rate (%)', ascending=False).index[:4].tolist()
    
    if len(top_4_agents) >= 3:
        # 평가 지표에 '상위 30% 방어력'을 새롭게 추가!
        categories = ['총 수익성(Return)', '안정성(Risk 방어)', '상위 30% 방어력', '트렌드 추종', '함정 회피력']
        N = len(categories)
        
        radar_df = pd.DataFrame(index=top_4_agents, columns=categories)
        
        for agent in top_4_agents:
            radar_df.loc[agent, '총 수익성(Return)'] = df_score.loc[agent, 'Avg Reward']
            radar_df.loc[agent, '안정성(Risk 방어)'] = 1 / (df_score.loc[agent, 'Risk (Std)'] + 1e-5)
            # 🌟 Win Rate 대신 Top 30% Rate를 레이더 차트의 주력 지표로 사용!
            radar_df.loc[agent, '상위 30% 방어력'] = df_score.loc[agent, 'Top 30% Rate (%)']
            
            action_file = os.path.join(agent_logs_dir, f"{agent}_actions.csv")
            if os.path.exists(action_file):
                df_act = pd.read_csv(action_file)
                late_trend = df_act.iloc[:, int(df_act.shape[1]*0.8):].isin([4,5]).mean().mean()
                radar_df.loc[agent, '트렌드 추종'] = late_trend
                avoid_switch = 1 - df_act.isin([6,7]).mean().mean()
                radar_df.loc[agent, '함정 회피력'] = avoid_switch

        # MinMax 스케일링 (0~1)
        for col in categories:
            col_min, col_max = radar_df[col].min(), radar_df[col].max()
            if col_max - col_min == 0: radar_df[col] = 1.0
            else: radar_df[col] = (radar_df[col] - col_min) / (col_max - col_min)
            
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        plt.title("MC_06: Top 4 모델 헥사곤 스탯 (Top N% 안정성 기준)", size=16, fontweight='bold', pad=20)
        plt.xticks(angles[:-1], categories)
        
        colors = ['b', 'r', 'g', 'm']
        for i, agent in enumerate(top_4_agents):
            values = radar_df.loc[agent].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=agent, color=colors[i])
            ax.fill(angles, values, color=colors[i], alpha=0.1)
            
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(master_output_dir, "MC_06_Radar_Chart.png"), dpi=300)
        plt.close()

    print(f"✅ 모든 심층 분석 시각화가 완료되었습니다!")