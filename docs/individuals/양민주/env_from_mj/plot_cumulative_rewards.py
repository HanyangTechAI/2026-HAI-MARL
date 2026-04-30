#!/usr/bin/env python
# docs/individuals/양민주/plot_cumulative_rewards.py
"""
21-Arm 실험 결과의 Cumulative Rewards 시각화

실행 방법:
python docs/individuals/양민주/plot_cumulative_rewards.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def plot_cumulative_rewards(result_dir):
    """
    각 에이전트의 cumulative rewards를 시각화
    
    Args:
        result_dir: 결과 폴더 경로
    """
    agent_logs_dir = os.path.join(result_dir, "agent_logs")
    
    # agent_logs 폴더에서 에이전트 목록 자동 추출
    agents = []
    for filename in os.listdir(agent_logs_dir):
        if filename.endswith("_rewards.csv"):
            agent_name = filename.replace("_rewards.csv", "")
            agents.append(agent_name)
    
    agents.sort()  # 정렬
    
    print(f"📋 발견된 에이전트: {len(agents)}개")
    for agent in agents:
        print(f"   - {agent}")
    
    # 색상 팔레트 (11개 이상 대응)
    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
        "#BC4B51", "#5F0F40", "#0F4C5C", "#E36414", "#9A031E",
        "#5F5AA2", "#2A9D8F", "#E76F51", "#264653", "#F4A261"
    ]
    
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(16, 9))
    
    print("📊 Cumulative Rewards 계산 중...")
    
    for idx, agent in enumerate(agents):
        rewards_file = os.path.join(agent_logs_dir, f"{agent}_rewards.csv")
        
        if not os.path.exists(rewards_file):
            print(f"⚠️  {agent} 파일을 찾을 수 없습니다: {rewards_file}")
            continue
        
        # CSV 읽기 (첫 행은 시드 번호들)
        df = pd.read_csv(rewards_file, header=0)
        
        # 각 시드(컬럼)별로 cumulative sum 계산
        cumulative_rewards = []
        for col in df.columns:
            rewards = df[col].values
            cumsum = np.cumsum(rewards)
            cumulative_rewards.append(cumsum)
        
        # 평균과 표준편차 계산
        cumulative_rewards = np.array(cumulative_rewards)  # shape: (num_seeds, horizon)
        mean_cumulative = np.mean(cumulative_rewards, axis=0)
        std_cumulative = np.std(cumulative_rewards, axis=0)
        
        steps = np.arange(len(mean_cumulative))
        
        # 색상 및 스타일 선택
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # 평균 라인 그리기
        ax.plot(steps, mean_cumulative, 
                label=agent, 
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=0.9)
        
        # 신뢰구간 (±1 std) 그리기
        ax.fill_between(steps, 
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        color=color,
                        alpha=0.12)
        
        print(f"✅ {agent}: 최종 평균 누적 보상 = {mean_cumulative[-1]:.1f}")
    
    # 그래프 꾸미기
    ax.set_xlabel("Time Steps (Days)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Cumulative Rewards", fontsize=14, fontweight='bold')
    ax.set_title("21-Arm 환경: 알고리즘별 누적 보상 비교 (20회 평균)", 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 축 범위 설정
    ax.set_xlim(0, len(steps))
    ax.set_ylim(bottom=0)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 저장
    output_path = os.path.join(result_dir, "MC_03_Cumulative_Rewards.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 저장 완료: {output_path}")
    
    plt.close()

def main():
    import sys
    
    # 명령줄 인자로 결과 폴더 경로 받기
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        # 기본값
        result_dir = os.path.join("docs", "individuals", "양민주", "results_21arms_20260416_171925")
    
    if not os.path.exists(result_dir):
        print(f"❌ 결과 폴더를 찾을 수 없습니다: {result_dir}")
        return
    
    print("="*70)
    print("🚀 Cumulative Rewards 시각화 시작")
    print("="*70)
    
    plot_cumulative_rewards(result_dir)
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()
