"""
MARL 실험 결과 시각화
"""
import matplotlib.pyplot as plt
import numpy as np

# 실험 결과 데이터 (experiment_marl.py 실행 결과)
experiments = {
    "실험 2 (7명)": {
        "Thompson_x7": 249.37,
        "Eps_0.2": 246.60,
        "Thompson_Weekend": 236.96,
        "Eps_0.1": 219.76,
        "UCB_0.1": 193.08,
        "UCB_0.15": 187.95,
        "Eps_0.05": 138.82,
    },
    "실험 3 (8명)": {
        "Eps_0.05": 248.57,
        "UCB_0.15": 233.59,
        "Thompson_CollisionAware": 201.53,
        "UCB_0.1": 197.02,
        "Eps_0.2": 187.96,
        "Thompson_x7": 167.08,
        "Eps_0.1": 136.21,
        "Thompson_Weekend": 125.63,
    },
    "실험 4 (10명)": {
        "UCB_0.1": 223.89,
        "Thompson_CollisionAware": 201.16,
        "Thompson_x5": 181.15,
        "Eps_0.2": 173.12,
        "Thompson_x10": 172.42,
        "Thompson_x7": 138.81,
        "Eps_0.05": 132.17,
        "Eps_0.1": 115.81,
        "Thompson_Weekend": 104.40,
        "UCB_0.15": 102.09,
    }
}

# 그래프 생성
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (exp_name, results) in enumerate(experiments.items()):
    ax = axes[idx]
    
    # 데이터 정렬 (높은 순)
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    agents = [name for name, _ in sorted_results]
    scores = [score for _, score in sorted_results]
    
    # Thompson 계열 색상 구분
    colors = []
    for name in agents:
        if "Thompson" in name:
            colors.append('#FF6B6B')  # 빨간색
        elif "Eps" in name:
            colors.append('#4ECDC4')  # 청록색
        else:
            colors.append('#95E1D3')  # 연한 청록색
    
    # 막대 그래프
    bars = ax.barh(agents, scores, color=colors, alpha=0.8)
    
    # Thompson 계열에 별표 추가
    for i, (name, score) in enumerate(sorted_results):
        if "Thompson" in name:
            ax.text(score + 5, i, '★', fontsize=16, va='center', color='red')
    
    ax.set_xlabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_title(exp_name, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 상위 3위 강조
    for i in range(min(3, len(bars))):
        bars[i].set_edgecolor('gold')
        bars[i].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('docs/individuals/양민주/experiment_results_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장 완료: docs/individuals/양민주/experiment_results_comparison.png")
plt.show()


# 충돌 인식 Thompson 성능 추이 그래프
fig, ax = plt.subplots(figsize=(10, 6))

exp_names = ["실험 2\n(7명)", "실험 3\n(8명)", "실험 4\n(10명)"]
thompson_collision_scores = [
    0,  # 실험 2에는 없음
    201.53,  # 실험 3
    201.16,  # 실험 4
]
thompson_basic_scores = [
    249.37,  # 실험 2 (Thompson_x7)
    167.08,  # 실험 3 (Thompson_x7)
    138.81,  # 실험 4 (Thompson_x7)
]
eps_best_scores = [
    246.60,  # 실험 2 (Eps_0.2)
    248.57,  # 실험 3 (Eps_0.05)
    173.12,  # 실험 4 (Eps_0.2)
]

x = np.arange(len(exp_names))
width = 0.25

bars1 = ax.bar(x - width, thompson_basic_scores, width, label='Thompson (기본)', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x, [0, thompson_collision_scores[1], thompson_collision_scores[2]], width, 
               label='Thompson (충돌 인식)', color='#C44569', alpha=0.8)
bars3 = ax.bar(x + width, eps_best_scores, width, label='Epsilon-Greedy (최고)', color='#4ECDC4', alpha=0.8)

ax.set_xlabel('실험 (에이전트 수)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
ax.set_title('Thompson 변형 알고리즘 성능 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(exp_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('docs/individuals/양민주/thompson_performance_trend.png', dpi=300, bbox_inches='tight')
print("✅ 그래프 저장 완료: docs/individuals/양민주/thompson_performance_trend.png")
plt.show()
