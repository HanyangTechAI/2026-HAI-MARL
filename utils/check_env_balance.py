# utils/check_env_balance.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 프로젝트 루트 경로를 시스템 패스에 추가 (utils 폴더에서 실행 시 모듈 인식 문제 해결)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from arms.arm_registry import STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY

def check_environment_balance():
    print("⏳ 환경(Arm) 밸런스 검증 시뮬레이션을 시작합니다...")
    
    # 공통 설정
    HORIZON = 1941
    # 🌟 GLOBAL_SCALER 완전 삭제! (Arm은 이제 스케일러를 모릅니다)
    np.random.seed(42) # 동일한 노이즈로 검증하기 위해 시드 고정
    
    # 데이터 경로
    SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
    SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
    SWITCH_FILE = os.path.join("data", "walmart", "extracted_data", "regime_switches.csv")
    
    arms = []
    arm_names = []

    # 1. Arm 조립 (수학 연산 없이 레지스트리 원본 값 그대로 주입!)
    for name in ["CA_HOBBIES_1", "CA_FOODS_1"]:
        p = STATIONARY_REGISTRY[name]
        arms.append(StationaryArm(name, p["mean"], p["variance"]))
        arm_names.append(name)

    for name in ["CA_FOODS_2", "TX_FOODS_2"]:
        p = EVENT_SHOCK_REGISTRY[name]
        arms.append(EventShockArm(name, p["base_mean"], p["base_variance"], SHOCKS_FILE, SEASON_FILE))
        arm_names.append(name)

    for name in ["CA_HOUSEHOLD_1", "WI_FOODS_2"]:
        p = TREND_REGISTRY[name]
        arms.append(TrendArm(name, p["start_mean"], p["slope"], p["variance"]))
        arm_names.append(name)

    for name in ["TX_HOUSEHOLD_1", "WI_HOUSEHOLD_1"]:
        p = SWITCH_REGISTRY[name]
        arms.append(SwitchArm(name, p["base_mean"], p["base_variance"], SWITCH_FILE))
        arm_names.append(name)

    # 2. 1941일 동안 모든 매장의 보상을 매일매일 기록
    rewards_log = np.zeros((HORIZON, len(arms)))
    
    for t in range(HORIZON):
        for i, arm in enumerate(arms):
            # 🌟 역산(복구) 불필요! Arm이 이미 1500, 7000 같은 쌩값을 뱉어줍니다.
            rewards_log[t, i] = arm.draw(t)

    # 3. 데이터프레임 변환 및 누적/통계 계산
    df = pd.DataFrame(rewards_log, columns=arm_names)
    total_rewards = df.sum()
    
    print("\n" + "="*50)
    print("🏆 각 매장을 1941일 내내 100% 독식했을 때의 총 보상 (이론적 최대치)")
    print("="*50)
    # 총 보상 순으로 정렬하여 출력
    total_rewards_sorted = total_rewards.sort_values(ascending=False)
    for name, val in total_rewards_sorted.items():
        print(f" - {name:<18} : 총 {int(val):>9,} 개 (일평균: {int(val/HORIZON):>5,} 개)")
    print("="*50)

    # ==========================================
    # 📊 시각화 1: 30일 이동평균 시계열 그래프
    # ==========================================
    plt.figure(figsize=(16, 8))
    df_ma = df.rolling(window=30, min_periods=1).mean()
    
    for name in arm_names:
        plt.plot(df_ma.index, df_ma[name], label=name, linewidth=2, alpha=0.8)
        
    plt.title("Walmart Golden Balance Dynamic Market (30-Day Moving Average)", fontsize=18, fontweight='bold')
    plt.xlabel("Days (Steps)", fontsize=14)
    plt.ylabel("Expected Daily Sales (Units)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("env_dynamics_30d_ma.png", dpi=300)
    
    # ==========================================
    # 📊 시각화 2: 총 누적 보상 바 차트
    # ==========================================
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(arms)))
    total_rewards_sorted.plot(kind='bar', color=colors, edgecolor='black')
    
    plt.title("Theoretical Maximum Cumulative Rewards (If selected 100% of the time)", fontsize=16, fontweight='bold')
    plt.xlabel("Arm Name", fontsize=12)
    plt.ylabel("Total Units Sold", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("env_total_rewards_bar.png", dpi=300)

    print("\n✅ 분석 완료! 현재 폴더에 두 장의 검증 그래프가 저장되었습니다:")
    print(" 1) env_dynamics_30d_ma.png (일일 변동성 및 트렌드/스위치 흐름)")
    print(" 2) env_total_rewards_bar.png (최종 기대 보상 서열)")

if __name__ == "__main__":
    check_environment_balance()