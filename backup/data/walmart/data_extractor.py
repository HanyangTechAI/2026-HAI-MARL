import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def load_and_aggregate_dept(filepath):
    print("⏳ 데이터 로딩 및 dept 단위 집계 중...")
    # 필요한 컬럼만 불러와 메모리 아끼기 (id류 + 날짜 d_1~d_1941)
    day_cols = [f'd_{i}' for i in range(1, 1942)]
    needed_cols = ['state_id', 'dept_id'] + day_cols
    df = pd.read_csv(filepath, usecols=needed_cols)
    
    # 💡 dept_id 단위로 집계
    agg_df = df.groupby(['state_id', 'dept_id'])[day_cols].sum()
    print("✅ 데이터 집계 완료!\n")
    return agg_df

def plot_all_21_depts(agg_df):
    """
    21개 dept 라인을 주(State) 단위로 3개의 Figure로 나누어 시각화합니다.
    """
    states = agg_df.index.get_level_values('state_id').unique()
    depts = agg_df.index.get_level_values('dept_id').unique()
    
    # 공통 색상 맵 설정 (7개 dept 구분을 위해)
    colors = plt.cm.tab10(np.linspace(0, 1, len(depts)))
    dept_color_map = {dept: colors[i] for i, dept in enumerate(depts)}

    for state in states:
        print(f"📊 {state} 지역 상세 그래프 그리는 중...")
        plt.figure(figsize=(18, 10))
        
        state_data = agg_df.xs(state, level='state_id')
        
        for dept, row in state_data.iterrows():
            # 노이즈 제거를 위한 30일 이동평균
            rolling_mean = row.rolling(window=30, min_periods=1).mean()
            
            plt.plot(rolling_mean.values, 
                     label=f'{dept}', 
                     color=dept_color_map[dept], 
                     linewidth=2)

        plt.title(f"Walmart Daily Sales Trend in {state} (by Dept_id, 30-Day Moving Average)", fontsize=18, fontweight='bold')
        plt.xlabel("Days (1 to 1941)", fontsize=14)
        plt.ylabel("Total Units Sold", fontsize=14)
        plt.legend(title="Department ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # 주 단위로 이미지 파일 저장
        save_path = f"{state}_21_dept_trends.png"
        plt.savefig(save_path, dpi=100)
        print(f"💾 저장 완료: {save_path}")
        
        # 화면에도 띄우기 (필요없으면 주석 처리)
        # plt.show()
        plt.close() # 메모리 해제

if __name__ == "__main__":
    CSV_PATH = "m5-forecasting-accuracy\sales_train_evaluation.csv" 
    if os.path.exists(CSV_PATH):
        aggregated_data = load_and_aggregate_dept(CSV_PATH)
        plot_all_21_depts(aggregated_data)
    else:
        print(f"🚨 에러: '{CSV_PATH}' 파일이 없습니다.")