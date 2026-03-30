# data/walmart/switch_extractor.py
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

def detect_regime_changes():
    # 📍 동적 경로 및 폴더 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sales_path = os.path.join(script_dir, "sales_train_evaluation.csv")
    output_dir = os.path.join(script_dir, "switched_data") # 🌟 요청하신 폴더 이름!
    os.makedirs(output_dir, exist_ok=True)

    print("⏳ [국면 전환 탐지] 구조적 파괴(Regime Change) 지점을 분석 중...")
    sales = pd.read_csv(sales_path)
    
    states = ['CA', 'TX', 'WI']
    depts = sales['dept_id'].unique()
    
    regime_list = []
    WINDOW = 50 # 60일 단위로 과거/미래 체질 비교
    THRESHOLD = 0.20 # 평균이 30% 이상 변해야 국면 전환으로 인정

    for state in states:
        for dept in depts:
            arm_name = f"{state}_{dept}"
            series = sales[(sales['state_id'] == state) & (sales['dept_id'] == dept)].loc[:, 'd_1':'d_1941'].sum()
            
            # 전체 기간의 기준 평균 (Base Mean)
            overall_mean = series.mean()
            if overall_mean < 10: # 수요가 너무 적은 매대는 무시
                continue

            # 과거 60일 평균과 미래 60일 평균을 계산
            past_mean = series.rolling(window=WINDOW, min_periods=WINDOW).mean()
            future_mean = series.shift(-WINDOW).rolling(window=WINDOW, min_periods=WINDOW).mean()
            
            # (미래 평균 - 과거 평균) / 전체 평균 = 체질 변화 비율
            change_ratio = (future_mean - past_mean) / overall_mean
            
            # 변화율의 절댓값이 가장 큰 피크점(Peak)들을 찾음 (최소 100일 간격 쿨타임)
            peaks, _ = find_peaks(abs(change_ratio.dropna()), height=THRESHOLD, distance=100)
            
            for peak_idx in peaks:
                # dropna()로 인해 밀린 인덱스 보정 (window 크기만큼)
                real_step = peak_idx + WINDOW 
                
                # 전환 전후의 실제 변화량 (Multiplier 계산)
                # Base Mean 대비 '새로운 국면의 평균'이 몇 배가 되었는가?
                new_regime_mean = future_mean.iloc[peak_idx]
                multiplier = round(new_regime_mean / overall_mean, 3)
                
                direction = "떡상(Up)" if change_ratio.iloc[peak_idx] > 0 else "폭락(Down)"
                
                regime_list.append({
                    'arm_name': arm_name,
                    'switch_step': real_step,
                    'multiplier': multiplier,
                    'direction': direction
                })
                
                print(f"🎯 [{arm_name}] 국면 전환 감지! Step {real_step} 기점: {direction} (배수: {multiplier}배)")

    # CSV로 예쁘게 저장
    output_csv = os.path.join(output_dir, "regime_switches.csv")
    pd.DataFrame(regime_list).to_csv(output_csv, index=False)
    print(f"\n✅ 추출 완료! '{output_csv}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    detect_regime_changes()