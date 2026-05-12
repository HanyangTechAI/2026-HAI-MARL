import os

# 시각화 모듈 임포트 (utils 폴더 안에 해당 파일들이 있다고 가정)
try:
    from utils.plot_trueskill import plot_trueskill_results
    from utils.plot_paper_figures import (
        setup_academic_style, 
        plot_fig1_market_dynamics, 
        plot_fig2_slippage_curve, 
        plot_fig3_episode_trajectory
    )
except ImportError as e:
    print(f"🚨 모듈 임포트 에러: {e}")
    print("utils 폴더 안에 plot_trueskill.py와 plot_paper_figures.py가 있는지 확인해주세요.")
    exit(1)

def main():
    print("="*60)
    print("🚀 [논문용 시각화 파이프라인 가동]")
    print("="*60)

    # ---------------------------------------------------------
    # 1. TrueSkill 결과 시각화 (CSV 기반 데이터 플롯)
    # ---------------------------------------------------------
    # ⚠️ 여기에 최근에 생성된 폴더명(타임스탬프)을 정확히 입력해주세요!
    TARGET_FOLDER = "TS_20260410_180442" # 예시 폴더명
    
    csv_path = os.path.join("output", TARGET_FOLDER, "Leaderboard_Detailed.csv")
    ts_output_dir = os.path.join("output", TARGET_FOLDER)

    print("\n[1단계] TrueSkill 배틀로얄 통계 시각화 진행 중...")
    if os.path.exists(csv_path):
        plot_trueskill_results(csv_path, ts_output_dir)
        # (plot_trueskill 내부에서 이미 완료 메시지를 출력합니다)
    else:
        print(f"❌ 오류: CSV 파일을 찾을 수 없습니다 -> {csv_path}")
        print("경로에 맞는 최신 타임스탬프 폴더명으로 'TARGET_FOLDER' 변수를 수정해주세요.")

    # ---------------------------------------------------------
    # 2. 논문 서론용 개념 Figure 생성 (CSV 불필요, 수학적 렌더링)
    # ---------------------------------------------------------
    print("\n[2단계] 논문 서론용 3대 핵심 개념도 렌더링 중...")
    paper_output_dir = os.path.join("output", "Paper_Figures")
    os.makedirs(paper_output_dir, exist_ok=True)
    
    setup_academic_style()
    plot_fig1_market_dynamics(paper_output_dir)
    plot_fig2_slippage_curve(paper_output_dir)
    plot_fig3_episode_trajectory(paper_output_dir)
    
    print(f"✅ 논문용 기초 피겨 3종이 {paper_output_dir} 에 저장되었습니다.")
    
    print("\n" + "="*60)
    print("🎉 모든 시각화 작업이 성공적으로 완료되었습니다!")
    print("="*60)

if __name__ == "__main__":
    main()