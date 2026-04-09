import re

# 개별 알고리즘 목록 정리
# 팀원분들 모두 각자 제작한 알고리즘은 제작 이후 여기에 추가해주세요.
from agents.epsilon_greedy import EpsilonGreedy
from agents.decaying_epsilon import DecayingEpsilon
from agents.ucb import UCBAgent
from agents.sliding_window_ucb import SlidingWindowUCB
from agents.as_ucb import AS_UCB
from agents.sw_as_ucb import SW_AS_UCB
from agents.softmax import SoftmaxAgent
from agents.wsls import WSLS
from agents.fft_ucb import FFT_UCB
from agents.periodic_ucb import PeriodicUCB
from agents.thompson_sampling import ThompsonSampling
from agents.thompson_weekly import ThompsonWeekly
from agents.thompson_collision_aware import ThompsonCollisionAware

def _get_param(agent_name: str, key: str, default: float, is_int: bool = False):
    """
    agent_name 에서 파라미터 값 추출
    (예: name="UCB_c0.05_w100", key="c" -> 0.05 반환)
    """
    # _{키워드} 뒤에 나오는 숫자(소수점 포함)를 찾음
    match = re.search(rf"_{key}([0-9.]+)", agent_name)
    if match:
        val = float(match.group(1))
        return int(val) if is_int else val
    return default

def get_agent(agent_name: str, nbArms: int):
    """
    Args:
        agent_name (str): _description_
        nbArms (int): _description_
        키워드 관련되서 파라미터 설정을 잘 해줘야 함.
    """

    # 1. 범용 파라미터 자동 추출 (순서 상관없음, 안 적혀있으면 기본값 적용)
    eps_val = _get_param(agent_name, "e", 0.05)           # Epsilon 값
    c_val   = _get_param(agent_name, "c", 0.05)           # UCB c 값
    w_size  = _get_param(agent_name, "w", 100, True)      # Window 크기
    p_val   = _get_param(agent_name, "p", 7, True)        # Period (주기)
    s_val   = _get_param(agent_name, "s", 10.0)           # Smoothing 파라미터
    tau_val = _get_param(agent_name, "t", 0.1)            # Softmax Temperature