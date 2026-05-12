import re

# 개별 알고리즘 목록 정리
# 팀원분들 모두 각자 제작한 알고리즘은 제작 이후 여기에 추가해주세요.
from agents.epsilon_greedy import EpsilonGreedy
from agents.decaying_epsilon import DecayingEpsilonGreedy
from agents.ucb import UCBAgent
from agents.sliding_window_ucb import SlidingWindowUCB
from agents.as_ucb import AS_UCB
from agents.sw_as_ucb import SW_AS_UCB
from agents.softmax import SoftmaxAgent
from agents.wsls import WSLS
from agents.fft_ucb import FFTPeriodicUCB
from agents.periodic_ucb import PeriodicUCB
from agents.thompson_sampling import ThompsonSampling
from agents.thompson_weekly import ThompsonWeekendWeekday
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

    # 범용 파라미터 자동 추출
    eps_val = _get_param(agent_name, "e", 0.05)           # Epsilon 값
    c_val   = _get_param(agent_name, "c", 0.05)           # UCB c 값
    w_size  = _get_param(agent_name, "w", 100, True)      # Window 크기
    p_val   = _get_param(agent_name, "p", 7, True)        # Period (주기)
    s_val   = _get_param(agent_name, "s", 10.0)           # Smoothing 파라미터
    tau_val = _get_param(agent_name, "t", 0.1)            # Softmax Temperature
    decay_val = _get_param(agent_name, "d", 0.995)        # 감쇠율 (DecayEps의 핵심, 기본값 0.995)
    scale_val = _get_param(agent_name, "r", 7.0)          # 보상 스케일 (Thompson의 핵심, 기본값 7.0)
    pen_val   = _get_param(agent_name, "cp", 0.5)         # 충돌 페널티 비율 (Collision Aware 핵심)
    
    # --- [융합 & 고급 UCB 계열] ---
    if agent_name.startswith("SW_AS_UCB"):
        return SW_AS_UCB(nbArms, window_size=w_size, period=p_val, c=c_val, smoothing=s_val, name=agent_name)
        
    elif agent_name.startswith("AS_UCB"):
        return AS_UCB(nbArms, period=p_val, c=c_val, smoothing=s_val, name=agent_name)
        
    elif agent_name.startswith("SW_UCB"):
        return SlidingWindowUCB(nbArms, window_size=w_size, c=c_val, name=agent_name)

    elif agent_name.startswith("FFT_UCB"):
        return FFTPeriodicUCB(nbArms, c=c_val, warmup_rounds=w_size, sin_weight=s_val, name=agent_name)
        
    elif agent_name.startswith("Periodic_UCB"):
        return PeriodicUCB(nbArms, period=p_val, name=agent_name)

    # --- [기본 UCB 계열] ---
    elif agent_name.startswith("UCB"):
        return UCBAgent(nbArms, c=c_val, name=agent_name)

    # --- [Epsilon & Heuristic 계열] ---
    elif agent_name.startswith("DecayEps"):
        # 초기 탐색률은 논문 실험 최고치였던 0.5로 고정하고, 감쇠율은 추출값(기본 0.995) 사용
        return DecayingEpsilonGreedy(nbArms, initial_epsilon=0.5, min_epsilon=eps_val, decay_rate=decay_val, name=agent_name)
        
    elif agent_name.startswith("Eps"):
        return EpsilonGreedy(nbArms, epsilon=eps_val, name=agent_name)
        
    elif agent_name.startswith("Softmax"):
        return SoftmaxAgent(nbArms, temperature=tau_val, name=agent_name)
        
    elif agent_name.startswith("WSLS"):
        return WSLS(nbArms, name=agent_name)

    # --- [Thompson Sampling 계열] ---
    elif agent_name.startswith("TS_Collision_Aware"):
        return ThompsonCollisionAware(nbArms, reward_scale=scale_val, collision_penalty_rate=pen_val, penalty_decay=0.95, name=agent_name)
        
    elif agent_name.startswith("TS_Weekly"):
        return ThompsonWeekendWeekday(nbArms, reward_scale=scale_val, name=agent_name)
        
    elif agent_name.startswith("TS"):
        return ThompsonSampling(nbArms, reward_scale=scale_val, name=agent_name)

    # 매칭되는 이름이 없을 경우 에러 발생
    raise ValueError(f"🚨 팩토리에 등록되지 않은 알고리즘 이름입니다: {agent_name}")