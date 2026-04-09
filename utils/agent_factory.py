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

def get_agent(agent_name: str, nbArms: int):
    """
    Args:
        agent_name (str): _description_
        nbArms (int): _description_
    """

    parts = agent_name.split("_")
    base_name = parts[0]