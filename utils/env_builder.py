# utils/env_builder.py
import os
import random
from envs.SMPyBandits.Environment.MAB import MAB
from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from arms.arm_registry import STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY

# 상수 경로
SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
SWITCH_FILE = os.path.join("data", "walmart", "extracted_data", "regime_switches.csv")

def build_dynamic_market(strategy="balanced"):
    """전략에 따라 8개의 시장을 차출하고 MAB 환경과 이름 리스트를 반환합니다."""
    sampled_arms = {"stationary": [], "shocks": [], "trends": [], "switches": []}
    
    if strategy == "balanced":
        sampled_arms["stationary"] = random.sample(list(STATIONARY_REGISTRY.keys()), 2)
        sampled_arms["shocks"] = random.sample(list(EVENT_SHOCK_REGISTRY.keys()), 2)
        sampled_arms["trends"] = random.sample(list(TREND_REGISTRY.keys()), 2)
        sampled_arms["switches"] = random.sample(list(SWITCH_REGISTRY.keys()), 2)
    else:
        all_keys = (
            [("stationary", k) for k in STATIONARY_REGISTRY.keys()] +
            [("shocks", k) for k in EVENT_SHOCK_REGISTRY.keys()] +
            [("trends", k) for k in TREND_REGISTRY.keys()] +
            [("switches", k) for k in SWITCH_REGISTRY.keys()]
        )
        for cat, name in random.sample(all_keys, 8):
            sampled_arms[cat].append(name)
            
    arm_configuration = []
    arm_names = [] # 🌟 어떤 매장이 차출되었는지 기록하기 위한 리스트
    
    for name in sampled_arms["stationary"]:
        p = STATIONARY_REGISTRY[name]
        arm_configuration.append(StationaryArm(name, p["mean"], p["variance"]))
        arm_names.append(name)
        
    for name in sampled_arms["shocks"]:
        p = EVENT_SHOCK_REGISTRY[name]
        arm_configuration.append(EventShockArm(name, p["base_mean"], p["base_variance"], SHOCKS_FILE, SEASON_FILE))
        arm_names.append(name)
        
    for name in sampled_arms["trends"]:
        p = TREND_REGISTRY[name]
        arm_configuration.append(TrendArm(name, p["start_mean"], p["slope"], p["variance"]))
        arm_names.append(name)
        
    for name in sampled_arms["switches"]:
        p = SWITCH_REGISTRY[name]
        arm_configuration.append(SwitchArm(name, p["base_mean"], p["base_variance"], SWITCH_FILE))
        arm_names.append(name)
        
    return MAB(arm_configuration), arm_names