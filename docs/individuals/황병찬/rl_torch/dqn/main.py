import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

### input parameter ####################
# env_name = "LunarLander-v2"
# model_dir = "./dqn_model.pt"
# train_steps = 100000
# test_steps = 1000
# ########################################

# # make environment
# env = gym.make(env_name)

# # train policy
# train_model = DQN("MlpPolicy", env, verbose=1)
# train_model.learn(total_timesteps=train_steps)
# train_model.save(model_dir)

# # evaluate policy
# test_model = DQN.load(model_dir)
# obs = env.reset()
# game_score = 0

# for i in range(test_steps):
#     action, _states = test_model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     game_score += reward
#     if done:
#       obs = env.reset()

# env.close()
# print(f"game_score : {game_score}")

env = gym.make('LunarLander-v3')

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("dqn_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
env = gym.make('LunarLander-v3', render_mode='human')
# 총 5번의 에피소드(착륙 시도) 진행
for episode in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

env.close()