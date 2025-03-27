import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

class CapitalAllocationEnv(gym.Env):
    def __init__(self, data):
        super(CapitalAllocationEnv, self).__init__()
        self.data = data
        self.index = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self):
        self.index = 0
        return self._get_obs()

    def _get_obs(self):
        slope, r2, atr_norm, _ = self.data[self.index]
        return np.array([slope, r2, atr_norm], dtype=np.float32)

    def step(self, action):
        long_alloc = float(np.clip(action[0], 0.0, 1.0))
        short_alloc = 1.0 - long_alloc

        _, _, _, reward_fn = self.data[self.index]
        reward = reward_fn(long_alloc, short_alloc)

        self.index += 1
        done = self.index >= len(self.data)

        obs = self._get_obs() if not done else np.zeros(3, dtype=np.float32)
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        pass

def reward_function(long_alloc, short_alloc, long_return, short_return):
    return long_alloc * long_return + short_alloc * short_return

env = CapitalAllocationEnv(data)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_allocation_agent")