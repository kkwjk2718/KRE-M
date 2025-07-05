# kre_m_rl_env.py
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.node import Node
from core.app import App
from market.amm import NodeAMM

class KREmMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.num_nodes = config.get("num_nodes", 2)
        self.num_apps = config.get("num_apps", 1) # 기본값도 1로 맞춰주는 것이 좋습니다.
        self._agent_ids = {f"app_{i}" for i in range(self.num_apps)}
        observation_size = 1 + self.num_nodes
        self.observation_space = Dict({
            agent_id: Box(low=0, high=np.inf, shape=(observation_size,), dtype=np.float32)
            for agent_id in self._agent_ids
        })
        self.action_space = Dict({
            agent_id: Discrete(self.num_nodes)
            for agent_id in self._agent_ids
        })
        self.nodes, self.apps, self.markets, self.timestep = [], {}, {}, 0

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.nodes = [Node(f"node-{i}", 10, 32) for i in range(self.num_nodes)]
        self.markets = {node.id: NodeAMM(node) for node in self.nodes}
        self.apps = {f"app_{i}": App(f"app_{i}", 2, 4, 100.0) for i in range(self.num_apps)}
        self.agents = list(self.apps.keys())
        return self._get_obs(), {}

    def step(self, action_dict):
        self.timestep += 1
        obs, rewards, dones, truncateds, infos = {}, {}, {}, {}, {}

        for agent_id, action in action_dict.items():
            node_to_buy_from = self.nodes[action]
            cost = self.markets[node_to_buy_from.id].buy("cpu", 1.0)
            
            # [수정된 부분] 보상 체계를 +10 / -10 으로 단순화
            if cost != -1.0:
                self.apps[agent_id].budget -= cost
                rewards[agent_id] = 10.0  # 성공 시 +10점
            else:
                rewards[agent_id] = -10.0 # 실패 시 -10점
        
        obs = self._get_obs()
        truncateds["__all__"] = self.timestep >= 50
        dones["__all__"] = False
        return obs, rewards, dones, truncateds, infos

    def _get_obs(self):
        all_obs = {}
        cpu_prices = [self.markets[node.id].get_price("cpu") for node in self.nodes]
        for agent_id, app in self.apps.items():
            all_obs[agent_id] = np.array([app.budget] + cpu_prices, dtype=np.float32)
        return all_obs