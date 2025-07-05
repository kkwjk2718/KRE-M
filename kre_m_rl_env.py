# kre_m_rl_env.py
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.node import Node
from core.app import App
from market.amm import NodeAMM

class KREmMultiAgentEnv(MultiAgentEnv):
    """
    KRE-M 시뮬레이터를 RLlib의 MultiAgentEnv 형식으로 래핑한 클래스입니다.
    """

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        # 시뮬레이션 파라미터
        self.num_nodes = config.get("num_nodes", 2)
        self.num_apps = config.get("num_apps", 5)

        # 에이전트 ID 설정
        self._agent_ids = {f"app_{i}" for i in range(self.num_apps)}

        # 상태 및 행동 공간 정의
        observation_size = 1 + self.num_nodes
        self.observation_space = Dict({
            agent_id: Box(low=0, high=np.inf, shape=(observation_size,), dtype=np.float32)
            for agent_id in self._agent_ids
        })
        self.action_space = Dict({
            agent_id: Discrete(self.num_nodes)
            for agent_id in self._agent_ids
        })
        
        self.nodes = []
        self.apps = {}
        self.markets = {}
        self.timestep = 0


    def reset(self, *, seed=None, options=None):
        """환경을 초기 상태로 리셋하고, 첫 번째 상태를 반환합니다."""
        # print("--- 환경 리셋 ---") # 로그가 너무 많아져서 주석 처리
        self.timestep = 0
        
        self.nodes = [Node(f"node-{i}", 10, 32) for i in range(self.num_nodes)]
        self.markets = {node.id: NodeAMM(node) for node in self.nodes}
        self.apps = {f"app_{i}": App(f"app_{i}", 2, 4, 100.0) for i in range(self.num_apps)}
        
        self.agents = list(self.apps.keys())
        
        return self._get_obs(), {}

    def step(self, action_dict):
        """에이전트들의 행동을 받아 한 스텝을 진행하고, 결과(상태, 보상, 종료 여부)를 반환합니다."""
        self.timestep += 1
        
        obs, rewards, dones, truncateds, infos = {}, {}, {}, {}, {}

        for agent_id, action in action_dict.items():
            app = self.apps[agent_id]
            # 예산이 없으면 행동하지 않음
            if app.budget <= 0:
                rewards[agent_id] = 0
                continue

            node_to_buy_from = self.nodes[action]
            market = self.markets[node_to_buy_from.id]

            cost = market.buy("cpu", 1.0)

            # [수정된 부분] 보상 체계 변경
            if cost != -1.0:
                app.budget -= cost
                # 성공 시: 성공 보너스 5점에서 비용을 뺀 값을 보상으로 제공
                rewards[agent_id] = 5.0 - cost 
            else:
                # 실패 시: 페널티 -10점
                rewards[agent_id] = -10.0
        
        obs = self._get_obs()

        is_truncated = self.timestep >= 50
        truncateds["__all__"] = is_truncated
        
        dones["__all__"] = False
        
        return obs, rewards, dones, truncateds, infos

    def _get_obs(self):
        """모든 에이전트의 현재 상태(관측)를 구성하여 반환합니다."""
        all_obs = {}
        cpu_prices = [self.markets[node.id].get_price("cpu") for node in self.nodes]

        for agent_id, app in self.apps.items():
            all_obs[agent_id] = np.array([app.budget] + cpu_prices, dtype=np.float32)
        return all_obs