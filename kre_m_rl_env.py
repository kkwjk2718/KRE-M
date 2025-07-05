# kre_m_rl_env.py
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
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
        # 여기에 노드 에이전트도 추가할 예정

        # 상태 및 행동 공간 정의 (지금은 앱 에이전트만)
        # 상태: [내 예산, node1_cpu_가격, node2_cpu_가격, ...]
        observation_size = 1 + self.num_nodes
        self.observation_space = Dict({
            agent_id: Box(low=0, high=float('inf'), shape=(observation_size,))
            for agent_id in self._agent_ids
        })

        # 행동: 어느 노드에서 CPU를 살지 선택
        self.action_space = Dict({
            agent_id: Discrete(self.num_nodes)
            for agent_id in self._agent_ids
        })
        
        # 내부 시뮬레이션 변수 초기화
        self.nodes = []
        self.apps = {}
        self.markets = {}
        self.timestep = 0


    def reset(self, *, seed=None, options=None):
        """환경을 초기 상태로 리셋하고, 첫 번째 상태를 반환합니다."""
        print("--- 환경 리셋 ---")
        self.timestep = 0
        
        # 노드, 앱, 시장 생성
        self.nodes = [Node(f"node-{i}", 10, 32) for i in range(self.num_nodes)]
        self.markets = {node.id: NodeAMM(node) for node in self.nodes}
        self.apps = {f"app_{i}": App(f"app_{i}", 2, 4, 100.0) for i in range(self.num_apps)}
        
        return self._get_obs(), {}

    def step(self, action_dict):
        """에이전트들의 행동을 받아 한 스텝을 진행하고, 결과(상태, 보상, 종료 여부)를 반환합니다."""
        self.timestep += 1
        print(f"\n--- Timestep {self.timestep} ---")
        
        obs, rewards, dones, truncateds, infos = {}, {}, {}, {}, {}

        for agent_id, action in action_dict.items():
            app = self.apps[agent_id]
            node_to_buy_from = self.nodes[action]
            market = self.markets[node_to_buy_from.id]

            # 행동: 선택한 노드에서 CPU 1개 구매 시도
            cost = market.buy("cpu", 1.0)

            # 보상 설계 (가장 간단한 버전)
            if cost != -1.0: # 구매 성공
                app.budget -= cost
                rewards[agent_id] = -cost # 지출한 비용이 마이너스 보상
            else: # 구매 실패
                rewards[agent_id] = -10.0 # 실패 시 큰 페널티
        
        # 모든 에이전트에 대한 정보 업데이트
        obs = self._get_obs()
        dones["__all__"] = self.timestep >= 50 # 50 스텝 후에 종료
        truncateds["__all__"] = False
        
        return obs, rewards, dones, truncateds, infos

    def _get_obs(self):
        """모든 에이전트의 현재 상태(관측)를 구성하여 반환합니다."""
        all_obs = {}
        # 모든 노드의 현재 CPU 가격 리스트
        cpu_prices = [self.markets[node.id].get_price("cpu") for node in self.nodes]

        for agent_id, app in self.apps.items():
            # 상태 = [내 예산, node1_cpu_가격, node2_cpu_가격, ...]
            all_obs[agent_id] = [app.budget] + cpu_prices
        return all_obs