# main.py
import time
from core.node import Node
from core.app import App
from market.amm import NodeAMM

# --- 시뮬레이션 환경 설정 ---
def setup_environment():
    """시뮬레이션에 필요한 노드, 앱, 시장을 생성하고 초기화합니다."""
    print("="*10 + " 환경 설정 시작 " + "="*10)
    
    # 2개의 노드 생성 (CPU 10개, 메모리 32GB)
    nodes = [Node(node_id="node-1", total_cpu=10, total_mem=32),
             Node(node_id="node-2", total_cpu=10, total_mem=32)]

    # 각 노드에 대한 시장(AMM) 생성
    markets = {node.id: NodeAMM(node) for node in nodes}

    # 5개의 앱 생성 (각기 다른 요구사항과 예산)
    apps = [App(app_id=f"app-{i}", required_cpu=2, required_mem=4, budget=100.0) for i in range(5)]
    
    print("="*10 + " 환경 설정 완료 " + "="*10 + "\n")
    return nodes, markets, apps

# --- 메인 시뮬레이션 루프 ---
def run_simulation(nodes, markets, apps):
    """설정된 환경에서 시뮬레이션을 실행합니다."""
    simulation_duration = 10  # 10 타임스텝 동안 실행
    
    print("="*10 + " 시뮬레이션 시작 " + "="*10)
    
    for timestep in range(simulation_duration):
        print(f"\n--- Timestep {timestep+1}/{simulation_duration} ---")
        
        # --- 아주 간단한 '더미(Dummy)' 앱 에이전트 로직 ---
        # 모든 앱이 첫 번째 노드에서 CPU 1개를 구매하려고 시도합니다.
        for app in apps:
            target_node_id = "node-1"
            resource_type = "cpu"
            amount_to_buy = 1.0

            if app.budget > 0:
                print(f"[App {app.id}]가 Node {target_node_id}에서 CPU 구매 시도...")
                market = markets[target_node_id]
                cost = market.buy(resource_type, amount_to_buy)

                if cost != -1.0: # 구매 성공 시
                    app.budget -= cost
                    app.allocated_cpu += amount_to_buy

        # 현재 상태 출력
        print("\n[현재 상태]")
        for node in nodes:
            print(f"  - {node}")
        for app in apps:
            print(f"  - {app}")
            
        time.sleep(1) # 각 스텝 사이에 1초 지연

    print("\n" + "="*10 + " 시뮬레이션 종료 " + "="*10)


if __name__ == "__main__":
    # 1. 환경 설정
    sim_nodes, sim_markets, sim_apps = setup_environment()
    
    # 2. 시뮬레이션 실행
    run_simulation(sim_nodes, sim_markets, sim_apps)