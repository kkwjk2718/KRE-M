# market/amm.py
from core.node import Node

class NodeAMM:
    """
    각 노드에 소속되어 해당 노드의 자원 가격을 결정하는 시장입니다.
    지금은 본딩 커브 없이, 고정된 가격을 반환하는 가장 단순한 형태로 시작합니다.
    """
    def __init__(self, node: Node, cpu_base_price: float = 1.0, mem_base_price: float = 0.5):
        self.node = node
        self.cpu_price = cpu_base_price
        self.mem_price = mem_base_price
        
        # 판매량 추적 (나중에 본딩 커브에 사용)
        self.cpu_sold = 0.0
        self.mem_sold = 0.0

        print(f"[AMM for Node {self.node.id}] 생성됨 | CPU 가격: {self.cpu_price}, MEM 가격: {self.mem_price}")

    def get_price(self, resource_type: str) -> float:
        """지정된 자원의 현재 가격을 반환합니다."""
        if resource_type == "cpu":
            return self.cpu_price
        elif resource_type == "mem":
            return self.mem_price
        else:
            raise ValueError("알 수 없는 자원 타입입니다.")

    def buy(self, resource_type: str, amount: float) -> float:
        """자원 구매를 처리하고 비용을 반환합니다."""
        price = self.get_price(resource_type)
        cost = price * amount
        
        if resource_type == "cpu":
            if self.node.available_cpu >= amount:
                self.node.available_cpu -= amount
                self.cpu_sold += amount
                print(f"  [BUY] Node {self.node.id}: CPU {amount} 구매 완료. 비용: {cost:.2f}")
                return cost
            else:
                print(f"  [FAIL] Node {self.node.id}: CPU 자원 부족")
                return -1.0 # 구매 실패
        
        elif resource_type == "mem":
            if self.node.available_mem >= amount:
                self.node.available_mem -= amount
                self.mem_sold += amount
                print(f"  [BUY] Node {self.node.id}: MEM {amount} 구매 완료. 비용: {cost:.2f}")
                return cost
            else:
                print(f"  [FAIL] Node {self.node.id}: MEM 자원 부족")
                return -1.0 # 구매 실패
        
        return -1.0