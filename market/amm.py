# market/amm.py
from core.node import Node

class NodeAMM:
    """
    각 노드에 소속되어 자원 가격을 결정하는 시장입니다.
    자원 판매량에 따라 가격이 변하는 '다항식 본딩 커브'를 사용합니다.
    """
    def __init__(self, node: Node, p_base: float = 0.1, alpha: float = 10.0, beta: float = 2.0):
        self.node = node
        
        # 본딩 커브 파라미터
        self.P_base = p_base  # 기본 가격
        self.alpha = alpha    # 가격 민감도
        self.beta = beta      # 곡선 계수

        # 판매량 추적
        self.cpu_sold = 0.0
        self.mem_sold = 0.0
        
        # 초기 가격 출력
        initial_cpu_price = self.get_price("cpu")
        initial_mem_price = self.get_price("mem")
        print(f"[AMM for Node {self.node.id}] 생성됨 | 초기 CPU 가격: {initial_cpu_price:.2f}, 초기 MEM 가격: {initial_mem_price:.2f}")

    def get_price(self, resource_type: str) -> float:
        """지정된 자원의 현재 가격을 본딩 커브에 따라 계산하여 반환합니다."""
        if resource_type == "cpu":
            # 판매 비율 계산 (0.0 ~ 1.0)
            sold_ratio = self.cpu_sold / self.node.total_cpu
            # 가격 공식: Price = P_base + α * (S_sold / S_total)^β
            price = self.P_base + self.alpha * (sold_ratio ** self.beta)
            return price
        
        elif resource_type == "mem":
            sold_ratio = self.mem_sold / self.node.total_mem
            price = self.P_base + self.alpha * (sold_ratio ** self.beta)
            return price
        
        else:
            raise ValueError("알 수 없는 자원 타입입니다.")

    def buy(self, resource_type: str, amount: float) -> float:
        """자원 구매를 처리하고 비용을 반환합니다."""
        # 현재 가격을 기준으로 비용 계산
        price = self.get_price(resource_type)
        cost = price * amount
        
        if resource_type == "cpu":
            if self.node.available_cpu >= amount:
                self.node.available_cpu -= amount
                self.cpu_sold += amount
                new_price = self.get_price(resource_type)
                print(f"  [BUY] Node {self.node.id}: CPU {amount} 구매 완료. 비용: {cost:.2f} (가격 변동: {price:.2f} -> {new_price:.2f})")
                return cost
            else:
                print(f"  [FAIL] Node {self.node.id}: CPU 자원 부족")
                return -1.0 # 구매 실패
        
        elif resource_type == "mem":
            if self.node.available_mem >= amount:
                self.node.available_mem -= amount
                self.mem_sold += amount
                new_price = self.get_price(resource_type)
                print(f"  [BUY] Node {self.node.id}: MEM {amount} 구매 완료. 비용: {cost:.2f} (가격 변동: {price:.2f} -> {new_price:.2f})")
                return cost
            else:
                print(f"  [FAIL] Node {self.node.id}: MEM 자원 부족")
                return -1.0 # 구매 실패
        
        return -1.0