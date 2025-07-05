# core/node.py

class Node:
    """
    KRE-M 경제 시스템의 '자원 공급자'입니다.
    자신의 CPU와 Memory 자원을 '자원 주식' 형태로 판매합니다.
    """
    def __init__(self, node_id: str, total_cpu: float, total_mem: float):
        self.id = node_id
        self.total_cpu = total_cpu
        self.total_mem = total_mem
        
        # 현재 사용 가능한 자원 (초기값은 전체 자원)
        self.available_cpu = total_cpu
        self.available_mem = total_mem

        print(f"[Node {self.id}] 생성됨 | CPU: {self.total_cpu}, MEM: {self.total_mem}")

    def __repr__(self):
        return (f"Node(id={self.id}, "
                f"cpu={self.available_cpu}/{self.total_cpu}, "
                f"mem={self.available_mem}/{self.total_mem})")