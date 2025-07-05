# core/app.py

class App:
    """
    KRE-M 경제 시스템의 '자원 수요자'입니다.
    자신의 예산을 사용하여 노드로부터 자원 주식을 구매합니다.
    """
    def __init__(self, app_id: str, required_cpu: float, required_mem: float, budget: float):
        self.id = app_id
        self.required_cpu = required_cpu
        self.required_mem = required_mem
        self.budget = budget
        
        # 현재 할당받은 자원
        self.allocated_cpu = 0.0
        self.allocated_mem = 0.0

        print(f"[App {self.id}] 생성됨 | 요구사항 CPU: {self.required_cpu}, MEM: {self.required_mem}, 예산: {self.budget}")

    def __repr__(self):
        return (f"App(id={self.id}, budget={self.budget:.2f}, "
                f"alloc_cpu={self.allocated_cpu}, alloc_mem={self.allocated_mem})")