# train.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from kre_m_rl_env import KREmMultiAgentEnv

# 1. Ray 초기화
# num_cpus: 워커 VM들의 총 CPU 코어 수와 비슷하게 설정
# object_store_memory: 훈련기 VM의 RAM 일부를 할당
ray.init(num_cpus=8, object_store_memory=10*1024*1024*1024) # 10GB

# 2. PPO 알고리즘 설정
config = (
    PPOConfig()
    .environment(KREmMultiAgentEnv, env_config={"num_nodes": 2, "num_apps": 5})
    .framework("torch")
    .rollouts(num_rollout_workers=4) # 데이터 수집에 사용할 워커(CPU 코어) 수
    .training(
        model={"fcnet_hiddens": [64, 64]}, # 신경망 모델의 은닉층 구조
        train_batch_size=1024 # 한 번의 학습 스텝에 사용할 데이터 샘플 크기
    )
    .multi_agent(
        # 모든 앱 에이전트가 동일한 정책(신경망)을 공유하도록 설정
        policies=["app_policy"],
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "app_policy")
    )
)

# 3. 훈련기(Trainer) 생성
algo = config.build()

# 4. 훈련 루프 실행
print("\n--- 훈련 시작 ---")
for i in range(10): # 10번의 훈련 반복
    result = algo.train()
    print(f"Iteration: {i+1}")
    print(f"  Mean Reward: {result['episode_reward_mean']:.2f}")
    # print(pretty_print(result)) # 더 상세한 로그를 보려면 이 줄의 주석을 해제하세요.

print("--- 훈련 종료 ---")

# 5. Ray 종료
ray.shutdown()