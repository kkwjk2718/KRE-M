# train.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from kre_m_rl_env import KREmMultiAgentEnv

# 1. Ray 초기화
ray.init(num_cpus=8, object_store_memory=10*1024*1024*1024) # 10GB

# 2. PPO 알고리즘 설정
config = (
    PPOConfig()
    .environment(KREmMultiAgentEnv, env_config={"num_nodes": 2, "num_apps": 5})
    .framework("torch")
    .env_runners(
        num_env_runners=4,
        batch_mode="complete_episodes"
    )
    .multi_agent(
        policies=["app_policy"],
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "app_policy")
    )
    .resources(num_gpus=1)
    .rl_module(
        model_config={"fcnet_hiddens": [64, 64]},
    )
    .training(
        lr=5e-4
    )
)

# 3. 훈련기(Trainer) 생성
algo = config.build()

# 4. 훈련 루프 실행 (50회 반복)
print("\n--- 훈련 시작 ---")
for i in range(50):
    result = algo.train()
    
    # [수정된 부분] 정확한 키 경로에서 평균 보상값 가져오기
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float('nan'))
    
    print(f"Iteration: {i+1} | Mean Reward: {mean_reward:.2f}")

print("\n--- 훈련 종료 ---")

# 5. Ray 종료
ray.shutdown()