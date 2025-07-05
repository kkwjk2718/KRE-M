# train.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from kre_m_rl_env import KREmMultiAgentEnv

# 1. Ray 초기화
ray.init(num_cpus=8, object_store_memory=10*1024*1024*1024) # 10GB

# 2. PPO 알고리즘 설정 (API 변경)
config = (
    PPOConfig()
    .environment(KREmMultiAgentEnv, env_config={"num_nodes": 2, "num_apps": 5})
    .framework("torch")
    # .rollouts() -> .env_runners() 로 변경
    .env_runners(
        num_env_runners=4 # num_rollout_workers -> num_env_runners 로 변경
    )
    .training(
        model={"fcnet_hiddens": [64, 64]},
        train_batch_size=1024
    )
    .multi_agent(
        policies=["app_policy"],
        policy_mapping_fn=(lambda agent_id, episode, worker, **kwargs: "app_policy")
    )
)

# 3. 훈련기(Trainer) 생성
algo = config.build()

# 4. 훈련 루프 실행
print("\n--- 훈련 시작 ---")
for i in range(10):
    result = algo.train()
    print(f"Iteration: {i+1}")
    print(f"  Mean Reward: {result['episode_reward_mean']:.2f}")

print("--- 훈련 종료 ---")

# 5. Ray 종료
ray.shutdown()