# train.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from kre_m_rl_env import KREmMultiAgentEnv

ray.init(num_cpus=8, object_store_memory=10*1024*1024*1024) # 10GB

config = (
    PPOConfig()
    .environment(KREmMultiAgentEnv, env_config={"num_nodes": 2, "num_apps": 5})
    .framework("torch")
    .env_runners(
        num_env_runners=4
    )
    .multi_agent(
        policies=["app_policy"],
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "app_policy")
    )
    .resources(num_gpus=1)
    .rl_module(
        model_config={"fcnet_hiddens": [64, 64]},
    )
)

algo = config.build()

print("\n--- 훈련 시작 ---")
for i in range(10):
    result = algo.train()
    
    # [수정된 부분] .get()을 사용하여 안전하게 키 값 가져오기
    mean_reward = result.get('episode_reward_mean', 0)
    
    print(f"Iteration: {i+1}")
    print(f"  Mean Reward: {mean_reward:.2f}")

print("--- 훈련 종료 ---")
ray.shutdown()