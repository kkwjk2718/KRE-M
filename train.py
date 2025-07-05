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
)

algo = config.build()

print("\n--- 훈련 시작 ---")
for i in range(10):
    result = algo.train()
    
    # [수정된 부분] result 딕셔너리의 모든 키와 주요 정보를 출력
    print(f"\nIteration: {i+1}")
    print(f"Total Timesteps: {result['timesteps_total']}")
    print(f"Result Keys: {list(result.keys())}")
    
    # Mean Reward가 있는지 확인하고 출력
    if 'episode_reward_mean' in result:
        print(f"  Mean Reward: {result['episode_reward_mean']:.2f}")
    else:
        print("  Mean Reward: (Not available in this result)")


print("\n--- 훈련 종료 ---")
ray.shutdown()