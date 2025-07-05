# train.py
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from kre_m_rl_env import KREmMultiAgentEnv

ray.init(num_cpus=8, object_store_memory=10*1024*1024*1024)

config = (
    PPOConfig()
    .environment(KREmMultiAgentEnv, env_config={"num_nodes": 2, "num_apps": 1})
    .framework("torch")
    # [수정된 부분] .env_runners() -> .rollouts()로 변경
    .rollouts(
        num_rollout_workers=4,
        batch_mode="complete_episodes"
    )
    .multi_agent(
        policies=["app_policy"],
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "app_policy")
    )
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )
    .resources(
        num_gpus=1
    )
    .training(
        lr=5e-4,
        model={"fcnet_hiddens": [64, 64]}
    )
)

algo = config.build()

print("\n--- 훈련 시작 ---")
for i in range(50):
    result = algo.train()
    
    mean_reward = result.get("episode_reward_mean", float('nan'))
    
    print(f"Iteration: {i+1} | Mean Reward: {mean_reward:.2f}")

print("--- 훈련 종료 ---")
ray.shutdown()