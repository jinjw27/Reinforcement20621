# evaluate_metrics.py
import numpy as np
from stable_baselines3 import PPO
from environment import FluidHorizonEnv

def evaluate_model(model_path="final_model.zip", num_episodes=10, max_steps=5000):
    """
    Evaluate a trained PPO model on the FluidHorizonEnv.
    Metrics: average reward, average survival time.
    """
    # 모델 불러오기
    model = PPO.load(model_path)
    env = FluidHorizonEnv(render_mode=None)

    rewards, survival_times = [], []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward, steps = 0, 0

        while not done and steps < max_steps:
            # 결정적 행동 선택 (재현성 보장)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        rewards.append(total_reward)
        survival_times.append(steps)
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, Steps={steps}")

    env.close()

    # 평균 지표 계산
    avg_reward = np.mean(rewards)
    avg_survival = np.mean(survival_times)

    print("\n=== Evaluation Summary ===")
    print(f"평균 보상 (Average Reward): {avg_reward:.2f}")
    print(f"평균 생존 시간 (Average Survival Time): {avg_survival:.2f} steps")

    return avg_reward, avg_survival

if __name__ == "__main__":
    evaluate_model("final_model.zip", num_episodes=10, max_steps=5000)
