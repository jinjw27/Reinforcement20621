import gymnasium as gym
from stable_baselines3 import PPO
import os

# Import the custom environment from your local file
from environment import FluidHorizonEnv

def evaluate_model(model_path, num_episodes=10):
    """
    Loads a trained PPO model and evaluates its performance
    by rendering the environment visually.

    Args:
        model_path (str): The path to the trained model file (e.g., 'final_model.zip').
        num_episodes (int): The number of episodes to run for evaluation.
    """
    try:
        # Load the trained model
        model = PPO.load(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}. Please make sure it exists.")
        return

    # Create the environment with render_mode='human' to show the game window
    env = FluidHorizonEnv(render_mode="human")

    print("\nStarting evaluation...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Use the model to predict the next action
            # deterministic=True ensures the agent's actions are consistent
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    # Path to the trained model file
    MODEL_PATH = "final_model.zip"
    evaluate_model(MODEL_PATH)
