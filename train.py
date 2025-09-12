import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the custom environment class from the provided file
from environment import FluidHorizonEnv

# --- Configuration ---
# Set the total number of timesteps for the training
TIMESTEPS = 1_000_000
# Define the directories for saving checkpoints and logs
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./tensorboard_logs/"

def main():
    """
    Trains a PPO agent on the Fluid Horizon environment, with seamless
    resuming from the latest checkpoint.
    """
    # Create the directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Instantiate the environment in headless mode for faster training
    env = FluidHorizonEnv(render_mode=None)
    # Wrap the environment with Monitor for better logging
    env = Monitor(env)

    # --- Policy Selection Logic ---
    # The observation space is a 1D vector (Box), so we use MlpPolicy.
    policy = "MlpPolicy"

    model = None
    latest_checkpoint = None

    # Search for the latest checkpoint to resume from
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if checkpoints:
        # Find the latest checkpoint file by sorting
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"✅ Resuming training from existing checkpoint: {latest_checkpoint}")
        # Load the model from the checkpoint
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log=LOG_DIR, verbose=1)
    else:
        print("🔍 No checkpoint found. Starting a new training run.")
        # Create a new PPO model from scratch
        model = PPO(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

    # Configure CheckpointCallback to save the model periodically
    # The save_freq is the number of steps between saves
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_model",
    )

    # Train the agent
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=checkpoint_callback,
        # Set reset_num_timesteps to False to continue the timestep counter from the loaded model
        reset_num_timesteps=latest_checkpoint is None,
    )

    # Save the final model after training is complete
    model.save("final_model.zip")
    print("✨ Training complete. Final model saved as final_model.zip")

if __name__ == "__main__":
    # This is the entry point for the training script
    main()
