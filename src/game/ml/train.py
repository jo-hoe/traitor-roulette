import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback

from src.game.ml.ml_environment import create_environment

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.losses = []
        self.std_devs = []
        self.final_bankrolls = []

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:  # Collect data every 1000 steps
            loss = self.model.logger.name_to_value['train/loss']
            self.losses.append(loss)
            
            # Evaluate std dev of rewards
            rewards = []
            for _ in range(100):  # Run 100 episodes to calculate std dev
                obs = self.training_env.reset()
                if isinstance(obs, tuple):  # New Gymnasium API
                    obs = obs[0]
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.training_env.step(action)
                    if len(step_result) == 5:  # New Gymnasium API
                        obs, reward, terminated, truncated, _ = step_result
                        done = terminated or truncated
                    else:  # Old Gym API
                        obs, reward, done, _ = step_result
                    episode_reward += reward
                rewards.append(episode_reward)
            self.std_devs.append(np.std(rewards))
            self.final_bankrolls.append(obs[0])
        return True

def train(model_file_path: str, initial_bankroll: int, total_timesteps: int):
    env = create_environment(initial_bankroll)
    model = A2C(
        "MlpPolicy",
        env,
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,    # Value function coefficient
        max_grad_norm=0.5,
        verbose=1
    )

    callback = CustomCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(model_file_path)
    
    # Plot loss and std dev
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(callback.losses)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(callback.std_devs)
    plt.title('Standard Deviation of Rewards')
    plt.xlabel('Iterations (x1000)')
    plt.ylabel('Std Dev')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()