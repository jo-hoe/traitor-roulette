import argparse

from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from src.common import generate_filepath
from stable_baselines3.common.callbacks import BaseCallback
from src.game.ml.ml_environment import bankroll_to_reward, create_environment


class BetPercentageCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(BetPercentageCallback, self).__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        if len(self.model.logger.name_to_value) > 0:
            critic_loss = self.model.logger.name_to_value.get(
                'train/critic_loss', None)
            actor_loss = self.model.logger.name_to_value.get(
                'train/actor_loss', None)
            if critic_loss is not None and actor_loss is not None:
                self.losses.append((self.n_calls, critic_loss, actor_loss))

        return super()._on_step()

    def _get_final_observations_before_reset(self):
        for env_idx in range(self.model.n_envs):
            # Get the terminal observation of the just finished trajectory
            yield self.locals["infos"][env_idx].get("terminal_observation")


def train(model_file_path: str, initial_bankroll: int, total_timesteps: int):
    env = create_environment(initial_bankroll)
    model = SAC(
        "MlpPolicy",
        env
    )

    callback = BetPercentageCallback()
    model.learn(total_timesteps=total_timesteps,
                callback=callback, progress_bar=True)

    model.save(model_file_path)

    # Plot loss function
    steps, critic_losses, actor_losses = zip(*callback.losses)

    plt.figure(figsize=(12, 10))
    plt.plot(steps, critic_losses, 'r-', label='Critic Loss')
    plt.plot(steps, actor_losses, 'b-', label='Actor Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('SAC Loss Functions')
    plt.legend()

    plt.savefig(generate_filepath('sac_losses.png'))


def plot_rewards(initial_bankroll: int):
    # Generate bankroll values from 0 to 204000
    bankroll_values = np.linspace(0, 204000, 500)

    # Calculate corresponding rewards
    reward_values = [bankroll_to_reward(bankroll)
                     for bankroll in bankroll_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(reward_values, bankroll_values,
             label='Reward vs Pot size', color='blue')
    plt.title('Pot size to Reward Mapping')
    plt.xlabel('Rewards')
    plt.ylabel('Pot size')
    plt.axvline(x=0, color='grey', linestyle='--',
                linewidth=0.7)  # Add a vertical line for y=0
    plt.axhline(y=initial_bankroll, color='green', linestyle='--', linewidth=0.7,
                label='Initial Pot size Level')  # Marker for initial bankroll
    plt.legend()
    plt.grid()
    plt.savefig(generate_filepath("training_reward_function.png"), dpi=600)


if __name__ == "__main__":
    default_total_timesteps = 1 << 15
    default_bankroll = 68000

    parser = argparse.ArgumentParser(
        description='Train a machine learning model on Traitor Roulette.')
    parser.add_argument('--total_timesteps', dest='total_timesteps',
                        default=default_total_timesteps, type=int,
                        help=f'set the number of trained time steps, default is {default_total_timesteps}')
    parser.add_argument('--bankroll', dest='bankroll',
                        default=default_bankroll, type=int,
                        help=f'set your initial bankroll should be a multiple of 2000, default is {default_bankroll}')
    args = parser.parse_args()

    plot_rewards(args.bankroll)

    train(generate_filepath("ppo_trained_model.zip"), initial_bankroll=args.bankroll,
          total_timesteps=args.total_timesteps)
