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

    plot_losses(callback.losses)
    plot_actor_losses(callback.losses)


def moving_average(data, window_size):
    """Calculate the moving average of a given data list."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_losses(losses: list):
    window_size = 32
    steps, critic_losses, actor_losses = zip(*losses)

    critic_avg = moving_average(critic_losses, window_size)
    actor_avg = moving_average(actor_losses, window_size)

    # Adjust steps for moving average
    # Align steps with the moving average length
    steps_avg = steps[window_size - 1:]

    plt.figure(figsize=(12, 10))

    plt.plot(steps, critic_losses, 'r-', label='Critic Loss', alpha=0.25)
    plt.plot(steps, actor_losses, 'b-', label='Actor Loss', alpha=0.25)

    plt.plot(steps_avg, critic_avg, 'r--',
             label='Critic Loss Average', linewidth=2)
    plt.plot(steps_avg, actor_avg, 'b--',
             label='Actor Loss Average', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('SAC Loss Functions with Moving Averages')
    plt.legend()

    plt.savefig(generate_filepath('ml_sac_losses.png'))

def plot_actor_losses(losses: list):
    window_size = 32
    steps, _, actor_losses = zip(*losses)

    actor_avg = moving_average(actor_losses, window_size)

    # Adjust steps for moving average
    # Align steps with the moving average length
    steps_avg = steps[window_size - 1:]

    plt.figure(figsize=(12, 10))

    plt.plot(steps, actor_losses, 'b-', label='Actor Loss', alpha=0.25)

    plt.plot(steps_avg, actor_avg, 'b--',
             label='Actor Loss Average', linewidth=2)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('SAC Loss Functions with Moving Averages')
    plt.legend()

    plt.savefig(generate_filepath('ml_actor_losses.png'))



def plot_reward_function(initial_bankroll: int):
    # Generate bankroll values from 0 to 204000
    bankroll_values = np.linspace(0, 204000, 500)

    # Calculate corresponding rewards
    reward_values = [bankroll_to_reward(bankroll)
                     for bankroll in bankroll_values]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(reward_values, bankroll_values,
             label='Reward vs Stack Size', color='blue')
    plt.title('Stack Size to Reward Mapping')
    plt.xlabel('Rewards')
    plt.ylabel('Stack Size')
    plt.axvline(x=0, color='grey', linestyle='--',
                linewidth=0.7)  # Add a vertical line for y=0
    plt.axhline(y=initial_bankroll, color='green', linestyle='--', linewidth=0.7,
                label='Initial Stack Size Level')  # Marker for initial bankroll
    plt.legend()
    plt.grid()
    plt.savefig(generate_filepath("ml_reward.png"), dpi=600)


if __name__ == "__main__":
    default_total_timesteps = 1 << 18
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

    plot_reward_function(args.bankroll)

    train(generate_filepath("ml_model.zip"), initial_bankroll=args.bankroll,
          total_timesteps=args.total_timesteps)
