import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from src.game.ml.ml_environment import create_environment
    
class BetPercentageCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(BetPercentageCallback, self).__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        if len(self.model.logger.name_to_value) > 0:
            critic_loss = self.model.logger.name_to_value.get('train/critic_loss', None)
            actor_loss = self.model.logger.name_to_value.get('train/actor_loss', None)
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
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

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
    plt.savefig('sac_losses.png')