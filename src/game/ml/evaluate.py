import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from src.game.ml.ml_environment import create_environment

def evaluate(model_path: str, initial_bankroll : int, num_episodes: int = 1000):
    model = SAC.load(model_path)
    total_rewards = []
    bet_sizes = [[] for _ in range(3)]  # List for each round
    final_bankrolls = []

    env = create_environment(initial_bankroll)

    for _ in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # New Gymnasium API
            obs = obs[0]
        done = False
        episode_reward = 0
        round_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            if len(step_result) == 5:  # New Gymnasium API
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # Old Gym API
                obs, reward, done, info = step_result
            
            episode_reward += reward
            
            if round_count < 3:
                bet_sizes[round_count].append(action[0])  # Assuming action[0] is the bet percentage
            round_count += 1
        
        total_rewards.append(episode_reward)
        final_bankrolls.append(obs[0])  # Assuming obs[0] is the bankroll

    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
    print(f"Standard deviation of rewards: {np.std(total_rewards)}")
    print(f"Best reward: {np.max(total_rewards)}")
    print(f"Worst reward: {np.min(total_rewards)}")
    print(f"Average final bankroll: {np.mean(final_bankrolls)}")

    # Plot average bet sizes per round
    plt.figure(figsize=(10, 6))
    rounds = ['Round 1', 'Round 2', 'Round 3']
    avg_bets = [np.mean(bets) for bets in bet_sizes]
    plt.bar(rounds, avg_bets)
    plt.title('Average Bet Size per Round')
    plt.ylabel('Bet Size (%)')
    plt.savefig('average_bets.png')
    plt.close()

    # Plot distribution of final bankrolls
    plt.figure(figsize=(10, 6))
    plt.hist(final_bankrolls, bins=50)
    plt.title('Distribution of Final Bankrolls')
    plt.xlabel('Bankroll')
    plt.ylabel('Frequency')
    plt.savefig('final_bankrolls.png')
    plt.close()