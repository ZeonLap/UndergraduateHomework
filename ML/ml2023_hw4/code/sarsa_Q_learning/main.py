import gym
import matplotlib.pyplot as plt
import os
from algorithms import QLearning, Sarsa
from utils import render_single_Q, evaluate_Q
from argparse import ArgumentParser


# Feel free to run your own debug code in main!

parser = ArgumentParser()

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--algorithm", type=str, default=0.9, choices=["Q_learning", "Sarsa"])

args = parser.parse_args()

def main():
    num_episodes = 5000
    env = gym.make('Taxi-v3')

    if args.algorithm == "Q_learning":
        Q1, Q_rewards = QLearning(env, num_episodes, lr=args.lr)
    elif args.algorithm == "Sarsa":
        Q1, Q_rewards = Sarsa(env, num_episodes, lr=args.lr)
    else:
        assert False, "Unknown Algorithm"

    # q_learning
    Q1, Q_rewards = QLearning(env, num_episodes, lr=args.lr)
    render_single_Q(env, Q1)
    evaluate_Q(env, Q1, 200)

    plt.plot(range(num_episodes), Q_rewards)
    plt.savefig(os.path.join("figs", f"{args.algorithm} with lr={args.lr}.png"))
    plt.title(f"{args.algorithm} with lr={args.lr}")
    plt.ylim(-100, 100)
    # plt.show()


if __name__ == '__main__':
    main()
