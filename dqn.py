# noinspection PyUnresolvedReferences
import growspace
import gym
import wandb
from stable_baselines3 import DQN

import config
from callbacks import WandbStableBaselines3Callback


def train_dqn_growpsace(save_model=False):
    wandb.run = config.tensorboard.run
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    env = gym.make(config.env_name)

    model = DQN("CnnPolicy", env, verbose=1, gradient_steps=20, optimize_memory_usage=True)
    model.learn(total_timesteps=config.num_updates, log_interval=1, callback=WandbStableBaselines3Callback())
    if save_model:
        model.save(f"dqn_{config.env_name}")


train_dqn_growpsace()
