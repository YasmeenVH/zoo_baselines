# noinspection PyUnresolvedReferences
import growspace
import gym

from stable_baselines import TRPO
import config
import wandb
from callbacks import WandbStableBaselines2Callback


def train_trpo(save_model=False):
    wandb.run = config.tensorboard.run
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    env = gym.make(config.env_name)

    model = TRPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=config.num_updates, callback=WandbStableBaselines2Callback())
    if save_model:
        model.save(f"trpo_{config.env_name}")


train_trpo()
