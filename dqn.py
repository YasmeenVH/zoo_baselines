# noinspection PyUnresolvedReferences
import growspace

import gym

from stable_baselines3 import DQN
import config
import wandb
from callbacks import WandbStableBaselines3Callback


def train_dqn_growpsace(save_model=False):
    wandb.run = config.tensorboard.run
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    env = gym.make(config.env_name)
    # policy, env, learning_rate = 0.0001, buffer_size = 1000000, learning_starts = 50000, batch_size = 32, tau = 1.0,
    # gamma = 0.99, train_freq = 4, gradient_steps = 1, replay_buffer_class = None, replay_buffer_kwargs = None,
    # optimize_memory_usage = False, target_update_interval = 10000, exploration_fraction = 0.1,
    # exploration_initial_eps = 1.0, exploration_final_eps = 0.05, max_grad_norm = 10, tensorboard_log = None,
    # create_eval_env = False, policy_kwargs = None, verbose = 0, seed = None, device = 'auto', _init_setup_model = True
    model = DQN("CnnPolicy", env, verbose=1, gradient_steps=20, optimize_memory_usage=True)
    model.learn(total_timesteps=config.num_updates, log_interval=1, callback=WandbStableBaselines3Callback())
    if save_model:
        model.save(f"dqn_{config.env_name}")


train_dqn_growpsace()
