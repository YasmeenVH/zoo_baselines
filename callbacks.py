import array2gif
import torch
import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback as BaseCallbackBaselines3
from stable_baselines.common.callbacks import BaseCallback as BaseCallbackBaselines2

import config
import wandb


class WandbStableBaselines3Callback(BaseCallbackBaselines3):

    def __init__(self, verbose=0, model_name=None, env_name=None):
        super(WandbStableBaselines3Callback, self).__init__(verbose)
        self.model_name = model_name
        self.env_name = env_name

        self.episode_rewards = []
        self.episode_length = []
        self.episode_branches = []
        self.episode_branch1 = []
        self.episode_branch2 = []
        self.episode_light_width = []
        self.episode_light_move = []
        self.episode_success = []
        self.episode_plantpixel = []

        self.count_time_steps = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        infos = self.locals["infos"]
        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_length.append(info['episode']['l'])
                wandb.log({"Reward_per_step": info['episode']['r']})

            if 'new_branches' in info.keys():
                self.episode_branches.append(info['new_branches'])

            if 'new_b1' in info.keys():
                self.episode_branch1.append(info['new_b1'])

            if 'new_b2' in info.keys():
                self.episode_branch2.append(info['new_b2'])

            if 'light_width' in info.keys():
                self.episode_light_width.append(info['light_width'])

            if 'light_move' in info.keys():
                self.episode_light_move.append(info['light_move'])

            if 'success' in info.keys():
                self.episode_success.append(info['success'])

            if 'plant_pixel' in info.keys():
                self.episode_plantpixel.append(info['plant_pixel'])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.render_gif_on_wandb()

    def render_gif_on_wandb(self):
        images = []
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
        obs = env.reset()
        with torch.no_grad():
            for i in range(500):
                action, _states = self.model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if dones:
                    break
                obs = obs.transpose(1, 0, 2)
                obs = obs[:, :, ::-1]
                images.append(obs)

        array2gif.write_gif(images, 'replay.gif', fps=4)
        config.tensorboard.run.log({"video": wandb.Video('replay.gif', fps=4, format="gif")}, commit=True)
        config.tensorboard.run.history._flush()

        # env.reset()
        env.close()


class WandbStableBaselines2Callback(BaseCallbackBaselines2):

    def __init__(self, verbose=0, model_name=None, env_name=None):
        super(WandbStableBaselines2Callback, self).__init__(verbose)
        self.model_name = model_name
        self.env_name = env_name

        self.episode_rewards = []
        self.episode_length = []
        self.episode_branches = []
        self.episode_branch1 = []
        self.episode_branch2 = []
        self.episode_light_width = []
        self.episode_light_move = []
        self.episode_success = []
        self.episode_plantpixel = []

        self.count_time_steps = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        info = self.locals["info"]
        assert type(info) == dict
        if 'reward' in self.locals.keys():
            self.episode_rewards.append(self.locals["reward"])
            wandb.log({"Reward_per_step": self.locals["reward"]})

        if 'new_branches' in info.keys():
            self.episode_branches.append(info['new_branches'])

        if 'new_b1' in info.keys():
            self.episode_branch1.append(info['new_b1'])

        if 'new_b2' in info.keys():
            self.episode_branch2.append(info['new_b2'])

        if 'light_width' in info.keys():
            self.episode_light_width.append(info['light_width'])

        if 'light_move' in info.keys():
            self.episode_light_move.append(info['light_move'])

        if 'success' in info.keys():
            self.episode_success.append(info['success'])

        if 'plant_pixel' in info.keys():
            self.episode_plantpixel.append(info['plant_pixel'])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.render_gif_on_wandb()

    def render_gif_on_wandb(self):
        images = []
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
        obs = env.reset()
        with torch.no_grad():
            for i in range(500):
                action, _states = self.model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if dones:
                    break
                obs = obs.transpose(1, 0, 2)
                obs = obs[:, :, ::-1]
                images.append(obs)

        array2gif.write_gif(images, 'replay.gif', fps=4)
        config.tensorboard.run.log({"video": wandb.Video('replay.gif', fps=4, format="gif")}, commit=True)
        config.tensorboard.run.history._flush()

        # env.reset()
        env.close()
