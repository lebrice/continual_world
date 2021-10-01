import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import gym
import numpy as np
from continual_world.envs import META_WORLD_TIME_HORIZON
from continual_world.utils.wrappers import ScaleReward, SuccessCounter
from gym import spaces
from gym.spaces import Box
from gym.utils.colorize import colorize
from gym.wrappers import TimeLimit
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.settings.rl import RLEnvironment, RLSetting
from sequoia.settings.rl.continual.objects import Actions, Observations, Rewards
from sequoia.settings.rl.environment import RLEnvironment
from sequoia.settings.rl.wrappers.task_labels import add_task_labels


@add_task_labels.register(Observations)
def _add_task_ids(obs: Observations, task_labels: Any) -> Observations:
    return dataclasses.replace(obs, task_labels=task_labels)


def wrap_sequoia_env(
    env: RLEnvironment,
    nb_tasks_in_env: int,
    add_task_ids: bool,
    max_episode_steps: Optional[int] = None,
    is_multitask: bool = False,
    scale_reward: bool = False,
    div_by_return: bool = False,
) -> "SequoiaToCWWrapper":
    # TODO: Implement a wrapper to mimic the `MultiTaskEnv` class from CW when the environment is
    # stationary.
    env = SequoiaToCWWrapper(
        env, nb_tasks_in_env=nb_tasks_in_env, add_task_labels=add_task_ids
    )
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = SuccessCounter(env)

    if scale_reward:
        env = ScaleReward(env, div_by_return=div_by_return)

    # TODO: Missing a 'name' property, which would usually be the task name in metaworld.
    # There doesn't seem to be a way to get the name of the task programmatically atm.
    # from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
    env.name = "fake-task-name"
    return env


def concat_x_and_t(observation: Observations, nb_tasks: int) -> np.ndarray:
    x = observation.x
    task_id = observation.task_labels
    if task_id is None:
        return x
    # NOTE: Assuming that the observations aren't batched for now.
    ts = np.zeros(nb_tasks)
    if task_id >= nb_tasks:
        # BUG: Getting a task_id greater than the number of tasks in TraditionalRLSetting?!
        # warnings.warn(
        #     RuntimeWarning(
        #         colorize(
        #             f"BUG: Getting a task id of {task_id} when we expected the total number of "
        #             f"tasks to be {nb_tasks}! Will change the task id to a value of {nb_tasks-1} "
        #             f" instead as a temporary fix. (obs = {observation})",
        #             "red",
        #         )
        #     )
        # )
        task_id = nb_tasks - 1
    ts[task_id] = 1
    return np.concatenate([x, ts], axis=-1)


class SequoiaToCWWrapper(gym.Wrapper):
    def __init__(
        self, env: RLEnvironment, nb_tasks_in_env: int, add_task_labels: bool = False
    ):
        """Create a wrapper around a gym.Env from Sequoia so it matches the format from cw.
        
        Parameters
        ----------
        env : RLEnvironment
            [description]
        nb_tasks_in_env : int
            The number of tasks that his environment will *actually* contain.
            TODO: Using this for the moment, since the `sac` function expects to get a single env
            that will go through all the tasks.
        """
        super().__init__(env=env)
        # TODO: Create a 'Box' space with the one-hot of the task ID in there.
        x_space = env.observation_space.x
        t_space = env.observation_space.task_labels
        from sequoia.common.spaces.sparse import Sparse

        if isinstance(t_space, Sparse):
            t_space = t_space.base
        self.onehot_len = t_space.n
        self.add_task_labels = add_task_labels
        if self.add_task_labels:
            self.observation_space = Box(
                low=np.concatenate([x_space.low, np.zeros(self.onehot_len)]),
                high=np.concatenate([x_space.high, np.ones(self.onehot_len)]),
            )
        else:
            self.observation_space = x_space

        if isinstance(self.env.action_space, spaces.Dict):
            self.action_space = self.env.action_space["y_pred"]
        if isinstance(self.env.reward_space, spaces.Dict):
            self.reward_space = self.env.reward_space["y_pred"]
        # Attributes to match the ContinualLearningEnv from continual_world.
        self.num_envs = nb_tasks_in_env
        self.cur_seq_idx: Optional[int] = None
        # self.steps_per_env = nb_tasks * self.env.max_steps
        # self.steps_limit = self.num_envs * self.steps_per_env

    def action(self, action: Actions):
        if isinstance(action, Actions):
            return action.y_pred
        return action

    def reward(self, reward: Rewards):
        if isinstance(reward, Rewards):
            return reward.y
        return reward

    def observation(self, observation: RLSetting.Observations) -> np.ndarray:
        x = observation.x
        if not self.add_task_labels:
            return x

        task_id = observation.task_labels
        self.cur_seq_idx = task_id
        ts = np.zeros(self.onehot_len)
        ts[task_id] = 1
        return np.concatenate([x, ts], axis=-1)

    def reset(self):
        return self.observation(super().reset())

    def step(self, action: Actions):
        action = self.action(action)
        obs, reward, done, info = super().step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        return obs, reward, done, info

    def pop_successes(self):
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes
