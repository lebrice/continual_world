import gym
import numpy as np
from gym import spaces
from gym.spaces import Box
from sequoia.common.gym_wrappers import IterableWrapper
from sequoia.settings.rl import RLEnvironment, RLSetting
from sequoia.settings.rl.continual.objects import (Actions, Observations,
                                                   Rewards)
from sequoia.settings.rl.environment import RLEnvironment

from continual_world.utils.wrappers import SuccessCounter


def wrap_sequoia_env(env: RLEnvironment, nb_tasks_in_env: int) -> gym.Env:
    env = SequoiaToCWWrapper(env, nb_tasks_in_env=nb_tasks_in_env)
    env = SuccessCounter(env)
    return env

def concat_x_and_t(observation: Observations, nb_tasks: int) -> np.ndarray:
    x = observation.x
    task_id = observation.task_labels
    # NOTE: Assuming that the observations aren't batched for now.
    ts = np.zeros(nb_tasks)
    ts[task_id] = 1
    return np.concatenate([x, ts], axis=-1)


class SequoiaToCWWrapper(gym.Wrapper):
    def __init__(self, env: RLEnvironment, nb_tasks_in_env: int):
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
        self.onehot_len = t_space.n
        self.observation_space = Box(
            low=np.concatenate([x_space.low, np.zeros(self.onehot_len)]),
            high=np.concatenate([x_space.high, np.ones(self.onehot_len)]),
        )
        if isinstance(self.env.action_space, spaces.Dict):
            self.action_space = self.env.action_space["y_pred"]
        if isinstance(self.env.reward_space, spaces.Dict):
            self.reward_space = self.env.reward_space["y_pred"]
        # Attributes to match the ContinualLearningEnv from continual_world.
        self.num_envs = nb_tasks_in_env
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
        task_id = observation.task_labels
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
