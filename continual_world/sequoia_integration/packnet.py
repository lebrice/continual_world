import warnings
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Dict, List, Optional, Tuple

import gym
import tensorflow as tf
import tqdm
from continual_world.config import AlgoConfig
from continual_world.methods.packnet import PackNetHelper
from continual_world.spinup.objects import BatchDict
from continual_world.utils.utils import reset_optimizer
from sequoia.settings.rl import TaskIncrementalRLSetting

from .base_sac_method import SAC, GradientsTuple

logger = get_logger(__name__)


class PackNet(SAC, target_setting=TaskIncrementalRLSetting):  # type: ignore
    # NOTE: This PackNet method requires task labels for now.

    @dataclass
    class Config(SAC.Config):
        packnet_retrain_steps: int = 1000  # TODO: Double-check the value here.

    def __init__(self, algo_config: "PackNet.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: "PackNet.Config"
        self.packnet_helper: PackNetHelper

    def configure(self, setting: TaskIncrementalRLSetting) -> None:
        super().configure(setting)
        packnet_models = [self.actor]
        if self.algo_config.regularize_critic:
            packnet_models.extend([self.critic1, self.critic2])
        self.packnet_helper = PackNetHelper(packnet_models)

    @tf.function
    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        obs1: tf.Tensor,
        obs2: tf.Tensor,
        acts: tf.Tensor,
        rews: tf.Tensor,
        done: tf.Tensor,
        episodic_batch: BatchDict = None,
    ) -> Tuple[GradientsTuple, Dict]:
        gradients, metrics = super().get_gradients(
            seq_idx,
            obs1=obs1,
            obs2=obs2,
            acts=acts,
            rews=rews,
            done=done,
            episodic_batch=episodic_batch,
        )

        actor_gradients, critic_gradients, alpha_gradient = gradients
        actor_gradients = self.packnet_helper.adjust_gradients(
            actor_gradients, self.actor.trainable_variables, tf.convert_to_tensor(seq_idx),
        )
        if self.algo_config.regularize_critic:
            critic_gradients = self.packnet_helper.adjust_gradients(
                critic_gradients, self.critic_variables, tf.convert_to_tensor(seq_idx)
            )
        gradients = GradientsTuple(actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics

    def fit(self, train_env: gym.Env, valid_env: gym.Env) -> None:
        self.set_view(-1)
        return super().fit(train_env, valid_env)
    
    def on_eval_episode_start(self, seq_idx: int, episode: int) -> None:
        super().on_eval_episode_start(seq_idx=seq_idx, episode=episode)
        self.set_view(seq_idx)

    def test_agent_on_env(
        self, test_env: gym.Env, deterministic: bool, max_episodes: int
    ) -> Tuple[List[bool], Dict[int, Dict[str, List[float]]]]:
        successes, metrics = super().test_agent_on_env(test_env, deterministic, max_episodes)
        # Reset the most up-to-date version of the model.
        self.set_view(-1)
        return successes, metrics

    def get_actions(
        self, observations: TaskIncrementalRLSetting.Observations, action_space: gym.Space
    ) -> TaskIncrementalRLSetting.Actions:
        if observations.task_labels is not None:
            task_id = observations.task_labels
            if task_id > self.current_task_idx:
                task_id = -1
            self.set_view(task_id)
        return super().get_actions(observations, action_space)

    def on_task_switch(self, task_id: Optional[int]):
        super().on_task_switch(task_id)

    def set_view(self, task_id: int) -> None:
        """ Set the "view" in the packnet helper if it isn't already set to that value. """
        if self.packnet_helper.current_view != task_id:
            if task_id == -1:
                logger.info(f"Resetting the view to the most recent model.")
            elif task_id > self.current_task_idx:
                logger.warning(
                    RuntimeWarning(
                        f"Being tested on future task ({task_id}) for which we don't yet have a pruned"
                        f" model! Using the latest model instead. "
                    )
                )
                task_id = -1
            else:
                logger.info(f"Setting view to pruned model for task id {task_id}")
            self.packnet_helper.set_view(task_id)

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        # NOTE: The base class doesn't currently do anything in this method, so it's safe to run it
        # before. However, if this were to be used in conjunction with another method, then things
        # could get a bit complicated: We'd have to figure out if this packnet stuff needs to go
        # before or after it.
        super().handle_task_boundary(task_id=task_id, training=training)
        # NOTE: Don't do anything at the start of training (when `task_id` == 0).
        # TODO: How should we handle a `task_id` value of `-1`? Should we just pretend it's a new
        # task? For now we don't handle it.
        if task_id == -1:
            warnings.warn(
                RuntimeWarning(
                    f"Unable to use PackNet effectively, since the task labels weren't given!"
                )
            )
        # NOTE: self.curent_task_idx is the *current* training task, while `task_id` is the new/next
        # task.
        # NOTE: We do nothing on the last task here.
        if not training:
            # If we're switching to a previous task, set the view.
            # NOTE: If we switch to a task we haven't encountered before, we use the current
            # (most up to date) view and raise a warning.
            self.set_view(task_id)

        if training and 1 <= task_id < self.num_tasks - 1:
            if task_id == 1:
                self.packnet_helper.set_freeze_biases_and_normalization(True)

            # Each task gets equal share of 'kernel' weights.
            if self.algo_config.packnet_fake_num_tasks is not None:
                num_tasks_left = self.algo_config.packnet_fake_num_tasks - task_id - 1
            else:
                num_tasks_left = self.num_tasks - task_id - 1
                # num_tasks_left = env.num_envs - self.current_task_idx - 1
            prune_perc = num_tasks_left / (num_tasks_left + 1)
            self.packnet_helper.prune(prune_perc, self.current_task_idx)

            reset_optimizer(self.optimizer)

            for _ in tqdm.tqdm(range(self.algo_config.packnet_retrain_steps), desc="finetuning"):
                batch = self.replay_buffer.sample_batch(self.algo_config.batch_size)
                self.learn_on_batch(tf.convert_to_tensor(self.current_task_idx), batch)

            reset_optimizer(self.optimizer)
