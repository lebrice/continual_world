from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from continual_world.methods.agem import AgemHelper
from continual_world.spinup.replay_buffers import EpisodicMemory
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting

from .base_sac_method import SAC, GradientsTuple, BatchDict


class AGEM(SAC):
    @dataclass
    class Config(SAC.Config):
        episodic_mem_per_task: int = 0
        episodic_batch_size: int = 0

    def __init__(self, algo_config: "AGEM.Config" = None):
        super().__init__(algo_config=algo_config)
        self.algo_config: AGEM.Config

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        episodic_mem_size = self.algo_config.episodic_mem_per_task * self.num_tasks
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
        )
        self.agem_helper = AgemHelper()

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        super().handle_task_boundary(task_id=task_id, training=training)
        # NOTE: Moved this to the subclasses.
        assert self.current_task_idx == task_id
        if task_id > 0:
            new_episodic_mem = self.replay_buffer.sample_batch(
                self.algo_config.episodic_mem_per_task
            )
            self.episodic_memory.store_multiple(**new_episodic_mem)
        # BUG: Gradients and var don't have the same shape!
        self.learn_on_batch = tf.function(self.learn_on_batch)
        self.get_gradients = tf.function(self.get_gradients)

    def sample_batches(self) -> Tuple[BatchDict, BatchDict]:
        batch, episodic_batch = super().sample_batches()
        if self.current_task_idx > 0:
            episodic_batch = self.episodic_memory.sample_batch(
                self.algo_config.episodic_batch_size
            )
        return batch, episodic_batch

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
            tf.convert_to_tensor(seq_idx),
            obs1=obs1,
            obs2=obs2,
            acts=acts,
            rews=rews,
            done=tf.convert_to_tensor(done),
            episodic_batch=None,
        )

        # Warning: we refer here to the int task_idx in the parent function, not
        # the passed seq_idx.
        # TODO: Does that make sense though?
        if self.current_task_idx > 0:
            if not episodic_batch:
                raise RuntimeError(
                    f"Need to pass episodic_batch to `get_gradients` for AGEM. ({self.current_task_idx}, {self.training})"
                )

            ref_gradients, _ = super().get_gradients(
                tf.convert_to_tensor(seq_idx),
                obs1=episodic_batch["obs1"],
                obs2=episodic_batch["obs2"],
                acts=episodic_batch["acts"],
                rews=episodic_batch["rews"],
                done=tf.convert_to_tensor(episodic_batch["done"]),
            )
            gradients, violation = self.agem_helper.adjust_gradients(
                gradients, ref_gradients
            )
            metrics["agem_violation"] = violation

        return gradients, metrics
