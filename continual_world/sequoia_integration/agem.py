from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from continual_world.methods.agem import AgemHelper
from continual_world.spinup.replay_buffers import EpisodicMemory
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting

from .base_sac_method import SACMethod, GradientsTuple, BatchDict


class AGEM(SACMethod):
    @dataclass
    class Config(SACMethod.Config):
        episodic_mem_per_task: int = 0
        episodic_batch_size: int = 0

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        episodic_mem_size = self.algo_config.episodic_mem_per_task * self.num_tasks
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
        )
        self.agem_helper = AgemHelper()

    def handle_task_boundary(self, old_task: int, new_task: int, training: bool) -> None:
        # NOTE: Moved this to the subclasses.
        if self.current_task_idx > 0:
            new_episodic_mem = self.replay_buffer.sample_batch(
                self.algo_config.episodic_mem_per_task
            )
            self.episodic_memory.store_multiple(**new_episodic_mem)
        super().handle_task_boundary(old_task=old_task, new_task=new_task, training=training)

    def get_gradients(
        self,
        seq_idx: int,
        obs1: tf.Tensor,
        obs2: tf.Tensor,
        acts: tf.Tensor,
        rews: tf.Tensor,
        done: bool,
        episodic_batch: BatchDict=None,
    ) -> Tuple[List[tf.Tensor], Dict]:
        gradients, metrics = super().get_gradients(
            seq_idx, obs1=obs1, obs2=obs2, acts=acts, rews=rews, done=done
        )
        # Warning: we refer here to the int task_idx in the parent function, not
        # the passed seq_idx.
        if self.current_task_idx > 0:
            if not episodic_batch:
                raise RuntimeError(f"Need to pass episodic_batch to `get_gradients` for AGEM.")

            ref_gradients, _ = super().get_gradients(
                seq_idx,
                obs1=episodic_batch["obs1"],
                obs2=episodic_batch["obs2"],
                acts=episodic_batch["acts"],
                rews=episodic_batch["rews"],
                done=episodic_batch["done"],
            )
            gradients, violation = self.agem_helper.adjust_gradients(gradients, ref_gradients)
            metrics["agem_violation"] = violation
        return gradients, metrics
