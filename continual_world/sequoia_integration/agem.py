from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import tensorflow as tf
from continual_world.methods.agem import AgemHelper
from continual_world.spinup.replay_buffers import EpisodicMemory
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting
from simple_parsing.helpers.hparams.hparam import categorical
from .base_sac_method import SAC, GradientsTuple, BatchDict
from logging import getLogger


logger = getLogger(__name__)


class AGEM(SAC):
    """Averaged Gradient Episodic Memory method.
    """

    __citation__ = """
    @misc{chaudhry2019efficient,
        title={Efficient Lifelong Learning with A-GEM}, 
        author={Arslan Chaudhry and Marc'Aurelio Ranzato and Marcus Rohrbach and Mohamed Elhoseiny},
        year={2019},
        eprint={1812.00420},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """

    @dataclass
    class Config(SAC.Config):
        """ Hyper-parameters specific to the AGEM method. """

        episodic_mem_per_task: int = 10_000
        episodic_batch_size: int = categorical(128, 256, default=128)

    def __init__(self, algo_config: "AGEM.Config" = None):
        super().__init__(algo_config=algo_config)
        self.algo_config: AGEM.Config
        self.episodic_memory: EpisodicMemory
        self.agem_helper: AgemHelper

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        # Also create the episodic memory buffer.
        episodic_mem_size = self.algo_config.episodic_mem_per_task * self.num_tasks
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
        )
        self.agem_helper = AgemHelper()

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        super().handle_task_boundary(task_id=task_id, training=training)
        # NOTE: Should we also do this agem step when the task IDS aren't available (task_id = -1)?
        # For now that's not a problem, because we always have task IDS during training.
        if training and task_id > 0:
            size = self.episodic_memory.size
            max_size = self.episodic_memory.max_size
            logger.info(f"Episodic memory buffer has {size}/{max_size} items.")
            # BUG: Overfilling the buffer raises an AssertionError.
            # NOTE: The episodic_memory is made to have a capacity that is large enough for at least
            # one task boundary per task. It's weird that we could encounter this 'full' problem.
            # TODO: Why/how does the AGEM in their repo know not to overfill this buffer?
            new_episodic_mem = self.replay_buffer.sample_batch(
                self.algo_config.episodic_mem_per_task,
                # min(self.replay_buffer.size, self.algo_config.episodic_mem_per_task)
            )
            n_new_items: int = new_episodic_mem["obs1"].shape[0]
            logger.info(f"Adding {n_new_items} new items to the episodic memory.")
            self.episodic_memory.store_multiple(**new_episodic_mem)

    def on_task_switch(self, task_id: Optional[int]):
        # NOTE: super().on_task_switch does things first, and then calls self.handle_task_boundary.
        super().on_task_switch(task_id)

        # self.learn_on_batch = tf.function(self.learn_on_batch)
        # self.get_gradients = tf.function(self.get_gradients)

    def sample_batches(self) -> Tuple[BatchDict, BatchDict]:
        batch, episodic_batch = super().sample_batches()
        if self.current_task_idx > 0:
            episodic_batch = self.episodic_memory.sample_batch(
                self.algo_config.episodic_batch_size
            )
        return batch, episodic_batch

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
        # TODO: This is from the original code, but need to clarify this, big time!
        # > Warning: we refer here to the int task_idx in the parent function, not
        # > the passed seq_idx.
        if self.current_task_idx > 0:
            if not episodic_batch:
                raise RuntimeError(
                    f"Need to pass episodic_batch to `get_gradients` for AGEM. "
                    f"({self.current_task_idx}, {self.training})"
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
