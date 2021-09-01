from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from continual_world.config import AlgoConfig
from continual_world.methods.packnet import PackNetHelper
from continual_world.utils.utils import reset_optimizer
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting
import tensorflow as tf

from .base_sac_method import GradientsTuple, SACMethod


class PackNet(SACMethod):
    @dataclass
    class Config(SACMethod.Config):
        packnet_retrain_steps: int = 1000  # TODO: Double-check the value here.

    def __init__(self, algo_config: "PackNet.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: AlgoConfig
        self.packnet_helper: PackNetHelper

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        packnet_models = [self.actor]
        if self.algo_config.regularize_critic:
            packnet_models.extend([self.critic1, self.critic2])
        self.packnet_helper = PackNetHelper(packnet_models)

    def get_gradients(
        self,
        seq_idx: int,
        obs1: tf.Tensor,
        obs2: tf.Tensor,
        acts: tf.Tensor,
        rews: tf.Tensor,
        done: bool,
    ) -> Tuple[GradientsTuple, Dict]:
        gradients, metrics = super().get_gradients(
            seq_idx, obs1=obs1, obs2=obs2, acts=acts, rews=rews, done=done
        )

        actor_gradients, critic_gradients, alpha_gradient = gradients
        actor_gradients = self.packnet_helper.adjust_gradients(
            actor_gradients, self.actor.trainable_variables, tf.convert_to_tensor(seq_idx)
        )
        if self.algo_config.regularize_critic:
            critic_gradients = self.packnet_helper.adjust_gradients(
                critic_gradients, self.critic_variables, tf.convert_to_tensor(seq_idx)
            )
        gradients = GradientsTuple(actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics

    def on_task_switch(self, task_id: Optional[int]):
        super().on_task_switch(task_id)
    
    def handle_task_boundary(self, old_task: int, new_task: int, training: bool) -> None:
        # NOTE: This block was adapted from the main loop in 'fit', so should it include the logic
        # from the original 'task switch'? (Resetting the optimizer etc)?
        if training and new_task < self.num_tasks - 1:
            # self.algo_config.cl_method == "packnet"
            # and (current_task_t + 1 == steps_per_task)
            # and self.current_task_idx < self.num_tasks - 1
        # ):
            if new_task == 0:
                self.packnet_helper.set_freeze_biases_and_normalization(True)

            # Each task gets equal share of 'kernel' weights.
            if self.algo_config.packnet_fake_num_tasks is not None:
                num_tasks_left = (
                    self.algo_config.packnet_fake_num_tasks - new_task - 1
                )
            else:
                num_tasks_left = self.num_tasks - new_task - 1
                # num_tasks_left = env.num_envs - self.current_task_idx - 1
            prune_perc = num_tasks_left / (num_tasks_left + 1)
            self.packnet_helper.prune(prune_perc, self.current_task_idx)

            reset_optimizer(self.optimizer)

            for _ in range(self.algo_config.packnet_retrain_steps):
                batch = self.replay_buffer.sample_batch(self.algo_config.batch_size)
                self.learn_on_batch(tf.convert_to_tensor(self.current_task_idx), batch)

            reset_optimizer(self.optimizer)
        super().handle_task_boundary(old_task=old_task, new_task=new_task, training=training)
