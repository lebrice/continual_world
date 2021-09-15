from dataclasses import dataclass
from typing import Dict, Type, Tuple
import tensorflow as tf

from sequoia.settings.rl.setting import RLSetting

from continual_world.config import AlgoConfig
from continual_world.methods.vcl import VclHelper, VclMlpActor
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting


from .base_sac_method import SAC


class VCL(SAC):
    @dataclass
    class Config(SAC.Config):
        vcl_first_task_kl: bool = True
        vcl_variational_ln: bool = False

    def __init__(self, algo_config: "VCL.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: AlgoConfig
        self.vcl_helper: VclHelper

        # Change the type of actor, compared to the base SAC method.
        self.actor_cl: Type[VclMlpActor] = VclMlpActor

    def get_actor_kwargs(self, setting: RLSetting) -> Dict:
        actor_kwargs = super().get_actor_kwargs(setting)
        actor_kwargs["variational_ln"] = self.algo_config.vcl_variational_ln
        return actor_kwargs

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        self.vcl_helper = VclHelper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )

    def get_auxiliary_loss(self, seq_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        aux_pi_loss, aux_value_loss = super().get_auxiliary_loss(seq_idx=seq_idx)

        reg_loss = self.vcl_helper.regularize(
            seq_idx, regularize_last_layer=self.algo_config.vcl_first_task_kl
        )
        reg_loss_coef = (
            self.algo_config.cl_reg_coef
            if seq_idx > 0 or self.algo_config.vcl_first_task_kl
            else 0.0
        )
        # reg_loss_coef = tf.cond(
        #     seq_idx > 0 or self.algo_config.vcl_first_task_kl,
        #     lambda: self.algo_config.cl_reg_coef,
        #     lambda: 0.0,
        # )
        reg_loss *= reg_loss_coef
        aux_pi_loss += reg_loss
        return aux_pi_loss, aux_value_loss

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        super().handle_task_boundary(task_id=task_id, training=training)
        
        
        if training and task_id > 0:
            self.vcl_helper.update_prior()

    # @tf.function
    def get_action(self, obs: tf.Tensor, deterministic: bool=tf.constant(False)) -> tf.Tensor:
        # NOTE: (from the original implementation):
        # Disabling multiple samples in VCL for faster evaluation
        return super().get_action(obs=obs, deterministic=deterministic)
        # mu, log_std, pi, logp_pi = self.actor(tf.expand_dims(obs, 0), samples_num=10)
        mu, log_std, pi, logp_pi = self.actor(tf.expand_dims(obs, 0))
        if deterministic:
            return mu[0]
        else:
            return pi[0]
