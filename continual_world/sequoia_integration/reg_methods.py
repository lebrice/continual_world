from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Final, Optional, Tuple

import tensorflow as tf
from continual_world.methods.regularization import (
    EWCHelper,
    L2Helper,
    MASHelper,
    RegularizationHelper,
)
from continual_world.methods.vcl import VclMlpActor
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting
from sequoia.settings.rl.setting import RLSetting

from .base_sac_method import SACMethod


class RegMethod(SACMethod, ABC):
    @dataclass
    class Config(SACMethod.Config):
        """ Hyper-Parameters of a regularization method for CRL. """

        cl_reg_coef: float = 0.0

    def __init__(self, algo_config: "RegMethod.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: RegMethod.Config
        self.reg_helper: RegularizationHelper

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        self.old_params = list(
            tf.Variable(tf.identity(param), trainable=False)
            for param in self.all_common_variables
        )
        self.reg_helper = self.get_reg_helper()

    @abstractmethod
    def get_reg_helper(self) -> RegularizationHelper:
        raise NotImplementedError()

    def get_auxiliary_loss(self, seq_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        aux_pi_loss, aux_value_loss = super().get_auxiliary_loss(seq_idx=seq_idx)
        reg_loss = self.reg_helper.regularize(self.old_params)
        # TODO: Could also avoid computing the reg loss if we're going to multiply it by 0 anyway.
        reg_loss_coef = tf.cond(
            seq_idx > 0, lambda: self.algo_config.cl_reg_coef, lambda: 0.0
        )
        reg_loss *= reg_loss_coef
        aux_pi_loss += reg_loss
        aux_value_loss += reg_loss
        return aux_pi_loss, aux_value_loss

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        super().handle_task_boundary(task_id=task_id, training=training)
        assert self.current_task_idx == task_id
        if training and task_id > 0:
            for old_param, new_param in zip(self.old_params, self.all_common_variables):
                old_param.assign(new_param)
            self.reg_helper.update_reg_weights(self.replay_buffer)


class L2RegMethod(RegMethod):
    @dataclass
    class Config(RegMethod.Config):
        pass

    def __init__(self, algo_config: "L2RegMethod.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: L2RegMethod.Config
        self.reg_helper: L2Helper

    def get_reg_helper(self) -> L2Helper:
        return L2Helper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )


class EWCRegMethod(RegMethod):
    @dataclass
    class Config(RegMethod.Config):
        pass

    def __init__(self, algo_config: "EWCRegMethod.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: EWCRegMethod.Config
        self.reg_helper: EWCHelper

    def get_reg_helper(self) -> EWCHelper:
        return EWCHelper(
            self.actor,
            self.critic1,
            self.critic2,
            self.algo_config.regularize_critic,
            self.algo_config.critic_reg_coef,
        )


class MASRegMethod(RegMethod):
    @dataclass
    class Config(RegMethod.Config):
        pass

    def __init__(self, algo_config: "MASRegMethod.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: MASRegMethod.Config
        self.reg_helper: MASHelper

    def get_reg_helper(self) -> MASHelper:
        return MASHelper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )
