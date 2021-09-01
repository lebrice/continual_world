from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Final, Optional

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
            tf.Variable(tf.identity(param), trainable=False) for param in self.all_common_variables
        )
        self.reg_helper = self.get_reg_helper()

    @abstractmethod
    def get_reg_helper(self) -> RegularizationHelper:
        raise NotImplementedError()

    def on_task_switch(self, task_id: Optional[int]):
        super().on_task_switch(task_id)
        # TODO: Check that it's ok to do this AFTER the parts of super().on_task_switch that follow
        # where this used to be placed.

    def handle_task_boundary(self, old_task: int, new_task: int, training: bool) -> None:
        if training and new_task > 0:
            for old_param, new_param in zip(self.old_params, self.all_common_variables):
                old_param.assign(new_param)
            self.reg_helper.update_reg_weights(self.replay_buffer)
        super().handle_task_boundary(old_task=old_task, new_task=new_task, training=training)


class L2RegMethod(RegMethod):
    @dataclass
    class Config(RegMethod.Config):
        pass

    def __init__(self, algo_config: "L2RegMethod.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: L2RegMethod.Config
        self.reg_helper: L2Helper

    def get_reg_helper(self) -> L2Helper:
        return L2Helper(self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic)


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
        return MASHelper(self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic)
