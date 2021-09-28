from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from continual_world.methods.regularization import (
    EWCHelper,
    L2Helper,
    MASHelper,
    RegularizationHelper,
)
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting
from sequoia.settings.rl.setting import RLSetting
from simple_parsing.helpers.hparams import categorical
from .base_sac_method import SAC


class RegMethod(SAC, ABC):
    """ Base class for regularization-based CL methods on top of the SAC backbone. """

    @dataclass
    class Config(SAC.Config):
        """ Hyper-Parameters of a regularization method for CRL. """

        # Regularization coefficient.
        cl_reg_coef: float = 1e-4

    def __init__(self, algo_config: "RegMethod.Config" = None):
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
        """ Create the regularization helper for this method. 
        
        Regularization methods need to implement this.
        """
        raise NotImplementedError()

    def get_auxiliary_losses(self, seq_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Get auxiliary losses for the actor and critic for the given task index.

        This is where the regularization methods do their work.
        """
        # NOTE: calling super()'s version even though it returns tf.zeros, because if we were to
        # eventually inherit from multiple regularization methods, then perhaps we could add the
        # losses.
        aux_actor_loss, aux_critic_loss = super().get_auxiliary_losses(seq_idx=seq_idx)
        if seq_idx > 0:
            reg_loss = self.algo_config.cl_reg_coef * self.reg_helper.regularize(
                self.old_params
            )
            aux_actor_loss += reg_loss
            aux_critic_loss += reg_loss
        return aux_actor_loss, aux_critic_loss
        # NOTE: Could also avoid computing the reg loss if we're going to multiply it by 0 anyway?
        # reg_loss = self.reg_helper.regularize(self.old_params)
        # reg_loss_coef = tf.cond(
        #     seq_idx > 0, lambda: self.algo_config.cl_reg_coef, lambda: 0.0
        # )
        # reg_loss *= reg_loss_coef
        # aux_actor_loss += reg_loss
        # aux_critic_loss += reg_loss
        # return aux_actor_loss, aux_critic_loss

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        super().handle_task_boundary(task_id=task_id, training=training)
        if training and task_id > 0:
            for old_param, new_param in zip(self.old_params, self.all_common_variables):
                old_param.assign(new_param)
            self.reg_helper.update_reg_weights(self.replay_buffer)


class L2Regularization(RegMethod):
    """ Simple L2 regularization method.
    
    Tries to prevent the weights from changing to much with respect to the old weights by adding an
    L2 penalty.
    """

    @dataclass
    class Config(RegMethod.Config):
        """ Hyper-parameters of the L2 regularization method. """

        # Regularization coefficient.
        cl_reg_coef: float = categorical(1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, default=1e5)

    def __init__(self, algo_config: "L2Regularization.Config" = None):
        super().__init__(algo_config=algo_config)
        self.algo_config: L2Regularization.Config
        self.reg_helper: L2Helper

    def get_reg_helper(self) -> L2Helper:
        return L2Helper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )


class EWC(RegMethod):
    """ Elastic Weight Consolidation method. """

    __citation__ = """
    @misc{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks}, 
        author={James Kirkpatrick and Razvan Pascanu and Neil Rabinowitz and Joel Veness and Guillaume Desjardins and Andrei A. Rusu and Kieran Milan and John Quan and Tiago Ramalho and Agnieszka Grabska-Barwinska and Demis Hassabis and Claudia Clopath and Dharshan Kumaran and Raia Hadsell},
        year={2017},
        eprint={1612.00796},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """

    @dataclass
    class Config(RegMethod.Config):
        """ Hyper-parameters of the EWC regularization method. """

        # EWC Regularization coefficient.
        cl_reg_coef: float = categorical(1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, default=1e4)

    def __init__(self, algo_config: "EWC.Config" = None):
        super().__init__(algo_config=algo_config)
        self.algo_config: EWC.Config
        self.reg_helper: EWCHelper

    def get_reg_helper(self) -> EWCHelper:
        return EWCHelper(
            self.actor,
            self.critic1,
            self.critic2,
            self.algo_config.regularize_critic,
            self.algo_config.critic_reg_coef,
        )


class MAS(RegMethod):
    """ Memory Aware Synapses Method. """

    __citation__ = """
    @misc{aljundi2018memory,
        title={Memory Aware Synapses: Learning what (not) to forget}, 
        author={Rahaf Aljundi and Francesca Babiloni and Mohamed Elhoseiny and Marcus Rohrbach and Tinne Tuytelaars},
        year={2018},
        eprint={1711.09601},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    """

    @dataclass
    class Config(RegMethod.Config):
        """ Hyper-parameters of the MAS regularization method. """

        # MAS Regularization coefficient.
        cl_reg_coef: float = categorical(1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, default=1e4)

    def __init__(self, algo_config: "MAS.Config" = None):
        super().__init__(algo_config=algo_config)
        self.algo_config: MAS.Config
        self.reg_helper: MASHelper

    def get_reg_helper(self) -> MASHelper:
        return MASHelper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )
