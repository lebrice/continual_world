from dataclasses import dataclass
from typing import Type

from continual_world.config import AlgoConfig
from continual_world.methods.vcl import VclHelper, VclMlpActor
from sequoia.settings.rl.discrete.setting import DiscreteTaskAgnosticRLSetting


from .base_sac_method import SACMethod


class VCL(SACMethod):
    @dataclass
    class Config(SACMethod.Config):
        packnet_retrain_steps: int = 0

    def __init__(self, algo_config: "VCL.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: AlgoConfig
        self.vcl_helper: VclHelper

        # Change the type of actor, compared to the base SAC method.
        self.actor_cl: Type[VclMlpActor] = VclMlpActor

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)
        self.vcl_helper = VclHelper(
            self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        )

    def handle_task_boundary(self, old_task: int, new_task: int, training: bool) -> None:
        # NOTE: Moved this to the subclasses.
        if training and new_task > 0:
            self.vcl_helper.update_prior()
        super().handle_task_boundary(old_task=old_task, new_task=new_task, training=training)
