from dataclasses import dataclass
from typing import Dict, Final

from sequoia.settings.rl.setting import RLSetting
from continual_world.sequoia_integration.sac_method import SACMethod
from continual_world.methods.vcl import VclMlpActor

# @dataclass
class L2Method(SACMethod):
    @dataclass
    class Config(SACMethod.Config):
        cl_method: Final[str] = "l2"


class VCL(SACMethod):
    @dataclass
    class Config(SACMethod.Config):
        cl_method: Final[str] = "vcl"

    def configure(self, setting: RLSetting) -> None:
        super().configure(setting)

    def get_actor_kwargs(self, setting: RLSetting) -> Dict:
        actor_kwargs = super().get_actor_kwargs(setting)
        actor_kwargs["variational_ln"] = self.algo_config.vcl_variational_ln
        return actor_kwargs

    def create_networks(self) -> None:
        self.actor_cl = VclMlpActor
        return super().create_networks()