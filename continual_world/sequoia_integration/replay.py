""" Replay method on top of the SAC backbone.

By default, it uses a different insertion strategy for the buffer, and doesn't clear it between
tasks.
"""
from dataclasses import dataclass
from logging import getLogger as get_logger

from continual_world.utils.utils import sci2int
from sequoia.settings.rl import DiscreteTaskAgnosticRLSetting
from sequoia.utils.utils import constant
from simple_parsing.helpers import choice, field

from .base_sac_method import SAC

logger = get_logger(__name__)


class Replay(SAC):
    """ Replay method.
    
    When the buffer is large enough to contain all samples, this is also called 'Perfect Memory'.
    """

    @dataclass
    class Config(SAC.Config):
        """ Hyper-Parameters """

        # Size of the replay buffer.
        replay_size: int = field(default=1_000_000, type=sci2int)

        # Strategy of inserting examples into the buffer
        buffer_type: str = choice("reservoir", "fifo", default="reservoir")  # type: ignore

        # Wether to reset the buffer after every task. In the case or Replay, we don't clear it.
        reset_buffer_on_task_change: bool = constant(False)

    def __init__(self, algo_config: "Replay.Config"):
        super().__init__(algo_config=algo_config)
        self.algo_config: "Replay.Config"

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        super().configure(setting)


class PerfectMemory(Replay):
    """ Replay method where all the training data is stored in the replay buffer. """

    @dataclass
    class Config(Replay.Config):
        # Size of the replay buffer.
        # NOTE: This value will be overwritten to be the max number of steps in the setting.
        replay_size: int = 1_000_000

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        self.algo_config.replay_size = setting.train_max_steps
        logger.info(
            f"Setting the replay buffer size to {setting.train_max_steps}, since this "
            f"method wants to store all training samples."
        )
        super().configure(setting)
