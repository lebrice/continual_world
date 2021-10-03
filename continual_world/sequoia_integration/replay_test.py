from .replay import Replay, PerfectMemory
from typing import Any, ClassVar, Dict, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests


class TestReplay(SACMethodTests):
    Method: ClassVar[Type[Replay]] = Replay
    non_default_config_values: ClassVar[Dict[str, Any]] = {
        "buffer_type": "reservoir",
        "reset_buffer_on_task_change": False,
    }


class TestPerfectMemory(TestReplay):
    Method: ClassVar[Type[Replay]] = PerfectMemory
