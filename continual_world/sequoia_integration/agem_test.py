from typing import ClassVar, Type, Dict, Any
from .base_sac_method_test import TestSACMethod as SACMethodTests
from .agem import AGEM


class TestAGEMMethod(SACMethodTests):
    Method: ClassVar[Type[AGEM]] = AGEM
    non_default_config_values: ClassVar[Dict[str, Any]] = {
        "episodic_mem_per_task": 10_000,
        "episodic_batch_size": 128,
    }
