from .packnet import PackNet
from typing import Any, ClassVar, Dict, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests


class TestPackNetMethod(SACMethodTests):
    Method: ClassVar[Type[PackNet]] = PackNet

    non_default_config_values: ClassVar[Dict[str, Any]] = {"packnet_retrain_steps": 1e3}
