from .packnet import PackNet
from typing import ClassVar, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests


class TestPackNetMethod(SACMethodTests):
    Method: ClassVar[Type[PackNet]] = PackNet
