from typing import ClassVar, Type
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests
from .sac_method import SACMethod


class TestSACMethod(MethodTests):
    Method: ClassVar[Type[SACMethod]] = SACMethod

    def method(self, config: Config):
        return self.Method()

