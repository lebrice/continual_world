from typing import ClassVar, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests
from .agem import AGEM


class TestAGEMMethod(SACMethodTests):
    Method: ClassVar[Type[AGEM]] = AGEM
