from typing import ClassVar, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests
from .reg_methods import RegMethod, L2Regularization, EWC, MAS


class RegMethodTests(SACMethodTests):
    Method: ClassVar[Type[RegMethod]]


class TestL2RegMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = L2Regularization


class TestEWCMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = EWC


class TestMASMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = MAS
