from typing import ClassVar, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests
from .reg_methods import RegMethod, L2RegMethod, EWCRegMethod, MASRegMethod


class RegMethodTests(SACMethodTests):
    Method: ClassVar[Type[RegMethod]]


class TestL2RegMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = L2RegMethod


class TestEWCMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = EWCRegMethod


class TestMASMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = MASRegMethod
