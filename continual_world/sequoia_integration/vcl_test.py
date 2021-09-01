from .vcl import VCL
from typing import ClassVar, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests


class TestVCLMethod(SACMethodTests):
    Method: ClassVar[Type[VCL]] = VCL
