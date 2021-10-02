from typing import ClassVar, Type, Dict, Any
from .vcl import VCL
from .base_sac_method_test import TestSACMethod as SACMethodTests


class TestVCLMethod(SACMethodTests):
    Method: ClassVar[Type[VCL]] = VCL
    non_default_config_values: ClassVar[Dict[str, Any]] = {"cl_reg_coef": 1e-4}
