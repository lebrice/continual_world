from typing import Any, ClassVar, Dict, Type
from .base_sac_method_test import TestSACMethod as SACMethodTests
from .reg_methods import RegMethod, L2Regularization, EWC, MAS
import pytest
from sequoia.common.config import Config


class RegMethodTests(SACMethodTests):
    Method: ClassVar[Type[RegMethod]]

    non_default_config_values: ClassVar[Dict[str, Any]] = {"cl_reg_coef": 1e-4}

    @pytest.fixture
    def method(self, config: Config):
        """ Fixture that provides the Method that will be used for testing. """
        algo_config = self.Method.Config(start_steps=50, update_after=50)
        return self.Method(algo_config=algo_config)


class TestL2RegMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = L2Regularization


class TestEWCMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = EWC


class TestMASMethod(RegMethodTests):
    Method: ClassVar[Type[RegMethod]] = MAS
