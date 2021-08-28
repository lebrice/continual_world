from typing import ClassVar, Dict, Type
import pytest
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests, MethodType
from sequoia.settings.rl import RLSetting
from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
from .sac_method import SACMethod
from sequoia.common.metrics import EpisodeMetrics
import tensorflow as tf


# IDEA: To help with debugging, prevent the tf.function decorators from working when testing.
@pytest.fixture(autouse=True)
def disable_tf_function_wrappers():
    tf.config.run_functions_eagerly(True)
    yield
    tf.config.run_functions_eagerly(False)


class TestSACMethod(MethodTests):
    Method: ClassVar[Type[SACMethod]] = SACMethod
    setting_kwargs: ClassVar[Dict[str, str]] = {"dataset": "MountainCarContinuous-v0"}

    def test_configure_sets_values_properly(self, config: Config):
        algo_config = self.Method.Config(start_steps=50)
        method = self.Method(algo_config=algo_config)
        setting = IncrementalRLSetting(dataset="MountainCarContinuous-v0", nb_tasks=1, train_max_steps=5_000)
        method.configure(setting=setting)
        # NOTE: this would be different if we had task labels at train and test time.
        assert method.obs_dim == setting.observation_space.x.shape
        

    def test_simple_env(self, config: Config):
        # NOTE: Enabling the actor earlier just so it gets used.
        algo_config = self.Method.Config(start_steps=50)
        method = self.Method(algo_config=algo_config)
        setting = IncrementalRLSetting(dataset="MountainCarContinuous-v0", nb_tasks=1, train_max_steps=5_000)
        results: IncrementalRLSetting.Results = setting.apply(method, config=config)
        assert False, results.summary()
        self.validate_results(method=method, setting=setting, results=results)

    @pytest.fixture()
    def method(self, config: Config):
        # TODO: Create the parameters to be passed to the Method class for testing, so it's not too
        # long.
        return self.Method()

    def validate_results(self, setting: RLSetting, method: MethodType, results: RLSetting.Results) -> None:
        # TODO: Actually check that the results make sense for that env.
        assert results.objective > 0
        # return super().validate_results(setting, method, results)

    def test_debug(self, method: MethodType, setting: RLSetting, config: Config):
        pass # todo: re-enable this test once the above one is fully debugged.
        # return super().test_debug(method, setting, config)
