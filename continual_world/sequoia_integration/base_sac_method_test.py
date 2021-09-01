from typing import ClassVar, Dict, Type
import pytest
from sequoia.common.config import Config
from sequoia.methods import random_baseline
from sequoia.methods.method_test import MethodTests, MethodType
from sequoia.settings.rl import RLSetting
from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
from .base_sac_method import SACMethod
from sequoia.common.metrics import EpisodeMetrics
import tensorflow as tf
from sequoia.methods.random_baseline import RandomBaselineMethod


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
        setting = IncrementalRLSetting(
            dataset="MountainCarContinuous-v0", nb_tasks=1, train_max_steps=5_000
        )
        method.configure(setting=setting)
        # NOTE: this would be different if we had task labels at train and test time.
        assert method.obs_dim == setting.observation_space.x.shape

    # TODO: Maybe we could use a simpler continuous action space environment?
    @pytest.mark.skip(
        reason=f"A bit long and unstable with this env. Would need more steps to actually see a "
        f"difference in performance in MountainCar between the SAC and random. "
    )
    @pytest.mark.timeout(60)
    def test_simple_env(self, config: Config):
        # NOTE: Enabling the actor earlier just so it gets used.
        algo_config = self.Method.Config(start_steps=50)
        method = self.Method(algo_config=algo_config)
        setting = IncrementalRLSetting(
            dataset="MountainCarContinuous-v0",
            nb_tasks=1,
            train_max_steps=2_000,
            train_steps_per_task=2_000,
            test_max_steps=2_000,
            test_steps_per_task=2_000,
        )
        random_results = setting.apply(RandomBaselineMethod(), config=config)
        setting = IncrementalRLSetting(
            dataset="MountainCarContinuous-v0",
            nb_tasks=1,
            train_max_steps=2_000,
            train_steps_per_task=2_000,
            test_max_steps=2_000,
            test_steps_per_task=2_000,
        )
        results: IncrementalRLSetting.Results = setting.apply(method, config=config)
        assert results.objective > random_results.objective
        # self.validate_results(method=method, setting=setting, results=results)

    @pytest.fixture
    def method(self, config: Config):
        # TODO: Create the parameters to be passed to the Method class for testing, so it's not too
        # long.
        algo_config = self.Method.Config(start_steps=50, update_after=50)
        return self.Method(algo_config=algo_config)

    def validate_results(
        self, setting: RLSetting, method: MethodType, results: RLSetting.Results
    ) -> None:
        # TODO: Use a different way of evaluating: Compare to the random baseline instead.
        assert results.objective > -100
        # assert results.objective > 0
        # return super().validate_results(setting, method, results)

    # NOTE: This is the 'main' test.
    # def test_debug(self, method: MethodType, setting: RLSetting, config: Config):
    #     return super().test_debug(method, setting, config)

    def test_from_args(self):
        from simple_parsing import ArgumentParser
        import shlex
        parser = ArgumentParser()
        self.Method.add_argparse_args(parser, dest="method")
        args = parser.parse_args([])
        method = self.Method.from_argparse_args(args, dest="method")
        assert isinstance(method, self.Method)
        # TODO: Not sure how to check for this here.
        # assert method.algo_config.cl_method ==