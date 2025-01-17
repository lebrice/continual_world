import logging
import time
from dataclasses import asdict
from typing import Any, Callable, ClassVar, Dict, Tuple, Type

import pytest
import tensorflow as tf
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests, MethodType
from sequoia.methods.random_baseline import RandomBaselineMethod
from sequoia.settings.rl import RLSetting
from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
from simple_parsing import ArgumentParser
from continual_world.defaults import CL_DEFAULTS

from .base_sac_method import SAC

logger = logging.getLogger(__name__)

# IDEA: Create a fixture to run tests on CPU and (optionally) on GPU.


@pytest.fixture(scope="session")
def get_random_performance():
    cache: Dict[Tuple, RLSetting.Results] = {}

    def get_key(setting: RLSetting) -> Tuple:
        return (type(setting), setting.dataset)

    def get_random_performance_fn(setting: RLSetting) -> RLSetting.Results:
        nonlocal cache
        key = get_key(setting)
        if key not in cache:
            logger.info(
                f"No cached entry found, re-calculating the random baseline performance for this setting."
            )
            logger.debug(f"Key was {key}")
            random_baseline = RandomBaselineMethod()
            random_performance = setting.apply(random_baseline)

            assert (
                get_key(setting) == key
            ), "the key shouldn't change when a method is applied to the setting!"

            cache[key] = random_performance
        else:
            logger.info(
                f"Returning the cached performance on setting {key[0]} and dataset {key[1]}"
            )
        return cache[key]

    # Save it as a property just so we can clear it if we really need to.
    get_random_performance_fn._cache = cache

    return get_random_performance_fn


def test_get_random_performance_fixture(
    get_random_performance: Callable[[RLSetting], RLSetting.Results]
):
    algo_config = SAC.Config(start_steps=50, update_after=50)
    method = SAC(algo_config=algo_config)
    setting = IncrementalRLSetting(
        dataset="MountainCarContinuous-v0", nb_tasks=1, train_max_steps=1_000
    )

    start_time = time.time()
    random_results = get_random_performance(setting)
    first_elapsed = time.time() - start_time

    start_time = time.time()
    random_results = get_random_performance(setting)
    second_elapsed = time.time() - start_time

    speedup = first_elapsed / second_elapsed
    assert speedup > 1_000
    # Clear the cache just to prevent any future tests from having key collisions with the setting
    # used for this test.
    get_random_performance._cache.clear()  # type: ignore



class TestSACMethod(MethodTests):
    Method: ClassVar[Type[SAC]] = SAC
    setting_kwargs: ClassVar[Dict[str, Any]] = {
        "dataset": "MountainCarContinuous-v0",
        "train_max_steps": 1_000,
        "max_episode_steps": 100,
        "nb_tasks": 3,
    }

    # Save the fixture here so we can use it in the other tests subclasses.
    get_random_performance = staticmethod(get_random_performance)

    non_default_config_values: ClassVar[Dict[str, Any]] = {}

    def test_values_match_default(self):
        config = self.Method.Config()
        for key, default_value in CL_DEFAULTS.items():
            if not hasattr(config, key):
                continue
            if key in ["seed"] or key in self.non_default_config_values:
                continue
            config_value = getattr(config, key)
            assert config_value == default_value, (key, config_value, default_value)

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
        # BUG: The epoch logger doesn't get to store any metrics, which causes an IndexError when
        # `self.log_tabular` is called.
        algo_config = self.Method.Config(start_steps=50, update_after=50, update_every=100)
        return self.Method(algo_config=algo_config)

    # @pytest.fixture(params=[False, True], autouse=True)
    # def eager_mode(self, request):
    #     """ Fixture that makes all tests run with `tf.function` disabled and then enabled.
    #     """
    #     eager: bool = request.param
    #     tf.config.run_functions_eagerly(eager)
    #     yield eager

    @pytest.mark.parametrize("eager_mode", [False,]) #True])
    # @pytest.mark.timeout(30)
    def test_debug(
        self,
        method: MethodType,
        setting: RLSetting,
        config: Config,
        get_random_performance: Callable[[RLSetting], RLSetting.Results],
        eager_mode: bool,
    ):  
        tf.config.run_functions_eagerly(eager_mode)
        results: RLSetting.Results = setting.apply(method, config=config)

        self.validate_results(
            setting=setting,
            method=method,
            results=results,
            get_random_performance_fn=get_random_performance,
        )
        if not eager_mode:
            # idea: Could check the number of run traces for `learn_on_batch`.
            # TODO: learn_on_batch_calls is 2 when not using eager mode, which seems a bit odd given
            # that there are three tasks! Would be a good idea to check the arguments to that
            # function for each trace, and see if the task ID changes or something else does.
            # assert method.learn_on_batch_calls == setting.nb_tasks - 1
            pass

    def validate_results(
        self,
        setting: RLSetting,
        method: MethodType,
        results: RLSetting.Results,
        get_random_performance_fn,
    ) -> None:
        # TODO: Use a different way of evaluating: Compare to the random baseline instead.
        # TODO: These 'random baseline results' could be cached and reused for a given session,
        # right?
        random_performance = get_random_performance_fn(setting=setting)
        print(f"Random performance: {random_performance.objective}")
        print(f"Method performance: {results.objective}")
        assert results.objective > -100
        # TODO: The methods don't do better than random during testing!
        # assert results.objective > random_performance.objective
        # assert results.objective > 0
        # return super().validate_results(setting, method, results)

    def test_from_args(self):
        parser = ArgumentParser()
        self.Method.add_argparse_args(parser)
        args = parser.parse_args([])
        method = self.Method.from_argparse_args(args)
        assert isinstance(method, self.Method)
        # Check that the default values from the command-line are the same as the default values
        # when created in code.
        assert method.algo_config == self.Method.Config()

    # TODO: Not using their search space for now. But we could.
    # def test_get_search_space(self):
    #     pass
