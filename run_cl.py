import argparse
from continual_world.envs import get_single_env, get_cl_env
from continual_world.methods.vcl import VclMlpActor
from continual_world.spinup.models import MlpActor, MlpCritic, PopArtMlpCritic
from continual_world.spinup.utils.logx import EpochLogger
from continual_world.spinup.sac import sac
from continual_world.task_lists import task_seq_to_task_list
from continual_world.utils.utils import get_activation_from_str, sci2int, str2bool

from dataclasses import asdict
from simple_parsing import ArgumentParser
from simple_parsing.helpers import choice, field, list_field
from dataclasses import dataclass
from typing import Optional, List, Tuple

from continual_world.config import TaskConfig, AlgoConfig

import gym
from gym.spaces import Box
import numpy as np
from gym import spaces


# from sequoia.settings.rl import RLSetting, RLEnvironment
# from sequoia.settings.rl.continual.objects import Observations, Actions, Rewards
# from sequoia.common.gym_wrappers import IterableWrapper
# from continual_world.sequoia_compat import wrap_sequoia_env


def main(logger: EpochLogger, task_config: TaskConfig, algo_config: AlgoConfig):
    assert (task_config.tasks is None) != (task_config.task_list is None)
    if task_config.tasks is not None:
        task_config.tasks = task_seq_to_task_list[task_config.tasks]
    else:
        task_config.tasks = task_config.task_list
    train_env = get_cl_env(
        task_config.tasks,
        task_config.steps_per_task,
        algo_config.scale_reward,
        algo_config.div_by_return,
        randomization=algo_config.randomization,
    )
    # TODO: Testing out if we can swap out their env with ours instead:
    # TODO: Move the MT10 benchmarks to DiscreteTaskAgnosticRLSetting rather than
    # IncrementalRLSetting so that we can have a single env for all tasks in sequence.
    # BUG: When using MT10 as the dataset, we get an error with `self.test_steps_per_task` at line 302 (assert)
    # from sequoia.settings.rl import IncrementalRLSetting
    # setting = IncrementalRLSetting(
    #     dataset="CW20",
    #     train_steps_per_task=task_config.steps_per_task,
    #     train_max_steps = 20 * task_config.steps_per_task,
    # )

    # train_env_sequoia = setting.train_dataloader()
    # train_env_sequoia = wrap_sequoia_env(train_env_sequoia, nb_tasks=1)

    # BUG: Need to limit the number of steps a bit apparently.
    # train_env = train_env_sequoia
    # NOTE: Changing this a bit
    # steps = train_env.max_steps - 1

    steps = task_config.steps_per_task * len(task_config.tasks)

    # Consider normalizing test envs in the future.
    num_tasks = len(task_config.tasks)
    test_envs = [
        get_single_env(
            task,
            one_hot_idx=i,
            one_hot_len=num_tasks,
            randomization=algo_config.randomization,
        )
        for i, task in enumerate(task_config.tasks)
    ]

    num_heads = num_tasks if algo_config.multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=algo_config.hidden_sizes,
        activation=get_activation_from_str(algo_config.activation),
        use_layer_norm=algo_config.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=algo_config.hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=algo_config.hidden_sizes,
        activation=get_activation_from_str(algo_config.activation),
        use_layer_norm=algo_config.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=algo_config.hide_task_id,
    )
    if algo_config.use_popart:
        assert algo_config.multihead_archs, "PopArt works only in the multi-head setup"
        critic_cl = PopArtMlpCritic
    else:
        critic_cl = MlpCritic

    if algo_config.cl_method == "vcl":
        actor_cl = VclMlpActor
        actor_kwargs["variational_ln"] = algo_config.vcl_variational_ln
    else:
        actor_cl = MlpActor

    sac(
        train_env,
        test_envs=test_envs,
        steps=steps,
        logger=logger,
        actor_cl=actor_cl,
        actor_kwargs=actor_kwargs,
        critic_cl=critic_cl,
        critic_kwargs=critic_kwargs,
        # Task config
        task_config=task_config,
        algo_config=algo_config,

        # seed=task_config.seed,
        # replay_size=task_config.replay_size,
        # # Algo config
        # batch_size=algo_config.batch_size,
        # buffer_type=algo_config.buffer_type,
        # # algo_config=algo_config,
        # # )
        # reset_buffer_on_task_change=algo_config.reset_buffer_on_task_change,
        # reset_optimizer_on_task_change=algo_config.reset_optimizer_on_task_change,
        # lr=algo_config.lr,
        # alpha=algo_config.alpha,
        # cl_method=algo_config.cl_method,
        # cl_reg_coef=algo_config.cl_reg_coef,
        # packnet_retrain_steps=algo_config.packnet_retrain_steps,
        # regularize_critic=algo_config.regularize_critic,
        # vcl_first_task_kl=algo_config.vcl_first_task_kl,
        # episodic_mem_per_task=algo_config.episodic_mem_per_task,
        # episodic_batch_size=algo_config.episodic_batch_size,
        # reset_critic_on_task_change=algo_config.reset_critic_on_task_change,
        # clipnorm=algo_config.clipnorm,
        # gamma=algo_config.gamma,
        # target_output_std=algo_config.target_output_std,
        # packnet_fake_num_tasks=algo_config.packnet_fake_num_tasks,
        # agent_policy_exploration=algo_config.agent_policy_exploration,
        # critic_reg_coef=algo_config.critic_reg_coef,
    )


def get_parser() -> Tuple[TaskConfig, AlgoConfig]:
    parser = ArgumentParser(description="Continual World")

    parser.add_arguments(TaskConfig, "task")
    parser.add_arguments(AlgoConfig, "algo")

    args = parser.parse_args()
    task: TaskConfig = args.task
    algo: AlgoConfig = args.algo

    return task, algo


if __name__ == "__main__":
    # args = vars(get_parser())
    task_config, algo_config = get_parser()
    args_for_epoch_logger = dict(**asdict(task_config), **asdict(algo_config))
    logger = EpochLogger(task_config.logger_output, config=args_for_epoch_logger)
    # del task_config.logger_output  # TODO: Why though?
    main(logger, task_config=task_config, algo_config=algo_config)
