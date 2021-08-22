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

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class TaskConfig:
    """ Configuration options for the task(s). """
    # Name of the sequence you want to run
    tasks: str = choice(*list(task_seq_to_task_list.keys()))  # type: ignore
    # Types of logger used.
    logger_output: List[Literal["neptune", "tensorboard", "tsv"]] = choice(  # type: ignore
        "neptune", "tensorboard", "tsv"
    )
    # Random seed used for running the experiments
    seed: int  # type: ignore
    # Numer of steps per task
    steps_per_task: int = field(default=int(1e6), type=sci2int)
    # Size of the replay buffer
    replay_size: int = field(default=int(1e6), type=sci2int)
    # Number of samples in each mini-batch sampled by SAC
    batch_size: int = 128
    # Hidden layers sizes in the base network
    hidden_sizes: List[int] = list_field(256, 256, 256, 256)
    # Strategy of inserting examples into the buffer
    buffer_type: str = choice("fifo", "reservoir", default="fifo")
    # List of tasks you want to run, by name or by the MetaWorld index
    task_list: Optional[List[str]] = None


@dataclass
class AlgoConfig:
    """ Configuration options for the Algorithm. """
    reset_buffer_on_task_change: bool = True
    reset_optimizer_on_task_change: bool = True
    reset_critic_on_task_change: bool = False
    activation: str = "lrelu"
    use_layer_norm: bool = True
    scale_reward: bool = False
    div_by_return: bool = False
    lr: float = 1e-3
    alpha: str = "auto"
    use_popart: bool = False
    cl_method: Optional[str] = choice(  # type: ignore
        None, "l2", "ewc", "mas", "vcl", "packnet", "agem", default=None
    )
    packnet_retrain_steps: int = 0
    regularize_critic: bool = False
    cl_reg_coef: float = 0.0
    vcl_first_task_kl: bool = True
    vcl_variational_ln: bool = False
    episodic_mem_per_task: int = 0
    episodic_batch_size: int = 0
    randomization: str = "random_init_all"
    multihead_archs: bool = True
    hide_task_id: bool = True
    clipnorm: Optional[float] = None
    gamma: float = 0.99
    target_output_std: float = 0.089
    packnet_fake_num_tasks: Optional[int] = None
    agent_policy_exploration: str2bool = False
    critic_reg_coef: float = 1.0


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
    from sequoia.settings.rl import IncrementalRLSetting
    setting = IncrementalRLSetting(dataset="CW10")
    train_env_sequoia = setting.train_dataloader()
    
    # TODO: Add a wrapper so that both can be exactly the same!
    assert False, (train_env.observation_space, train_env_sequoia.observation_space)

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
    steps = task_config.steps_per_task * len(task_config.tasks)

    num_heads = num_tasks if algo_config.multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=task_config.hidden_sizes,
        activation=get_activation_from_str(algo_config.activation),
        use_layer_norm=algo_config.use_layer_norm,
        num_heads=num_heads,
        hide_task_id=algo_config.hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=task_config.hidden_sizes,
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
        test_envs,
        logger,
        steps=steps,
        actor_cl=actor_cl,
        actor_kwargs=actor_kwargs,
        critic_cl=critic_cl,
        critic_kwargs=critic_kwargs,
        seed=task_config.seed,
        replay_size=task_config.replay_size,
        batch_size=task_config.batch_size,
        buffer_type=task_config.buffer_type,
        # algo_config=algo_config,
        # )
        reset_buffer_on_task_change=algo_config.reset_buffer_on_task_change,
        reset_optimizer_on_task_change=algo_config.reset_optimizer_on_task_change,
        lr=algo_config.lr,
        alpha=algo_config.alpha,
        cl_method=algo_config.cl_method,
        cl_reg_coef=algo_config.cl_reg_coef,
        packnet_retrain_steps=algo_config.packnet_retrain_steps,
        regularize_critic=algo_config.regularize_critic,
        vcl_first_task_kl=algo_config.vcl_first_task_kl,
        episodic_mem_per_task=algo_config.episodic_mem_per_task,
        episodic_batch_size=algo_config.episodic_batch_size,
        reset_critic_on_task_change=algo_config.reset_critic_on_task_change,
        clipnorm=algo_config.clipnorm,
        gamma=algo_config.gamma,
        target_output_std=algo_config.target_output_std,
        packnet_fake_num_tasks=algo_config.packnet_fake_num_tasks,
        agent_policy_exploration=algo_config.agent_policy_exploration,
        critic_reg_coef=algo_config.critic_reg_coef,
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
    del task_config.logger_output  # TODO: Why though?
    main(logger, task_config=task_config, algo_config=algo_config)
