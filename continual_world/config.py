from dataclasses import dataclass
from simple_parsing.helpers import field, list_field, choice
from continual_world.utils.utils import get_activation_from_str, sci2int, str2bool
from typing import List, Optional

from simple_parsing.helpers.serialization.serializable import Serializable
from continual_world.task_lists import task_seq_to_task_list

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


@dataclass
class TaskConfig(Serializable):
    """ Configuration options for the task(s). """

    # Name of the sequence you want to run
    tasks: str = choice(*list(task_seq_to_task_list.keys()))  # type: ignore

    # Random seed used for running the experiments
    seed: int = 123
    # Numer of steps per task
    steps_per_task: int = field(default=int(1e6), type=sci2int)

    # List of tasks you want to run, by name or by the MetaWorld index
    task_list: Optional[List[str]] = None
    max_ep_len: int=200

    num_test_eps_stochastic: int = 10
    num_test_eps_deterministic: int = 1



# TODO: This AlgoConfig is more like a CLAlgoConfig, since `run_single` doesnt have these params.


@dataclass
class AlgoConfig(Serializable):
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
    agent_policy_exploration: bool = False
    critic_reg_coef: float = 1.0

    polyak: float = 0.995
    """
    Interpolation factor in polyak averaging for target networks. Target networks are updated
    towards main networks according to:
    
    ```
    theta_target = rho * theta_target + (1-rho) * theta
    ```
    where `rho` is polyak. (Always between 0 and 1, usually close to 1.)
    """

    #    .. math:: \\theta_{\\text{targ}} \\leftarrow
        # \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
    
    # Hidden layers sizes in the base network
    hidden_sizes: List[int] = list_field(256, 256, 256, 256)

    # Size of the replay buffer
    replay_size: int = field(default=int(1e6), type=sci2int)
    
    # Strategy of inserting examples into the buffer
    buffer_type: str = choice("fifo", "reservoir", default="fifo")  # type: ignore
    
    # Number of samples in each mini-batch sampled by SAC
    batch_size: int = 128

    # Number of steps for uniform-random action selection, before running real policy. Helps
    # exploration.
    start_steps: int = 10_000
    
    # Number of env interactions to collect before starting to do gradient descent updates. Ensures
    # replay buffer is full enough for useful updates.
    update_after: int = 1_000
    
    # Number of env interactions that should elapse between gradient descent updates.
    # Note: Regardless of how long you wait between updates, the ratio of env steps to gradient
    # steps is locked to 1.
    update_every: int = 50
    
    # How often (in terms of gap between epochs) to save the current policy and value function.
    save_freq_epochs: int = 100
