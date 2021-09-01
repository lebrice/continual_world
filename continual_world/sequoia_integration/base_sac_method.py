import itertools
import logging
import math
import os
import random
import time
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import gym
import numpy as np
import tensorflow as tf
from continual_world.config import TaskConfig
from continual_world.methods.vcl import VclHelper, VclMlpActor
from continual_world.sequoia_integration.wrappers import (
    SequoiaToCWWrapper,
    concat_x_and_t,
    wrap_sequoia_env,
)
from continual_world.spinup.models import Actor, MlpActor, MlpCritic, PopArtMlpCritic
from continual_world.spinup.replay_buffers import (
    EpisodicMemory,
    ReplayBuffer,
    ReservoirReplayBuffer,
)
from continual_world.spinup.utils.logx import EpochLogger
from continual_world.task_lists import task_seq_to_task_list
from continual_world.utils.utils import (
    get_activation_from_str,
    reset_optimizer,
    reset_weights,
    sci2int,
    str2bool,
)
from gym import spaces
from sequoia.common.spaces.typed_dict import TypedDictSpace
from sequoia.methods import Method
from sequoia.settings.base.setting import SettingType
from sequoia.settings.rl import DiscreteTaskAgnosticRLSetting
from sequoia.settings.rl.incremental.setting import IncrementalRLSetting
from sequoia.settings.rl.setting import RLSetting
from simple_parsing import ArgumentParser
from simple_parsing.helpers import choice, field, list_field
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from simple_parsing.helpers.serialization.serializable import Serializable
from tensorflow.python.types.core import Value
from tqdm import tqdm

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# weights_reg_methods = ["l2", "ewc", "mas"]
# exp_replay_methods = ["agem"]
logger = logging.getLogger(__name__)


class BatchDict(TypedDict):
    obs1: tf.Tensor
    obs2: tf.Tensor
    acts: tf.Tensor
    rews: tf.Tensor
    done: bool


class GradientsTuple(NamedTuple):
    actor_gradients: List[tf.Tensor]
    critic_gradients: List[tf.Tensor]
    alpha_gradient: List[tf.Tensor]


# TODO: switch this to the `DiscreteTaskAgnosticRLSetting` if/when we add support for passing the
# envs of each task to that setting.


class SACMethod(Method, target_setting=IncrementalRLSetting):  # type: ignore
    family: ClassVar[str] = "continual_world"

    @dataclass
    class Config(HyperParameters):
        """ Configuration options for the Algorithm. """

        # Which CL method to use.
        # NOTE: This has been converted into what is essentially an 'Abstract' property that is
        # fixed by the different subclasses.
        # cl_method: Optional[str] = choice(  # type: ignore
        #     None, "l2", "ewc", "mas", "vcl", "packnet", "agem", default=None
        # )

        # Wether to clear the replay buffer when a task boundary is reached during training.
        reset_buffer_on_task_change: bool = True
        # Wether to reset the optimzier when a task boundary is reached during training.
        reset_optimizer_on_task_change: bool = True
        reset_critic_on_task_change: bool = False
        activation: str = "lrelu"
        use_layer_norm: bool = True
        # scale_reward: bool = False
        # div_by_return: bool = False
        lr: float = 1e-3
        alpha: str = "auto"
        use_popart: bool = False
        regularize_critic: bool = False
        cl_reg_coef: float = 0.0
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

    def __init__(self, algo_config: "SACMethod.Config" = None):
        super().__init__()
        self.algo_config = algo_config or self.Config()
        # NOTE: These attributes are set in self.configure. This is just here for type hints.
        self.task_config: TaskConfig

        self.actor_cl: Type[Actor] = MlpActor
        self.critic_cl: Type[MlpCritic] = MlpCritic

        self.critic_variables: List[tf.Variable]
        self.all_variables: List[tf.Variable]
        self.all_common_variables: List[tf.Variable]

        self.actor: Actor
        self.critic1: Union[MlpCritic, PopArtMlpCritic]
        self.critic2: Union[MlpCritic, PopArtMlpCritic]
        self.target_critic1: Union[MlpCritic, PopArtMlpCritic]
        self.target_critic2: Union[MlpCritic, PopArtMlpCritic]

        self.optimizer: tf.keras.optimizers.Optimizer

        # The arguments to be pased to `self.actor_cl` (the actor class constructor).
        self.actor_kwargs: Dict = {}
        self.critic_kwargs: Dict = {}
        self.current_task_idx: int = -1
        self.logger: EpochLogger
        self.replay_buffer: Union[ReplayBuffer, ReservoirReplayBuffer]

        self.target_entropy: np.ndarray
        self.all_log_alpha: tf.Variable
        self.auto_alpha: bool = False

        self.add_task_ids: bool = False

        # The number of tasks that the train and validation environments actually contain.
        self.nb_tasks_in_train_env: int = 1
        self.nb_tasks_in_valid_env: int = 1

        # Wether the tasks follow some non-stationary distribution, or if they are uniformly sampled
        self.stationary_context: bool = False

    def configure(self, setting: DiscreteTaskAgnosticRLSetting) -> None:
        """ Configure the Method before training begins on the envs of the Setting. """
        # IDEA: Copy over the info from the setting into the `TaskConfig` class which contains the
        # values that would have been used by the `sac` function.
        # TODO: Use the values of `run_mt` when `setting.stationary_context` is True.
        # NOTE: Using the arguments from `cl_example.sh` for now.
        # --seed 0 \
        # --steps_per_task 2e3 \
        # --tasks DOUBLE_PMO1 \
        # --cl_method ewc \
        # --cl_reg_coef 1e4 \
        # --logger_output tsv tensorboard

        self.task_config = TaskConfig(
            seed=0,
            tasks="DOUBLE_PMO1",
            logger_output=["tsv", "tensorboard"],
            steps_per_task=setting.train_steps_per_task,
        )

        self.num_tasks = setting.nb_tasks
        self.stationary_context = setting.stationary_context
        self.nb_tasks_in_train_env = (
            setting.nb_tasks if setting.stationary_context else 1
        )
        self.nb_tasks_in_valid_env = (
            setting.nb_tasks if setting.stationary_context else 1
        )

        self.current_task_idx = -1

        self.logger = EpochLogger(
            self.task_config.logger_output,
            config=dict(**asdict(self.task_config), **asdict(self.algo_config)),
        )
        # Wether or not we add the task IDS to the observations.
        self.add_task_ids = (
            setting.task_labels_at_train_time and setting.task_labels_at_test_time
        )
        self.obs_dim = np.prod(setting.observation_space.x.shape)
        if setting.task_labels_at_train_time and setting.task_labels_at_test_time:
            # The task ids will be concatenated to the observations.
            # TODO: Actually check that the algos can work without the task ids.
            self.obs_dim += setting.nb_tasks

        if isinstance(setting.action_space, spaces.Dict):
            action_space = setting.action_space["y_pred"]
        else:
            action_space = setting.action_space
        if not isinstance(action_space, spaces.Box):
            raise NotImplementedError(
                f"This Method expects a continuous action space (an instance of `gym.spaces.Box`)"
            )
        self.act_dim = np.prod(action_space.shape)
        # This implementation assumes all dimensions share the same bound!
        assert np.all(action_space.high == action_space.high[0])

        random.seed(self.task_config.seed)
        tf.random.set_seed(self.task_config.seed)
        np.random.seed(self.task_config.seed)

        if self.algo_config.use_popart:
            assert (
                self.algo_config.multihead_archs
            ), "PopArt works only in the multi-head setup"
            self.critic_cl = PopArtMlpCritic
        else:
            self.critic_cl = MlpCritic

        self.replay_buffer = self.create_replay_buffer()

        self.actor_kwargs = self.get_actor_kwargs(setting)
        self.critic_kwargs = self.get_critic_kwargs(setting)

        self.create_networks()

        # For reference on automatic alpha tuning, see
        # "Automating Entropy Adjustment for Maximum Entropy" section
        # in https://arxiv.org/abs/1812.05905
        self.auto_alpha = False
        if self.algo_config.alpha == "auto":
            self.auto_alpha = True
            self.all_log_alpha = tf.Variable(
                np.ones((self.num_tasks, 1), dtype=np.float32), trainable=True
            )
            if self.algo_config.target_output_std is None:
                self.target_entropy = -np.prod(setting.action_space.shape).astype(
                    np.float32
                )
            else:
                target_1d_entropy = np.log(
                    self.algo_config.target_output_std * math.sqrt(2 * math.pi * math.e)
                )
                self.target_entropy = self.act_dim * target_1d_entropy

        # TODO: Split these off into different methods based on the value of `cl_method`.
        # cl_method = self.algo_config.cl_method
        # Setup CL methods: Moved to the subclasses.
        # if cl_method == "packnet":
        #     packnet_models = [self.actor]
        #     if self.algo_config.regularize_critic:
        #         packnet_models.extend([self.critic1, self.critic2])
        #     self.packnet_helper = PackNetHelper(packnet_models)
        # elif cl_method in weights_reg_methods:
        #     self.old_params = list(
        #         tf.Variable(tf.identity(param), trainable=False)
        #         for param in self.all_common_variables
        #     )
        #     self.reg_helper = self.get_reg_helper()
        # if cl_method == "vcl":
        #     self.vcl_helper = VclHelper(
        #         self.actor, self.critic1, self.critic2, self.algo_config.regularize_critic
        #     )
        # if cl_method == "agem":
        #     episodic_mem_size = self.algo_config.episodic_mem_per_task * self.num_tasks
        #     self.episodic_memory = EpisodicMemory(
        #         obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
        #     )
        #     self.agem_helper = AgemHelper()

    def fit(self, train_env: gym.Env, valid_env: gym.Env,) -> None:
        """Train and validate using the environments created by the Setting.

        Parameters
        ----------
        train_env : gym.Env
            Training environment.
        valid_env : gym.Env
            Validation environment.
        """
        steps = train_env.max_steps
        env: SequoiaToCWWrapper = wrap_sequoia_env(
            train_env,
            nb_tasks_in_env=self.nb_tasks_in_train_env,
            add_task_ids=self.add_task_ids,
            is_multitask=self.stationary_context,
        )
        test_envs: List[SequoiaToCWWrapper] = [
            wrap_sequoia_env(
                valid_env,
                nb_tasks_in_env=self.nb_tasks_in_valid_env,
                add_task_ids=self.add_task_ids,
                is_multitask=self.stationary_context,
            )
        ]

        env.seed(self.task_config.seed)
        env.action_space.seed(self.task_config.seed)
        start_time = time.time()
        obs, ep_ret, ep_len = env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        # learn_on_batch = self.get_learn_on_batch()
        current_task_t = 0
        reg_weights = None

        for t in tqdm(range(steps), desc="Training"):
            # NOTE: Moving this 'on task change' to `on_task_switch`.
            # TODO: This might actually make some sense, if we consider multi-task and traditional
            # envs, which will change tasks always. However, the task distribution is nonstationary
            # at train time, so we'd be better off just considering all tasks as being the same, no?

            # NOTE: Not doing this for now, because we don't want to call `on_task_switch` when in a
            # stationary context, and we also expect to have `on_task_switch` be called by the
            # Setting anyways if the task ids are available.
            # task_id = getattr(env, "cur_seq_idx", -1)
            # if self.current_task_idx != task_id:
            #     self.on_task_switch(task_id)

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if current_task_t > self.algo_config.start_steps or (
                self.algo_config.agent_policy_exploration and self.current_task_idx > 0
            ):
                action = self.get_action(tf.convert_to_tensor(obs))
            else:
                action = env.action_space.sample()

            # Step the env
            next_obs, reward, done, info = env.step(action)
            ep_ret += reward
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done_to_store = done
            if ep_len == self.task_config.max_ep_len or info.get("TimeLimit.truncated"):
                done_to_store = False

            # Store experience to replay buffer
            self.replay_buffer.store(obs, action, reward, next_obs, done_to_store)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = next_obs

            # End of trajectory handling
            if done or (ep_len == self.task_config.max_ep_len):
                self.logger.store({"train/return": ep_ret, "train/ep_length": ep_len})
                ep_ret, ep_len = 0, 0
                if t < steps - 1:
                    obs = env.reset()

            # Update handling
            if (
                current_task_t >= self.algo_config.update_after
                and current_task_t % self.algo_config.update_every == 0
            ):
                # NOTE: @lebrice: Interesting: Perform one update per step, even when `update_every`
                # is an interval, i.e. "perform 50 updates every 50 steps".

                for j in range(self.algo_config.update_every):
                    batch, episodic_batch = self.sample_batches()
                    results = self.learn_on_batch(
                        tf.convert_to_tensor(self.current_task_idx),
                        batch,
                        episodic_batch,
                    )
                    self.logger.store(
                        {
                            "train/q1_vals": results["q1"],
                            "train/q2_vals": results["q2"],
                            "train/log_pi": results["logp_pi"],
                            "train/loss_pi": results["pi_loss"],
                            "train/loss_q1": results["q1_loss"],
                            "train/loss_q2": results["q2_loss"],
                            "train/loss_reg": results["reg_loss"],
                            "train/agem_violation": results["agem_violation"],
                        }
                    )

                    for i in range(self.num_tasks):
                        if self.auto_alpha:
                            self.logger.store(
                                {
                                    f"train/alpha/{i}": float(
                                        tf.math.exp(self.all_log_alpha[i][0])
                                    )
                                }
                            )
                        # if self.critic_cl is PopArtMlpCritic: # (NOTE: using isinstance instead)
                        if isinstance(self.critic1, PopArtMlpCritic):
                            self.logger.store(
                                {
                                    f"train/popart_mean/{i}": self.critic1.moment1[i][
                                        0
                                    ],
                                    f"train/popart_std/{i}": self.critic1.sigma[i][0],
                                }
                            )

            # End of epoch wrap-up
            if ((t + 1) % self.task_config.log_every == 0) or (t + 1 == steps):
                epoch = (
                    t + 1 + self.task_config.log_every - 1
                ) // self.task_config.log_every

                # Save model
                if (epoch % self.algo_config.save_freq_epochs == 0) or (t + 1 == steps):
                    dir_prefixes = []
                    if self.current_task_idx == -1:
                        dir_prefixes.append("./checkpoints")
                    else:
                        dir_prefixes.append(
                            "./checkpoints/task{}".format(self.current_task_idx)
                        )
                        if self.current_task_idx == self.num_tasks - 1:
                            dir_prefixes.append("./checkpoints")

                    for prefix in dir_prefixes:
                        self.actor.save_weights(os.path.join(prefix, "actor"))
                        self.critic1.save_weights(os.path.join(prefix, "critic1"))
                        self.target_critic1.save_weights(
                            os.path.join(prefix, "target_critic1")
                        )
                        self.critic2.save_weights(os.path.join(prefix, "critic2"))
                        self.target_critic2.save_weights(
                            os.path.join(prefix, "target_critic2")
                        )

                # Test the performance of the deterministic version of the agent.
                self.test_agent(test_envs)

                # Log info about epoch
                self.logger.log_tabular("epoch", epoch)
                self.logger.log_tabular("train/return", with_min_and_max=True)
                self.logger.log_tabular("train/ep_length", average_only=True)
                self.logger.log_tabular("total_env_steps", t + 1)
                self.logger.log_tabular("current_task_steps", current_task_t + 1)
                self.logger.log_tabular("train/q1_vals", with_min_and_max=True)
                self.logger.log_tabular("train/q2_vals", with_min_and_max=True)
                self.logger.log_tabular("train/log_pi", with_min_and_max=True)
                self.logger.log_tabular("train/loss_pi", average_only=True)
                self.logger.log_tabular("train/loss_q1", average_only=True)
                self.logger.log_tabular("train/loss_q2", average_only=True)
                for i in range(self.num_tasks):
                    if self.auto_alpha:
                        self.logger.log_tabular(
                            "train/alpha/{}".format(i), average_only=True
                        )
                    # if self.critic_cl is PopArtMlpCritic:
                    if isinstance(self.critic1, PopArtMlpCritic):
                        self.logger.log_tabular(
                            "train/popart_mean/{}".format(i), average_only=True
                        )
                        self.logger.log_tabular(
                            "train/popart_std/{}".format(i), average_only=True
                        )
                self.logger.log_tabular("train/loss_reg", average_only=True)
                self.logger.log_tabular("train/agem_violation", average_only=True)

                # TODO: We assume here that SuccessCounter is outermost wrapper.
                avg_success = np.mean(env.pop_successes())
                self.logger.log_tabular("train/success", avg_success)
                if "seq_idx" in info:
                    self.logger.log_tabular("train/active_env", info["seq_idx"])

                self.logger.log_tabular("walltime", time.time() - start_time)
                self.logger.dump_tabular()

            current_task_t += 1

    def sample_batches(self) -> Tuple[BatchDict, Optional[BatchDict]]:
        batch = self.replay_buffer.sample_batch(self.algo_config.batch_size)
        episodic_batch = None
        return batch, episodic_batch

    def create_replay_buffer(self) -> ReplayBuffer:
        # Create experience buffer
        if self.algo_config.buffer_type == "fifo":
            return ReplayBuffer(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                size=self.algo_config.replay_size,
            )
        elif self.algo_config.buffer_type == "reservoir":
            return ReservoirReplayBuffer(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                size=self.algo_config.replay_size,
            )
        else:
            raise NotImplementedError(self.algo_config.buffer_type)

    def get_actor_kwargs(self, setting: RLSetting) -> Dict:
        num_heads = self.num_tasks if self.algo_config.multihead_archs else 1
        actor_kwargs = dict(
            hidden_sizes=self.algo_config.hidden_sizes,
            activation=get_activation_from_str(self.algo_config.activation),
            use_layer_norm=self.algo_config.use_layer_norm,
            num_heads=num_heads,
            hide_task_id=self.algo_config.hide_task_id,
        )
        # Share information about action space with policy architecture
        actor_kwargs["action_space"] = setting.action_space
        actor_kwargs["input_dim"] = self.obs_dim

        # if self.algo_config.cl_method == "vcl":
        #     self.actor_kwargs["variational_ln"] = self.algo_config.vcl_variational_ln
        return actor_kwargs

    def get_critic_kwargs(self, setting: RLSetting) -> Dict:
        num_heads = self.num_tasks if self.algo_config.multihead_archs else 1
        critic_kwargs = dict(
            hidden_sizes=self.algo_config.hidden_sizes,
            activation=get_activation_from_str(self.algo_config.activation),
            use_layer_norm=self.algo_config.use_layer_norm,
            num_heads=num_heads,
            hide_task_id=self.algo_config.hide_task_id,
        )
        critic_kwargs["input_dim"] = self.obs_dim + self.act_dim
        return critic_kwargs

    def create_networks(self) -> None:
        # Create actor and critic networks

        self.actor = self.actor_cl(**self.actor_kwargs)

        self.critic1 = self.critic_cl(**self.critic_kwargs)
        self.target_critic1 = self.critic_cl(**self.critic_kwargs)
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.critic2 = self.critic_cl(**self.critic_kwargs)
        self.target_critic2 = self.critic_cl(**self.critic_kwargs)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.critic_variables = (
            self.critic1.trainable_variables + self.critic2.trainable_variables
        )
        self.all_variables = self.actor.trainable_variables + self.critic_variables
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.algo_config.lr)

        # FIXME: Remove this, debugging.
        self._last_task_ids = None

    def get_actions(
        self,
        observations: DiscreteTaskAgnosticRLSetting.Observations,
        action_space: gym.Space,
    ) -> Union[DiscreteTaskAgnosticRLSetting.Actions, Any]:
        # BUG: In TraditionalRLSetting, we get some weird task ids during test time, makes no sense.
        if observations.task_labels != self._last_task_ids:
            print(f"New task ids: {observations.task_labels} (obs = {observations})")
        if observations.done:
            print(f"Obs has done=True: {observations}")
        self._last_task_ids = observations.task_labels

        obs = concat_x_and_t(observations, nb_tasks=self.num_tasks)
        action: tf.Tensor = self.get_action(tf.convert_to_tensor(obs))
        action_np = action.numpy()
        if isinstance(action_space, TypedDictSpace):
            return action_space.dtype(**action_np)
        return action_np

    @tf.function
    def learn_on_batch(
        self, seq_idx: int, batch: BatchDict, episodic_batch: BatchDict = None
    ) -> Dict:
        gradients, metrics = self.get_gradients(
            seq_idx, **batch, episodic_batch=episodic_batch
        )

        if self.algo_config.clipnorm is not None:
            actor_gradients, critic_gradients, alpha_gradient = gradients
            gradients = GradientsTuple(
                tf.clip_by_global_norm(actor_gradients, self.algo_config.clipnorm)[0],
                tf.clip_by_global_norm(critic_gradients, self.algo_config.clipnorm)[0],
                tf.clip_by_norm(alpha_gradient, self.algo_config.clipnorm),
            )

        self.apply_update(*gradients)
        return metrics

    def get_learn_on_batch(self):
        # TODO: Not sure why this was wrapped in a function
        return self.learn_on_batch
        # return learn_on_batch

    @tf.function
    def get_log_alpha(self, obs1):
        return tf.squeeze(
            tf.linalg.matmul(obs1[:, -self.num_tasks :], self.all_log_alpha)
        )

    @tf.function
    def get_action(self, o, deterministic=tf.constant(False)) -> tf.Tensor:
        # if self.algo_config.cl_method == "vcl":
        #     # Disable multiple samples in VCL for faster evaluation
        #     action_fn = self.get_action  # = vcl_get_stable_action
        # else:
        #     action_fn = self.get_action
        mu, log_std, pi, logp_pi = self.actor(tf.expand_dims(o, 0))
        if deterministic:
            return mu[0]
        else:
            return pi[0]

    @tf.function
    def vcl_get_stable_action(self, o, deterministic=tf.constant(False)) -> tf.Tensor:
        mu, log_std, pi, logp_pi = self.actor(tf.expand_dims(o, 0), samples_num=10)
        if deterministic:
            return mu[0]
        else:
            return pi[0]

    def get_gradients(
        self,
        seq_idx: int,
        obs1: tf.Tensor,
        obs2: tf.Tensor,
        acts: tf.Tensor,
        rews: tf.Tensor,
        done: bool,
        episodic_batch: BatchDict = None,
    ) -> Tuple[GradientsTuple, Dict]:
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(obs1)
            else:
                log_alpha = tf.math.log(self.algo_config.alpha)

            # Main outputs from computation graph
            mu, log_std, pi, logp_pi = self.actor(obs1)
            q1 = self.critic1(obs1, acts)
            q2 = self.critic2(obs1, acts)

            # compose q with pi, for pi-learning
            q1_pi = self.critic1(obs1, pi)
            q2_pi = self.critic2(obs1, pi)

            # get actions and log probs of actions for next states, for Q-learning
            _, _, pi_next, logp_pi_next = self.actor(obs2)

            # target q values, using actions from *current* policy
            target_q1 = self.target_critic1(obs2, pi_next)
            target_q2 = self.target_critic2(obs2, pi_next)

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            if self.critic_cl is PopArtMlpCritic:
                q_backup = tf.stop_gradient(
                    self.critic1.normalize(
                        rews
                        + self.algo_config.gamma
                        * (1 - done)
                        * (
                            self.critic1.unnormalize(min_target_q, obs2)
                            - tf.math.exp(log_alpha) * logp_pi_next
                        ),
                        obs1,
                    )
                )
            else:
                q_backup = tf.stop_gradient(
                    rews
                    + self.algo_config.gamma
                    * (1 - done)
                    * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
                )

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi - min_q_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = q1_loss + q2_loss

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(logp_pi + self.target_entropy)
                )

            aux_pi_loss, aux_value_loss = self.get_auxiliary_loss(seq_idx)
            reg_loss = aux_pi_loss + aux_value_loss
            pi_loss += aux_pi_loss
            value_loss += aux_value_loss

        # Compute gradients
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        if self.critic_cl is PopArtMlpCritic:
            # Stats are shared between critic1 and critic2.
            # We keep them only in critic1.
            self.critic1.update_stats(q_backup, obs1)

        gradients = GradientsTuple(
            actor_gradients=actor_gradients,
            critic_gradients=critic_gradients,
            alpha_gradient=alpha_gradient,
        )
        metrics = dict(
            pi_loss=pi_loss,
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q1=q1,
            q2=q2,
            logp_pi=logp_pi,
            reg_loss=reg_loss,
            agem_violation=0,
        )
        return gradients, metrics

    def get_auxiliary_loss(self, seq_idx: int) -> Tuple[tf.Tensor, tf.Tensor]:
        aux_pi_loss = tf.zeros([1])
        aux_value_loss = tf.zeros([1])
        return aux_pi_loss, aux_value_loss

    def apply_update(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
    ):
        self.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_variables))

        if self.auto_alpha:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])

        # Polyak averaging for target variables
        for v, target_v in zip(
            self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(
                self.algo_config.polyak * target_v + (1 - self.algo_config.polyak) * v
            )
        for v, target_v in zip(
            self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(
                self.algo_config.polyak * target_v + (1 - self.algo_config.polyak) * v
            )

    def test_agent(self, test_envs: List[gym.Env]):
        # TODO: parallelize test phase if we hit significant added walltime.
        # TODO: Adapt this a bit.

        # Dict that for each mode (deterministic or not) stores wether an episode was a success or
        # not.
        successes_per_mode: Dict[bool, List[bool]] = defaultdict(list)
        for deterministic, max_episodes in [
            (False, self.task_config.num_test_eps_stochastic),
            (True, self.task_config.num_test_eps_deterministic),
            (True, None),
        ]:
            for test_env in test_envs:
                env_successes, metrics = self.test_agent_on_env(
                    test_env, deterministic=deterministic, max_episodes=max_episodes
                )
                successes_per_mode[deterministic].extend(env_successes)
                # TODO: Log the episode metrics (metrics).
                # self.logger.log_tabular(f"{prefix}/return", with_min_and_max=True)
                # self.logger.log_tabular(f"{prefix}/ep_length", average_only=True)

        for deterministic, successes in successes_per_mode.items():
            mode = "deterministic" if deterministic else "stochastic"
            self.logger.log_tabular(f"test/{mode}/average_success", np.mean(successes))

    def on_test_episode_start(self, seq_idx: int, episode: int) -> None:
        pass

    def on_test_loop_end(self) -> None:
        pass

    def test_agent_on_env(
        self,
        test_env: gym.Env,
        deterministic: bool,
        max_episodes: int = None,
        key_prefix: str = None,
    ) -> Tuple[List[bool], Dict[int, Dict[str, List[float]]]]:
        mode = "deterministic" if deterministic else "stochastic"
        # NOTE: Changed this next line, since we may be calling this with a single env for task 4
        # for example.
        metrics: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        total_steps = 0
        for episode in range(max_episodes) if max_episodes else itertools.count():
            # TODO: The number of test episodes is too large actually! Need to check instead
            # the `max_steps` property.
            truncated = False

            if total_steps >= test_env.max_steps or test_env.is_closed():
                truncated = True
                break

            obs, done, ep_ret, ep_len = test_env.reset(), False, 0, 0
            total_steps += 1

            # Update the seq_idx at the beginning of each episode, since we might be switching tasks
            # within one env in Sequoia.
            seq_idx = test_env.cur_seq_idx
            if seq_idx is None:
                seq_idx = -1
            # prefix = key_prefix or f"test/{mode}/{seq_idx}/{test_env.name}"

            self.on_test_episode_start(seq_idx=seq_idx, episode=episode)

            while not (done or (ep_len == self.task_config.max_ep_len)):
                if test_env.is_closed() or total_steps >= test_env.max_steps:
                    truncated = True
                    break

                assert obs is not None
                obs, reward, done, _ = test_env.step(
                    self.get_action(
                        tf.convert_to_tensor(obs), tf.constant(deterministic)
                    )
                )
                total_steps += 1
                ep_ret += reward
                ep_len += 1

            if truncated:
                print(f"Not logging episode {episode} since it was truncated.")
            else:
                # In this case, we only log stuff when it's the result of a 'full' test
                # episode.
                # self.logger.store(
                #     {f"{prefix}/return": ep_ret, f"{prefix}/ep_length": ep_len}
                # )
                metrics[seq_idx]["return"].append(ep_ret)
                metrics[seq_idx]["ep_length"].append(ep_len)

        self.on_test_loop_end()

        env_success = test_env.pop_successes()
        return env_success, metrics

    def on_task_switch(self, task_id: Optional[int]):
        """Called by the Setting when reaching a task boundary.

        When task labels are available in the setting, we also receive the index of the new task.
        Otherwise, `task_id` will be None. 
        
        NOTE: We can also check if this task boundary is happening during training or testing using
        `self.training`.

        Parameters
        ----------
        task_id : Optional[int]
            [description]
        """
        logger.info(
            f"on_task_switch called with task_id = {task_id} (training={self.training})"
        )
        if self.current_task_idx == task_id:
            logger.info(
                f"Ignoring call to `on_task_switch` since task_id ({task_id}) is the same as "
                f"before ({self.current_task_idx}). (self.training={self.training})."
            )
            return

        if task_id is None:
            task_id = -1

        if self.training:
            # Reset the number of training steps taken in the current task.
            self.current_task_t = 0

        self.current_task_idx = task_id

        # TODO: Can get a bit complicated to try and give an `old_task` and a `new_task`, because
        # there are boundaries between test/train in TaskIncrementalRL for example.
        self.handle_task_boundary(task_id=self.current_task_idx, training=self.training)

        if not self.training:
            # Don't do the rest if self.training is False.
            logger.info(
                "Not resetting the models/buffers/optimizers since task boundary is reached during "
                "testing."
            )
            return

        if self.algo_config.reset_buffer_on_task_change:
            assert self.algo_config.buffer_type == "fifo"
            self.replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                size=self.algo_config.replay_size,
            )

        if self.algo_config.reset_critic_on_task_change:
            reset_weights(self.critic1, self.critic_cl, self.critic_kwargs)
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(self.critic2, self.critic_cl, self.critic_kwargs)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if self.algo_config.reset_optimizer_on_task_change:
            reset_optimizer(self.optimizer)

        # Update variables list and update function in case model changed.
        # E.g: For VCL after the first task we set trainable=False for layer
        # normalization. We need to recompute the graph in order for TensorFlow
        # to notice this change.
        self.learn_on_batch = self.get_learn_on_batch()
        self.all_variables = self.actor.trainable_variables + self.critic_variables
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )

    def handle_task_boundary(self, task_id: int, training: bool) -> None:
        pass

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser, dest: str = "") -> None:
        # TODO: Use the values from `defaults.py` depending on what kind of Setting we need to set
        # arguments for. (Also need to detect the setting type on which this is going to be used!)
        prefix = f"{dest}." if dest else ""
        parser.add_arguments(cls.Config, f"{prefix}algo_config")

    @classmethod
    def from_argparse_args(cls, args: Namespace, dest: str = ""):
        args = getattr(args, dest) if dest else args
        algo_config: SACMethod.Config = getattr(args, "algo_config")
        return cls(algo_config=algo_config)

    @classmethod
    def is_applicable(cls, setting: Union[SettingType, Type[SettingType]]) -> bool:
        """Returns wether this Method is applicable to the given setting.
        """
        # NOTE: Normally we'd just rely on `isinstance(target_setting, cls.target_setting)`, but in
        # this case we also want to avoid settings with don't have a continuous action space.
        if isinstance(setting, RLSetting):
            action_space: gym.Space = setting.action_space
            if isinstance(action_space, spaces.Dict):
                action_space = action_space["y_pred"]
            # Need continuous action spaces in this method for now.
            return isinstance(action_space, spaces.Box)
        return super().is_applicable(setting)


def main():
    from sequoia.common.config import Config
    from sequoia.settings.rl import IncrementalRLSetting, TraditionalRLSetting

    algo_config = SACMethod.Config(start_steps=50)
    method = SACMethod(algo_config=algo_config)
    setting = TraditionalRLSetting(
        dataset="MountainCarContinuous-v0",
        nb_tasks=1,
        train_max_steps=5_000,
        train_steps_per_task=5_000,
        add_done_to_observations=True,
    )

    results: TraditionalRLSetting.Results = setting.apply(
        method, config=Config(debug=True)
    )
    # config = Config(debug=True)
    # setting = IncrementalRLSetting(
    #     dataset="CW20",
    #     train_steps_per_task=2_000,
    #     train_max_steps=20 * 2_000,
    #     test_steps_per_task=2_000,
    #     test_max_steps=20 * 2_000,
    #     nb_tasks=20,
    # )
    # method = SACMethod()
    # results = setting.apply(method, config=config)
    print(results.summary())


if __name__ == "__main__":
    main()
