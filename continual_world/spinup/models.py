from typing import Callable, NamedTuple, Tuple
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from continual_world.envs import MW_ACT_LEN, MW_OBS_LEN
import warnings

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(input_tensor=2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(input_dim, hidden_sizes, activation, use_layer_norm=False):
    model = tf.keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(hidden_sizes[0]))
    if use_layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(tf.nn.tanh))
    else:
        model.add(tf.keras.layers.Activation(activation))
    for size in hidden_sizes[1:]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    return model


def _choose_head(out: tf.Tensor, obs: tf.Tensor, num_heads: int):
    batch_size = tf.shape(out)[0]
    out = tf.reshape(out, [batch_size, -1, num_heads])

    # NOTE: This next line assumed that the obs has task labels, but that isn't always the case!
    # obs = tf.reshape(obs[:, -num_heads:], [batch_size, num_heads, 1])

    output_head_coefficients: tf.Tensor

    # NOTE: AutoGraph isn't able to convert this check.
    if obs.shape[1] > num_heads:
        # If the shape[1] is greater than num_heads, that means the task labels are present.
        task_labels_onehot = obs[:, -num_heads:]
        output_head_coefficients = tf.reshape(task_labels_onehot, [batch_size, num_heads, 1])
    else:
        # raise RuntimeError(f"Can't choose heads without having access to the task labels!")
        # TODO: IDEA: (@lebrice) How about we use all the heads then, with a coefficient per head?
        warnings.warn(RuntimeWarning("Using all the output heads since we don't have task labels."))
        output_head_coefficients = tf.ones([batch_size, num_heads, 1]) / num_heads
    return tf.squeeze(out @ output_head_coefficients, axis=2)


from abc import ABC, abstractmethod
from gym.spaces import Space, Box


class ActorOutput(NamedTuple):
    mu: tf.Tensor
    log_std: tf.Tensor
    pi: tf.Tensor
    logp_pi: tf.Tensor


class Actor(Model, ABC):
    def __init__(
        self,
        input_dim: int,
        action_space: Box,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.num_heads = num_heads
        self.hide_task_id = hide_task_id

        if self.hide_task_id:
            # NOTE: This is specific to meta-world and is never used with Sequoia.
            self.input_dim = MW_OBS_LEN

    @abstractmethod
    def call(self, x: tf.Tensor) -> ActorOutput:
        pass


class MlpActor(Actor):
    def __init__(
        self,
        input_dim,
        action_space,
        hidden_sizes=(256, 256),
        activation=tf.tanh,
        use_layer_norm=False,
        num_heads=1,
        hide_task_id=False,
    ):
        super().__init__(
            input_dim=input_dim,
            action_space=action_space,
            activation=activation,
            use_layer_norm=use_layer_norm,
            num_heads=num_heads,
            hide_task_id=hide_task_id,
        )
        input_dim = self.input_dim
        self.core = mlp(input_dim, hidden_sizes, activation, use_layer_norm=use_layer_norm)
        self.head_mu = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                tf.keras.layers.Dense(action_space.shape[0] * num_heads),
            ]
        )
        self.head_log_std = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                tf.keras.layers.Dense(action_space.shape[0] * num_heads),
            ]
        )

    def call(self, x: tf.Tensor) -> ActorOutput:
        obs = x
        if self.hide_task_id:
            # NOTE: This is specific to meta-world and is never used with Sequoia.
            x = x[:, :MW_OBS_LEN]
        x = self.core(x)
        mu = self.head_mu(x)
        log_std = self.head_log_std(x)

        if self.num_heads > 1:
            mu = _choose_head(mu, obs, self.num_heads)
            log_std = _choose_head(log_std, obs, self.num_heads)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        pi = mu + tf.random.normal(tf.shape(input=mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        # Make sure actions are in correct range
        action_scale = self.action_space.high[0]
        mu *= action_scale
        pi *= action_scale

        return ActorOutput(mu=mu, log_std=log_std, pi=pi, logp_pi=logp_pi)

    @property
    def common_variables(self):
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return (
                self.core.trainable_variables
                + self.head_mu.trainable_variables
                + self.head_log_std.trainable_variables
            )


class MlpCritic(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ):
        super().__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = num_heads
        self.input_dim = input_dim
        if self.hide_task_id:
            # NOTE: This is specific to meta-world and is never used with Sequoia.
            self.input_dim = MW_OBS_LEN + MW_ACT_LEN
        
        self.core = mlp(input_dim, hidden_sizes, activation, use_layer_norm=use_layer_norm)
        self.head = tf.keras.Sequential(
            [Input(shape=(hidden_sizes[-1],)), tf.keras.layers.Dense(num_heads)]
        )

    def call(self, x, a):
        obs = x
        if self.hide_task_id:
            # NOTE: This is specific to meta-world and is never used with Sequoia.
            x = x[:, :MW_OBS_LEN]
        x = self.head(self.core(tf.concat([x, a], axis=-1)))
        if self.num_heads > 1:
            x = _choose_head(x, obs, self.num_heads)
        x = tf.squeeze(x, axis=1)
        return x

    @property
    def common_variables(self):
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return self.core.trainable_variables + self.head.trainable_variables


class PopArtMlpCritic(MlpCritic):
    # PopArt is a method for normalizing returns, especially useful
    # in multi-task learning. For reference, see https://arxiv.org/abs/1602.07714
    # and https://arxiv.org/abs/1809.04474v1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.moment1 = tf.Variable(tf.zeros((self.num_heads, 1)), trainable=False)
        self.moment2 = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)
        self.sigma = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)

        # TODO: Expose as a hyperparameter?
        self.beta = 3e-4

    @tf.function
    def unnormalize(self, x, obs):
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return x * sigma + moment1

    @tf.function
    def normalize(self, x, obs):
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return (x - moment1) / sigma

    @tf.function
    def update_stats(self, returns, obs):
        task_counts = tf.reduce_sum(obs[:, -self.num_heads :], axis=0)
        batch_moment1 = tf.reduce_sum(
            tf.expand_dims(returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)
        batch_moment2 = tf.reduce_sum(
            tf.expand_dims(returns * returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)

        update_pos = tf.expand_dims(tf.cast(task_counts > 0, tf.float32), 1)
        new_moment1 = self.moment1 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment1, 1) - self.moment1)
        )
        new_moment2 = self.moment2 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment2, 1) - self.moment2)
        )
        new_sigma = tf.math.sqrt(new_moment2 - new_moment1 * new_moment1)
        new_sigma = tf.clip_by_value(new_sigma, 1e-4, 1e6)

        # Update weights of the last layer.
        l = self.head.layers[-1]
        l.kernel.assign(l.kernel * tf.transpose(self.sigma) / tf.transpose(new_sigma))
        l.bias.assign(
            (l.bias * tf.squeeze(self.sigma) + tf.squeeze(self.moment1 - new_moment1))
            / tf.squeeze(new_sigma)
        )

        self.moment1.assign(new_moment1)
        self.moment2.assign(new_moment2)
        self.sigma.assign(new_sigma)
