from typing import List, NamedTuple, TypeVar, Generic

import tensorflow as tf

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
import numpy as np


class BatchDict(TypedDict):
    obs1: np.ndarray
    obs2: np.ndarray
    acts: np.ndarray
    rews: np.ndarray
    done: bool


class GradientsTuple(NamedTuple):
    actor_gradients: List[tf.Tensor]
    critic_gradients: List[tf.Tensor]
    alpha_gradient: List[tf.Tensor]

