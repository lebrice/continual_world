from functools import singledispatch
import warnings
from typing import Any, Union, List, Dict
import numpy as np
from collections import defaultdict
import tensorflow as tf


@singledispatch
def to_numpy(v):
    return v.numpy() if isinstance(v, tf.Tensor) else np.asarray(v)


@to_numpy.register(dict)
def _dict_to_numpy(v: Dict) -> Dict:
    return {k: to_numpy(value) for k, value in v.items()}


def average_metrics(list_of_dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    dict_of_lists = defaultdict(list)
    for log_dict in self.log_queue:
        for k, v in log_dict.items():
            dict_of_lists[k].append(v)
    concat_values = { 
        k: (np.concatenate(values) if values[0].shape else np.asarray(values))
        for k, values in dict_of_lists.items()
    }
    mean_values = {
        k: np.mean(values) for k, values in concat_values.items()
    }
    return mean_values

def clamp_value_between_bounds(
    obj: Any, attribute: str, reason: str, min_val: Union[int, float] = None, max_val: Union[int, float] = None
) -> None:
    """ Clamps the value of a given attribute between the given bounds. Raises a warning if it does.
    """
    value = getattr(obj, attribute)
    if min_val is not None and value < min_val:
        warnings.warn(
            RuntimeWarning(
                f"Increasing value of '{attribute}' from {value} --> {min_val}"
                + (f" ({reason})" if reason else "")
            )
        )
        setattr(obj, attribute, min_val)
    if max_val is not None and value > max_val:
        warnings.warn(
            RuntimeWarning(
                f"Decreasing value of '{attribute}' from {value} --> {max_val}"
                + (f" ({reason})" if reason else "")
            )
        )
        setattr(obj, attribute, max_val)
