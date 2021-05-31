import abc
import functools
from dataclasses import asdict, dataclass, astuple
from typing import NamedTuple, Generic, TypeVar, Type

import numpy as np
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


class TipData(NamedTuple):
    partials: np.ndarray
    counts: np.ndarray

def jaxify_args(f):
    @functools.wraps(f)
    def _f(*args):
        return f(*map(jnp.array, args))

    return _f


@jaxify_args
def order_events(times, node_heights, node_is_sample):
    """Given sample times, discretization times, and node heights, return a sequence which records
    the number of lineages from one event to the next.

    Returns:
        Sequence [(t[0], change[0]), ..., (t[k], change[k])] giving the net change in the number of lineages
        at each sampling event.

    Notes:
        Assumes the following:
            - times[0] = min(sample_heights) = 0
            - times[-1] >= max(node_heights) > max(sample_times)
            - len(sample_times) = 1 + len(node_heights)

    Examples:
         >>> node_is_sample = [True, True, False, True, True, False]
         >>> event_list = order_events([0, .5, 1.0], [0., 0., .1, .7, .7, .9], node_is_sample).tolist()
         >>> len(event_list)
         9
         >>> # two lineages sampled at time zero
         >>> sorted(event_list[:3])  # doctest: +NUMBER
         [[0., 0.], [0., 1.], [0., 2.]]
         >>> # at time t=.1, a coalescence occurs
         >>> event_list[3]  # doctest: +NUMBER
         [.1, 1.]
         >>> # at time t=.5, nothing happens
         >>> event_list[4]  # doctest: +NUMBER
         [.5, 1.]
         >>> # at time t=.7, two new samples are drawn
         >>> event_list[5:7]  # doctest: +NUMBER
         [[.7, 2.], [.7, 3.]]
         >>> # at time t=.9, a coalescent occurs
         >>> event_list[7]  # doctest: +NUMBER
         [.9, 2.]
         >>> # at time t=1, nothing happens
         >>> event_list[8]  # doctest: +NUMBER
         [1., 2.]
    """
    assert len(node_heights) == len(node_is_sample)
    i = node_heights.argsort()
    node_heights = node_heights[i]
    node_is_sample = node_is_sample[i]
    merged = jnp.concatenate([times, node_heights])
    j = merged.argsort()
    delta = jnp.concatenate([jnp.zeros_like(times), 2 * node_is_sample - 1])
    return jnp.array([merged[j], jnp.cumsum(delta[j])]).T


class RateFunction(NamedTuple):
    t: np.ndarray
    c: np.ndarray
