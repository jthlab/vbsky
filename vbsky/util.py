import csv
import functools
from typing import Dict, NamedTuple, Tuple

import jax
import numpy as np
import tskit
from Bio import SeqIO
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap
from jax.lib import pytree

from .tree_data import TreeData


class TipData(NamedTuple):
    partials: np.ndarray
    counts: np.ndarray


def jaxify_args(f):
    @functools.wraps(f)
    def _f(*args):
        return f(*map(jnp.array, args))

    return _f


def get_dates(dates_file):
    reader = csv.reader(open(dates_file, "r"))
    node_dates_dict = {row[0]: float(row[1]) for row in reader}
    return node_dates_dict


DNA_MAP = dict(zip("acgt", np.eye(4)))


def get_partials(alignment: Dict[int, SeqIO.SeqRecord]) -> np.ndarray:
    """Convert an alignment to array of tip partial likelihoods."""
    num_samples = len(alignment)
    assert num_samples > 0
    num_sites = len(alignment[0].seq)
    partials = np.zeros((num_sites, num_samples, 4), dtype=int)

    for s in range(num_samples):
        for i, c in enumerate(alignment[s].seq):
            # all missing values are encoded with [1,1,1,1]
            partials[i, s] = DNA_MAP.get(c.lower(), [1, 1, 1, 1])
    return partials


def parse_tsk_tree(tree: tskit.Tree) -> Tuple[TreeData, jnp.ndarray]:
    G = tree.genotype_matrix()  # [M, n]


def scan_with_init(f, init, xs, length=None, reverse=False):
    """Wrapper around jax.lax.scan that adds the initial value(s) onto the returned sequence.

    Examples:
        # >>> scan_with_init(lambda carry, x: (x, x), -1, jnp.arange(3))  # doctest: +NORMALIZE_WHITESPACE
        # (DeviceArray(2, dtype=int32), DeviceArray([-1, 0, 1, 2], dtype=int32))
        # >>> # compute k-chooses-2
        # >>> scan_with_init(lambda carry, x: (carry + x, carry + x), 1, jnp.arange(2, 10))[1]  # doctest: +NORMALIZE_WHITESPACE
        # DeviceArray([ 1, 3, 6, 10, 15, 21, 28, 36, 45], dtype=int32)
        # >>> # 10 + 9, 10 + 9 + 8, ...
        >>> scan_with_init(lambda carry, x: (carry + x, carry + x), 10, jnp.arange(1, 10), reverse=True)[1]  # doctest: +NORMALIZE_WHITESPACE
        DeviceArray([55, 54, 52, 49, 45, 40, 34, 27, 19, 10], dtype=int32)
        >>> scan_with_init(lambda carry, x: (x, x), {'a': -1}, {'a': jnp.arange(3)})  # doctest: +NORMALIZE_WHITESPACE
        ({'a': DeviceArray(2, dtype=int32)}, {'a': DeviceArray([-1, 0, 1, 2], dtype=int32)})
    """
    carry, ys = jax.lax.scan(f, init, xs, length, reverse)
    if reverse:
        ys, init = init, ys
    ys = jax.tree_util.tree_multimap(jnp.append, init, ys)
    return carry, ys


def id_assert(x, cond, msg):
    def _f(x):
        assert cond(x), msg

    return id_tap(_f, x)


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


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = pytree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


class RateFunction(NamedTuple):
    t: np.ndarray
    c: np.ndarray
