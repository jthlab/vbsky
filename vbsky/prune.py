"""Pruning algorithm and efficient gradients"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from . import substitution
from .tree_data import TreeData

scan_ = jax.lax.scan


def _compute_postorder_partials(
    branch_lengths: jnp.ndarray,
    Q: substitution.SubstitutionModel,
    tip_partials: jnp.ndarray,
    td: TreeData,
    rescale: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(tip_partials) == td.n

    def f(tup, parent):
        post, const = tup
        children = jnp.take(td.parent_child, parent, 0)
        t = jnp.take(branch_lengths, children, 0)  # [2]
        A = jnp.take(post, children, 0)  # [2, 4]
        Pprod = Q.expm_action(t[0], A[0]) * Q.expm_action(t[1], A[1])
        if rescale:
            c = Pprod.sum()
        else:
            c = 1.0
        Pprod /= c
        return (
            jax.ops.index_update(post, parent, Pprod),
            jax.ops.index_update(const, parent, jnp.log(c) + const[children].sum()),
        ), None

    _, A = tip_partials.shape
    init = (
        jnp.concatenate([tip_partials, jnp.zeros([td.n - 1, A])]),
        jnp.zeros(2 * td.n - 1),
    )
    return scan_(f, init, jnp.arange(td.n, 2 * td.n - 1))[0]


def _compute_preorder_partials(
    branch_lengths,
    Q: substitution.SubstitutionModel,
    postorder_partials,
    log_post_const,
    td: TreeData,
    rescale: bool = True,
):
    assert len(postorder_partials) == 2 * td.n - 1

    def f(tup, child):
        pre, const = tup
        parent = jnp.take(td.child_parent, child)
        sib = jnp.take(td.siblings, child)
        # eq (7) from paper
        A0 = Q.expm_action(branch_lengths[sib], postorder_partials[sib])
        A = pre[parent] * A0
        B = Q.expm_action(branch_lengths[child], A, right=False)
        if rescale:
            c = B.sum()
        else:
            c = 1.0
        B /= c
        return (
            jax.ops.index_update(pre, child, B),
            jax.ops.index_update(
                const, child, jnp.log(c) + const[parent] + log_post_const[sib]
            ),
        ), None

    M, A = postorder_partials.shape  # M = 2 * N - 1
    init = (
        jax.ops.index_update(jnp.zeros([M, A]), jax.ops.index[-1, :], Q.pi),
        jnp.zeros(M),
    )
    (pre, const), _ = scan_(f, init, td.postorder[:-1], reverse=True)
    return pre, const


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5))
def prune_loglik(
    branch_lengths: jnp.ndarray,
    Q: substitution.SubstitutionModel,
    tip_partials: jnp.ndarray,
    td: TreeData,
    rescale: bool = True,
    dbg: bool = False,
) -> float:
    """Compute log-likelihood using tree-pruning algorithm.

    Args:
        branch_lengths: Vector whose i-th entry contains the length of the branch above node i.
        Q: Substitution model.
        tip_partials: Array of length [n, 4] containing partial likelihoods at each tip.
        td: Data for tree.

    Returns:
        Log-likelihood of data under for given data/tree/substitution model.
    """
    # dimension checks
    assert Q.P.ndim == 2
    A = Q.P.shape[0]
    assert Q.P.shape[1] == A
    assert tip_partials.ndim == 2
    assert tip_partials.shape == (td.n, A)
    assert branch_lengths.ndim == 1
    assert len(branch_lengths) == 2 * (td.n - 1)

    pop, log_consts = _compute_postorder_partials(
        branch_lengths, Q, tip_partials, td, rescale
    )  # [2 N - 1, A]
    return jnp.log(pop[-1] @ Q.pi) + log_consts[-1]


@prune_loglik.defjvp
def prune_loglik_jvp(td: TreeData, rescale: bool, dbg: bool, primals, tangents):
    branch_lengths, Q, tip_partials = primals
    branch_lengths_dot, *_ = tangents
    post, log_post_const = _compute_postorder_partials(
        branch_lengths, Q, tip_partials, td, rescale
    )  # [2 N - 1, A]
    P = post[-1] @ Q.pi  # overall likelihood, i.e. pi * P(root | pi)
    log_P = jnp.log(P) + log_post_const[-1]
    pre, log_pre_const = _compute_preorder_partials(
        branch_lengths, Q, post, log_post_const, td, rescale
    )
    grad = jnp.einsum("nu,uv,nv->n", pre[:-1], Q.Q, post[:-1]) * jnp.exp(
        log_pre_const[:-1] + log_post_const[:-1] - log_P
    )  # eq (9) from paper
    # grads for Q, pi, tip_partials currently unsupported
    return log_P, jnp.dot(grad, branch_lengths_dot)
