from functools import partial
from typing import Any
from collections.abc import Callable

import jax
from jax import vmap, jit, value_and_grad
import jax.numpy as jnp
from jax.experimental.host_callback import id_print
from jax.flatten_util import ravel_pytree

from vbsky.prob import VF
from vbsky.prob.distribution import Distribution
from vbsky.bdsky import loglik
from vbsky.substitution import SubstitutionModel
from vbsky.tree_data import TreeData
from vbsky.util import TipData


def unpack(samples):
    unpacked_samples = {}
    for k, v in samples.items():
        if isinstance(v, dict):
            unpacked_samples |= v
        else:
            unpacked_samples[k] = v

    unpacked_samples["rho"] = jnp.concatenate(
        [jnp.zeros_like(unpacked_samples["delta"][:, :-1]), unpacked_samples["rho_m"]],
        axis=1,
    )

    return unpacked_samples


def loss(
    params: dict[str, Any],
    flows: VF,
    tree_data: tuple[TreeData],
    tip_data: tuple[TipData],
    rng: jax.random.PRNGKey,
    Q: SubstitutionModel,
    c: tuple[tuple[bool, bool], tuple[bool, bool, bool]],
    dbg: bool,
    equidistant_intervals: bool,
    _params_prior_loglik: Callable[..., float],
    n_trees: int
):
    # approximate the loglik by monte carlo
    c1, c2 = c
    samples = {}
    elbo1 = 0.0
    # elbo = E_{x~q(theta)} log p(x) - log q(x;theta)
    if dbg:
        params = id_print(params, what="params")
    for k, v in flows.items():
        p = params[k]
        rng, subrng = jax.random.split(rng)
        # sample from each component of the variational prior
        samples[k] = v.sample(subrng, p, 1)
        # p_prime = jax.lax.stop_gradient(p)
        if c1[0]:
            s1 = vmap(v.log_pdf, (None, 0))(p, samples[k])
            e = jnp.mean(s1)
            if dbg:
                e = id_print(e, what=f"entropy[{k}]")
            elbo1 -= e

    if dbg:
        samples = id_print(samples, what="samples")

    elbo2 = 0.0
    if c1[1]:
        s2 = vmap(loglik, (0,) + (None,) * 9)(
            unpack(samples), tree_data, tip_data, Q, c2, dbg, True, equidistant_intervals, _params_prior_loglik, n_trees
        )
        elbo2 += jnp.mean(s2)
    if dbg:
        elbo1, elbo2 = id_print((elbo1, elbo2), what="elbo")
    return -(elbo1 + elbo2)


def step(i, velocity, M, flows, params, tree_data, tip_data, Q, c, dbg):
    x0, unravel = ravel_pytree(params)

    def obj(x, rng):
        p = unravel(x)
        return loss(p, flows, tree_data, tip_data, rng, Q, c, M, dbg)

    rng = jax.random.PRNGKey(i)
    while True:
        try:
            f0, g0 = obj(x0, rng)
            assert jnp.isfinite(f0).all()
            g0, _ = ravel_pytree(g0)
            assert jnp.isfinite(g0).all()
            p = -g0 / jnp.linalg.norm(g0)
            assert jnp.isfinite(p).all()
            break
        except AssertionError:
            breakpoint()
            print("randomizing to avoid nans")
            _, rng = jax.random.split(rng)

    mass = 0.9
    if velocity is None:
        velocity = jnp.zeros_like(p)
    velocity = mass * velocity + p
    p = velocity

    m = jnp.dot(p, g0)
    t = -0.5 * m

    def cond(ss):
        f, g = obj(x0 + ss * p, rng)
        g, _ = ravel_pytree(g)
        f = id_print(f, what="f")
        cond1 = jnp.isfinite(f)
        cond2 = jnp.isfinite(g).all()
        armijo = f0 - f >= ss * t - 1e-7
        # fail = jnp.isclose(ss, 0.0)
        stop = cond1 & cond2 & armijo  # | fail
        return ~stop

    def body(ss):
        ss = id_print(ss, what="ss")
        return ss * 0.5

    # ss_star = jax.lax.while_loop(cond, body, 1.0 / (i + 1))
    ss = 1 / (1 + i)
    while cond(ss):
        ss = body(ss)
    x_star = x0 + ss * p
    f, g = obj(x_star, rng)
    i, f = id_print((i, f), what="i,f")
    return f, unravel(x_star), g, velocity
