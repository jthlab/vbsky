from functools import partial
from typing import Any

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


@partial(jit, static_argnums=(1, 7, 8))
@value_and_grad
def loss(
    params: dict[str, Any],
    flows: VF,
    tree_data: TreeData,
    tip_data: TipData,
    rng: jax.random.PRNGKey,
    Q: SubstitutionModel,
    c: jnp.ndarray,
    M: int,
    dbg: bool,
):
    # approximate the loglik by monte carlo
    samples = {}
    elbo1 = 0.0
    # elbo = E_{x~q(theta)} log p(x) - log q(x;theta)
    for k, v in flows.items():
        p = params[k]
        rng, subrng = jax.random.split(rng)
        # sample from each component of the variational prior
        samples[k] = v.sample(subrng, p, M)
        e = vmap(v.log_pdf, (None, 0))(p, samples[k]).mean()
        if dbg:
            e = id_print(e, what=f"entropy[{k}]")
        elbo1 -= e

    elbo2 = vmap(loglik, (0,) + (None,) * 6)(
        unpack(samples), tree_data, tip_data, Q, c, dbg, True
    ).mean()
    # elbo1, elbo2 = id_print((elbo1, elbo2), what="elbo")
    return -(elbo1 + elbo2)


def step(i, velocity, M, flows, params, tree_data, tip_data, Q, c, dbg):
    x0, unravel = ravel_pytree(params)

    def obj(x):
        p = unravel(x)
        return loss(p, flows, tree_data, tip_data, jax.random.PRNGKey(i), Q, c, M, dbg)

    f0, g0 = obj(x0)
    g0, _ = ravel_pytree(g0)
    p = -g0 / jnp.linalg.norm(g0)

    mass = 0.9
    if velocity is None:
        velocity = jnp.zeros_like(p)
    velocity = mass * velocity + p
    p = velocity

    m = jnp.dot(p, g0)
    t = -0.5 * m

    def cond(ss):
        f, g = obj(x0 + ss * p)
        g, _ = ravel_pytree(g)
        f = id_print(f, what="f")
        cond1 = jnp.isfinite(f)
        cond2 = jnp.isfinite(g).all()
        armijo = f0 - f >= ss * t
        fail = jnp.isclose(ss, 0.0)
        stop = (cond1 & cond2 & armijo) | fail
        return ~stop

    def body(ss):
        ss = id_print(ss, what="ss")
        return ss * 0.5

    # ss_star = jax.lax.while_loop(cond, body, 1.0 / (i + 1))
    ss = 1 / (1 + i)
    while cond(ss):
        ss = body(ss)
    x_star = x0 + ss * p
    f, g = obj(x_star)
    i, f = id_print((i, f), what="i,f")
    return f, unravel(x_star), g, velocity
