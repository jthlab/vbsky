from collections.abc import Callable
from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.host_callback import id_print
from jax.scipy.special import xlog1py, xlogy, logit, expit

from . import prune
from .substitution import SubstitutionModel
from .tree_data import TreeData
from .util import TipData, order_events

from functools import partial


def _tree_loglik(
    lam: jnp.ndarray,
    psi: jnp.ndarray,
    mu: jnp.ndarray,
    rho: jnp.ndarray,
    xs: jnp.ndarray,
    times: jnp.ndarray,
    td: TreeData,
    dbg: bool = False,
    condition_on_survival: bool = True,
):
    """Likelihood of the birth-death skyline model.

    Notes:
         See Theorem 1 of Stadler et al. (PNAS 2013)
    """
    # Note that the paper has time running forward from t0=0 (root) to t_m (most recent sampling time). in contrast,
    # all of our data structures measure time from the present backwards. to better adhere to the notation in the bdsky
    # paper, all node times are therefore transformed to be xs[i] = tm - node_heights[i].

    # Calculate parameters A, B, p, q
    # 1. A_i is calculated directly
    if dbg:
        lam, psi, mu, rho = id_print((lam, psi, mu, rho), what="lam,psi,mu,rho")

    m = len(lam)
    assert m == len(lam) == len(psi) == len(mu)
    assert m + 1 == len(times)

    # 2. Recursion for B_i and {p,q}_{i+1}(t_i).
    def f(tup, d):
        pi1, logit_pi1 = tup
        # pi1 = p_{i+1}(t_i)
        B_i = ((1 - 2 * (1 - d["rho"]) * pi1) * d["lam"] + d["mu"] + d["psi"]) / d["A"]
        Adt = d["A"] * d["dt"]  # A_i (t_i - t_{i-1})
        # f is numerically tricky since B_i can be close to (or equal) -1,1 while Adt can be >> 1
        f = ((1 + B_i) - jnp.exp(-Adt) * (1 - B_i)) / (
            (1 + B_i) + jnp.exp(-Adt) * (1 - B_i)
        )
        g = f  # jnp.where(jnp.isfinite(f), f, -1.0)
        p_i = jnp.minimum(
            (d["lam"] + d["mu"] + d["psi"] - d["A"] * g) / (2 * d["lam"]), 1.0
        )
        q = d["dt"] * jnp.minimum(d["lam"] - d["mu"], 0.0)
        logit_pi = jnp.where(
            jnp.isfinite(logit(p_i)),
            logit(p_i),
            jnp.where(
                jnp.isclose(d["psi"], 0.0) & jnp.isclose(d["rho"], 0.0),
                jnp.where(
                    d["lam"] < d["mu"],
                    logit_pi1
                    - jnp.log(pi1)
                    - q
                    - (d["lam"] - jnp.exp(q) * d["mu"])
                    * (1.0 - pi1)
                    / (d["lam"] - d["mu"]),
                    jnp.log(d["mu"] / abs(d["lam"] - d["mu"])),
                ),
                logit(p_i),
            ),
        )
        # p_i, logit_pi, _ = id_print(
        #     (
        #         p_i,
        #         logit_pi,
        #         {
        #             "logit(p_i)": logit(p_i),
        #             "expit(...)": expit(logit_pi),
        #             "pi1": pi1,
        #             "logit_pi1": logit_pi1,
        #             "q": q,
        #             "d": d,
        #             "B_i": B_i,
        #             "f": f,
        #         },
        #     ),
        #     what="logit(p_i)",
        # )
        return (p_i, logit_pi), B_i

    dt = jnp.diff(times)
    A = jnp.sqrt((lam - mu - psi) ** 2 + (4 * lam * psi))
    if dbg:
        A = id_print(A, what="A")
    # xs = id_print(xs)
    (p1_0, logit_p1_0), B = jax.lax.scan(
        f,
        (1.0, 0.0),
        {
            "A": A,
            "lam": lam,
            "mu": mu,
            "psi": psi,
            "rho": rho,
            "dt": dt,
            "i": jnp.arange(len(A)),
        },
        reverse=True,
    )

    if dbg:
        p1_0, logit_p1_0, B = id_print((p1_0, logit_p1_0, B), what="p1_0,B")

    x = xs[td.n :]  # transmission times
    y = xs[: td.n]  # times of sampled nodes
    # Indices into the rate functions for each vertex
    I_helper = vmap(jnp.searchsorted, (None, 0, None))
    # I{x,y} = ell({x,y}_i) in paper
    Ix, Iy = [I_helper(times, u, "right") for u in (x, y)]
    if dbg:
        times, Ix, Iy, x, y = id_print(
            (times, Ix, Iy, x, y), what="times, Ix, Iy, x, y"
        )

    @jnp.vectorize
    def log_q(i, t):
        A_i, B_i = [jnp.take(x, i - 1) for x in (A, B)]
        t_i = jnp.take(times, i)
        Adt = A_i * (t - t_i)
        if dbg:
            Adt, B_i = id_print((Adt, B_i), what="Adt,B_i")
        return jnp.log(4) + Adt - 2 * jnp.log(abs(jnp.exp(Adt) * (1 - B_i) + (1 + B_i)))
        m1 = jnp.isclose(B_i, -1.0)
        safe = jnp.where(m1, 0.0, B_i)
        f = jnp.where(
            m1, Adt + jnp.log(2.0), jnp.log(abs(jnp.exp(Adt) * (1 - safe) + (1 + safe)))
        )
        return jnp.log(4) + Adt - 2 * f

    # now calculate n_i, the number of d2 vertices at each time point.
    node_is_sample = td.n <= jnp.arange(2 * td.n - 1)
    ev = order_events(times, xs, node_is_sample)
    # number of degree-2 vertices at times -- by construction, no d2 vertices at t_M. also, add one to lineage counts
    # because there is a branch extending up from the root to time t0.
    n_i = jnp.append(
        2 + ev[jnp.searchsorted(ev[:, 0], times[1:-1], side="left"), 1], 0.0
    )  # times[1], ..., times[m]

    # number of deterministic samples at each time point
    near_t_i = jnp.isclose(xs[: td.n, None], times[None, 1:]) & (rho[None] > 0)
    N_i = near_t_i.sum(0)
    sampled_leaves = ~near_t_i.max(1)

    if dbg:
        n_i, N_i, sampled_leaves = id_print(
            (n_i, N_i, sampled_leaves), what="n_i,N_i,sampled_leaves"
        )

    # Finally, compute each term in the likelihood (Thm. 1)
    # First the easy one:
    l1 = log_q(1, times[0])
    if dbg:
        l1 = id_print(l1, what="log[q_1(t_0)]")
    if condition_on_survival:
        l1 = 2 * jnp.log1p(-p1_0)
        # l1_0 = jnp.log1p(-p1_0)
        # l1_1 = jnp.where(logit_p1_0 > 10, -logit_p1_0, -jnp.log1p(jnp.exp(logit_p1_0)))
        # l1_2 = jnp.where(jnp.isfinite(l1_0), l1_0, l1_1)
        # if dbg:
        #     l1_0, l1_1, l1_2 = id_print((l1_0, l1_1, l1_2), what="l1_0, l1_1, l1_2")
        # # l1_1 = jnp.log1p(-p1_0)
        # l1 -= l1_2
    if dbg:
        l1 = id_print(l1, what="log[q_1(t_0) / (1 - p_1(t_0))]")
    loglik = l1

    # \prod_{i=1}^{N+n-1} lam[\ell(x_i)] q_\ell(x_i)(x_i)
    l2 = log_q(Ix, x) + jnp.take(jnp.log(lam), Ix - 1)
    if dbg:
        l2 = id_print(l2, what=r"\prod_{i=1}^{N+n-1} lam[\ell(x_i)] q_\ell(x_i)(x_i)")
    loglik += l2.sum()

    # \prod_{i=1}^n \psi(y_i)(y_i) / q_\ell(y_i)(y_i)
    l3 = xlogy(sampled_leaves, jnp.take(psi, Iy - 1)) - sampled_leaves * log_q(Iy, y)
    if dbg:
        l3 = id_print(l3, what=r"\prod_{i=1}^n \psi(y_i)(y_i) / q_\ell(y_i)(y_i)")
    loglik += l3.sum()

    # \prod_{i=1}^m [(1-rho_i)q_{i+1}(t_i)]^{n_i}
    # Includes non-sensical q_{m+1}(t_m) term, but n_m === 0 so it does not matter.
    i = jnp.arange(1, m)
    log_qi1_ti = log_q(i + 1, times[1:-1])
    l41 = xlog1py(n_i, -rho)
    l42 = n_i[:-1] * log_qi1_ti
    if dbg:
        l41, l42 = id_print((l41, l42), what=r"(1-rho[i])^n_i, q_{i+1}(ti)^n_i")
    loglik += l41.sum() + l42.sum()

    # \prod_{i=1}^m rho_i^N_i
    l5 = xlogy(N_i, rho)
    if dbg:
        l5 = id_print(l5, what=r"rho_i^N_i")
    loglik += l5.sum()

    return loglik


def _bdsky_transform(params):
    # R: R0
    # delta: becoming uninfectious (death + sampling)
    # s: sampling
    lam = params["R"] * params["delta"]  # birth
    psi = params["s"] * params["delta"]  # sampling
    mu = params["delta"] - psi  # death
    rho = params["rho"]
    return lam, psi, mu, rho


def _lognorm_logpdf(log_x, mu, sigma):
    return jax.scipy.stats.norm.logpdf((log_x - mu) / sigma) - log_x - jnp.log(sigma)


# def _params_prior_loglik(params):
#     # TODO move this
#     # uninformative gamma prior on tau
#     tau = params["precision"][0]
#     ll = jax.scipy.stats.gamma.logpdf(tau, a=0.001, scale=1 / 0.001)

#     # loc = params["grid"][0]
#     # scale = (params["grid"][-1] - params["grid"][0])
#     # ll += jax.scipy.stats.uniform.logpdf(params["grid"][1], loc=loc, scale=scale)
#     # ll += jax.scipy.stats.beta.logpdf(params["s"], 1, 9).sum()
#     # ll += jax.scipy.stats.gamma.logpdf(params["R"], 1.5).sum()
#     # ll += jax.scipy.stats.gamma.logpdf(params["delta"], 4).sum()

#     ll += jax.scipy.stats.beta.logpdf(params["rho"][-1], 1, 999)

#     # marginal priors
    
#     mus = [0, 4.1, -1.2]
#     sigmas = [1, 0.5, 0.05]
#     for i,k in enumerate(["R", "delta", "origin"]):
#         log_rate = jnp.log(params[k])
#         ll += _lognorm_logpdf(log_rate, mu=mus[i], sigma=sigmas[i]).sum()
#     #     ll -= (tau / 2) * (jnp.diff(log_rate) ** 2).sum()
#     #     m = len(log_rate)
#     #     ll += xlogy((m - 1) / 2, tau / (2 * jnp.pi))
#     #     # gmrf with precision tau
#     #     for rate in bdsky_transform(params):
#     #             m = len(rate)
#     #             ll -= (tau / 2) * (jnp.diff(jnp.log(1e-8 + rate)) ** 2).sum()
#     #             ll += xlogy((m - 1) / 2, tau / (2 * jnp.pi))
#     #     gmrf_ll -= jnp.log(delta).sum() / 2  # missing from original skyride paper?
#     return ll


def loglik(
    params,
    tr_d: TreeData,
    tp_d: TipData,
    Q: SubstitutionModel,
    c: tuple[bool, bool, bool],
    dbg: bool = False,
    condition_on_survival: bool = True,
    equidistant_intervals: bool = True,
    _params_prior_loglik: Callable[..., float] = lambda x: 0.,
):
    # params["x1"] = params["x1rh"][:1]
    # params["root_height"] = params["x1rh"][1:]
    # There should be one proportion for each internal branch except the root.
    assert len(params["proportions"]) == tr_d.n - 2
    assert len(params["R"]) == len(params["s"]) == len(params["delta"])
    assert params["root_proportion"].ndim == params["root_proportion"].size == 1

    # transform to the bdsky model parameters
    loglik = 0.0
    if c[0]:
        params_prior_ll = _params_prior_loglik(params)
        loglik += params_prior_ll

    lam, psi, mu, rho = _bdsky_transform(params)

    # params["root_height"] = id_print(params["root_height"], what="root_height")

    # Convert proportions and root height to internal node heights.
    root_height = 1.0 * (  # params["root_proportion"][0] * (
        params["origin"][0] + params["origin_start"][0] - tr_d.sample_times.max()
    )
    node_heights = tr_d.height_transform(
        root_height,
        params["proportions"],
    )

    # Convert node heights to branch lengths
    branch_lengths = node_heights[tr_d.child_parent[:-1]] - node_heights[:-1]

    # likelihood of tree under bdsky prior
    # create time points: grid of m equispaced intervals or prespecified grid

    tm = params["origin"][0] + params["origin_start"][0]

    if equidistant_intervals:
        m = len(params["R"])
        times = jnp.linspace(0, tm, m + 1)
    else:
        times = tm - params["grid"]
        times = jax.ops.index_update(times, 0, 0)
        
    # times = id_print(times)
    xs = tm - node_heights
    if c[1]:
        tree_prior_ll = _tree_loglik(
            lam, psi, mu, rho, xs, times, tr_d, dbg, condition_on_survival
        )
    loglik += tree_prior_ll

    # likelihood of data given tree: map across all columns of the alignment
    # branch_lengths = id_print(branch_lengths, what="branch_lengths")
    if c[2]:
        data_ll = (
            vmap(prune.prune_loglik, (None, None, 0, None, None, None))(
                branch_lengths * params["clock_rate"][0],
                Q,
                tp_d.partials,
                tr_d,
                True,
                dbg,
            )
            * tp_d.counts
        ).sum()

    loglik += data_ll

    return loglik
