import jax
import jax.numpy as jnp
from jax import vmap
from jax.experimental.host_callback import id_print
from jax.scipy.special import xlog1py, xlogy

from . import prune
from .substitution import SubstitutionModel
from .tree_data import TreeData
from .util import TipData, order_events


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
    def f(pi1, d):
        # pi1 = p_{i+1}(t_i)
        B_i = ((1 - 2 * (1 - d["rho"]) * pi1) * d["lam"] + d["mu"] + d["psi"]) / d["A"]

        Adt = d["A"] * d["dt"]  # A_i (t_i - t_{i-1})
        # fix a numerical issue
        bad = jnp.isclose(B_i, -1.0)
        B_safe = jnp.where(bad, 0.0, B_i)
        f = jnp.where(
            bad,
            -1.0,
            ((1 + B_safe) - jnp.exp(-Adt) * (1 - B_safe))
            / ((1 + B_safe) + jnp.exp(-Adt) * (1 - B_safe)),
        )
        # if dt >> 1, psi=0 then f~=1, A=|lam-mu| and p_i~=(lam+mu-|lam-mu|)/(2 lam) = min(mu/lam,1.)
        # this causes problems when we condition on survival (division by 1-p1).
        p_i = (d["lam"] + d["mu"] + d["psi"] - d["A"] * f) / (2 * d["lam"])
        # p_i = 0.5 * (1.0 + (d["mu"] + d["psi"] - d["A"] * f) / d["lam"])
        # p_i = 2 * (1 + (d["mu"] + d["psi"] - d["A"] * f) / (4 * d["lam"]))
        # 1mp_i = (d["lam"] - d["mu"] - d["psi"] + d["A"] * f) / (2 * d["lam"])
        # if True:
        #     p_i, _ = id_print(
        #         (
        #             p_i,
        #             {
        #                 "p_i": p_i,
        #                 "B_i": B_i,
        #                 "d": d,
        #                 "f": f,
        #                 "lam/(lam+mu)": d["lam"] / (d["lam"] + d["mu"]),
        #             },
        #         )
        #     )
        return p_i, B_i

    # 1-p1_0 is the probability that the initial carrier has >0 sampled lineages. this probability is lower bounded
    # by the probability that the first event is a birth instead of death: 1-p1_0 >= lam/(lam+mu)

    dt = jnp.diff(times)
    A = jnp.sqrt((lam - mu - psi) ** 2 + (4 * lam * psi))
    if dbg:
        A = id_print(A, what="A")
    # xs = id_print(xs)
    p1_0, B = jax.lax.scan(
        f,
        1.0,
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
        p1_0, B = id_print((p1_0, B), what="p1_0,B")

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
        # return jnp.log(4) + Adt - 2 * jnp.log(abs(jnp.exp(Adt) * (1 - B_i) + (1 + B_i)))
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
        1 + ev[jnp.searchsorted(ev[:, 0], times[1:-1], side="left"), 1], 0.0
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
    if condition_on_survival:
        l1 -= jnp.log1p(-p1_0)
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
    return jax.scipy.stats.norm.logpdf((log_x - mu) / sigma) - log_x


def _params_prior_loglik(params):
    # TODO move this
    # uninformative gamma prior on tau
    tau = params["precision"][0]
    ll = jax.scipy.stats.gamma.logpdf(tau, a=0.001, scale=1 / 0.001)
    ll += jax.scipy.stats.beta.logpdf(params["rho"][-1], 1, 9999)

    # marginal priors
    for k in ["R", "delta", "x1"]:
        log_rate = jnp.log(params[k])
        ll += _lognorm_logpdf(log_rate, mu=1.0, sigma=1.25).sum()
        # ll -= (tau / 2) * (jnp.diff(log_rate) ** 2).sum()
        # m = len(log_rate)
        # ll += xlogy((m - 1) / 2, tau / (2 * jnp.pi))
    #     # gmrf with precision tau
    #     for rate in bdsky_transform(params):
    #             m = len(rate)
    #             ll -= (tau / 2) * (jnp.diff(jnp.log(1e-8 + rate)) ** 2).sum()
    #             ll += xlogy((m - 1) / 2, tau / (2 * jnp.pi))
    #     gmrf_ll -= jnp.log(delta).sum() / 2  # missing from original skyride paper?
    return ll


def loglik(
    params,
    tr_d: TreeData,
    tp_d: TipData,
    Q: SubstitutionModel,
    c: jnp.ndarray = jnp.ones(3),
    dbg: bool = False,
    condition_on_survival: bool = True,
):
    # There should be one proportion for each internal branch except the root.
    assert len(params["proportions"]) == tr_d.n - 2
    assert len(params["R"]) == len(params["s"]) == len(params["delta"])
    assert params["root_height"].ndim == params["root_height"].size == 1

    # transform to the bdsky model parameters
    params_prior_ll = _params_prior_loglik(params)

    lam, psi, mu, rho = _bdsky_transform(params)

    # params["root_height"] = id_print(params["root_height"], what="root_height")

    # Convert proportions and root height to internal node heights.
    root_height = params["root_height"][0]
    node_heights = tr_d.height_transform(
        root_height,
        params["proportions"],
    )

    # Convert node heights to branch lengths
    branch_lengths = node_heights[tr_d.child_parent[:-1]] - node_heights[:-1]

    # likelihood of tree under bdsky prior
    # create time points: grid of m equispaced intervals
    m = len(params["R"])
    tm = root_height + params["x1"][0]
    times = jnp.linspace(0, tm, m + 1)
    # times = id_print(times)
    xs = tm - node_heights
    tree_prior_ll = _tree_loglik(
        lam, psi, mu, rho, xs, times, tr_d, dbg, condition_on_survival
    )

    # likelihood of data given tree: map across all columns of the alignment
    # branch_lengths = id_print(branch_lengths, what="branch_lengths")
    data_ll = (
        vmap(prune.prune_loglik, (None, None, 0, None, None))(
            branch_lengths * params["clock_rate"][0],
            Q,
            tp_d.partials,
            tr_d,
            dbg,
        )
        * tp_d.counts
    ).sum()

    if dbg:
        params_prior_ll, tree_prior_ll, data_ll = id_print(
            (params_prior_ll, tree_prior_ll, data_ll), what="lls"
        )

    return c[0] * params_prior_ll + c[1] * tree_prior_ll + c[2] * data_ll
