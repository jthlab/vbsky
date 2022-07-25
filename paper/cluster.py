# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: vbsky
#     language: python
#     name: vbsky
# ---

# +
# # %load_ext nb_black
# # %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import jax.numpy as jnp
import jax
from jax import vmap
from jax.scipy.special import xlogy

jax.config.update("jax_enable_x64", True)
import scipy.linalg

from collections import defaultdict

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from ete3 import Tree
from datetime import datetime, MINYEAR
import time

from vbsky.fasta import SeqData
from vbsky.bdsky import _lognorm_logpdf
from vbsky.prob import VF
from vbsky.prob.distribution import PointMass
from vbsky.prob.transform import (
    Transform,
    Compose,
    Affine,
    Blockwise,
    Positive,
    ZeroOne,
    DiagonalAffine,
    Householder,
    Shift,
    Scale,
    Bounded,
    Exp,
    Softplus,
    Concat,
)
from vbsky.prob.distribution import Constant
from vbsky.prob import arf

from vbsky.plot import *

pos = Compose(DiagonalAffine, Exp)
plus = Compose(DiagonalAffine, Positive)
z1 = Compose(DiagonalAffine, ZeroOne)


# +
def _params_prior_loglik(params):
    ll = 0
    tau = {"R": params["precision_R"][0], "s": params["precision_s"][0]}
    ll += jax.scipy.stats.gamma.logpdf(tau["R"], a=0.001, scale=1 / 0.001)
    ll += jax.scipy.stats.gamma.logpdf(tau["s"], a=0.001, scale=1 / 0.001)

    ll += jax.scipy.stats.beta.logpdf(params["s"], 0.02, 0.98).sum()

    #     mus = [0.5, 4.1, -2]
    #     sigmas = [1, 0.5, 0.5]

    mus = [1.0, -0.5]
    sigmas = [1, 1e-2]

    for i, k in enumerate(["R", "origin"]):
#     for i, k in enumerate(["R"]):
        log_rate = jnp.log(params[k])
        ll += _lognorm_logpdf(log_rate, mu=mus[i], sigma=sigmas[i]).sum()

    for k in ["R", "s"]:
        log_rate = jnp.log(params[k])
        if k in ["R", "delta", "s"]:
            ll -= (tau[k] / 2) * (jnp.diff(log_rate) ** 2).sum()
            m = len(log_rate)
            ll += xlogy((m - 1) / 2, tau[k] / (2 * jnp.pi))
    return ll


def _params_prior_loglik_less_smooth(params):
    ll = 0
    tau = {"R": params["precision_R"][0], "s": params["precision_s"][0]}
    ll += jax.scipy.stats.gamma.logpdf(tau["R"], a=10, scale=0.1 / 10)
    ll += jax.scipy.stats.gamma.logpdf(tau["s"], a=10, scale=0.1 / 10)

    ll += jax.scipy.stats.beta.logpdf(params["s"], 20, 980).sum()

    #     mus = [0.5, 4.1, -2]
    #     sigmas = [1, 0.5, 0.5]

    mus = [1.0, -1.2]
    sigmas = [1, 1e-2]

    for i, k in enumerate(["R", "origin"]):
        #     for i, k in enumerate(["R"]):
        log_rate = jnp.log(params[k])
        ll += _lognorm_logpdf(log_rate, mu=mus[i], sigma=sigmas[i]).sum()

    for k in ["R", "s"]:
        log_rate = jnp.log(params[k])
        if k in ["R", "delta", "s"]:
            ll -= (tau[k] / 2) * (jnp.diff(log_rate) ** 2).sum()
            m = len(log_rate)
            ll += xlogy((m - 1) / 2, tau[k] / (2 * jnp.pi))
    return ll


def _params_prior_loglik_bias(params):
    ll = 0
    tau = {"R": params["precision_R"][0], "s": params["precision_s"][0]}
    ll += jax.scipy.stats.gamma.logpdf(tau["R"], a=0.001, scale=1 / 0.001)
    ll += jax.scipy.stats.gamma.logpdf(tau["s"], a=0.001, scale=1 / 0.001)

    ll += jax.scipy.stats.beta.logpdf(params["s"], 20, 980).sum()

    #     mus = [0.5, 4.1, -2]
    #     sigmas = [1, 0.5, 0.5]

    mus = [1.0, -1.2]
    sigmas = [1, 0.1]

    #     for i, k in enumerate(["R", "origin"]):
    for i, k in enumerate(["R"]):
        log_rate = jnp.log(params[k])
        ll += _lognorm_logpdf(log_rate, mu=mus[i], sigma=sigmas[i]).sum()

    for k in ["R", "s"]:
        log_rate = jnp.log(params[k])
        if k in ["R", "delta", "s"]:
            ll -= (tau[k] / 2) * (jnp.diff(log_rate) ** 2).sum()
            m = len(log_rate)
            ll += xlogy((m - 1) / 2, tau[k] / (2 * jnp.pi))
    return ll


priors = {
    "original": _params_prior_loglik,
    "less": _params_prior_loglik_less_smooth,
    "bias": _params_prior_loglik_bias,
}


def default_flows(data, m, rate):

    local_flows = [
        {"proportions": Transform(td.n - 2, z1), "root_proportion": Transform(1, z1)}
        for td in data.tds
    ]

    global_flows = VF(
        origin=Transform(1, pos),
#         origin=Constant(0.3),
        origin_start=Constant(data.earliest),
        # delta=Transform(m, pos),
        delta=Constant(np.repeat(36.5, m)),
        R=Transform(m, pos),
        rho_m=Constant(0),
        s=Transform(m, z1),
        #         s=Constant(np.repeat(0.02, m)),
        # precision=Constant(1.0),
        precision_R=Transform(1, pos),
        precision_s=Transform(1, pos),
        clock_rate=Constant(rate),
    )
    return global_flows, local_flows


def fixed_origin_flows(data, m, rate):

    local_flows = [
        {"proportions": Transform(td.n - 2, z1), "root_proportion": Transform(1, z1)}
        for td in data.tds
    ]

    global_flows = VF(
        origin=Constant(0.3),
        origin_start=Constant(data.earliest),
        delta=Constant(np.repeat(36.5, m)),
        R=Transform(m, pos),
        rho_m=Constant(0),
        s=Transform(m, z1),
        precision_R=Transform(1, pos),
        precision_s=Transform(1, pos),
        clock_rate=Constant(rate),
    )
    return global_flows, local_flows


rate = 1.12e-3

# +
external_state_df = pd.read_csv("covid/external_state.csv")
external_global_df = pd.read_csv("covid/external_global.csv")
external = {}

def add_key(k, df, col):
    external[k] = df.loc[
        (df[col].str.lower() == k) & (~(df["rt"].isna()))
    ]

    external[k]["date"] = pd.to_datetime(external[k]["date"])

def add_key2(k, df, col):
    external[k] = df.loc[
        (df[col].str.lower() == k) & (~(df["rt"].isna())) & (df["Province_State"].isna())
    ]

    external[k]["date"] = pd.to_datetime(external[k]["date"])

for k in ["florida", "michigan"]:
    add_key(k, external_state_df, "stateName")
    
for k in ["united kingdom", "us"]:
    add_key2(k, external_global_df, "Country_Region")
external["uk"] = external["united kingdom"]    
external["usa"] = external["us"]


def plot_external(ax, region, color=None):
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    ax.plot(
        external[region]["date"],
        external[region]["rt"],
        label="Surveillance Data",
        color=color,
    )
    ax.fill_between(
        external[region]["date"],
        external[region]["rt_lower"],
        external[region]["rt_upper"],
        alpha=0.1,
        label="_nolegend_",
        color=color,
    )


# +
def plot_one(res, ax, param, m, start, top, end, x0, label, ci, title):
    y0 = []
    for x1, x2, y in zip(start, top, res.global_posteriors[param]):
        intervals = np.linspace(x1, x2, m + 1)
        t = (end - intervals)[::-1]
        y0.append(interp1d(t[1:], y, kind="nearest", bounds_error=False)(x0))
    q25, q50, q75 = np.nanquantile(np.array(y0), q=[0.025, 0.5, 0.975], axis=0)

    color = next(ax._get_lines.prop_cycler)["color"]
    year = np.floor(x0).astype(int)
    x = year.astype(str).astype("datetime64[Y]") + np.around((x0-year) * 365.245 * 24 * 3600).astype('timedelta64[s]')
    x = x.astype("datetime64[ns]")
    ax.plot(x, q50, label=label, color=color)

    if ci == "fill":
        ax.fill_between(x, q25, q75, alpha=0.1, label="_nolegend_", color=color)
    if ci == "lines":
        ax.plot(x, q25, "--", label="_nolegend_", alpha=0.25, color=color)
        ax.plot(x, q75, "--", label="_nolegend_", alpha=0.25, color=color)
    # plt.xlim(reversed(plt.xlim()))
    # ax.set_xlabel("Year")
    ax.set_title(title, size=14)
    if param == "R":
        ax.set_ylim(0, 4)
        ax.axhline(y=1, linestyle="--", color="r")
    ax.set_xlim(left_end, right_end)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)


def plot_by_param(res, data, axs, m, regions, param, **kwargs):
    if "label" not in kwargs:
        kwargs["label"] = "VBSKY"
    if "ci" not in kwargs:
        kwargs["ci"] = "fill"
    if "ntips" not in kwargs:
        kwargs["ntips"] = 200

    for ax, r in zip(axs, regions):
        
        start, top, end, x0 = plot_helper(res[r], data[r], kwargs["ntips"])
        if r == "usa" or r == "uk":
            title = r.upper()
        else:
            title = r.title()
        plot_one(
            res[r], ax, param, m, start, top, end, x0, kwargs["label"], kwargs["ci"], title
        )


# -

regions = ["florida", "michigan", "usa", "uk"]

fasta = {}
fasta["florida"] = AlignIO.read("covid/audacity_alns/audacity_fl.fa", format="fasta")
fasta["michigan"] = AlignIO.read("covid/audacity_alns/audacity_mi.fa", format="fasta")
fasta["usa"] = AlignIO.read("covid/audacity_alns/audacity_usa.fa", format="fasta")
fasta["uk"] = AlignIO.read("covid/audacity_alns/audacity_uk.fa", format="fasta")

left_end = pd.Timestamp("2020-01-01")
right_end = pd.Timestamp("2021-12-08")

data = {k: SeqData(v, right_end=right_end) for k, v in fasta.items()}

# ## Run Analysis

# +
n_tips = 200
temp_folder = "covid/temp"
tree_path = "covid/temp/subsample.trees"

stratified = False
stratify_by = None
cluster = True

for k, v in data.items():
    audacity = True
    if "florida" in k:
        audacity_tree_path = "covid/trees/global_fl.tree"
    elif "michigan" in k:
        audacity_tree_path = "covid/trees/global_mi.tree"
    elif "usa" in k: 
        audacity_tree_path = "covid/trees/global_usa.tree"
    else:
        audacity_tree_path = "covid/trees/global_uk.tree"

    n_trees = min(int(np.ceil(v.n / n_tips)), 50)
    v.prep_data(
        n_tips,
        n_trees,
        temp_folder,
        tree_path,
        audacity=audacity,
        audacity_tree_path=audacity_tree_path,
        stratified=stratified,
        stratify_by=stratify_by,
        cluster = cluster
    )

# +
m = 50

for k, v in data.items():
    global_flows, local_flows = fixed_origin_flows(v, m, rate)
    v.setup_flows(global_flows, local_flows)

rng = jax.random.PRNGKey(6)
res = {}
n_iter = 10
threshold = 0.001
step_size = 1.0
for k, v in data.items():
    start = time.time()
    res[k] = v.loop(
        priors["bias"], rng, n_iter, step_size=step_size, threshold=threshold
    )

# +
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = axs.flatten()

plot_by_param(res, data, axs, m, regions, "R")
for i, r in enumerate(regions):
    plot_external(axs[i], r)

axs[0].legend(loc="lower right")
    
fig.set_size_inches(20, 14)
plt.tight_layout()
fig.savefig("covid/figures/all/cluster_R.pdf", format="pdf")

# +
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)

for r in regions:
    start, top, end, x0 = plot_helper(res[r], data[r], 200)
    if r == "usa" or r == "uk":
        title = r.upper()
    else:
        title = r.title()
    plot_one(res[r], ax, "s", m, start, top, end, x0, title, "fill", "")

plt.legend()
fig.savefig("covid/figures/all/cluster_s.pdf", format="pdf")
