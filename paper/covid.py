# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
# -

#
# ## Data Preprocessing After Download from Gisaid

# +
# Filter from original alignment
# from tqdm import tqdm

# def filter_alignment(filter_by, name, ref):
#     sids = []
#     seqs = []
#     with open(ref, "rt") as file:
#         with open(f"covid/alns/{name}.fa", "w") as file2:
#             it = enumerate(file)
#             for i, line in tqdm(it):
#                 if i % 2 == 0:
#                     flag = False
#                     for kw in filter_by:
#                         if kw in line:
#                             flag = True
#                             break
#                     if flag:
#                         desc = line.split("|")
#                         date = desc[2]
#                         if date[-5:-3] == "00":
#                             continue
#                         file2.write(line)
#                         file2.write(next(it)[1])

# ref = "covid/chunks/x00.fa"
# # ref = "covid/msa_0406.fa"
# filter_alignment(["USA"], "usa", ref)
# filter_alignment(["United Kingdom", "England", "Scotland", "Wales", "Northern Ireland"], "uk", ref)
# ref = "covid/usa.fa"
# filter_alignment(["USA/MI"], "mi", ref)
# filter_alignment(["USA/FL"], "fl", ref)

# +
# # Set up audacity tree


# def filter_audacity_tree(subset, name):
#     global_tree = Tree("covid/global.tree", format=1)
#     leaves = set([leaf.name for leaf in global_tree])

#     seqs = []
#     to_prune = []

#     for s in subset:
#         desc = s.description.split("|")
#         if desc[1] in leaves:
#             seqs.append(s)
#             to_prune.append(desc[1])

#     global_tree.prune(to_prune, preserve_branch_length=True)
#     global_tree.write(outfile=f"covid/global_{name}.tree")
#     SeqIO.write(MultipleSeqAlignment(seqs), f"covid/alns/audacity_{name}.fa", "fasta")


# florida = AlignIO.read("covid/alns/fl.fa", format="fasta")
# filter_audacity_tree(florida, "fl")

# michigan = AlignIO.read("covid/alns/mi.fa", format="fasta")
# filter_audacity_tree(michigan, "mi")

# usa = AlignIO.read("covid/alns/usa.fa", format="fasta")
# filter_audacity_tree(usa, "usa")

# usa = AlignIO.read("covid/alns/uk.fa", format="fasta")
# filter_audacity_tree(usa, "uk")
# -

# ## Helper Functions

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
# -

# ## Plot Function and Variables

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


def plot_external(ax, region):
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
left_end = pd.Timestamp("2020-01-01")
right_end = pd.Timestamp("2021-12-08")

regions = ["florida", "michigan", "usa", "uk"]
# -

# ## Run Analysis

fasta = {}
fasta["florida"] = AlignIO.read("covid/audacity_alns/audacity_fl.fa", format="fasta")
fasta["michigan"] = AlignIO.read("covid/audacity_alns/audacity_mi.fa", format="fasta")
fasta["usa"] = AlignIO.read("covid/audacity_alns/audacity_usa.fa", format="fasta")
fasta["uk"] = AlignIO.read("covid/audacity_alns/audacity_uk.fa", format="fasta")

data = {k: SeqData(v, right_end=right_end) for k, v in fasta.items()}

sns.kdeplot(data["florida"].dates, alpha=0.7, label="Florida")
sns.kdeplot(data["michigan"].dates, alpha=0.7, label="Michigan")
sns.kdeplot(data["usa"].dates, alpha=0.7, label="USA")
sns.kdeplot(data["uk"].dates, alpha=0.7, label="UK")
plt.xticks(rotation=30)
_ = plt.legend(loc="upper left")
plt.savefig("covid/figures/all/sample_times.pdf", format="pdf")

# +
# %%time 
n_tips = 200
temp_folder = "covid/temp"
tree_path = "covid/temp/subsample.trees"

stratified = False
stratify_by = None

prep_times = {}
for k, v in data.items():
    audacity = True
    if "florida" in k:
        audacity_tree_path = "covid/trees/global_fl.tree"
    elif "michigan" in k:
        audacity_tree_path = "covid/trees/global_mi.tree"
    elif "usa" in k: 
        audacity_tree_path = "covid/trees/global_usa.tree"
    else:
        audacity = False
        audacity_tree_path = "covid/trees/global_uk.tree"
        
    start = time.time()
    if stratified:
        stratify_by = defaultdict(list)
        for s, d in zip(v.sids, v.dates):
            days = (v.max_date - d).days
            stratify_by[(d.year, (d.month-1)//3)].append(s)
            
    n_trees = min(int(np.ceil(v.n / n_tips)), 50)
#     n_trees = 5
    v.prep_data(
        n_tips,
        n_trees,
        temp_folder,
        tree_path,
        audacity=audacity,
        audacity_tree_path=audacity_tree_path,
        stratified=stratified,
        stratify_by=stratify_by
    )
    prep_times[k] = time.time() - start
# -

# ## Uninformative smoothness

# +
m = 50

for k, v in data.items():
    global_flows, local_flows = default_flows(v, m, rate)
    v.setup_flows(global_flows, local_flows)

rng = jax.random.PRNGKey(6)
res = {}
n_iter = 10
threshold = 0.001
step_size = 1.0
opt_times = {}
for k, v in data.items():
    start = time.time()
    res[k] = v.loop(
        priors["original"], rng, n_iter, step_size=step_size, threshold=threshold
    )
    opt_times[k] = time.time() - start

import pickle
with open('prep_times.pickle', 'wb') as handle:
    pickle.dump(prep_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('opt_times.pickle', 'wb') as handle:
    pickle.dump(opt_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

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


# +
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = axs.flatten()

plot_by_param(res, data, axs, m, regions, "R")
for i, r in enumerate(regions):
    plot_external(axs[i], r)

axs[0].legend(loc="lower right")
    
fig.set_size_inches(20, 14)
plt.tight_layout()
fig.savefig("covid/figures/all/smooth_R.pdf", format="pdf")

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
fig.savefig("covid/figures/all/smooth_s.pdf", format="pdf")
# -

# ## Less Smoothing

# +
m = 50

for k, v in data.items():
    global_flows, local_flows = default_flows(v, m, rate)
    v.setup_flows(global_flows, local_flows)

rng = jax.random.PRNGKey(6)
res_less = {}
n_iter = 10
threshold = 0.001
step_size = 1.0
for k, v in data.items():
    res_less[k] = v.loop(
        priors["less"], rng, n_iter, step_size=step_size, threshold=threshold
    )

# +
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = axs.flatten()

plot_by_param(res_less, data, axs, m, regions, "R")
for i, r in enumerate(regions):
    plot_external(axs[i], r)

axs[0].legend(loc="lower right")
    
fig.set_size_inches(20, 14)
plt.tight_layout()
fig.savefig("covid/figures/all/less_smooth_strong_prior_R.pdf", format="pdf")

# +
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)

for r in regions:
    start, top, end, x0 = plot_helper(res_less[r], data[r], 200)
    if r == "usa" or r == "uk":
        title = r.upper()
    else:
        title = r.title()
    plot_one(res_less[r], ax, "s", m, start, top, end, x0, title, "fill", "")

plt.legend()
fig.savefig("covid/figures/all/less_smooth_strong_prior_s.pdf", format="pdf")
# -

# ## Biased Sampling

data2 = {k: SeqData(v, right_end=right_end) for k, v in fasta.items()}

# +
n_tips = 200
temp_folder = "covid/temp"
tree_path = "covid/temp/subsample.trees"

stratified = True
stratify_by = None

for k, v in data2.items():
    audacity = True
    if "florida" in k:
        audacity_tree_path = "covid/trees/global_fl.tree"
    elif "michigan" in k:
        audacity_tree_path = "covid/trees/global_mi.tree"
    elif "usa" in k: 
        audacity_tree_path = "covid/trees/global_usa.tree"
    else:
        audacity = False
        audacity_tree_path = "covid/trees/global_uk.tree"

    if stratified:
        stratify_by = defaultdict(list)
        for s, d in zip(v.sids, v.dates):
            days = (v.max_date - d).days
            stratify_by[(d.year, (d.month - 1) // 3)].append(s)

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
    )

# +
m = 50

for k, v in data2.items():
    global_flows, local_flows = fixed_origin_flows(v, m, rate)
    v.setup_flows(global_flows, local_flows)

rng = jax.random.PRNGKey(6)
res_bias = {}
n_iter = 10
threshold = 0.001
step_size = 1.0
for k, v in data2.items():
    res_bias[k] = v.loop(
        priors["bias"], rng, n_iter, step_size=step_size, threshold=threshold
    )

# +
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs = axs.flatten()

plot_by_param(res_bias, data2, axs, m, regions, "R")
for i, r in enumerate(regions):
    plot_external(axs[i], r)

axs[0].legend(loc="lower right")
    
fig.set_size_inches(20, 14)
plt.tight_layout()
fig.savefig("covid/figures/all/bias_R_strong_prior_s.pdf", format="pdf")

# +
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)


for r in regions:
    start, top, end, x0 = plot_helper(res_bias[r], data2[r], 200)
    if r == "usa" or r == "uk":
        title = r.upper()
    else:
        title = r.title()
    plot_one(res_bias[r], ax, "s", m, start, top, end, x0, title, "fill", "")

plt.legend()
fig.savefig("covid/figures/all/bias_s_strong_prior_s.pdf", format="pdf")
# -

# ## Parameter Tuning (ntrees and ntips)

# fasta = {}
# fasta["usa"] = AlignIO.read("covid/audacity_alns/audacity_usa.fa", format="fasta")
# fasta["uk"] = AlignIO.read("covid/audacity_alns/audacity_uk.fa", format="fasta")

# # +
# nts = [10, 25, 50, 100, 150]
# # nts = [10, 25]
# regions = ["usa", "uk"]
# data_trees = {f"{b}_{n}": SeqData(fasta[b], right_end=right_end) for n in nts for b in regions}

# n_tips = 200
# temp_folder = "covid/temp"
# tree_path = "covid/temp/subsample.trees"
# stratified = False

# for b in regions:
#     if "usa" in b: 
#         audacity = True
#         audacity_tree_path = "covid/trees/global_usa.tree"
#     else:
#         audacity = False
#         audacity_tree_path = "covid/trees/global_uk.tree"

#     for n in nts:
#         n_trees = n
#         data_trees[f"{b}_{n}"].prep_data(
#             n_tips,
#             n_trees,
#             temp_folder,
#             tree_path,
#             audacity=audacity,
#             audacity_tree_path=audacity_tree_path,
#             stratified=stratified,
#         )

# # +
# m = 50

# for k, v in data_trees.items():
#     global_flows, local_flows = default_flows(v, m, rate)
#     v.setup_flows(global_flows, local_flows)

# rng = jax.random.PRNGKey(6)
# res_trees = {}
# n_iter = 10
# threshold = 0.001
# step_size = 1.0
# for k, v in data_trees.items():
#     res_trees[k] = v.loop(
#         priors["original"], rng, n_iter, step_size=step_size, threshold=threshold
#     )

# # +
# fig, axs = plt.subplots(2)
# fig.set_size_inches(10, 7)

# for i, b in enumerate(regions):
#     if "usa" in b: 
#         audacity = True
#         audacity_tree_path = "covid/trees/global_usa.tree"
#     else:
#         audacity = False
#         audacity_tree_path = "covid/trees/global_uk.tree"
#     for n in nts:
#         start, top, end, x0 = plot_helper(
#             res_trees[f"{b}_{n}"], data_trees[f"{b}_{n}"], 200
#         )
#         title = b.upper()
#         plot_one(
#             res_trees[f"{b}_{n}"],
#             axs[i],
#             "R",
#             m,
#             start,
#             top,
#             end,
#             x0,
#             n,
#             "lines",
#             title,
#         )

# for ax in axs:
#     ax.set_xlim(left_end, right_end)
# axs[0].legend(loc="lower left")

# fig.savefig("covid/figures/all/n_trees_R.pdf", format="pdf")

# # +
# fig, axs = plt.subplots(2)
# fig.set_size_inches(10, 7)

# for i, b in enumerate(regions):
#     for n in nts:
#         start, top, end, x0 = plot_helper(
#             res_trees[f"{b}_{n}"], data_trees[f"{b}_{n}"], 200
#         )
#         title = b.upper()
#         plot_one(
#             res_trees[f"{b}_{n}"],
#             axs[i],
#             "s",
#             m,
#             start,
#             top,
#             end,
#             x0,
#             n,
#             "lines",
#             title,
#         )

# for ax in axs:
#     ax.set_xlim(left_end, right_end)
#     ax.set_yscale('log')
# axs[0].legend(loc="lower left")

# fig.savefig("covid/figures/all/n_trees_s.pdf", format="pdf")

# # +
# n_tipss = [50, 100, 200, 400]
# # n_tipss = [50, 100]
# regions = ["usa", "uk"]
# data_tips = {f"{b}_{n}": SeqData(fasta[b], right_end=right_end) for n in n_tipss for b in regions}

# n_trees = 50
# temp_folder = "covid/temp"
# tree_path = "covid/temp/subsample.trees"
# stratified = False

# for b in regions:
#     if "usa" in b: 
#         audacity = True
#         audacity_tree_path = "covid/trees/global_usa.tree"
#     else:
#         audacity = False
#         audacity_tree_path = "covid/trees/global_uk.tree"
#     for n in n_tipss:
#         n_tips = n
#         data_tips[f"{b}_{n}"].prep_data(
#             n_tips,
#             n_trees,
#             temp_folder,
#             tree_path,
#             audacity=audacity,
#             audacity_tree_path=audacity_tree_path,
#             stratified=stratified,
#         )

# # +
# m = 50

# for k, v in data_tips.items():
#     global_flows, local_flows = default_flows(v, m, rate)
#     v.setup_flows(global_flows, local_flows)

# rng = jax.random.PRNGKey(6)
# res_tips = {}
# n_iter = 10
# threshold = 0.001
# step_size = 1.0
# for k, v in data_tips.items():
#     res_tips[k] = v.loop(
#         _params_prior_loglik, rng, n_iter, step_size=step_size, threshold=threshold
#     )

# # +
# fig, axs = plt.subplots(2)
# fig.set_size_inches(10, 7)

# for i, b in enumerate(regions):
#     for n in n_tipss:
#         start, top, end, x0 = plot_helper(
#             res_tips[f"{b}_{n}"], data_tips[f"{b}_{n}"], n
#         )
#         title = b.upper()
#         plot_one(
#             res_tips[f"{b}_{n}"], axs[i], "R", m, start, top, end, x0, n, "lines", title
#         )

# for ax in axs:
#     ax.set_xlim(left_end, right_end)
# axs[0].legend(loc="lower left")

# fig.savefig("covid/figures/all/n_tips_R.pdf", format="pdf")

# # +
# fig, axs = plt.subplots(2)
# fig.set_size_inches(10, 7)

# for i, b in enumerate(regions):
#     for n in n_tipss:
#         start, top, end, x0 = plot_helper(
#             res_tips[f"{b}_{n}"], data_tips[f"{b}_{n}"], n
#         )
#         title = b.upper()
#         plot_one(
#             res_tips[f"{b}_{n}"], axs[i], "s", m, start, top, end, x0, n, "lines", title
#         )

# for ax in axs:
#     ax.set_xlim(left_end, right_end)
# axs[0].legend(loc="lower left")

# fig.savefig("covid/figures/all/n_tips_s.pdf", format="pdf")
# # -

# for i, b in enumerate(regions):
#     for n in n_tipss:
#         sns.kdeplot(
#             res_tips[f"{b}_{n}"].global_posteriors["origin"].flatten(), label=f"{b}_{n}"
#         )
# plt.xlim(0, 2)
# plt.legend()

# # ## BEAST

# # +
# beast = {}
# ks = ["florida", "michigan", "usa"]
# # ks = ["florida", "michigan"]
# ns = [100, 500]
# sampling = ["random", "stratified"]

# for k in ks:
#     for n in ns:
#         for s in sampling:
#             beast[f"{k}_{n}_{s}"] = AlignIO.read(
#                 f"covid/beast/covid/{k}_{n}_{s}.fa", format="fasta"
#             )

# dates = defaultdict(list)
# names = defaultdict(list)
# for k, v in beast.items():
#     for s in v:
#         date = datetime.strptime(s.id.split("_")[-1], "%Y-%m-%d")
#         dates[k].append(date)
#         names[k].append(s.id)
# # -

# beast_data = {k: SeqData(v, names=names[k], dates=dates[k]) for k, v in beast.items()}


# def parse_log(fname):
#     with open(fname) as fp:
#         for line in fp:
#             line = line.strip()
#             if line.startswith("Sample"):
#                 keys = line.split("\t")
#                 sample_dict = {k: [] for k in keys}
#                 idx_dict = {j: k for j, k in enumerate(keys)}
#             elif not line.startswith("#") and len(line) != 0:
#                 for j, h in enumerate(line.split("\t")):
#                     sample_dict[idx_dict[j]].append(float(h))

#         for k, v in sample_dict.items():
#             sample_dict[k] = np.array(v[1:])
#     return sample_dict


# short_dicts = {}
# for k in ks:
#     for n in ns:
#         for s in sampling:
#             short_dicts[f"{k}_{n}_{s}"] = parse_log(
#                 f"covid/beast/covid/short/logs/{k}_{n}_{s}.log"
#             )

# long_dicts = {}
# for k in ks:
#     for n in ns:
#         for s in sampling:
#             long_dicts[f"{k}_{n}_{s}"] = parse_log(
#                 f"covid/beast/covid/long/logs/{k}_{n}_{s}.log"
#             )


# def plot_beast(sample_dict, ax, param, end, label, ci, title):
#     x0 = np.linspace(2020, end, 100)

#     ys = []
#     m = 0
#     for k, v in sample_dict.items():
#         if param in k:
#             ys.append(v)
#             m += 1
#     ys = np.array(ys).T
#     y0 = []

#     try:
#         top = sample_dict["origin_BDSKY_Serial"]
#     except:
#         top = sample_dict["origin_BDSKY_Contemp"]
#     for x, y in zip(top, ys):
#         intervals = np.linspace(0, x, m + 1)
#         t = (end - intervals)[::-1]
#         y0.append(interp1d(t[1:], y, kind="nearest", fill_value="extrapolate")(x0))
#     q25, q50, q75 = np.quantile(np.array(y0), q=[0.025, 0.5, 0.975], axis=0)

#     color = next(ax._get_lines.prop_cycler)["color"]
#     ax.plot(x0, q50, label=label, color=color)
#     if ci == "fill":
#         ax.fill_between(x0, q25, q75, alpha=0.1, label="_nolegend_", color=color)
#     if ci == "lines":
#         ax.plot(x0, q25, "--", label="_nolegend_", alpha=0.25, color=color)
#         ax.plot(x0, q75, "--", label="_nolegend_", alpha=0.25, color=color)
#     # plt.xlim(reversed(plt.xlim()))
#     ax.set_xlabel("Year")
#     ax.set_title(title)


# # +
# def beast_comparison_plot_helper(k, param, ax, s, length="short"):
#     ns = [100, 500]
#     bdsky_label = {"random": "Serial", "stratified": "Contemp"}
#     plot_label = {"random": "random", "stratified": "contemporary"}
#     color = next(ax._get_lines.prop_cycler)["color"]
#     if k != "usa" and param in "reproductiveNumber_BDSKY":
#         ax.plot(external[k]["time"], external[k]["rt"], label="Surveillance Data")
#         ax.fill_between(
#             external[k]["time"],
#             external[k]["rt_lower"],
#             external[k]["rt_upper"],
#             alpha=0.1,
#             label="_nolegend_",
#         )
#     else:
#         color = next(ax._get_lines.prop_cycler)["color"]

#     for n in ns:
#         if length == "short":
#             logs = short_dicts[f"{k}_{n}_{s}"]
#         else:
#             logs = long_dicts[f"{k}_{n}_{s}"]

#         plot_beast(
#             logs,
#             ax,
#             f"{param}_{bdsky_label[s]}",
#             beast_data[f"{k}_{n}_{s}"].end,
#             label=f"BEAST - {n} tips, {plot_label[s]}",
#             ci="fill",
#             title="",
#         )


# def beast_comparison_plot(k, param, length):
#     if param == "reproductiveNumber_BDSKY":
#         fig, axs = plt.subplots(2)
#         beast_comparison_plot_helper(k, param, axs[0], "random", length)
#         beast_comparison_plot_helper(k, param, axs[1], "stratified", length)
#         height = 7
#     else:
#         fig, ax = plt.subplots()
#         beast_comparison_plot_helper(k, param, ax, "random", length)
#         axs = [ax]
#         height = 3.5

#     for ax in axs:
#         if param == "reproductiveNumber_BDSKY":
#             ax.set_ylim(0, 2)
#             ax.set_xlim(2020, 2021.9)
#             ax.legend(loc="lower left")

#     fig.set_size_inches(10, height)
#     if k == "usa":
#         part1 = k.upper()
#     else:
#         part1 = k.title()

#     vbsky_label = {"reproductiveNumber_BDSKY": "R", "samplingProportion_BDSKY": "s"}

#     _title = f"{part1} - {vbsky_label[param]} - {length.title()}"
#     fig.suptitle(_title)

#     fig.savefig(
#         f"covid/figures/beast/{k}_{vbsky_label[param]}_{length}.pdf", format="pdf"
#     )


# # -

# for k in ["florida", "michigan", "usa"]:
#     for length in ["short", "long"]:
#         for param in ["reproductiveNumber_BDSKY", "samplingProportion_BDSKY"]:
#             beast_comparison_plot(k, param, length)

# # ## Case count data

# usa_df = pd.read_csv("covid/us_cases.csv")
# states_cases = pd.read_csv("covid/states_cases.csv")
# florida_df = states_cases.loc[states_cases["state"] == "Florida"]
# michigan_df = states_cases.loc[states_cases["state"] == "Michigan"]

# florida_df.index[0]


# def format_df(df):
#     df["daily"] = df["cases"].diff()
#     df.loc[df.index[0], "daily"] = df.loc[df.index[0], "cases"]
#     df["date"] = pd.to_datetime(df["date"])
#     df["decimal_date"] = df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365
#     df["seven_day"] = df["daily"].rolling(7).mean()


# format_df(usa_df)
# format_df(florida_df)
# format_df(michigan_df)

# florida_df.loc[florida_df["seven_day"] < 0]

# florida_df.loc[
#     (florida_df["decimal_date"] < 2021.427397) & (florida_df["decimal_date"] > 2021.415)
# ]

# fig, axs = plt.subplots(3)
# titles = ["Florida", "Michigan", "USA"]
# for i, df in enumerate([florida_df, michigan_df, usa_df]):
#     axs[i].plot(df["decimal_date"], df["seven_day"])
#     axs[i].set_title(titles[i])
#     axs[i].set_ylim(bottom=0)
# fig.set_size_inches(10, 10.5)
# fig.savefig(f"covid/figures/all/case_count.pdf", format="pdf")
