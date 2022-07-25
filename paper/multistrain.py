# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %load_ext nb_black
# # %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")

import numpy as np
import pandas as pd

import jax.numpy as jnp
import jax
from jax import vmap
from jax.scipy.special import xlogy

jax.config.update("jax_enable_x64", True)
import scipy.linalg

from collections import defaultdict
import gzip
import pickle

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from ete3 import Tree
from datetime import datetime, MINYEAR

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

# ## Helper Functions

# +

def _params_prior_loglik_bias(params):
    ll = 0
    tau = {"R": params["precision_R"][0], "s": params["precision_s"][0]}
    ll += jax.scipy.stats.gamma.logpdf(tau["R"], a=10000, scale=0.01)
    ll += jax.scipy.stats.gamma.logpdf(tau["s"], a=10000, scale=0.01)

    ll += jax.scipy.stats.beta.logpdf(params["s"], 2, 98).sum()

    mus = [0]
    sigmas = [1]

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
        origin=Constant(0.1),
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

# -

# ## Import Covid sequence data

fasta = {}
regions = ["florida", "michigan", "usa", "uk"]
short = {"michigan": "mi", "florida": "fl", "usa":"usa", "uk":"uk"}
variants = ["alpha", "delta", "omicron"]
for r in regions:
    fasta[r] = {}
    for v in variants:
        fasta[r][v] = AlignIO.read(
            f"covid/strains/audacity_alns/audacity_{short[r]}_{v}.fa", format="fasta"
        )

strains = {}
for r in regions:
    strains[r] = {}
    for v in variants:
        if v == "alpha":
            left_end = pd.Timestamp("2020-09-01")
        elif v == "delta":
            left_end = pd.Timestamp("2021-02-01")
        else:
            left_end = pd.Timestamp("2021-11-01")
        strains[r][v] = SeqData(fasta[r][v], left_end = left_end)

# ## Run Analysis

# +
n_tips = 200
temp_folder = "covid/temp"
tree_path = "covid/temp/subsample.trees"
audacity = True
stratified = False
stratify_by = None

for k1 in strains.keys():
    for k2, v in strains[k1].items():
        print(k1, k2)
        audacity_tree_path = f"covid/trees/global_{short[k1]}_{k2}.tree"

        if stratified:
            stratify_by = defaultdict(list)
            for s, d in zip(v.sids, v.dates):
                days = (v.max_date - d).days
                stratify_by[(d.year, (d.month - 1) // 3)].append(s)

        n_trees = min(int(np.ceil(v.n / n_tips)), 20)
        #         n_trees = 5

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
rate = 1.12e-3
m = 20

for k in strains.keys():
    for v in strains[k].values():
        global_flows, local_flows = fixed_origin_flows(v, m, rate)
        v.setup_flows(global_flows, local_flows)

rng = jax.random.PRNGKey(6)
res = {}
n_iter = 10
step_size = 1.0
threshold = 0.001
for k in strains.keys():
    res1 = {}
    for k1, v in strains[k].items():
        print(k, k1)
        res1[k1] = v.loop(
            _params_prior_loglik_bias,
            rng,
            n_iter,
            step_size=step_size,
            threshold=threshold,
        )
    res[k] = res1

# -

# ## Plot

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

left_end = pd.Timestamp("2020-09-20")
right_end = pd.Timestamp("2022-03-15")

# +
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

for i, b in enumerate(["florida", "michigan", "usa", "uk"]):
    for strain in ["alpha", "delta", "omicron"]:
        start, top, end, x0 = plot_helper(res[b][strain], strains[b][strain], 200)
        if b == "usa" or b == "uk":
            title = b.upper()
        else:
            title = b.title()
        plot_one(
            res[b][strain],
            axs[i],
            "R",
            m,
            start,
            top,
            end,
            x0,
            strain.title(),
            "fill",
            title,
        )

axs[0].legend(loc="lower left")
fig.set_size_inches(20, 14)
plt.tight_layout()

fig.savefig("covid/figures/strains/strain_R_audacity.pdf", format="pdf")

# +
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

for i, b in enumerate(["florida", "michigan", "usa", "uk"]):
    for strain in ["alpha", "delta", "omicron"]:
        start, top, end, x0 = plot_helper(res[b][strain], strains[b][strain], 200)
        if b == "usa" or b == "uk":
            title = b.upper()
        else:
            title = b.title()
        plot_one(
            res[b][strain],
            axs[i],
            "s",
            m,
            start,
            top,
            end,
            x0,
            strain.title(),
            "fill",
            title,
        )

axs[0].legend(loc="lower left")
fig.set_size_inches(20, 14)
plt.tight_layout()

fig.savefig("covid/figures/strains/strain_s_audacity.pdf", format="pdf")
# -

# ## BEAST

# +
# def sample_beast_tips(
#     k1, k2, n=200, stratified=False, stratify_by=None, stratified_handle=""
# ):
#     d = strains[k1][k2]
#     dates_dict = {d1: d2 for d1, d2 in zip(d.sids, d.dates)}
#     if stratified:
#         if stratify_by is None:
#             stratify_by = d.sample_months
#         key_list = [k for k, v in stratify_by.items() if len(v) > 50]

#         seqs = []
#         ell = len(key_list)
#         mod = n % ell
#         floor = n // ell
#         for i in range(ell):
#             sequence_pool = stratify_by[key_list[i]]
#             if i < mod:
#                 size = floor + 1
#             else:
#                 size = floor
#             seqs.append(np.random.choice(sequence_pool, size=size, replace=False))
#         seqs = np.concatenate(seqs)
#         aln_sample = MultipleSeqAlignment([d.seqs[s] for s in seqs])
#     else:
#         inds = np.random.choice(len(d.aln._records), size=n, replace=False)
#         aln_sample = MultipleSeqAlignment([d.aln[int(inds[0])]])
#         for j in range(1, n):
#             aln_sample.append(d.aln[int(inds[j])])

#     for r in aln_sample:
#         sp = r.description.split("|")
#         r.id = sp[0] + "_" + dates_dict[r.name].strftime("%Y-%m-%d")
#     #         r.description = ""

#     handle = f"covid/beast/multistrain/{k1}_{k2}_beast"
#     if stratified:
#         handle += stratified_handle
#     handle += ".fa"
#     with open(handle, "w") as output_handle:
#         count = AlignIO.write(aln_sample, output_handle, "fasta")

# +
# stratified = False
# # stratified_handle = "_quarters"
# stratified_handle = ""
# n = 200

# for k1, v in strains.items():
#     #     stratify_by = defaultdict(list)
#     #     for s, d in zip(v.sids, v.dates):
#     #         days = (v.max_date - d).days
#     #         stratify_by[(d.year, (d.month - 1) // 3)].append(s)
#     for k2 in v.keys():
#         # sample_beast_tips(k, n, stratified, stratify_by, stratified_handle)
#         sample_beast_tips(k1, k2, n, stratified)

# +
# import xml.etree.ElementTree as ET

# +
# def edit_template(k1, k2):
#     aln = AlignIO.read(f"covid/beast/multistrain/{k1}_{k2}_beast.fa", "fasta")

#     tree = ET.parse(f"covid/beast/multistrain/template.xml")
#     root = tree.getroot()

#     value = ""
#     for rec, child in zip(aln, root.find("data")):
#         seq = str(rec.seq)
#         child.set("id", f"seq_{rec.name}")
#         child.set("taxon", rec.name)
#         child.set("value", seq)
#         value += f"{rec.name}={rec.name.split('_')[1]},"
#     root.find("run").find("state").find("tree").find("trait").set("value", value[:-1])

#     for log in root.find("run").findall("logger"):
#         if log.get("id") == "tracelog":
#             log.set("fileName", f"covid/beast/multistrain/logs/{k1}_{k2}_beast.log")
#         if log.get("id") == "treelog.t:template":
#             log.set("fileName", f"covid/beast/multistrain/trees/{k1}_{k2}_beast.log")

#     xml_str = ET.tostring(root, encoding="unicode")
#     #     xml_str.replace('sim6"', f'sim{i}"')
#     with open(f"covid/beast/multistrain/{k1}_{k2}_beast.xml", "w") as xml:
#         xml.write(xml_str)

# +
# for k1 in strains.keys():
#     for k2 in strains[k1].keys():
#         edit_template(k1, k2)
