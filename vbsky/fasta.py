import dataclasses
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import StringIO
from typing import Dict, NamedTuple, Tuple, Union
from collections import namedtuple, defaultdict
import multiprocessing as mp
import sh

import jax.numpy as jnp
import numpy as np
from numpy import linalg as la

from functools import partial

import jax
from jax import vmap, grad, jit, value_and_grad
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves, tree_flatten, tree_multimap
from jax.experimental import optimizers

from Bio import SeqIO, AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from ete3 import Tree

from datetime import datetime, MINYEAR, timedelta

from tqdm import tqdm

from .upgma import get_distance_matrix
from .tree_data import TreeData
from .util import RateFunction, TipData
from .substitution import encode_partials, HKY, JC69
from .optim import loss, unpack

from .prob import VF
from .prob.distribution import PointMass
from .prob.transform import (
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
from .prob.distribution import Constant
from .prob import arf

pos = Compose(DiagonalAffine, Exp)
plus = Compose(DiagonalAffine, Positive)
z1 = Compose(DiagonalAffine, ZeroOne)

rate = 1.10e-3
days_in_year = 365.245
local_names = {"proportions", "root_proportion"}

ret = namedtuple("ret", ["global_posteriors", "local_posteriors", "fs"])

if mp.cpu_count() == 1:
    n_processes = 1
else:
    n_processes = mp.cpu_count() - 1

@value_and_grad
def _loss(p, *args):
    f1 = loss(p, *args)
    f2 = ravel_pytree(p)[0]
    return f1  # + jnp.dot(f2, f2)

def get_gisaid_dates(aln):
    dates = []
    for s in aln:
        desc = s.description.split("|")
        date = desc[2]

        if date[-5:-3] == "00":
            continue

        if date[-2:] == "00":
            date = date.replace("-00", "-15")
        date = datetime.strptime(date, "%Y-%m-%d")
        delta = timedelta(days=2 - date.weekday())
        date += delta
        dates.append(date)
    return dates

def get_gisaid_names(aln):
    names = []
    for s in aln:
        desc = s.description.split("|")
        names.append(desc[1])
    return names

class SeqData:
    def __init__(
        self, aln, names=None, dates=None, left_end=datetime(MINYEAR, 1, 1), right_end=datetime.today(),
    ):
        self.sids = []
        self.sids_dict = {}
        self.seqs = {}
        self.dates = []

        if names is None:
            names = get_gisaid_names(aln)
        if dates is None:
            dates = get_gisaid_dates(aln)
        
        self.date_float = isinstance(dates[0], float)

        if self.date_float:
            self.max_date = 0
            self.min_date = float("inf")
            if isinstance(left_end, datetime):
                left_end = 0
            if isinstance(right_end, datetime):
                right_end = float("inf")

        else:
            self.max_date = datetime(MINYEAR, 1, 1)
            self.min_date = datetime.today()

        for s, name, date in zip(aln, names, dates):

            if date >= left_end and date <= right_end:
                self.sids.append(name)
                self.dates.append(date)
                if self.max_date < self.dates[-1]:
                    self.max_date = self.dates[-1]
                if self.min_date > self.dates[-1]:
                    self.min_date = self.dates[-1]
                self.seqs[name] = s 
                self.sids_dict[s.description] = name

        self.sample_times = {}
        self.sample_months = defaultdict(list)

        if self.date_float:
            #fixme no default option for stratify when dates are floats
            for s, d in zip(self.sids, self.dates):
                self.sample_times[s] = self.max_date - d
            self.earliest = self.max_date - self.min_date

        else:
            for s, d in zip(self.sids, self.dates):
                days = (self.max_date - d).days
                self.sample_times[s] = days / days_in_year
                self.sample_months[(d.year, d.month)].append(s)

            self.earliest = (self.max_date - self.min_date).days / days_in_year
        self.aln = MultipleSeqAlignment(list(self.seqs.values()))

    @property
    def n(self):
        return len(self.aln)

    @property
    def end(self):
        if self.date_float:
            end = self.max_date

        else:
            year = self.max_date.year
            year_start = datetime(year, 1, 1)
            end = year + (self.max_date - year_start).days / days_in_year

        return end

    def sample_trees(
        self,
        n_tips,
        n_trees,
        temp_folder,
        tree_path,
        single_theta=False,
        audacity=False,
        audacity_tree_path="",
        stratified=False,
        stratify_by=None
    ):
        r, c = np.tril_indices(n_tips, -1)
        constructor = DistanceTreeConstructor()

        self.tree_path = tree_path
        self.alns = []
        self.trees = []

        print("Readying trees")

        if stratified:
            if stratify_by is None:
                stratify_by = self.sample_months
            key_list = [k for k,v in stratify_by.items() if len(v) > 50]

        p = ProcessPoolExecutor()
        futs = []
        for i in tqdm(range(n_trees)):
            if stratified:
                sequence_pool = stratify_by[key_list[i%len(key_list)]]
                try:
                    names = np.random.choice(sequence_pool, size=n_tips, replace=False).tolist()
                except:
                    names = sequence_pool

            else:
                names = np.random.choice(self.sids, size=n_tips, replace=False).tolist()
            
            aln_sample = MultipleSeqAlignment([self.seqs[s] for s in names])

            AlignIO.write(
                aln_sample,
                f"{temp_folder}/temp.fa",
                "fasta",
            )
            times_sample = np.array([self.sample_times[name] for name in names])

            if audacity:
                futs.append(p.submit(_prune_tree, audacity_tree_path, names))

            else:
                # fixme doesn't parallelize over this currently
                snp_dists = sh.Command("snp-dists")
                try:
                    with open(f"{temp_folder}/tab.tsv", "w") as tsv:
                        tsv.write(str(snp_dists("-b", f"{temp_folder}/temp.fa")))
                except sh.ErrorReturnCode_2 as e:
                    print(e.stderr.decode())

                pw_dist = np.loadtxt(
                    f"{temp_folder}/tab.tsv",
                    skiprows=1,
                    usecols=np.arange(1, n_tips + 1),
                ).astype(int)[r, c]
                dm, omega = get_distance_matrix(
                    pw_dist, names, times_sample, single_theta
                )
                tree = constructor.upgma(dm)
                self.trees.append(tree)

            self.alns.append(aln_sample)

        if futs:
            print("Processing audacity trees")
            for f in tqdm(futs):
                tree = f.result()
                self.trees.append(tree)

        if audacity:
            print("Writing audacity pruned trees")
            with open(tree_path, "w") as file:
                for tree in self.trees:
                    file.write(tree.write() + "\n")
        else:
            print("Writing phylo trees")
            Phylo.write(self.trees, self.tree_path, "newick")

    def process_trees(self):
        print("Processing trees")
        with ProcessPoolExecutor() as p:
            futs = [p.submit(_process_tree, line, self.sample_times) for line in open(self.tree_path, "rt")]
            res = [f.result() for f in tqdm(futs)]
        self.tds = [r[0] for r in res]
        self.node_mappings = [r[1] for r in res]

    def process_tips(self):
        self.tip_data_cs = []
        self.max_partial_count = 0

        print("Readying tip data")
        # Add parallelization
        with ProcessPoolExecutor() as p:
            futs = [p.submit(_process_tips, aln, nm, self.sids_dict) for aln, nm in zip(self.alns, self.node_mappings)]
            res = [f.result() for f in tqdm(futs)]
        self.tip_data_cs = [r[0] for r in res]
        self.max_partial_count = np.max([r[1] for r in res])

    def pad_tips(self):
        for i, tip_data_c in enumerate(self.tip_data_cs):
            pad_size = self.max_partial_count - tip_data_c.partials.shape[0]
            pad = np.tile([1, 1, 1, 1], (pad_size, tip_data_c.partials.shape[1], 1))
            partials = np.concatenate([tip_data_c.partials, pad])
            counts = np.concatenate([tip_data_c.counts, np.zeros(pad_size)])
            self.tip_data_cs[i] = TipData(partials, counts)

    def prep_data(
        self,
        n_tips,
        n_trees,
        temp_folder,
        tree_path,
        single_theta=False,
        audacity=False,
        audacity_tree_path="",
        stratified=False,
        stratify_by=None
    ):
        self.sample_trees(
            n_tips,
            n_trees,
            temp_folder,
            tree_path,
            single_theta,
            audacity,
            audacity_tree_path,
            stratified,
            stratify_by
        )
        self.process_trees()
        self.process_tips()
        self.pad_tips()

    def setup_flows(self, global_flows=None, local_flows=None):
        if global_flows is not None:
            self.global_flows = global_flows
        else:
            self.global_flows = VF(
                origin=Transform(1, pos),
                origin_start=Constant(self.earliest),
                delta=Constant(np.repeat(36.5, 10)),
                R=Transform(10, pos),
                rho_m=Constant(0),
                s=Constant(np.repeat(0.02, 10)),
                precision=Transform(1, pos),
                clock_rate=Constant(rate),
            )
        if local_flows is not None:
            self.local_flows = local_flows
        else:
            self.local_flows = [
                {
                    "proportions": Transform(td.n - 2, z1),
                    "root_proportion": Transform(1, z1),
                }
                for td in self.tds
            ]

    def loop(self, _params_prior_loglik, rng, n_iter=10, step_size=1.0, Q=HKY(2.7), threshold=0.01):
        opt_init, opt_update, get_params = optimizers.adagrad(step_size=step_size)

        @partial(jit, static_argnums=(5, 6))
        def step(
            opt_state,
            local_opt_state,
            rng,
            i,
            M,
            c,
            dbg,
            td,
            tip_data_c,
        ):
            p = get_params(opt_state)
            local_p = get_params(local_opt_state)
            p = p | local_p

            def f(j, tup):
                f_df0, rng = tup
                rng, subrng = jax.random.split(rng)
                f_df1 = _loss(
                    p,
                    self.flows,
                    td,
                    tip_data_c,
                    subrng,
                    Q,
                    c,
                    dbg,
                    True,
                    _params_prior_loglik,
                )
                f_df1 = tree_map(jnp.nan_to_num, f_df1)
                return tree_multimap(jnp.add, f_df0, f_df1), rng

            init = ((0.0, tree_map(jnp.zeros_like, p)), rng)
            f_df, rng = jax.lax.fori_loop(0, M, f, init)
            f, g = tree_map(lambda x: x / M, f_df)
            g = tree_map(jnp.nan_to_num, g)
            global_g = {k: g[k] for k in g if k not in local_names}
            local_g = {k: g[k] for k in g if k in local_names}
            return (
                f,
                g,
                opt_update(i, global_g, opt_state),
                opt_update(i, local_g, local_opt_state),
                rng,
            )

        params = {k: v.params for k, v in self.global_flows.items()}
        params = tree_map(lambda x: np.random.normal(size=x.shape) * 0.1, params)

        local_params = [{k: v.params for k, v in lf.items()} for lf in self.local_flows]
        local_params = [
            tree_map(lambda x: np.random.normal(size=x.shape) * 0.1, lp)
            for lp in local_params
        ]

        opt_state = opt_init(params)
        local_opt_state = [opt_init(lp) for lp in local_params]

        n_trees = len(self.tds)
        fs = [[] for _ in range(n_trees)]
        gs = [[] for _ in range(n_trees)]

        br = False
        prev = np.zeros(len(self.tds))
        with tqdm(total=n_iter * len(self.tds)) as pbar:
            for i in range(n_iter):
                for j, (td, tip_data_c) in enumerate(zip(self.tds, self.tip_data_cs)):
                    self.flows = self.global_flows | self.local_flows[j]
                    
                    f, g, opt_state1, local_opt_state1, rng1 = step(
                        opt_state,
                        local_opt_state[j],
                        rng,
                        i,
                        10,
                        ((True, True), (True, True, True)),
                        False,
                        td,
                        tip_data_c,
                    )
                    assert np.isfinite(f), f
                    if i > 1:
                        prev = np.array([fz[-2] for fz in fs])
                        curr = np.array([fz[-1] for fz in fs])
                        diff = np.abs(prev-curr)/prev
                        if diff.mean() < threshold:
                            break
                            br = True

                    fs[j].append(f)
                    gs[j].append(jnp.linalg.norm(ravel_pytree(g)[0]))
                    opt_state = opt_state1
                    local_opt_state[j] = local_opt_state1
                    rng = rng1

                    pbar.update(1)
                

                if br:
                    break
                

        p_star = get_params(opt_state)
        local_p_star = [get_params(lop) for lop in local_opt_state]

        priors, global_posteriors = [
            unpack(
                {
                    k: self.global_flows[k].sample(r, pp[k], 10000)
                    for r, k in zip(
                        jax.random.split(rr, len(self.global_flows)), self.global_flows
                    )
                }
            )
            for rr, pp in zip(jax.random.split(rng), (params, p_star))
        ]

        local_posteriors = []
        for j, flows in enumerate(self.local_flows):
            _, lp = [
                {
                    k: flows[k].sample(r, pp[k], 10000)
                    for r, k in zip(jax.random.split(rr, len(flows)), flows)
                }
                for rr, pp in zip(
                    jax.random.split(rng), (local_params[j], local_p_star[j])
                )
            ]
            local_posteriors.append(lp)

        return ret(global_posteriors, local_posteriors, fs)

def _prune_tree(audacity_tree_path, names):
    tree = Tree(audacity_tree_path, format=1)
    tree.prune(names, preserve_branch_length=True)
    return tree

def _process_tree(line, sample_times):
    td, node_mapping = TreeData.from_newick(
        line.strip(), return_node_mapping=True
    )
    reverse_node_mapping = dict(map(reversed, node_mapping.items()))
    td.sample_times[:] = [
        sample_times[reverse_node_mapping[i]] for i in range(td.n)
    ]
    return td, node_mapping

def _process_tips(msa, node_mapping, sids_dict):
    sids, seqs = zip(*[(s.description, str(s.seq)) for s in msa])
    sids = [sids_dict[s] for s in sids]
    tip_partials0 = np.array([encode_partials(seq) for seq in seqs]).transpose(
        [1, 0, 2]
    )
    perm = np.argsort([node_mapping[s] for s in sids])
    tip_partials = tip_partials0[:, perm]
    tip_data_c = TipData(*np.unique(tip_partials, axis=0, return_counts=True))
    return tip_data_c, tip_data_c.partials.shape[0]