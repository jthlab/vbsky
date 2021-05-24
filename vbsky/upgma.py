import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations

from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo.Consensus import bootstrap_trees, bootstrap
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

def get_distance_matrix(aln, times, model="identity", single_theta=False):
    calculator = DistanceCalculator(model)
    dm = calculator.get_distance(aln)
    y = np.concatenate([mat[:-1] for mat in dm.matrix])

    n = len(aln)
    nc2 = int(n * (n-1) / 2)
    inds = np.tril_indices(n,-1)

    dt = np.abs(times[:,None] - times)[inds]
    
    uniq_times = np.sort(np.unique(times))[::-1]
    times_dict = {uniq_times[i]:i for i in range(len(uniq_times))}

    if single_theta:
        reg = LinearRegression(fit_intercept=True).fit(dt.reshape(-1,1),y)
    else:
        larger = np.where(times[inds[0]] >= times[inds[1]], inds[0], inds[1])
        lsm = np.array([times_dict[times[ell]] for ell in larger])
        X = np.zeros((nc2, len(times_dict)))
        X[(np.arange(nc2), lsm)] = 1
        X = np.column_stack((X,dt))
        reg = LinearRegression(fit_intercept=False).fit(X,y)

    omega = reg.coef_[-1]

    adjusted_y = y + omega * (times[inds[0]] + times[inds[1]]) 
    mat = np.zeros((n,n))
    mat[inds] = adjusted_y
    mat_formatted = [r[:i+1].tolist() for i,r in enumerate(mat)]
    dm.matrix = mat_formatted
    return dm

def supgma_tree(aln, times, model="identity", single_theta=False):
    constructor = DistanceTreeConstructor()
    dm = get_distance_matrix(aln, times, model, single_theta)
    tree = constructor.upgma(dm)
    return tree

def supgma_bootstrap(aln, times, model="identity", single_theta=False, bootstraps=100):
    alns = bootstrap(aln, bootstraps)
    trees = [supgma_tree(a, times, model, single_theta) for a in alns]
    return trees

def supgma_subsample(aln, times_dict, model="identity", single_theta=False, n_tips=50, n_trees=5):
    alns = []
    trees = []

    for _ in range(n_trees):
        inds = np.random.choice(len(aln._records), size=n_tips, replace=False)
        aln_sample = MultipleSeqAlignment([aln[int(inds[0])]])
        for i in range(1, n_tips):
            aln_sample.append(aln[int(inds[i])])
        times_sample = np.array([times_dict[s.name] for s in aln_sample])
        trees.append(supgma_tree(aln_sample, times_sample, model, single_theta))
        alns.append(aln_sample)

    return trees, alns

    


def main():
    a = SeqRecord(Seq("AACGTGGCCACAT"), id="a")
    b = SeqRecord(Seq("AAGGTCGCCACAC"), id="b")
    c = SeqRecord(Seq("CAGTTCGCCACAA"), id="c")
    d = SeqRecord(Seq("GAGATTTCCGCCT"), id="d")
    times = np.array([0,0,1,1])
    aln = MultipleSeqAlignment([a,b,c,d])

    
    tree = supgma_tree(aln, times)
    trees = supgma_bootstrap(aln, times)

if __name__ == "__main__":
    main()