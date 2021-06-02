import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations

from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceCalculator, DistanceTreeConstructor
from Bio.Phylo.Consensus import bootstrap_trees, bootstrap
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

def get_distance_matrix(aln, times, single_theta=False):
    """
    Get DistanceMatrix object from alignment and sample times

    Notes:
        See Step 1 of Drummond and Rodrigo (2000)
    """

    names = [s.name for s in aln]
    n = len(aln)
    nc2 = int(n * (n-1) / 2)

    r, c = np.tril_indices(len(aln), -1)
    np_aln = np.array(aln)
    pw_dist = (
        (np_aln[r] != np_aln[c])
        * (np_aln[r] != "-")
        * (np_aln[c] != "-")
        * (np_aln[r] != "N")
        * (np_aln[c] != "N")
    )
    y = pw_dist.sum(axis=1) / len(np_aln[0])

    dt = np.abs(times[:,None] - times)[(r,c)]
    
    uniq_times = np.sort(np.unique(times))[::-1]
    times_dict = {uniq_times[i]:i for i in range(len(uniq_times))}

    if single_theta:
        reg = LinearRegression(fit_intercept=True).fit(dt.reshape(-1,1),y)
    else:
        larger = np.where(times[r] >= times[c], r, c)
        lsm = np.array([times_dict[times[ell]] for ell in larger])
        X = np.zeros((nc2, len(times_dict)))
        X[(np.arange(nc2), lsm)] = 1
        X = np.column_stack((X,dt))
        reg = LinearRegression(fit_intercept=False).fit(X,y)

    omega = reg.coef_[-1]

    adjusted_y = y + omega * (times[r] + times[c] - 2 * uniq_times[-1]) 
    mat = np.zeros((n,n))
    mat[(r,c)] = adjusted_y
    mat_formatted = [row[:i+1].tolist() for i,row in enumerate(mat)]
    dm = DistanceMatrix(names)
    dm.matrix = mat_formatted
    return dm, omega

def supgma_tree(aln, times, single_theta=False):
    """
    Returns single tree using serial upgma
    """
    constructor = DistanceTreeConstructor()
    dm, omega = get_distance_matrix(aln, times, single_theta)
    tree = constructor.upgma(dm)
    for a, time in zip(aln, times):
        leaf = next(tree.find_clades(a.name))
        leaf.branch_length -= omega * time 
    return tree

def supgma_bootstrap(aln, times, single_theta=False, bootstraps=100):
    """
    Returns bootstrapped trees using serial upgma
    """
    alns = bootstrap(aln, bootstraps)
    trees = [supgma_tree(a, times, single_theta) for a in alns]
    return trees

def supgma_subsample(aln, times_dict, single_theta=False, n_tips=50, n_trees=5):
    """
    Returns trees using subsample from the alignment
    """
    alns = []
    trees = []

    for i in range(n_trees):
        print(i)
        inds = np.random.choice(len(aln._records), size=n_tips, replace=False)
        aln_sample = MultipleSeqAlignment([aln[int(inds[0])]])
        for j in range(1, n_tips):
            aln_sample.append(aln[int(inds[j])])
        times_sample = np.array([times_dict[s.name] for s in aln_sample])
        trees.append(supgma_tree(aln_sample, times_sample, single_theta))
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