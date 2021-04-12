import dataclasses
import itertools
from dataclasses import dataclass
from io import StringIO
from typing import Dict, NamedTuple, Tuple, Union

import jax
import networkx as nx
import numpy as np
import tskit
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class


class TreeData(NamedTuple):
    """Data structure containing topological aspects of a binary tree."""

    postorder: np.ndarray
    child_parent: np.ndarray
    parent_child: np.ndarray
    sample_times: np.ndarray

    def __post_init__(self):
        assert self.sample_times.ndim == 1
        n = len(self.sample_times)
        num_nodes = 2 * n - 1
        assert (
            num_nodes
            == len(self.child_parent)
            == len(self.parent_child)
            == len(self.postorder)
            == len(self.lower_sampling_times)
        )
        assert (
            1
            == self.child_parent.ndim
            == self.postorder.ndim
            == self.lower_sampling_times.ndim
        )
        assert 2 == self.parent_child.ndim

    @property
    def leaves(self):
        return np.arange(self.n)

    @property
    def internal_nodes(self):
        return np.arange(self.n, self.N)

    def root(self):
        return self.preorder[0]

    @property
    def n(self):
        "The number of leaf nodes/tips represented by this tree."
        return (self.N + 1) // 2

    @property
    def N(self):
        return len(self.postorder)

    @property
    def preorder(self):
        return self.postorder[::-1]

    @property
    def siblings(self):
        siblings = jnp.zeros(self.N, dtype=int)
        for i in 0, 1:
            siblings = jax.ops.index_update(
                siblings,
                jax.ops.index[self.parent_child[:, i]],
                self.parent_child[:, 1 - i],
            )
        return siblings

    @property
    def lower_sampling_times(self):
        """For each internal node, compute the most recent sampling time of any descendant node (TMRD; time to most
        recent descendent.

        Returns:
            Array mapping each node to its TMRD. (For leaf nodes, this is its sampling time.)

        Examples:
            For homochronous sampling, all TMRDs are zero:
            >>> TreeData.from_newick('(a,b)').lower_sampling_times.tolist()  # doctest: +NUMBER
            [0., 0., 0.]

            If one leaf is sampled earlier then the other, then the TMRD of the root will equal the sampling time of the
            more recently sampled leaf:
            >>> TreeData.from_newick('(a:1,b:2)').lower_sampling_times.tolist()  # doctest: +NUMBER
            [1., 0., 1.]

            A less trivial example:
            >>> TreeData.from_newick('((a:1,b:2):3,c:4)').lower_sampling_times.tolist()  # doctest: +NUMBER
            [1., 0., 1., 1., 1.]
        """
        lowers = jnp.concatenate([self.sample_times, np.zeros(self.n - 1)])

        def _f(carry, i):
            target = jax.lax.cond(
                i >= self.n,
                lambda i: jnp.take(
                    carry, jnp.take(self.parent_child, i, axis=0), axis=0
                ).max(),
                lambda i: carry[i],
                i,
            )
            return jax.ops.index_update(carry, i, target), None

        lowers, _ = jax.lax.scan(_f, lowers, self.postorder)
        return lowers

    def to_newick(self, branch_lengths=None, node_mapping={}):
        """Convert tree to newick representation.

        Examples:
            >>> t = TreeData.from_newick("(A:1,B:2)")

            Order is not necessarily preserved:
            >>> t.to_newick()
            '(1,0)'

            With branch lengths:
            >>> t.to_newick([1,2], node_mapping={0: 'A', 1: 'B'})
            '(B:2,A:1)'
        """

        def _bl(s, i):
            try:
                return s + ":" + str(branch_lengths[i])
            except:
                return s

        def _f(i):
            if i < self.n:  # leaf node
                return _bl(str(node_mapping.get(i, i)), i)
            else:
                s = ",".join(map(_f, self.parent_child[i]))
                return _bl("(" + s + ")", i)

        return _f(self.preorder[0])

    @classmethod
    def from_newick(
        cls,
        newick: str,
        return_node_mapping: bool = True,
        return_branch_lengths: bool = False,
    ) -> Union[Tuple["TreeData", dict], Tuple["TreeData", dict, np.ndarray]]:
        """Construct tree data from a Newick tree. Assumes only leaf nodes are named.

        Returns:
            The constructed tree data, and a mapping of node names to integer node labels in the underlying data
            structure.

        Examples:
             >>> td, node_mapping = TreeData.from_newick("((human:1,chimp:2):1,gorilla:2.5);")
             >>> node_mapping['human']  # doctest: +ELLIPSIS
             0
             >>> node_mapping['chimp']
             1
             >>> node_mapping['gorilla']
             2
             >>> td.sample_times  # doctest: +NORMALIZE_WHITESPACE
             array([1. , 0. , 0.5])
             >>> td.n
             3

            If branch lengths are omitted, a branch of length 1 extends from each parent to child.
            >>> TreeData.from_newick("((human,chimp),gorilla)")[0].sample_times
            array([0., 0., 1.])

            Polytomies are arbitrarily broken
            >>> TreeData.from_newick("(A,B,C,D)")[0].to_newick()
            '(((D,C),B),A)'

        """

        from Bio import Phylo

        tree = Phylo.read(StringIO(newick), "newick", rooted=True)
        G = Phylo.to_networkx(tree)

        # Clades aren't sortable which causes problems with JAX, so relabel them with text strings.
        i = itertools.count()
        mapping = {
            clade: (
                clade.name if clade.name is not None else f"internal{next(i)}",
                clade.branch_length or 1,
            )
            for clade in G.nodes
        }
        mapping_names = {k: v[0] for k, v in mapping.items()}
        mapping_blens = dict(mapping.values())

        G1 = nx.relabel_nodes(G, mapping_names)
        for k in mapping_blens:
            try:
                u, v = next(iter(G1.in_edges(k)))
                G1[u][v]["weight"] = mapping_blens[k]
            except StopIteration:
                pass  # root node has no in edges

        poly = [u for u in G1.nodes if G1.out_degree(u) > 2]

        # break polytomies since they break our method.
        # FIXME dendropy does this automatically
        for u in poly:
            edges = list(G1.out_edges(u))
            n0 = u
            for _, v in edges[1:-1]:
                n = f"poly{next(i)}"
                G1.add_edge(n0, n)
                G1.add_edge(n, v)
                G1.remove_edge(u, v)
                n0 = n
            v = edges[-1][1]
            G1.add_edge(n0, v)
            G1.remove_edge(u, v)

        ret = td, nm = cls.from_nx(G1, "weight")
        if not return_branch_lengths:
            return ret
        q = [(n, nm[n]) for n in G1.nodes]
        bl = np.zeros(2 * td.n - 2)
        while q:
            n, j = q.pop()
            try:
                (u, v) = next(iter(G1.in_edges(n)))
                assert v == n
                bl[j] = G1[u][v]["weight"]
                q.append((u, td.child_parent[j]))
            except StopIteration:
                pass  # root node has no in edges
        return ret + (bl,)

    def inverse_height_transform(self, heights) -> Tuple[float, np.ndarray]:
        """Inverse of height_transform(): given node heights, return the root height and proportions that
        recover these heights.

        Examples:
            >>> td = TreeData.from_newick("((human,chimp),gorilla)")
            >>> heights = [0., 0., 0., 2., 5.]
            >>> root_height, proportions = td.inverse_height_transform(heights)
            >>> root_height
            5.0
            >>> proportions.tolist()  # doctest: +NUMBER
            [0.4]
            >>> td.height_transform(root_height, proportions).tolist() == heights
            True


        """
        assert len(heights) == self.N
        # From below, p[i] = (height[i] - height[d[i]]) / (height[pa[i]] - height[d[i]])
        root_height = heights[self.preorder[0]]
        internal_nodes = self.preorder[self.preorder >= self.n][1:]  # omit root node
        h = heights[internal_nodes]
        h_d = jnp.take(self.lower_sampling_times, internal_nodes, axis=0)
        h_p = jnp.take(
            heights, jnp.take(self.child_parent, internal_nodes, axis=0), axis=0
        )
        proportions = (h - h_d) / (h_p - h_d)
        # these have been permuted by nodes. now invert the permutation:
        proportions = proportions[internal_nodes.argsort()]
        return root_height, proportions

    def height_transform(
        self,
        root_height: float,
        proportions: jnp.ndarray,
    ):
        """Transform vectors of proportions and root height to node heights. The transformation is:

            height[i] = height[d[i]] + p[i] * (height[pa[i]] - height[d[i]])

        where i is an internal node, d[i] is the max sample time of any descendant, and pa[i] is its parent [1].

        Args:
            proportions: The vector p[i] in the above formula (n)
            root_height: The height of the root node.

        Returns:
            Vector of length 2 * n - 1: giving the height of each node in the tree.


        Examples:
            For the following tree:

                            4
                        3     \
                       / \     \
                      1   0     2

            There is one internal branch 3-4 to proportion. We set its height to halfway between the sampling
            time of 1 and the root height: .5 * (2.5 + 3) = 2.75.

            >>> td = TreeData.from_newick("((humans,chimp),gorilla);")
            >>> root_height = 3.0
            >>> proportions = [.5]
            >>> td.height_transform(root_height, proportions).tolist()  # doctest: +NUMBER
            [0., 0., 0., 1.5, 3.0]

        References:
            [1] FIXME Fourment et al (phylostan)
        """

        assert len(proportions) == self.N - self.n - 1

        lst = self.lower_sampling_times

        def _f(heights, i):
            h_d = lst[i]
            pa = jnp.take(self.child_parent, i, axis=0)
            p = jnp.take(proportions, i - self.n, axis=0)
            h = (1.0 - p) * h_d + p * heights[pa]
            heights = jax.ops.index_update(heights, i, h)
            return heights, None

        def _ff(heights, i):
            return jax.lax.cond(
                i >= self.n,
                lambda heights: _f(heights, i),
                lambda heights: (heights, None),
                heights,
            )

        # initialize node heights array
        # the first n heights are those of the leaves, and are simply given by lower_times[:n].
        # the rest need to be computed. The last node in our ordering is the root, whose time is directly parametrized.
        heights = jnp.concatenate(
            [
                self.lower_sampling_times[: self.n],
                jnp.zeros(self.n - 2),
                jnp.atleast_1d(root_height) + self.sample_times.max(),
            ]
        )
        if len(self.internal_nodes) > 1:
            # needed to cover an annoying edge case of tree with just 2 leaves; jax.lax.scan throws
            heights, _ = jax.lax.scan(
                _ff,
                heights,
                self.preorder[1:]  # omit root
                # here i want to write self.preorder[self.preorder >= self.n][1:], but the dimensions of this are not
                # statically known to jax
            )
        return heights

    def bl_to_nh(self, blens: jnp.ndarray) -> jnp.ndarray:
        "Convert a vector of branch lengths above each node to a vector of node heights"
        ret = np.zeros(2 * self.n - 1)
        ret[: self.n] = self.sample_times
        for i in self.postorder[:-1]:
            p = self.child_parent[i]
            ret[p] = ret[i] + blens[i]
        return ret

    @classmethod
    def from_tsk_tree(cls, tree: tskit.Tree) -> Tuple["TreeData", dict]:
        """Construct tree data from a tskit tree.

        Example:
            >>> import msprime as msp
            >>> tree = msp.simulate(sample_size=2).first()
            >>> td, node_mapping = TreeData.from_tsk_tree(tree)
            >>> td.n
            2
            >>> node_mapping
            {0: 0, 1: 1, 2: 2}
            >>> td.postorder
            array([0, 1, 2])
            >>> td.sample_times  # doctest: +NUMBER
            array([0., 0.])

            With historical samples:
            >>> tree = msp.simulate(samples=[msp.Sample(0, x) for x in [0., 2.]]).first()
            >>> td, node_mapping = TreeData.from_tsk_tree(tree)
            >>> td.n
            2
            >>> node_mapping
            {0: 0, 1: 1, 2: 2}
            >>> td.postorder
            array([0, 1, 2])
            >>> td.sample_times
            array([0., 2.])
        """
        t = tree.split_polytomies()
        G = nx.DiGraph(t.as_dict_of_dicts())
        return cls.from_nx(G)

    @classmethod
    def from_nx(
        cls,
        G: nx.DiGraph,
        branch_length_key: str = "branch_length",
    ) -> Tuple["TreeData", dict]:
        """Construct tree data from a networkx directed graph.

        # TODO currently no tip data

        Args:
            G: A directed graph containing the tree information.
            branch_length_key: Attribute name of branch length. If this is missing, every edge is assigned length 1.

        Returns:
            A constructed TreeData instance.

        Notes:
            Leaf nodes of G have indegree 1 and outdegree 0. Edge weights are used to infer the sampling times. The
            leaf node which is furthest from the root is sampled at time t=0.

        Examples:
            Construct TreeData from a simple cherry:
            >>> G = nx.DiGraph()
            >>> G.add_edges_from([('root', 'leaf1'), ('root', 'leaf2')])
            >>> td, node_mapping = TreeData.from_nx(G)
            >>> td.sample_times

            The number of leaf nodes in the tree is:
            >>> td.n
            2

            The nodes are renumbered such that 0,...,n-1 are the leaves, and n,...,2n-1 are the internal nodes. Hence:
            >>> td.leaves
            array([0, 1])

            The mapping between original and relabeled nodes is returned:
            >>> node_mapping
            {'leaf1': 0, 'leaf2': 1, 'root': 2}

            The remaining properties capture aspects of the tree:
            >>> td.postorder
            array([0, 1, 2])
            >>> td.preorder
            array([2, 1, 0])
            >>> td.sample_times
            array([0., 1.])
            >>> td.child_parent  # doctest: +NORMALIZE_WHITESPACE
            array([ 2, 2, -1])
            >>> td.parent_child  # doctest: +NORMALIZE_WHITESPACE
            array([[-1, -1], [-1, -1], [ 1, 0]])
        """

        def is_leaf(x):
            return G.out_degree(x) == 0 and G.in_degree(x) == 1

        n = sum(map(is_leaf, G.nodes()))
        leaf_iter = iter(range(n))
        internal_iter = iter(range(n, 2 * n - 1))
        node_remap = {}

        for x in nx.dfs_postorder_nodes(G):
            if is_leaf(x):
                # this is a leaf node, so it gets numbered from 0 ... n - 1
                node_remap[x] = next(leaf_iter)
            else:
                # this is a tip node, so it gets numbered n ... 2 * n - 1
                node_remap[x] = next(internal_iter)

        G = nx.relabel_nodes(G, mapping=node_remap)
        postorder = np.array(list(nx.dfs_postorder_nodes(G)))
        M = len(G)

        root = next(x for x in G.nodes if G.out_degree(x) == 2 and G.in_degree(x) == 0)
        leaves = list(filter(is_leaf, G.nodes))
        root_distance = nx.shortest_path_length(
            G, source=root, weight=branch_length_key
        )
        m = max(root_distance[k] for k in leaves)
        # sample times for each leaf node
        sample_times = np.array([m - root_distance[x] for x in range(n)])

        # map each child node to its parent
        child_parent = np.full(M, -1)
        for k in G.pred:
            try:
                child_parent[k] = next(iter(G.pred[k]))
            except StopIteration:
                pass  # the node has no predecessors, i.e. it's the root

        parent_child = np.zeros([M, 2], int) - 1
        for i in range(M):
            parent = child_parent[i]
            if parent == -1:
                continue
            parent_child[parent, 1] = parent_child[parent, 0]
            parent_child[parent, 0] = i

        return (
            cls(
                postorder=postorder,
                child_parent=child_parent,
                parent_child=parent_child,
                sample_times=sample_times,
            ),
            node_remap,
        )
