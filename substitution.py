import re
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import msprime as msp
import numpy as np

# copied from beast.evolution.datatype.Nucleotide
BEAST_PARTIALS = """
{0}, // A
{1}, // C
{2}, // G
{3}, // T
{3}, // U
{0, 2}, // R
{1, 3}, // Y
{0, 1}, // M
{0, 3}, // W
{1, 2}, // S
{2, 3}, // K
{1, 2, 3}, // B
{0, 2, 3}, // D
{0, 1, 3}, // H
{0, 1, 2}, // V
{0, 1, 2, 3}, // N
{0, 1, 2, 3}, // X
{0, 1, 2, 3}, // -
{0, 1, 2, 3}, // ?
"""

PARTIAL_DICT = {
    s[-1]: np.eye(4)[list(eval(re.match(r"\{[^}]+\}", s)[0]))].sum(0)
    for s in BEAST_PARTIALS.splitlines()[1:]
}


def encode_partials(col: str) -> np.ndarray:
    """Encode tip partials for a single alignment column.

    Args:
        col: string of length n.

    Returns:
        array of dimension [n,4] containing the tip partial encoding

    """
    return np.array(list(map(PARTIAL_DICT.__getitem__, col)))


class SubstitutionModel(NamedTuple):
    pi: np.ndarray
    P: np.ndarray
    d: np.ndarray
    Pinv: np.ndarray

    @property
    def Q(self):
        return self.P @ (self.d[:, None] * self.Pinv)

    def expm(self, t: float) -> np.ndarray:
        return self.P @ jnp.diag(jnp.exp(t * self.d)) @ self.Pinv

    def expm_action(self, t: float, p: np.ndarray, right: bool = True):
        if right:
            ret = self.P @ (jnp.exp(t * self.d) * (self.Pinv @ p))
        else:
            ret = ((p @ self.P) * jnp.exp(t * self.d)) @ self.Pinv
        # in this setting, should always return a distribution \in [0,1], but small numerical errors can crop up.
        return jnp.clip(ret, 1e-40, 1.0)


def HKY(kappa: float = 1) -> SubstitutionModel:
    "HKY(kappa) substitution matrix."
    mm = msp.HKYMutationModel(kappa)
    Q = mm.transition_matrix
    Q -= np.diag(Q.sum(1))
    pi = mm.root_distribution
    d, P = np.linalg.eigh(Q)
    assert np.allclose(P @ np.diag(d) @ P.T, Q)

    class HKYSubstitutionModel(SubstitutionModel):
        pass

    ret = HKYSubstitutionModel(pi, P, d, P.T)
    assert np.allclose(ret.Q, Q)
    return ret


def JC69() -> SubstitutionModel:
    "JC69 substitution model"
    return HKY(1.0)
