import math

import numpy as np
import pytest
from braket.circuits import gates
from braket.pennylane_plugin import (
    CY,
    ISWAP,
    PSWAP,
    XX,
    XY,
    YY,
    ZZ,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    V,
)

testdata = [
    (V, gates.V, []),
    (CY, gates.CY, []),
    (CPhaseShift, gates.CPhaseShift, [math.pi]),
    (CPhaseShift00, gates.CPhaseShift00, [math.pi]),
    (CPhaseShift01, gates.CPhaseShift01, [math.pi]),
    (CPhaseShift10, gates.CPhaseShift10, [math.pi]),
    (ISWAP, gates.ISwap, []),
    (PSWAP, gates.PSwap, [math.pi]),
    (XY, gates.XY, [math.pi]),
    (XX, gates.XX, [math.pi]),
    (YY, gates.YY, [math.pi]),
    (ZZ, gates.ZZ, [math.pi]),
]


@pytest.mark.parametrize("pl_op, braket_gate, params", testdata)
def test_matrices(pl_op, braket_gate, params):
    """ Tests that the matrices of the custom operations are correct.
    """
    assert np.allclose(pl_op._matrix(*params), braket_gate(*params).to_matrix())
