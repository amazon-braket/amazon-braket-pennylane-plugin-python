# Copyright 2019-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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
    """Tests that the matrices of the custom operations are correct."""
    assert np.allclose(pl_op._matrix(*params), braket_gate(*params).to_matrix())
