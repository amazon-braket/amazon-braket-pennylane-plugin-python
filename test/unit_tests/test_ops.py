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
from unittest.mock import patch

import autograd
import numpy as np
import pytest
from braket.circuits import gates

from braket.pennylane_plugin import (
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
)

testdata_fixed = [(ISWAP, gates.ISwap)]

testdata_parametrized = [
    (CPhaseShift, gates.CPhaseShift),
    (CPhaseShift00, gates.CPhaseShift00),
    (CPhaseShift01, gates.CPhaseShift01),
    (CPhaseShift10, gates.CPhaseShift10),
    (PSWAP, gates.PSwap),
    (XY, gates.XY),
    (XX, gates.XX),
    (YY, gates.YY),
    (ZZ, gates.ZZ),
]


@pytest.mark.parametrize("pl_op, braket_gate", testdata_fixed)
def test_fixed_ops(pl_op, braket_gate):
    """Tests that the matrices and decompositions of fixed custom operations are correct."""
    assert np.allclose(pl_op._matrix(), braket_gate().to_matrix())
    _assert_decomposition(pl_op, [])


@pytest.mark.parametrize("pl_op, braket_gate", testdata_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
def test_parametrized_ops(pl_op, braket_gate, angle):
    """Tests that the matrices and decompositions of parametrized custom operations are correct."""
    assert np.allclose(pl_op._matrix(angle), braket_gate(angle).to_matrix())
    _assert_decomposition(pl_op, [angle])


@patch("braket.pennylane_plugin.ops.np", new=autograd.numpy)
@pytest.mark.parametrize("pl_op, braket_gate", testdata_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
def test_param_shift(pl_op, braket_gate, angle):
    """Tests that the parameter-shift rules of custom operations yield the correct derivatives."""
    op = pl_op(angle, wires=list(range(pl_op.num_wires)))
    param_shifted = sum(
        [shift[0] * pl_op._matrix(angle + shift[2]) for shift in op.get_parameter_shift(0)]
    )
    direct_calculation = autograd.deriv(pl_op._matrix)(angle)
    assert np.allclose(param_shifted, direct_calculation)


def _assert_decomposition(pl_op, params):
    num_wires = pl_op.num_wires
    dimension = 2 ** num_wires
    num_indices = 2 * num_wires
    wires = list(range(num_wires))

    contraction_parameters = [
        np.reshape(np.eye(dimension), [2] * num_indices),
        list(range(num_indices)),
    ]
    index_substitutions = {i: i + num_wires for i in range(num_wires)}
    next_index = num_indices
    # Heterogeneous matrix chain multiplication using tensor contraction
    for gate in reversed(pl_op.decomposition(*params, wires=wires)):
        gate_wires = gate.wires.tolist()

        # Upper indices, which will be traced out
        contravariant = [index_substitutions[i] for i in gate_wires]

        # Lower indices, which will replace the contracted indices in the matrix
        covariant = list(range(next_index, next_index + len(gate_wires)))

        indices = contravariant + covariant
        # `gate.matrix` as type-(len(contravariant), len(covariant)) tensor
        gate_tensor = np.reshape(gate.matrix, [2] * len(indices))

        contraction_parameters += [gate_tensor, indices]
        next_index += len(gate_wires)
        index_substitutions.update({gate_wires[i]: covariant[i] for i in range(len(gate_wires))})

    # Ensure matrix is in the correct order
    new_indices = wires + [index_substitutions[i] for i in range(num_wires)]
    contraction_parameters.append(new_indices)

    actual_matrix = np.reshape(np.einsum(*contraction_parameters), [dimension, dimension])
    assert np.allclose(actual_matrix, pl_op._matrix(*params))
