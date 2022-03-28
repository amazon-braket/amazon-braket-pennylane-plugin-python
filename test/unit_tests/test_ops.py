# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
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

import itertools
import math
from unittest.mock import patch

import numpy as np
import pennylane as qml
import pytest
import scipy
import tensorflow as tf
from autograd import deriv
from autograd import numpy as anp
from braket.circuits import gates
from numpy import float64

from braket.pennylane_plugin import ECR, PSWAP, XY, CPhaseShift00, CPhaseShift01, CPhaseShift10

gates_2q_parametrized = [
    (CPhaseShift00, gates.CPhaseShift00),
    (CPhaseShift01, gates.CPhaseShift01),
    (CPhaseShift10, gates.CPhaseShift10),
    (PSWAP, gates.PSwap),
    (XY, gates.XY),
]

gates_2q_non_parametrized = [
    (ECR, gates.ECR),
]

observables_1q = [
    obs.compute_matrix() for obs in [qml.Hadamard, qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]
]
observables_2q = [
    np.kron(obs1, obs2) for obs1, obs2 in itertools.product(observables_1q, observables_1q)
]


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
def test_ops_parametrized(pl_op, braket_gate, angle):
    """Tests that the matrices and decompositions of parametrized custom operations are correct."""
    assert np.allclose(pl_op.compute_matrix(angle), braket_gate(angle).to_matrix())
    _assert_decomposition(pl_op, params=[angle])


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_parametrized)
@pytest.mark.parametrize(
    "angle", [tf.Variable(((i + 1) * math.pi / 12), dtype=float64) for i in range(12)]
)
def test_ops_parametrized_tf(pl_op, braket_gate, angle):
    """Tests that the matrices and decompositions of parametrized custom operations
    are correct using tensorflow interface.
    """
    pl_op.compute_matrix(angle)
    _assert_decomposition(pl_op, params=[angle])


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_non_parametrized)
def test_ops_non_parametrized(pl_op, braket_gate):
    """Tests that the matrices and decompositions of non-parametrized custom operations are
    correct.
    """
    assert np.allclose(pl_op.compute_matrix(), braket_gate().to_matrix())
    _assert_decomposition(pl_op)


@patch("braket.pennylane_plugin.ops.np", new=anp)
@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
@pytest.mark.parametrize("observable", observables_2q)
def test_param_shift_2q(pl_op, braket_gate, angle, observable):
    """Tests that the parameter-shift rules of custom operations yield the correct derivatives."""
    op = pl_op(angle, wires=[0, 1])
    if op.grad_recipe[0]:
        shifts = op.grad_recipe[0]
    else:
        cs, ss = qml.gradients.generate_shift_rule(op.parameter_frequencies[0])
        shifts = [[c, 1, s] for c, s in zip(cs, ss)]

    summands = []
    for shift in shifts:
        shifted = pl_op.compute_matrix(angle + shift[2])
        summands.append(
            shift[0] * np.matmul(np.matmul(np.transpose(np.conj(shifted)), observable), shifted)
        )

    from_shifts = sum(summands)

    def conj_obs_gate(angle):
        mat = pl_op.compute_matrix(angle)
        return anp.matmul(anp.matmul(anp.transpose(anp.conj(mat)), observable), mat)

    direct_calculation = deriv(conj_obs_gate)(angle)

    assert np.allclose(from_shifts, direct_calculation)


def _assert_decomposition(pl_op, params=None):
    num_wires = pl_op.num_wires
    dimension = 2**num_wires
    num_indices = 2 * num_wires
    wires = list(range(num_wires))

    contraction_parameters = [
        np.reshape(np.eye(dimension), [2] * num_indices),
        list(range(num_indices)),
    ]
    index_substitutions = {i: i + num_wires for i in range(num_wires)}
    next_index = num_indices

    # get decomposed gates
    decomposed_op = (
        pl_op.compute_decomposition(*params, wires=wires)
        if params
        else pl_op.compute_decomposition(wires=wires)
    )

    # Heterogeneous matrix chain multiplication using tensor contraction
    for gate in reversed(decomposed_op):
        gate_wires = gate.wires.tolist()

        # Upper indices, which will be traced out
        contravariant = [index_substitutions[i] for i in gate_wires]

        # Lower indices, which will replace the contracted indices in the matrix
        covariant = list(range(next_index, next_index + len(gate_wires)))

        indices = contravariant + covariant
        # `qml.matrix(gate)` as type-(len(contravariant), len(covariant)) tensor
        gate_tensor = np.reshape(qml.matrix(gate), [2] * len(indices))

        contraction_parameters += [gate_tensor, indices]
        next_index += len(gate_wires)
        index_substitutions.update({gate_wires[i]: covariant[i] for i in range(len(gate_wires))})

    # Ensure matrix is in the correct order
    new_indices = wires + [index_substitutions[i] for i in range(num_wires)]
    contraction_parameters.append(new_indices)

    actual_matrix = np.reshape(np.einsum(*contraction_parameters), [dimension, dimension])
    pl_op_matrix = pl_op.compute_matrix(*params) if params else pl_op.compute_matrix()

    assert np.allclose(actual_matrix, pl_op_matrix)


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
def test_gate_adjoint_parametrized(pl_op, braket_gate, angle):
    op = pl_op(angle, wires=[0, 1])
    assert np.allclose(
        np.dot(pl_op.compute_matrix(angle), qml.matrix(pl_op.adjoint(op))), np.identity(4)
    )


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_non_parametrized)
def test_gate_adjoint_non_parametrized(pl_op, braket_gate):
    op = pl_op(wires=[0, 1])
    assert np.allclose(
        np.dot(pl_op.compute_matrix(), qml.matrix(pl_op.adjoint(op))), np.identity(4)
    )


@pytest.mark.parametrize("pl_op, braket_gate", gates_2q_parametrized)
@pytest.mark.parametrize("angle", [(i + 1) * math.pi / 12 for i in range(12)])
def test_gate_generator(pl_op, braket_gate, angle):
    op = pl_op(angle, wires=[0, 1])
    if op.name != "PSWAP":
        assert np.allclose(
            qml.matrix(op), scipy.linalg.expm(1j * angle * qml.matrix(op.generator()))
        )
