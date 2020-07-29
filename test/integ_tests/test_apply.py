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

"""Tests that application of operations works correctly in the plugin devices"""
import numpy as np
import pennylane as qml
import pytest
from conftest import U2, U

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

np.random.seed(42)


# =========================================================

# list of all non-parametrized single-qubit gates,
# along with the PennyLane operation name
single_qubit = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.T, V]

# List of all non-parametrized single-qubit gates with inverses.
single_qubit_inverse = [qml.S, qml.T, V]

# list of all parametrized single-qubit gates
single_qubit_param = [qml.PhaseShift, qml.RX, qml.RY, qml.RZ]

# list of all non-parametrized two-qubit gates
two_qubit = [qml.CNOT, qml.CZ, qml.SWAP, CY, ISWAP]

# list of all three-qubit gates
three_qubit = [qml.CSWAP, qml.Toffoli]

# list of all parametrized two-qubit gates
two_qubit_param = [CPhaseShift, CPhaseShift00, CPhaseShift01, CPhaseShift10, PSWAP, XY, XX, YY, ZZ]


@pytest.mark.parametrize("shots", [8192])
class TestHardwareApply:
    """Test application of PennyLane operations on hardware simulators."""

    def test_basis_state(self, device, tol):
        """Test basis state initialization"""
        dev = device(4)
        state = np.array([0, 0, 1, 0])
        ops = qml.BasisState.decomposition(state, wires=[0, 1, 2, 3])

        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(4))
        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        assert np.allclose(res, expected, **tol)

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test state vector preparation"""
        dev = device(1)
        state = init_state(1)
        ops = qml.QubitStateVector.decomposition(state, wires=[0])

        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(1))
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("op", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, device, op, tol):
        """Test single-qubit gates without parameters"""
        dev = device(1)
        state = init_state(1)

        ops = qml.QubitStateVector.decomposition(state, wires=[0])
        ops.append(op(wires=[0]))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(1))
        expected = np.abs(op._matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("op", single_qubit_inverse)
    def test_single_qubit_no_parameters_inverse(self, init_state, device, op, tol):
        """Test inverses of single-qubit gates without parameters, where applicable"""
        dev = device(1)
        state = init_state(1)

        ops = qml.QubitStateVector.decomposition(state, wires=[0])
        gate = op(wires=[0]).inv()
        ops.append(gate)
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(1))
        expected = np.abs(gate.matrix @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, op, theta, tol):
        """Test parametrized single-qubit gates"""
        dev = device(1)
        state = init_state(1)

        ops = qml.QubitStateVector.decomposition(state, wires=[0])
        ops.append(op(theta, wires=[0]))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(1))
        expected = np.abs(op._matrix(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("op", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, op, tol):
        """Testno parameter two qubit gates"""
        dev = device(2)
        state = init_state(2)

        ops = qml.QubitStateVector.decomposition(state, wires=[0, 1])
        ops.append(op(wires=[0, 1]))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(2))
        expected = np.abs(op._matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, device, op, theta, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)

        dev.pre_measure()
        ops = qml.QubitStateVector.decomposition(state, wires=[0, 1])
        ops.append(op(theta, wires=[0, 1]))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(2))
        expected = np.abs(op._matrix(theta) @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("op", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, op, tol):
        dev = device(3)
        state = init_state(3)

        dev.pre_measure()
        ops = qml.QubitStateVector.decomposition(state, wires=[0, 1, 2])
        ops.append(op(wires=[0, 1, 2]))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(3))
        expected = np.abs(op._matrix() @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)

        ops = qml.QubitStateVector.decomposition(state, wires=list(range(N)))
        ops.append(qml.QubitUnitary(mat, wires=list(range(N))))
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(N))
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, **tol)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_rotation(self, init_state, device, tol):
        """Test three axis rotation gate"""
        dev = device(1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        test_op = qml.Rot(a, b, c, wires=[0])

        ops = qml.QubitStateVector.decomposition(state, wires=[0])
        ops.append(test_op)
        dev.apply(ops)
        dev.generate_samples()

        res = dev.probability(wires=range(1))
        expected = np.abs(test_op.matrix @ state) ** 2
        assert np.allclose(res, expected, **tol)
