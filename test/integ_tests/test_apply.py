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

"""Tests that gates are correctly applied in the plugin device"""

import numpy as np
import pennylane as qml
import pytest
from conftest import K2, U2, K, U

from braket.pennylane_plugin import PSWAP, CPhaseShift00, CPhaseShift01, CPhaseShift10

np.random.seed(42)

# =========================================================

# list of all non-parametrized single-qubit gates
single_qubit = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.S, qml.SX, qml.T]

# list of all parametrized single-qubit gates
single_qubit_param = [qml.PhaseShift, qml.RX, qml.RY, qml.RZ]

# list of all non-parametrized two-qubit gates
two_qubit = [qml.CNOT, qml.CY, qml.CZ, qml.SWAP, qml.ISWAP]

# list of all three-qubit gates
three_qubit = [qml.CSWAP, qml.Toffoli]

# list of all parametrized two-qubit gates
two_qubit_param = [
    qml.ControlledPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    PSWAP,
    qml.IsingXY,
    qml.IsingXX,
    qml.IsingYY,
    qml.IsingZZ,
]

# list of all single-qubit single-parameter noise operations
single_qubit_noise = [
    qml.AmplitudeDamping,
    qml.PhaseDamping,
    qml.DepolarizingChannel,
    qml.BitFlip,
    qml.PhaseFlip,
]


@pytest.mark.parametrize("shots", [8192])
class TestHardwareApply:
    """Test application of PennyLane operations on hardware simulators."""

    def test_basis_state(self, device, tol):
        """Test basis state initialization"""
        dev = device(4)
        state = np.array([0, 0, 1, 0])

        @qml.qnode(dev)
        def circuit():
            qml.BasisState.compute_decomposition(state, wires=[0, 1, 2, 3])
            return qml.probs(wires=range(4))

        expected = np.zeros([2**4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1
        assert np.allclose(circuit(), expected, **tol)

    def test_qubit_state_vector(self, init_state, device, tol):
        """Test state vector preparation"""
        dev = device(1)
        state = init_state(1)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep.compute_decomposition(state, wires=[0])
            return qml.probs(wires=range(1))

        assert np.allclose(circuit(), np.abs(state) ** 2, **tol)

    @pytest.mark.parametrize("op", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, device, op, tol):
        """Test single-qubit gates without parameters"""
        dev = device(1)
        state = init_state(1)
        TestHardwareApply.assert_op_and_inverse(op, dev, state, [0], tol, [])

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, device, op, theta, tol):
        """Test parametrized single-qubit gates"""
        dev = device(1)
        state = init_state(1)
        TestHardwareApply.assert_op_and_inverse(op, dev, state, [0], tol, [theta])

    @pytest.mark.parametrize("op", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, device, op, tol):
        """Test two qubit gates with no parameters"""
        dev = device(2)
        state = init_state(2)
        TestHardwareApply.assert_op_and_inverse(op, dev, state, [0, 1], tol, [])

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, device, op, theta, tol):
        """Test PauliX application"""
        dev = device(2)
        state = init_state(2)
        dev.pre_measure()
        TestHardwareApply.assert_op_and_inverse(op, dev, state, [0, 1], tol, [theta])

    @pytest.mark.parametrize("op", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, device, op, tol):
        dev = device(3)
        state = init_state(3)
        dev.pre_measure()
        TestHardwareApply.assert_op_and_inverse(op, dev, state, [0, 1, 2], tol, [])

    @pytest.mark.parametrize("mat", [U, U2])
    def test_qubit_unitary(self, init_state, device, mat, tol):
        N = int(np.log2(len(mat)))
        dev = device(N)
        state = init_state(N)
        wires = list(range(N))
        TestHardwareApply.assert_op_and_inverse(qml.QubitUnitary, dev, state, wires, tol, [mat])

    def test_rotation(self, init_state, device, tol):
        """Test three axis rotation gate"""
        dev = device(1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        TestHardwareApply.assert_op_and_inverse(qml.Rot, dev, state, [0], tol, [a, b, c])

    @pytest.mark.parametrize("prob", [0.0, 0.42])
    @pytest.mark.parametrize("op", single_qubit_noise)
    def test_single_qubit_noise(self, init_state, dm_device, op, prob, tol):
        """Test parametrized single-qubit noise operations"""
        dev = dm_device(1)
        state = init_state(1)
        TestHardwareApply.assert_noise_op(op, dev, state, [0], tol, [prob])

    @pytest.mark.parametrize("gamma", [0.0, 0.42])
    @pytest.mark.parametrize("prob", [0.0, 0.42])
    def test_generalized_amplitude_damping(self, init_state, dm_device, gamma, prob, tol):
        """Test parametrized GeneralizedAmplitudeDamping"""
        dev = dm_device(1)
        state = init_state(1)
        TestHardwareApply.assert_noise_op(
            qml.GeneralizedAmplitudeDamping, dev, state, [0], tol, [gamma, prob]
        )

    @pytest.mark.parametrize("kraus", [K, K2])
    def test_qubit_channel(self, init_state, dm_device, kraus, tol):
        N = int(np.log2(len(kraus[0])))
        dev = dm_device(N)
        state = init_state(N)
        wires = list(range(N))
        TestHardwareApply.assert_noise_op(qml.QubitChannel, dev, state, wires, tol, [kraus])

    @staticmethod
    def assert_op_and_inverse(op, dev, state, wires, tol, op_args):
        @qml.qnode(dev)
        def circuit():
            qml.StatePrep.compute_decomposition(state, wires=wires)
            op(*op_args, wires=wires)
            return qml.probs(wires=wires)

        assert np.allclose(circuit(), np.abs(op.compute_matrix(*op_args) @ state) ** 2, **tol)

        @qml.qnode(dev)
        def circuit_inv():
            qml.StatePrep.compute_decomposition(state, wires=wires)
            qml.adjoint(op(*op_args, wires=wires))
            return qml.probs(wires=wires)

        assert np.allclose(
            circuit_inv(), np.abs(op.compute_matrix(*op_args).conj().T @ state) ** 2, **tol
        )

    @staticmethod
    def assert_noise_op(op, dev, state, wires, tol, op_args):
        @qml.qnode(dev)
        def circuit():
            qml.StatePrep.compute_decomposition(state, wires=wires)
            op(*op_args, wires=wires)
            return qml.probs(wires=wires)

        def expected_probs():
            initial_dm = np.outer(state, state.conj())
            final_dm = sum(
                matrix @ initial_dm @ matrix.conj().T
                for matrix in op(*op_args, wires=wires).kraus_matrices()
            )
            return np.diagonal(final_dm)

        assert np.allclose(circuit(), expected_probs(), **tol)
