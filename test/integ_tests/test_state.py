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

"""Tests that samples are correctly computed in the plugin device"""
import numpy as np
import pennylane as qml
import pytest

np.random.seed(42)


@pytest.mark.parametrize("shots", [0])
class TestState:
    """Tests for the state return type"""

    def test_full_state(self, local_device, shots, tol):
        """Tests if the correct state vector or full density matrix is returned"""
        dev = local_device(1)
        supports_sv = "StateVector" in dev._braket_result_types

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.T(wires=0)
            return qml.state()

        output = circuit()
        expected_sv = np.array([np.sqrt(0.5), 0.5 + 0.5j])
        if supports_sv:
            assert np.allclose(output, expected_sv, **tol)
        else:
            assert np.allclose(output, np.kron(expected_sv, expected_sv.conj()), **tol)

    def test_reduced_density_matrix(self, local_device, shots, tol):
        """Tests if the correct reduced density matrix is returned"""
        dev = local_device(2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CRX(np.pi / 2, wires=[0, 1])
            return qml.density_matrix(wires=[0])

        output = circuit()
        expected_dm = np.array([[0.5, 1j / np.sqrt(8)], [-1j / np.sqrt(8), 0.5]])
        assert np.allclose(output, expected_dm, **tol)
