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

"""Tests that variances are correctly computed in the plugin device"""

import numpy as np
import pennylane as qp
import pytest

np.random.seed(42)


@pytest.mark.parametrize("shots", [None, 8192])
class TestVar:
    """Tests for the variance"""

    def test_var(self, device, shots, tol):
        """Tests for variance calculation"""
        dev = device(2)

        phi = 0.543
        theta = 0.6543

        @qp.qnode(dev)
        def circuit():
            qp.RX(phi, wires=[0])
            qp.RY(theta, wires=[0])
            return qp.var(qp.PauliZ(wires=[0]))

        expected = 0.25 * (3 - np.cos(2 * theta) - 2 * np.cos(theta) ** 2 * np.cos(2 * phi))
        assert np.allclose(circuit(), expected, **tol)

    def test_var_hermitian(self, device, shots, tol):
        """Tests for variance calculation using an arbitrary Hermitian observable"""
        dev = device(2)

        phi = 0.543
        theta = 0.6543
        H = np.array([[4, -1 + 6j], [-1 - 6j, 2]])

        @qp.qnode(dev)
        def circuit():
            qp.RX(phi, wires=[0])
            qp.RY(theta, wires=[0])
            return qp.var(qp.Hermitian(H, wires=[0]))

        expected = 0.5 * (
            2 * np.sin(2 * theta) * np.cos(phi) ** 2
            + 24 * np.sin(phi) * np.cos(phi) * (np.sin(theta) - np.cos(theta))
            + 35 * np.cos(2 * phi)
            + 39
        )
        assert np.allclose(circuit(), expected, **tol)


@pytest.mark.parametrize("shots", [None, 8192])
class TestTensorVar:
    """Tests for variance of tensor observables"""

    def test_paulix_pauliy(self, device, shots, tol):
        """Test that a tensor product involving PauliX and PauliY works correctly"""
        dev = device(3)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliX(0) @ qp.PauliY(2))

        expected = (
            8 * np.sin(theta) ** 2 * np.cos(2 * varphi) * np.sin(phi) ** 2
            - np.cos(2 * (theta - phi))
            - np.cos(2 * (theta + phi))
            + 2 * np.cos(2 * theta)
            + 2 * np.cos(2 * phi)
            + 14
        ) / 16
        assert np.allclose(circuit(), expected, **tol)

    def test_pauliz_hadamard(self, device, shots, tol):
        """Test that a tensor product involving PauliZ and PauliY and hadamard works correctly"""
        dev = device(3)

        theta = 0.432
        phi = 0.123
        varphi = -0.543

        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliZ(wires=[0]) @ qp.Hadamard(wires=[1]) @ qp.PauliY(wires=[2]))

        expected = (
            3
            + np.cos(2 * phi) * np.cos(varphi) ** 2
            - np.cos(2 * theta) * np.sin(varphi) ** 2
            - 2 * np.cos(theta) * np.sin(phi) * np.sin(2 * varphi)
        ) / 4

        assert np.allclose(circuit(), expected, **tol)

    def test_hermitian(self, device, shots, tol):
        """Test that a tensor product involving qp.Hermitian works correctly"""
        dev = device(3)

        theta = 0.432
        phi = 0.123
        varphi = -0.543
        A = np.array(
            [
                [-6, 2 + 1j, -3, -5 + 2j],
                [2 - 1j, 0, 2 - 1j, -5 + 4j],
                [-3, 2 + 1j, 0, -4 + 3j],
                [-5 - 2j, -5 - 4j, -4 - 3j, -6],
            ]
        )

        @qp.qnode(dev)
        def circuit():
            qp.RX(theta, wires=[0])
            qp.RX(phi, wires=[1])
            qp.RX(varphi, wires=[2])
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[1, 2])
            return qp.var(qp.PauliZ(wires=[0]) @ qp.Hermitian(A, wires=[1, 2]))

        expected = (
            1057
            - np.cos(2 * phi)
            + 12 * (27 + np.cos(2 * phi)) * np.cos(varphi)
            - 2 * np.cos(2 * varphi) * np.sin(phi) * (16 * np.cos(phi) + 21 * np.sin(phi))
            + 16 * np.sin(2 * phi)
            - 8 * (-17 + np.cos(2 * phi) + 2 * np.sin(2 * phi)) * np.sin(varphi)
            - 8 * np.cos(2 * theta) * (3 + 3 * np.cos(varphi) + np.sin(varphi)) ** 2
            - 24 * np.cos(phi) * (np.cos(phi) + 2 * np.sin(phi)) * np.sin(2 * varphi)
            - 8
            * np.cos(theta)
            * (
                4
                * np.cos(phi)
                * (
                    4
                    + 8 * np.cos(varphi)
                    + np.cos(2 * varphi)
                    - (1 + 6 * np.cos(varphi)) * np.sin(varphi)
                )
                + np.sin(phi)
                * (
                    15
                    + 8 * np.cos(varphi)
                    - 11 * np.cos(2 * varphi)
                    + 42 * np.sin(varphi)
                    + 3 * np.sin(2 * varphi)
                )
            )
        ) / 16
        assert np.allclose(circuit(), expected, **tol)
