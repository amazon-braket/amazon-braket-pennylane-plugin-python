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


"""Tests that gradients are correctly computed in the plugin device via braket"""

import math
import random

import pennylane as qml
import pytest
from pennylane import numpy as np

ABS_TOLERANCE = 1e-5


def adj_grad_test_helper(device, circuit, parameters):
    @qml.qnode(device, diff_method="device")
    def circuit_braket_ag(*parameters):
        return circuit(*parameters)

    # pl uses braket gradient calculation by default! so we have to specify
    # pl's parameter shift method
    @qml.qnode(device, diff_method="parameter-shift")
    def circuit_with_pl(*parameters):
        return circuit(*parameters)

    braket_grad = qml.grad(circuit_braket_ag)(*parameters)
    pl_grad = qml.grad(circuit_with_pl)(*parameters)

    assert len(braket_grad)
    # assert grad isn't all zeros
    assert np.array(
        [not math.isclose(derivative, 0, abs_tol=ABS_TOLERANCE) for derivative in pl_grad]
    ).any()
    assert np.allclose(braket_grad, pl_grad)
    braket_jac = qml.jacobian(circuit_braket_ag)(*parameters)
    pl_jac = qml.jacobian(circuit_with_pl)(*parameters)
    # assert jac isn't all zeros
    assert len(braket_jac)
    assert np.array(
        [not math.isclose(derivative, 0, abs_tol=ABS_TOLERANCE) for derivative in braket_jac]
    ).any()
    assert np.allclose(braket_jac, pl_jac)


@pytest.mark.parametrize("shots", [None])
class TestAdjointGradient:
    def test_grad(self, on_demand_sv_device):
        dev = on_demand_sv_device(2)

        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(1))

        x = np.array(0.1, requires_grad=False)
        y = np.array(0.2, requires_grad=True)
        z = np.array(0.3, requires_grad=True)

        qml.disable_return()
        adj_grad_test_helper(dev, circuit, [x, y, z])
        qml.enable_return()

    def test_grad_large_circuit(self, on_demand_sv_device):
        num_qubits = 8
        # having > 10 parameterized gates is important; we've
        # had a bug with string sorting of the gradients
        # that this test should catch
        num_params = 11
        dev = on_demand_sv_device(num_qubits)

        def circuit(x):
            # seed this so it's always the same circuit generated
            random.seed(2)
            # put qubits into an interesting state so we have non-zero
            # derivatives
            for i in range(num_qubits):
                rand_val = random.randint(0, 2)
                rand_param = random.choice(x)
                if rand_val == 0:
                    qml.RX(rand_param, wires=i)
                elif rand_val == 1:
                    qml.RY(rand_param, wires=i)
                else:
                    qml.RZ(rand_param, wires=i)
                if random.randint(0, 3) == 0:
                    # cnot i with some random other qubit
                    qml.CNOT(wires=[i, random.choice([j for j in range(num_qubits) if i != j])])

            # use every parameter at least once
            for i in range(num_params):
                rand_val = random.randint(0, 2)
                qubit = random.randrange(num_qubits)
                if rand_val == 0:
                    qml.RX(x[i], wires=qubit)
                elif rand_val == 1:
                    qml.RY(x[i], wires=qubit)
                else:
                    qml.RZ(x[i], wires=qubit)
            return qml.expval(qml.PauliZ(1))

        # x = [.1,.2,....]
        x = np.array([i / 10 for i in range(num_params)], requires_grad=True)
        x[1] = np.array(x[1], requires_grad=False)

        qml.disable_return()
        adj_grad_test_helper(dev, circuit, [x])
        qml.enable_return()

    # this test runs on sv1, dm1, and the local simulator to validate that
    # calls to qml.grad() without a specified diff_method succeed
    def test_default_diff_method(self, device):
        dev = device(2)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(z, wires=1)
            return qml.expval(qml.PauliZ(1))

        x = np.array(0.1, requires_grad=False)
        y = np.array(0.2, requires_grad=True)
        z = np.array(0.3, requires_grad=True)

        qml.disable_return()
        qml.grad(circuit)(x, y, z)
        qml.enable_return()
