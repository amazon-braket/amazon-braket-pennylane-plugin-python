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

import pennylane as qml
import pennylane.numpy as np
import pytest

seed = 42
np.random.seed(seed)
H = qml.PauliZ(0) @ qml.PauliZ(1)
wires = 2


@pytest.mark.parametrize("shots", [5000])
class TestShadowExpval:
    """Test shadow_expval computation of expectation values."""

    def test_shadow_expval(self, device, shots):
        dev = device(wires)
        if dev.short_name == "braket.aws.qubit":
            pytest.skip(
                "SV1 needs batch execution to execute in reasonable time, "
                "but parallel shadow_expval is currently broken"
            )

        @qml.qnode(dev)
        def shadow_circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.shadow_expval(H, seed=seed)

        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.expval(H)

        x = np.array(1.2)
        shadow_e = shadow_circuit(x)
        e = circuit(x)
        assert np.allclose([shadow_e], [e], atol=0.2)
