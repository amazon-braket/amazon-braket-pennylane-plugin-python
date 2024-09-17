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
from pennylane import numpy as np

s3 = ("my-bucket", "my-prefix")

dev_sim = qml.device("braket.simulator", s3_destination_folder=s3, wires=2)
dev_managed_sim = qml.device(
    "braket.aws.qubit",
    device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    wires=3,
    shots=0,
)
dev_qpu = qml.device(
    "braket.rigetti",
    s3_destination_folder=s3,
    poll_timeout_seconds=1800,
    shots=10000,
    wires=2,
)


@qml.qnode(dev_sim)
def circuit(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(a, wires=1)
    return qml.expval(qml.PauliZ(1))


print(circuit(0.543))


@qml.qnode(dev_qpu)
def circuit(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(a, wires=1)
    return qml.expval(qml.PauliZ(1))


@qml.qnode(dev_managed_sim)
def circuit(a):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(a, wires=1)
    return qml.expval(qml.PauliZ(1))


x = np.array(0.543, requires_grad=False)
print(circuit(x))
print(qml.grad(circuit)(x))
print(qml.jacobian(circuit)(x))
