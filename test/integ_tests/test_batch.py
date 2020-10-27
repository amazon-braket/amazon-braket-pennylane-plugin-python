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

"""Tests for batch execution of jobs on AWS"""
import pennylane as qml
import pytest
from pennylane import numpy as np

from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice


@pytest.mark.parametrize("shots", [0])
def test_batch_execution_of_gradient(device, shots, mocker):
    """Test that the output of a parallelized execution of batch circuits to evaluate the
    gradient is correct in comparison to default.qubit."""
    qml.enable_tape()
    qubits = 2
    layers = 2

    dev_aws = device(qubits)

    if isinstance(dev_aws, BraketLocalQubitDevice):
        pytest.skip("Parallelized batch execution is only supported on the remote AWS device")

    dev_aws._parallel = True

    dev_default = qml.device("default.qubit", wires=qubits)

    def func(weights):
        qml.templates.StronglyEntanglingLayers(weights, wires=range(qubits))
        return qml.expval(qml.PauliZ(0))

    qnode_aws = qml.QNode(func, dev_aws)
    qnode_default = qml.QNode(func, dev_default)

    weights = qml.init.strong_ent_layers_uniform(layers, qubits)

    dfunc_aws = qml.grad(qnode_aws)
    dfunc_default = qml.grad(qnode_default)

    spy1 = mocker.spy(BraketAwsQubitDevice, "_batch_execute_async")
    spy2 = mocker.spy(BraketAwsQubitDevice, "execute")

    res_aws = dfunc_aws(weights)
    res_default = dfunc_default(weights)

    assert np.allclose(res_aws, res_default)
    spy1.assert_called_once()
    spy2.assert_called_once()  # For a forward pass

    expected_circuits = qubits * layers * 3 * 2
    assert len(spy1.call_args_list[0][0][1]) == expected_circuits

    qml.disable_tape()
