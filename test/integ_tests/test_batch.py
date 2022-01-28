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

"""Tests for batch execution of jobs on AWS"""

import pennylane as qml
import pytest
from braket.aws import AwsDevice
from pennylane import numpy as np

from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice


@pytest.mark.parametrize("shots", [None])
def test_batch_execution_of_gradient(device, shots, mocker):
    """Test that the output of a parallelized execution of batch circuits to evaluate the
    gradient is correct in comparison to default.qubit."""
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

    shape = qml.templates.StronglyEntanglingLayers.shape(layers, qubits)
    weights = np.random.random(shape)

    dfunc_aws = qml.grad(qnode_aws)
    dfunc_default = qml.grad(qnode_default)

    spy1 = mocker.spy(BraketAwsQubitDevice, "execute")
    spy2 = mocker.spy(BraketAwsQubitDevice, "batch_execute")
    spy3 = mocker.spy(AwsDevice, "run_batch")

    res_aws = dfunc_aws(weights)
    res_default = dfunc_default(weights)

    if qml.version() >= "0.20.0":
        assert np.allclose(res_aws, res_default)
        spy1.assert_not_called()
        assert len(spy2.call_args_list) == 2
        assert len(spy3.call_args_list) == 2

        expected_circuits = qubits * layers * 3 * 2
        assert len(spy2.call_args_list[0][0][1]) == 1  # First batch_execute called for forward pass
        assert (
            len(spy2.call_args_list[1][0][1]) == expected_circuits
        )  # Then called for backward pass
    else:
        assert np.allclose(res_aws, res_default)
        spy1.assert_called_once()  # For a forward pass
        spy2.assert_called_once()
        spy3.assert_called_once()

        expected_circuits = qubits * layers * 3 * 2
        assert len(spy2.call_args_list[0][0][1]) == expected_circuits


@pytest.mark.parametrize("shots", [None])
def test_batch_execution_of_gradient_torch(device, shots, mocker):
    """Test that the output of a parallelized execution of batch circuits to evaluate the
    gradient is correct in comparison to default.qubit when using the torch interface."""
    try:
        import torch
    except ImportError:
        pytest.skip("This test requires installation of torch")

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

    qnode_aws = qml.QNode(func, dev_aws, interface="torch")
    qnode_default = qml.QNode(func, dev_default, interface="torch")

    weights_aws = torch.tensor(
        qml.init.strong_ent_layers_uniform(layers, qubits, seed=1967), requires_grad=True
    )
    weights_default = torch.tensor(
        qml.init.strong_ent_layers_uniform(layers, qubits, seed=1967), requires_grad=True
    )

    spy1 = mocker.spy(BraketAwsQubitDevice, "execute")
    spy2 = mocker.spy(BraketAwsQubitDevice, "batch_execute")
    spy3 = mocker.spy(AwsDevice, "run_batch")

    out_aws = qnode_aws(weights_aws)
    out_default = qnode_default(weights_default)

    out_aws.backward()
    out_default.backward()

    res_aws = weights_aws.grad
    res_default = weights_default.grad

    assert np.allclose(res_aws, res_default)
    spy1.assert_called_once()  # For a forward pass
    spy2.assert_called_once()
    spy3.assert_called_once()

    expected_circuits = qubits * layers * 3 * 2
    assert len(spy2.call_args_list[0][0][1]) == expected_circuits


@pytest.mark.parametrize("shots", [None])
def test_batch_execution_of_gradient_tf(device, shots, mocker):
    """Test that the output of a parallelized execution of batch circuits to evaluate the
    gradient is correct in comparison to default.qubit when using the tf interface."""
    tf = pytest.importorskip("tensorflow", minversion="2.4")

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

    qnode_aws = qml.QNode(func, dev_aws, interface="tf")
    qnode_default = qml.QNode(func, dev_default, interface="tf")

    weights_aws = tf.Variable(qml.init.strong_ent_layers_uniform(layers, qubits, seed=1967))
    weights_default = tf.Variable(qml.init.strong_ent_layers_uniform(layers, qubits, seed=1967))

    spy1 = mocker.spy(BraketAwsQubitDevice, "execute")
    spy2 = mocker.spy(BraketAwsQubitDevice, "batch_execute")
    spy3 = mocker.spy(AwsDevice, "run_batch")

    with tf.GradientTape() as tape:
        out_aws = qnode_aws(weights_aws)

    res_aws = tape.gradient(out_aws, weights_aws)

    with tf.GradientTape() as tape:
        out_default = qnode_default(weights_default)

    res_default = tape.gradient(out_default, weights_default)

    assert np.allclose(res_aws, res_default)
    spy1.assert_called_once()  # For a forward pass
    spy2.assert_called_once()
    spy3.assert_called_once()

    expected_circuits = qubits * layers * 3 * 2
    assert len(spy2.call_args_list[0][0][1]) == expected_circuits
