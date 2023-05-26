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

import json
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

import braket.ir as ir
import numpy as anp
import pennylane as qml
import pytest
from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask, AwsQuantumTaskBatch
from braket.circuits import Circuit, FreeParameter, Gate, Noise, Observable, result_types
from braket.circuits.noise_model import GateCriteria, NoiseModel, NoiseModelInstruction
from braket.device_schema import DeviceActionType
from braket.device_schema.gate_model_qpu_paradigm_properties_v1 import (
    GateModelQpuParadigmProperties,
)
from braket.device_schema.openqasm_device_action_properties import OpenQASMDeviceActionProperties
from braket.device_schema.pulse.pulse_device_action_properties_v1 import PulseDeviceActionProperties
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.simulator import BraketSimulator
from braket.task_result import GateModelTaskResult
from braket.tasks import GateModelQuantumTaskResult
from pennylane import QuantumFunctionError, QubitDevice
from pennylane import numpy as np
from pennylane.tape import QuantumScript, QuantumTape

import braket.pennylane_plugin.braket_device
from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice, __version__
from braket.pennylane_plugin.braket_device import BraketQubitDevice, Shots

SHOTS = 10000

ACTION_PROPERTIES = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {"name": "StateVector", "observables": None, "minShots": 0, "maxShots": 0},
                {
                    "name": "AdjointGradient",
                    "observables": ["x", "y", "z", "h", "i"],
                    "minShots": 0,
                    "maxShots": 0,
                },
            ],
        }
    )
)

ACTION_PROPERTIES_DM_DEVICE = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {"name": "StateVector", "observables": None, "minShots": 0, "maxShots": 0},
            ],
            "supportedPragmas": [
                "braket_noise_bit_flip",
                "braket_noise_depolarizing",
                "braket_noise_kraus",
                "braket_noise_pauli_channel",
                "braket_noise_generalized_amplitude_damping",
                "braket_noise_amplitude_damping",
                "braket_noise_phase_flip",
                "braket_noise_phase_damping",
                "braket_noise_two_qubit_dephasing",
                "braket_noise_two_qubit_depolarizing",
                "braket_unitary_matrix",
                "braket_result_type_sample",
                "braket_result_type_expectation",
                "braket_result_type_variance",
                "braket_result_type_probability",
                "braket_result_type_density_matrix",
            ],
        }
    )
)

GATE_MODEL_RESULT = GateModelTaskResult(
    **{
        "measurements": [[0, 0], [0, 0], [0, 0], [1, 1]],
        "measuredQubits": [0, 1],
        "taskMetadata": {
            "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
            "id": "task_arn",
            "shots": 100,
            "deviceId": "default",
        },
        "additionalMetadata": {
            "action": {
                "braketSchemaHeader": {"name": "braket.ir.openqasm.program", "version": "1"},
                "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
            },
        },
    }
)

RESULT = GateModelQuantumTaskResult.from_string(
    json.dumps(
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.gate_model_task_result",
                "version": "1",
            },
            "measurements": [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            "resultTypes": [
                {"type": {"targets": [0], "type": "probability"}, "value": [0.5, 0.5]},
                {
                    "type": {"observable": ["x"], "targets": [1], "type": "expectation"},
                    "value": 0.0,
                },
                {"type": {"observable": ["y"], "targets": [2], "type": "variance"}, "value": 0.1},
                {
                    "type": {"observable": ["z"], "targets": [3], "type": "sample"},
                    "value": [1, -1, 1, 1],
                },
            ],
            "measuredQubits": [0, 1, 2, 3],
            "taskMetadata": {
                "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
                "id": "task_arn",
                "shots": 0,
                "deviceId": "default",
            },
            "additionalMetadata": {
                "action": {
                    "braketSchemaHeader": {"name": "braket.ir.openqasm.program", "version": "1"},
                    "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
                },
            },
        }
    )
)
TASK = Mock()
TASK.result.return_value = RESULT
type(TASK).id = PropertyMock(return_value="task_arn")
TASK.state.return_value = "COMPLETED"
TASK_BATCH = Mock()
TASK_BATCH.results.return_value = [RESULT, RESULT]
type(TASK_BATCH).tasks = PropertyMock(return_value=[TASK, TASK])
SIM_TASK = Mock()
SIM_TASK.result.return_value.additional_metadata.simulatorMetadata.executionDuration = 1234
SIM_TASK.result.return_value.result_types = []
type(SIM_TASK).id = PropertyMock(return_value="task_arn")
SIM_TASK.state.return_value = "COMPLETED"
CIRCUIT = (
    Circuit()
    .h(0)
    .cnot(0, 1)
    .i(2)
    .i(3)
    .probability(target=[0])
    .expectation(observable=Observable.X(), target=1)
    .variance(observable=Observable.Y(), target=2)
    .sample(observable=Observable.Z(), target=3)
)

DEVICE_ARN = "baz"


def test_reset():
    """Tests that the members of the device are cleared on reset."""
    dev = _aws_device(wires=2)
    dev._circuit = CIRCUIT
    dev._task = TASK

    dev.reset()
    assert dev.circuit is None
    assert dev.task is None


def test_apply():
    """Tests that the correct Braket gate is applied for each PennyLane operation."""
    dev = _aws_device(wires=2)
    circuit = dev.apply([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])])
    assert circuit == Circuit().h(0).cnot(0, 1)


def test_apply_unique_parameters():
    """Tests that apply with unique_params=True applies the correct parametrized gates."""
    dev = _aws_device(wires=2)
    circuit = dev.apply(
        [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.RX(np.pi, wires=0),
            qml.RY(np.pi, wires=0),
            # note the gamma/p ordering doesn't affect the naming of the parameters below.
            qml.GeneralizedAmplitudeDamping(gamma=0.1, p=0.9, wires=0),
            qml.GeneralizedAmplitudeDamping(p=0.9, gamma=0.1, wires=0),
        ],
        use_unique_params=True,
    )
    expected = Circuit().h(0).cnot(0, 1).rx(0, FreeParameter("p_0"))
    expected = expected.ry(0, FreeParameter("p_1"))
    expected = expected.generalized_amplitude_damping(
        0,
        gamma=FreeParameter("p_2"),
        probability=FreeParameter("p_3"),
    )
    expected = expected.generalized_amplitude_damping(
        0,
        gamma=FreeParameter("p_4"),
        probability=FreeParameter("p_5"),
    )
    assert circuit == expected


def test_apply_unused_qubits():
    """Tests that the correct circuit is created when not all wires in the device are used."""
    dev = _aws_device(wires=4)
    operations = [qml.Hadamard(wires=1), qml.CNOT(wires=[1, 2]), qml.RX(np.pi / 2, wires=2)]
    rotations = [qml.RY(np.pi, wires=1)]
    circuit = dev.apply(operations, rotations)

    assert circuit == Circuit().h(1).cnot(1, 2).rx(2, np.pi / 2).ry(1, np.pi).i(0).i(3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_apply_unsupported():
    """Tests that apply() throws NotImplementedError when it encounters an unknown gate."""
    dev = _aws_device(wires=2)
    mock_op = Mock()
    mock_op.name = "foo"
    mock_op.parameters = []

    operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), mock_op]
    dev.apply(operations)


def test_apply_unwrap_tensor():
    """Test that apply() unwraps tensors from the PennyLane version of NumPy into standard NumPy
    arrays (or floats)"""
    dev = _aws_device(wires=1)

    a = anp.array(0.6)  # array
    b = np.array(0.5, requires_grad=True)  # tensor

    operations = [qml.RY(a, wires=0), qml.RX(b, wires=[0])]
    rotations = []
    circuit = dev.apply(operations, rotations)

    angles = [op.operator.angle for op in circuit.instructions]
    assert not any([isinstance(angle, np.tensor) for angle in angles])


@patch.object(AwsDevice, "run")
def test_execute(mock_run):
    mock_run.return_value = TASK
    dev = _aws_device(wires=4, foo="bar")

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.QubitUnitary(1 / np.sqrt(2) * np.tensor([[1, 1], [1, -1]], requires_grad=True), wires=0)
        qml.RX(0.432, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))

    # If the tape is constructed with a QNode, only the parameters marked requires_grad=True
    # will appear
    circuit._trainable_params = [0]

    results = dev.execute(circuit)

    assert np.allclose(
        results[0], RESULT.get_value_by_result_type(result_types.Probability(target=[0]))
    )
    assert np.allclose(
        results[1],
        RESULT.get_value_by_result_type(
            result_types.Expectation(observable=Observable.X(), target=1)
        ),
    )
    assert np.allclose(
        results[2],
        RESULT.get_value_by_result_type(result_types.Variance(observable=Observable.Y(), target=2)),
    )
    assert np.allclose(
        results[3],
        RESULT.get_value_by_result_type(result_types.Sample(observable=Observable.Z(), target=3)),
    )
    assert dev.task == TASK
    EXPECTED_CIRC = (
        Circuit()
        .h(0)
        .unitary([0], 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))
        .rx(0, 0.432)
        .cnot(0, 1)
        .i(2)
        .i(3)
        .probability(target=[0])
        .expectation(observable=Observable.X(), target=1)
        .variance(observable=Observable.Y(), target=2)
        .sample(observable=Observable.Z(), target=3)
    )
    mock_run.assert_called_with(
        EXPECTED_CIRC,
        s3_destination_folder=("foo", "bar"),
        shots=SHOTS,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
        inputs={},
    )


@patch.object(AwsDevice, "run")
def test_execute_legacy(mock_run):
    mock_run.return_value = TASK
    dev = _aws_device(wires=4, foo="bar")

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.QubitUnitary(1 / np.sqrt(2) * np.tensor([[1, 1], [1, -1]], requires_grad=True), wires=0)
        qml.RX(0.432, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))

    # If the tape is constructed with a QNode, only the parameters marked requires_grad=True
    # will appear
    circuit._trainable_params = [0]

    results = dev._execute_legacy(circuit)

    assert np.allclose(
        results[0], RESULT.get_value_by_result_type(result_types.Probability(target=[0]))
    )
    assert np.allclose(
        results[1],
        RESULT.get_value_by_result_type(
            result_types.Expectation(observable=Observable.X(), target=1)
        ),
    )
    assert np.allclose(
        results[2],
        RESULT.get_value_by_result_type(result_types.Variance(observable=Observable.Y(), target=2)),
    )
    assert np.allclose(
        results[3],
        RESULT.get_value_by_result_type(result_types.Sample(observable=Observable.Z(), target=3)),
    )
    assert dev.task == TASK
    EXPECTED_CIRC = (
        Circuit()
        .h(0)
        .unitary([0], 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))
        .rx(0, 0.432)
        .cnot(0, 1)
        .i(2)
        .i(3)
        .probability(target=[0])
        .expectation(observable=Observable.X(), target=1)
        .variance(observable=Observable.Y(), target=2)
        .sample(observable=Observable.Z(), target=3)
    )
    mock_run.assert_called_with(
        EXPECTED_CIRC,
        s3_destination_folder=("foo", "bar"),
        shots=SHOTS,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
        inputs={},
    )


CIRCUIT_1 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.432, wires=0),
        qml.RY(0.543, wires=0),
    ],
    measurements=[qml.expval(qml.PauliX(1))],
)
CIRCUIT_1.trainable_params = [0]

CIRCUIT_2 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.432, wires=0),
        qml.RY(0.543, wires=0),
    ],
    measurements=[qml.expval(2 * qml.PauliX(0) @ qml.PauliY(1))],
)
CIRCUIT_2.trainable_params = [0, 1]

CIRCUIT_3 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.432, wires=0),
        qml.RY(0.543, wires=0),
    ],
    measurements=[
        qml.expval(2 * qml.PauliX(0) @ qml.PauliY(1) + 0.75 * qml.PauliY(0) @ qml.PauliZ(1)),
    ],
)
CIRCUIT_3.trainable_params = [0, 1]

CIRCUIT_4 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.432, wires=0),
        qml.RY(0.543, wires=0),
    ],
    measurements=[qml.expval(qml.PauliX(1))],
)
CIRCUIT_4.trainable_params = []

PARAMS_5 = np.array([0.432, 0.543], requires_grad=True)
CIRCUIT_5 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(PARAMS_5[0], wires=0),
        qml.RY(PARAMS_5[1], wires=0),
    ],
    measurements=[qml.var(qml.PauliX(0) @ qml.PauliY(1))],
)
CIRCUIT_5.trainable_params = [0, 1]

PARAM_6 = np.tensor(0.432, requires_grad=True)
CIRCUIT_6 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.QubitUnitary(
            1 / np.sqrt(2) * np.tensor([[1, 1], [1, -1]], requires_grad=True), wires=0
        ),
        qml.RX(PARAM_6, wires=0),
        qml.QubitUnitary(1 / np.sqrt(2) * anp.array([[1, 1], [1, -1]]), wires=0),
        qml.CNOT(wires=[0, 1]),
    ],
    measurements=[qml.expval(qml.PauliX(1))],
)


@patch.object(AwsDevice, "run")
@pytest.mark.parametrize(
    "pl_circ, expected_braket_circ, wires, expected_inputs, result_types, expected_pl_result",
    [
        (
            CIRCUIT_1,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, FreeParameter("p_0"))
            .ry(0, 0.543)
            .adjoint_gradient(observable=Observable.X(), target=1, parameters=["p_0"]),
            2,
            {"p_0": 0.432},
            [
                {
                    "type": {
                        "observable": "x()",
                        "targets": [[0]],
                        "parameters": ["p_0", "p_1"],
                        "type": "adjoint_gradient",
                    },
                    "value": {
                        "gradient": {"p_0": -0.194399, "p_1": 0.9316158},
                        "expectation": 0.0,
                    },
                },
            ],
            [
                (
                    np.tensor([0.0], requires_grad=True),
                    np.tensor([-0.194399, 0.9316158], requires_grad=True),
                )
            ],
        ),
        (
            CIRCUIT_2,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, FreeParameter("p_0"))
            .ry(0, FreeParameter("p_1"))
            .adjoint_gradient(
                observable=(2 * Observable.X() @ Observable.Y()),
                target=[0, 1],
                parameters=["p_0", "p_1"],
            ),
            2,
            {"p_0": 0.432, "p_1": 0.543},
            [
                {
                    "type": {
                        "observable": "2.0 * x() @ y()",
                        "targets": [[0, 1]],
                        "parameters": ["p_0", "p_1"],
                        "type": "adjoint_gradient",
                    },
                    "value": {
                        "gradient": {"p_0": -0.01894799, "p_1": 0.9316158},
                        "expectation": 0.0,
                    },
                },
            ],
            [
                (
                    np.tensor([0.0], requires_grad=True),
                    np.tensor([-0.01894799, 0.9316158], requires_grad=True),
                )
            ],
        ),
        (
            CIRCUIT_3,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, FreeParameter("p_0"))
            .ry(0, FreeParameter("p_1"))
            .adjoint_gradient(
                observable=(
                    2 * Observable.X() @ Observable.Y() + 0.75 * Observable.Y() @ Observable.Z()
                ),
                target=[[0, 1], [0, 1]],
                parameters=["p_0", "p_1"],
            ),
            2,
            {"p_0": 0.432, "p_1": 0.543},
            [
                {
                    "type": {
                        "observable": "2.0 * x() @ y() + .75 * y() @ z()",
                        "targets": [[0, 1], [0, 1]],
                        "parameters": ["p_0", "p_1"],
                        "type": "adjoint_gradient",
                    },
                    "value": {
                        "gradient": {"p_0": -0.01894799, "p_1": 0.9316158},
                        "expectation": 0.0,
                    },
                },
            ],
            [
                (
                    np.tensor([0.0], requires_grad=True),
                    np.tensor([-0.01894799, 0.9316158], requires_grad=True),
                )
            ],
        ),
    ],
)
def test_execute_with_gradient(
    mock_run,
    pl_circ,
    expected_braket_circ,
    wires,
    expected_inputs,
    result_types,
    expected_pl_result,
):
    task = Mock()
    type(task).id = PropertyMock(return_value="task_arn")
    task.state.return_value = "COMPLETED"
    task.result.return_value = get_test_result_object(rts=result_types)
    mock_run.return_value = task
    dev = _aws_device(wires=wires, foo="bar", shots=0, device_type=AwsDeviceType.SIMULATOR)

    results = dev.execute(pl_circ, compute_gradient=True)

    assert dev.task == task

    mock_run.assert_called_with(
        expected_braket_circ,
        s3_destination_folder=("foo", "bar"),
        shots=0,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
        inputs=expected_inputs,
    )
    assert (results[0] == expected_pl_result[0][0]).all()
    assert (results[1] == expected_pl_result[0][1]).all()


@patch.object(AwsDevice, "run")
def test_execute_tracker(mock_run):
    """Asserts tracker stores information during execute when active"""
    mock_run.side_effect = [TASK, SIM_TASK, SIM_TASK, TASK]
    dev = _aws_device(wires=4, foo="bar")

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.probs(wires=(0,))

    callback = Mock()
    with qml.Tracker(dev, callback=callback) as tracker:
        dev.execute(circuit)
        dev.execute(circuit)
        dev.execute(circuit)
    dev.execute(circuit)

    latest = {
        "executions": 1,
        "shots": SHOTS,
        "braket_task_id": "task_arn",
        "braket_simulator_ms": 1234,
        "braket_simulator_billed_ms": 3000,
    }
    history = {
        "executions": [1, 1, 1],
        "shots": [SHOTS, SHOTS, SHOTS],
        "braket_task_id": ["task_arn", "task_arn", "task_arn"],
        "braket_simulator_ms": [1234, 1234],
        "braket_simulator_billed_ms": [3000, 3000],
    }
    totals = {
        "executions": 3,
        "shots": 3 * SHOTS,
        "braket_simulator_ms": 2468,
        "braket_simulator_billed_ms": 6000,
    }
    assert tracker.latest == latest
    assert tracker.history == history
    assert tracker.totals == totals

    callback.assert_called_with(latest=latest, history=history, totals=totals)


def _aws_device_mock_init(self, arn, aws_session):
    self._arn = arn


@patch.object(AwsDevice, "__init__", _aws_device_mock_init)
@patch.object(AwsDevice, "aws_session", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "type", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "properties")
@pytest.mark.parametrize(
    "action_props, shots, expected_use_grouping",
    [
        (
            OpenQASMDeviceActionProperties.parse_raw(
                json.dumps(
                    {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
                        "supportedResultTypes": [
                            {
                                "name": "StateVector",
                                "observables": None,
                                "minShots": 0,
                                "maxShots": 0,
                            },
                        ],
                    }
                )
            ),
            0,
            True,
        ),
        (
            OpenQASMDeviceActionProperties.parse_raw(
                json.dumps(
                    {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
                        "supportedResultTypes": [
                            {
                                "name": "StateVector",
                                "observables": None,
                                "minShots": 0,
                                "maxShots": 0,
                            },
                            {
                                "name": "AdjointGradient",
                                "observables": ["x", "y", "z", "h", "i"],
                                "minShots": 0,
                                "maxShots": 0,
                            },
                        ],
                    }
                )
            ),
            10,
            True,
        ),
        (
            OpenQASMDeviceActionProperties.parse_raw(
                json.dumps(
                    {
                        "actionType": "braket.ir.openqasm.program",
                        "version": ["1"],
                        "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
                        "supportedResultTypes": [
                            {
                                "name": "StateVector",
                                "observables": None,
                                "minShots": 0,
                                "maxShots": 0,
                            },
                            {
                                "name": "AdjointGradient",
                                "observables": ["x", "y", "z", "h", "i"],
                                "minShots": 0,
                                "maxShots": 0,
                            },
                        ],
                    }
                )
            ),
            0,
            # Should be disabled only when AdjGrad is present and shots = 0
            False,
        ),
    ],
)
def test_use_grouping(
    properties_mock, type_mock, session_mock, action_props, shots, expected_use_grouping
):
    """Tests that grouping is enabled except when AdjointGradient is present"""
    properties_mock.action = {DeviceActionType.OPENQASM: action_props}
    properties_mock.return_value.action.return_value = {DeviceActionType.OPENQASM: action_props}
    type_mock.return_value = AwsDeviceType.SIMULATOR
    device = BraketAwsQubitDevice(
        wires=1,
        device_arn=DEVICE_ARN,
        aws_session=Mock(),
        shots=shots,
    )
    assert device.use_grouping == expected_use_grouping


def test_pl_to_braket_circuit():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))

    braket_circuit_true = (
        Circuit()
        .rx(0, 0.2)
        .rx(1, 0.3)
        .cnot(0, 1)
        .add_result_type(result_types.Expectation(observable=Observable.Z(), target=0))
    )

    braket_circuit = dev._pl_to_braket_circuit(tape)

    assert braket_circuit_true == braket_circuit


def test_pl_to_braket_circuit_compute_gradient():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit
    with a gradient and unique parameters when compute_gradient is True"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))

    expected_braket_circuit = (
        Circuit()
        .rx(0, FreeParameter("p_0"))
        .rx(1, FreeParameter("p_1"))
        .cnot(0, 1)
        .add_result_type(
            result_types.AdjointGradient(
                observable=Observable.Z(), target=0, parameters=["p_0", "p_1"]
            )
        )
    )

    actual_braket_circuit = dev._pl_to_braket_circuit(
        tape,
        compute_gradient=True,
        trainable_indices=frozenset(dev._get_trainable_parameters(tape).keys()),
    )

    assert expected_braket_circuit == actual_braket_circuit


def test_pl_to_braket_circuit_compute_gradient_hamiltonian_tensor_product_terms():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit"""
    """when the Hamiltonian has multiple tensor product ops and we compute the gradient"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(
            qml.Hamiltonian(
                (2, 3),
                (
                    qml.PauliX(wires=0) @ qml.PauliX(wires=1),
                    qml.PauliY(wires=0) @ qml.PauliY(wires=1),
                ),
            )
        )

    braket_obs = 2 * Observable.X() @ Observable.X() + 3 * Observable.Y() @ Observable.Y()
    braket_circuit_true = (
        Circuit()
        .rx(0, FreeParameter("p_0"))
        .rx(1, FreeParameter("p_1"))
        .cnot(0, 1)
        .add_result_type(
            result_types.AdjointGradient(
                observable=braket_obs, target=[[0, 1], [0, 1]], parameters=["p_0", "p_1"]
            )
        )
    )

    braket_circuit = dev._pl_to_braket_circuit(
        tape,
        compute_gradient=True,
        trainable_indices=frozenset(dev._get_trainable_parameters(tape).keys()),
    )

    assert braket_circuit_true == braket_circuit


def test_pl_to_braket_circuit_gradient_fails_with_multiple_observables():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit
    with a gradient and unique parameters when compute_gradient is True"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))
        qml.expval(qml.PauliZ(0))
    with pytest.raises(
        ValueError,
        match="Braket can only compute gradients for circuits with a single expectation"
        " observable, not ",
    ):
        dev._pl_to_braket_circuit(tape, compute_gradient=True)


def test_pl_to_braket_circuit_gradient_fails_with_invalid_observable():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit
    with a gradient and unique parameters when compute_gradient is True"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.var(qml.PauliZ(0))
    with pytest.raises(
        ValueError,
        match="Braket can only compute gradients for circuits with a single expectation"
        " observable, not a",
    ):
        dev._pl_to_braket_circuit(tape, compute_gradient=True)


def test_pl_to_braket_circuit_hamiltonian():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.Hamiltonian((2, 3), (qml.PauliX(wires=0), qml.PauliY(wires=1))))

    braket_circuit_true = (
        Circuit()
        .rx(0, 0.2)
        .rx(1, 0.3)
        .cnot(0, 1)
        .expectation(Observable.X(), [0])
        .expectation(Observable.Y(), [1])
    )

    braket_circuit = dev._pl_to_braket_circuit(tape)

    assert braket_circuit_true == braket_circuit


def test_pl_to_braket_circuit_hamiltonian_tensor_product_terms():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit"""
    """when the Hamiltonian has multiple tensor product ops"""
    dev = _aws_device(wires=2, foo="bar")

    with QuantumTape() as tape:
        qml.RX(0.2, wires=0)
        qml.RX(0.3, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(
            qml.Hamiltonian(
                (2, 3),
                (
                    qml.PauliX(wires=0) @ qml.PauliX(wires=1),
                    qml.PauliY(wires=0) @ qml.PauliY(wires=1),
                ),
            )
        )

    braket_circuit_true = (
        Circuit()
        .rx(0, 0.2)
        .rx(1, 0.3)
        .cnot(0, 1)
        .expectation(Observable.X() @ Observable.X(), [0, 1])
        .expectation(Observable.Y() @ Observable.Y(), [0, 1])
    )

    braket_circuit = dev._pl_to_braket_circuit(tape)

    assert braket_circuit_true == braket_circuit


def test_bad_statistics():
    """Test if a QuantumFunctionError is raised for an invalid return type"""
    dev = _aws_device(wires=1, foo="bar")
    observable = qml.Identity(wires=0, do_queue=False)
    observable.return_type = None

    with pytest.raises(QuantumFunctionError, match="Unsupported return type specified"):
        dev.statistics(None, [observable])


def test_batch_execute_non_parallel(monkeypatch):
    """Test if the batch_execute() method simply calls the inherited method if parallel=False"""
    dev = _aws_device(wires=2, foo="bar", parallel=False)
    assert dev.parallel is False

    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "batch_execute", lambda self, circuits: 1967)
        res = dev.batch_execute([])
        assert res == 1967


@patch.object(AwsDevice, "run")
def test_batch_execute_non_parallel_tracker(mock_run):
    """Tests tracking for a non-parallel batch"""
    mock_run.return_value = TASK
    dev = _aws_device(wires=2, foo="bar", parallel=False)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.probs(wires=(0,))

    callback = Mock()
    with qml.Tracker(dev, callback=callback) as tracker:
        dev.batch_execute([circuit, circuit])
    dev.batch_execute([circuit])

    latest = {"batches": 1, "batch_len": 2}
    history = {
        "executions": [1, 1],
        "shots": [SHOTS, SHOTS],
        "batches": [1],
        "batch_len": [2],
        "braket_task_id": ["task_arn", "task_arn"],
    }
    totals = {"executions": 2, "shots": 2 * SHOTS, "batches": 1, "batch_len": 2}
    assert tracker.latest == latest
    assert tracker.history == history
    assert tracker.totals == totals

    callback.assert_called_with(latest=latest, history=history, totals=totals)


@patch.object(AwsDevice, "run_batch")
def test_batch_execute_parallel(mock_run_batch):
    """Test batch_execute(parallel=True) correctly calls batch execution methods in Braket SDK"""
    mock_run_batch.return_value = TASK_BATCH
    dev = _aws_device(wires=4, foo="bar", parallel=True)
    assert dev.parallel is True

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))

    circuits = [circuit, circuit]
    batch_results = dev.batch_execute(circuits)
    for results in batch_results:
        assert np.allclose(
            results[0], RESULT.get_value_by_result_type(result_types.Probability(target=[0]))
        )
        assert np.allclose(
            results[1],
            RESULT.get_value_by_result_type(
                result_types.Expectation(observable=Observable.X(), target=1)
            ),
        )
        assert np.allclose(
            results[2],
            RESULT.get_value_by_result_type(
                result_types.Variance(observable=Observable.Y(), target=2)
            ),
        )
        assert np.allclose(
            results[3],
            RESULT.get_value_by_result_type(
                result_types.Sample(observable=Observable.Z(), target=3)
            ),
        )

    mock_run_batch.assert_called_with(
        [CIRCUIT, CIRCUIT],
        s3_destination_folder=("foo", "bar"),
        shots=SHOTS,
        max_parallel=None,
        max_connections=AwsQuantumTaskBatch.MAX_CONNECTIONS_DEFAULT,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
    )


@patch.object(AwsDevice, "run_batch")
def test_batch_execute_parallel_tracker(mock_run_batch):
    """Asserts tracker updates during parallel execution"""

    mock_run_batch.return_value = TASK_BATCH
    type(TASK_BATCH).unsuccessful = PropertyMock(return_value={})
    dev = _aws_device(wires=1, foo="bar", parallel=True)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.probs(wires=(0,))

    circuits = [circuit, circuit]

    callback = Mock()
    with qml.Tracker(dev, callback=callback) as tracker:
        dev.batch_execute(circuits)
    dev.batch_execute(circuits)

    latest = {"batches": 1, "executions": 2, "shots": 2 * SHOTS}
    history = {
        "batches": [1],
        "executions": [2],
        "shots": [2 * SHOTS],
        "braket_task_id": ["task_arn", "task_arn"],
    }
    totals = {"batches": 1, "executions": 2, "shots": 2 * SHOTS}
    assert tracker.latest == latest
    assert tracker.history == history
    assert tracker.totals == totals

    callback.assert_called_with(latest=latest, history=history, totals=totals)


@patch.object(AwsDevice, "run_batch")
def test_batch_execute_partial_fail_parallel_tracker(mock_run_batch):
    """Asserts tracker updates during a partial failure of parallel execution"""

    FAIL_TASK = Mock()
    FAIL_TASK.result.return_value = None
    type(FAIL_TASK).id = PropertyMock(return_value="failed_task_arn")
    FAIL_TASK.state.return_value = "FAILED"
    FAIL_BATCH = Mock()
    FAIL_BATCH.results.side_effect = RuntimeError("tasks failed to complete")
    type(FAIL_BATCH).tasks = PropertyMock(return_value=[SIM_TASK, FAIL_TASK])
    type(FAIL_BATCH).unsuccessful = PropertyMock(return_value={"failed_task_arn"})

    mock_run_batch.return_value = FAIL_BATCH
    dev = _aws_device(wires=1, foo="bar", parallel=True)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.probs(wires=(0,))

    circuits = [circuit, circuit]

    callback = Mock()
    try:
        with qml.Tracker(dev, callback=callback) as tracker:
            dev.batch_execute(circuits)
        dev.batch_execute(circuits)
    except RuntimeError:
        pass

    latest = {"batches": 1, "executions": 1, "shots": 1 * SHOTS}
    history = {
        "batches": [1],
        "executions": [1],
        "shots": [1 * SHOTS],
        "braket_task_id": ["task_arn"],
        "braket_failed_task_id": ["failed_task_arn"],
        "braket_simulator_ms": [1234],
        "braket_simulator_billed_ms": [3000],
    }
    totals = {
        "batches": 1,
        "executions": 1,
        "shots": 1 * SHOTS,
        "braket_simulator_ms": 1234,
        "braket_simulator_billed_ms": 3000,
    }
    assert tracker.latest == latest
    assert tracker.history == history
    assert tracker.totals == totals

    callback.assert_called_with(latest=latest, history=history, totals=totals)


@pytest.mark.parametrize("old_return_type", [True, False])
@patch.object(AwsDevice, "run")
def test_execute_all_samples(mock_run, old_return_type):
    result = GateModelQuantumTaskResult.from_string(
        json.dumps(
            {
                "braketSchemaHeader": {
                    "name": "braket.task_result.gate_model_task_result",
                    "version": "1",
                },
                "measurements": [[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]],
                "resultTypes": [
                    {
                        "type": {"observable": ["h", "i"], "targets": [0, 1], "type": "sample"},
                        "value": [1, -1, 1, 1],
                    },
                    {
                        "type": {
                            "observable": [[[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]],
                            "targets": [2],
                            "type": "sample",
                        },
                        "value": [1, -1, 1, 1],
                    },
                ],
                "measuredQubits": [0, 1, 3],
                "taskMetadata": {
                    "braketSchemaHeader": {
                        "name": "braket.task_result.task_metadata",
                        "version": "1",
                    },
                    "id": "task_arn",
                    "shots": 0,
                    "deviceId": "default",
                },
                "additionalMetadata": {
                    "action": {
                        "braketSchemaHeader": {
                            "name": "braket.ir.openqasm.program",
                            "version": "1",
                        },
                        "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
                    },
                },
            }
        )
    )
    task = Mock()
    task.result.return_value = result
    mock_run.return_value = task
    dev = _aws_device(wires=3)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.sample(qml.Hadamard(0) @ qml.Identity(1))
        qml.sample(qml.Hermitian(np.array([[0, 1], [1, 0]]), wires=[2]))

    if old_return_type:
        qml.disable_return()
    results = dev.execute(circuit)
    qml.enable_return()

    assert len(results) == 2
    assert results[0].shape == (4,)
    assert results[1].shape == (4,)


@pytest.mark.parametrize("old_return_type", [True, False])
@patch.object(AwsDevice, "run")
def test_execute_some_samples(mock_run, old_return_type):
    """Tests that a combination with sample returns correctly and does not put single-number
    results in superflous arrays"""
    result = GateModelQuantumTaskResult.from_string(
        json.dumps(
            {
                "braketSchemaHeader": {
                    "name": "braket.task_result.gate_model_task_result",
                    "version": "1",
                },
                "measurements": [[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]],
                "resultTypes": [
                    {
                        "type": {"observable": ["h", "i"], "targets": [0, 1], "type": "sample"},
                        "value": [1, -1, 1, 1],
                    },
                    {
                        "type": {"observable": ["z"], "targets": [2], "type": "expectation"},
                        "value": 0.0,
                    },
                ],
                "measuredQubits": [0, 1, 3],
                "taskMetadata": {
                    "braketSchemaHeader": {
                        "name": "braket.task_result.task_metadata",
                        "version": "1",
                    },
                    "id": "task_arn",
                    "shots": 0,
                    "deviceId": "default",
                },
                "additionalMetadata": {
                    "action": {
                        "braketSchemaHeader": {
                            "name": "braket.ir.openqasm.program",
                            "version": "1",
                        },
                        "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
                    },
                },
            }
        )
    )
    if old_return_type:
        qml.disable_return()
    task = Mock()
    task.result.return_value = result
    mock_run.return_value = task
    dev = _aws_device(wires=3)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.sample(qml.Hadamard(0) @ qml.Identity(1))
        qml.expval(qml.PauliZ(2))

    results = dev.execute(circuit)
    qml.enable_return()

    assert len(results) == 2
    assert results[0].shape == (4,)
    assert isinstance(results[0], np.ndarray)
    assert results[1] == 0.0


@pytest.mark.xfail(raises=ValueError)
@patch.object(AwsDevice, "name", new_callable=mock.PropertyMock)
def test_non_circuit_device(name_mock):
    """Tests that BraketDevice cannot be instantiated with a non-circuit AwsDevice"""
    _bad_aws_device(wires=2)


def test_simulator_default_shots():
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = _aws_device(wires=2, device_type=AwsDeviceType.SIMULATOR, shots=Shots.DEFAULT)
    assert dev.shots is None
    assert dev.analytic


def test_simulator_0_shots():
    """Tests that simulator devices are analytic if ``shots`` is zero"""
    dev = _aws_device(wires=2, device_type=AwsDeviceType.SIMULATOR, shots=0)
    assert dev.shots is None
    assert dev.analytic


def test_simulator_none_shots():
    """Tests that simulator devices are analytic if ``shots`` is None"""
    dev = _aws_device(wires=2, device_type=AwsDeviceType.SIMULATOR, shots=None)
    assert dev.shots is None
    assert dev.analytic


@pytest.mark.parametrize("backend", ["default", "braket_sv", "braket_dm"])
def test_local_default_shots(backend):
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = BraketLocalQubitDevice(wires=2, backend=backend)
    assert dev.shots is None
    assert dev.analytic


@pytest.mark.parametrize("backend", ["default", "braket_sv", "braket_dm"])
def test_local_zero_shots(backend):
    """Test that the local simulator device is analytic if ``shots=0``"""
    dev = BraketLocalQubitDevice(wires=2, backend=backend, shots=0)
    assert dev.shots is None
    assert dev.analytic


@pytest.mark.parametrize("backend", ["default", "braket_sv", "braket_dm"])
def test_local_none_shots(backend):
    """Tests that the simulator devices are analytic if ``shots`` is specified to be `None`."""
    dev = BraketLocalQubitDevice(wires=2, backend=backend, shots=None)
    assert dev.shots is None
    assert dev.analytic


@patch.object(LocalSimulator, "run")
@pytest.mark.parametrize("shots", [0, 1000])
@pytest.mark.parametrize("backend", ["default", "braket_sv", "braket_dm"])
def test_local_qubit_execute(mock_run, shots, backend):
    """Tests that the local qubit device is run with the correct arguments"""
    mock_run.return_value = TASK
    dev = BraketLocalQubitDevice(wires=4, backend=backend, shots=shots, foo="bar")

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))

    dev.execute(circuit)
    mock_run.assert_called_with(CIRCUIT, shots=shots, foo="bar", inputs={})


def test_qpu_default_shots():
    """Tests that QPU devices have the right default value for ``shots``"""
    dev = _aws_device(wires=2, shots=Shots.DEFAULT)
    assert dev.shots == AwsDevice.DEFAULT_SHOTS_QPU
    assert not dev.analytic


@pytest.mark.xfail(raises=ValueError)
def test_qpu_0_shots():
    """Tests that QPUs can not be instantiated with 0 shots"""
    _aws_device(wires=2, shots=0)


@pytest.mark.xfail(raises=ValueError)
def test_invalid_device_type():
    """Tests that BraketDevice cannot be instantiated with an unknown device type"""
    _aws_device(wires=2, device_type="foo", shots=None)


def test_wires():
    """Test if the apply method supports custom wire labels"""

    wires = ["A", 0, "B", -1]
    dev = _aws_device(wires=wires, device_type=AwsDeviceType.SIMULATOR, shots=None)

    ops = [qml.RX(0.1, wires="A"), qml.CNOT(wires=[0, "B"]), qml.RY(0.3, wires=-1)]
    target_wires = [[0], [1, 2], [3]]
    circ = dev.apply(ops)

    for op, targets in zip(circ.instructions, target_wires):
        wires = op.target
        for w, t in zip(wires, targets):
            assert w == t


def test_supported_ops_set(monkeypatch):
    """Test that the supported operations are set correctly when the device is
    created."""

    test_ops = ["TestOperation"]
    with monkeypatch.context() as m:
        m.setattr(braket.pennylane_plugin.braket_device, "supported_operations", lambda x: test_ops)
        dev = _aws_device(wires=2)
        assert dev.operations == test_ops


def test_projection():
    """Test that the Projector observable is correctly supported."""
    wires = 2
    dev = BraketLocalQubitDevice(wires=wires)

    thetas = [1.5, 1.6]
    p_01 = np.cos(thetas[0] / 2) ** 2 * np.sin(thetas[1] / 2) ** 2
    p_10 = np.sin(thetas[0] / 2) ** 2 * np.cos(thetas[1] / 2) ** 2

    def f(thetas, **kwargs):
        [qml.RY(thetas[i], wires=i) for i in range(wires)]

    projector_01 = qml.Projector([0, 1], wires=range(wires))
    projector_10 = qml.Projector([1, 0], wires=range(wires))

    # 01 case
    @qml.qnode(dev)
    def f_01(thetas, measure_type):
        f(thetas)
        return measure_type(projector_01)

    expval_01 = f_01(thetas, qml.expval)
    assert np.allclose(expval_01, p_01)

    var_01 = f_01(thetas, qml.var)
    assert np.allclose(var_01, p_01 - p_01**2)

    samples = f_01(thetas, qml.sample, shots=100).tolist()
    assert set(samples) == {0, 1}

    # 10 case
    @qml.qnode(dev)
    def f_10(thetas, measure_type):
        f(thetas)
        return measure_type(projector_10)

    exp_10 = f_10(thetas, qml.expval)
    assert np.allclose(exp_10, p_10)

    var_10 = f_10(thetas, qml.var)
    assert np.allclose(var_10, p_10 - p_10**2)

    samples = f_10(thetas, qml.sample, shots=100).tolist()
    assert set(samples) == {0, 1}


@pytest.mark.xfail(raises=AttributeError)
def test_none_device():
    """Tests that an error is thrown when device is not given"""
    dev = DummyLocalQubitDevice(wires=2, device=None, shots=1000)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0, 1])
    dev.execute(circuit)


@pytest.mark.xfail(raises=NotImplementedError)
def test_run_task_unimplemented():
    """Tests that an error is thrown when _run_task is not implemented"""
    dummy = DummyCircuitSimulator()
    dev = DummyLocalQubitDevice(wires=2, device=dummy, shots=1000)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0, 1])
    dev.execute(circuit)


@patch("braket.pennylane_plugin.braket_device.AwsDevice")
def test_add_braket_user_agent_invoked(aws_device_mock):
    aws_device_mock_instance = aws_device_mock.return_value
    aws_device_mock_instance.properties.action = {DeviceActionType.OPENQASM: ACTION_PROPERTIES}
    aws_device_mock_instance.type = AwsDeviceType.SIMULATOR
    BraketAwsQubitDevice(wires=2, device_arn="foo", shots=10)
    expected_user_agent = f"BraketPennylanePlugin/{__version__}"
    aws_device_mock_instance.aws_session.add_braket_user_agent.assert_called_with(
        expected_user_agent
    )


@patch.object(AwsDevice, "run")
@pytest.mark.parametrize("old_return_type", [True, False])
@pytest.mark.parametrize(
    "pl_circ, expected_braket_circ, wires, expected_inputs, result_types, expected_pl_result",
    [
        (
            CIRCUIT_1,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, FreeParameter("p_0"))
            .ry(0, 0.543)
            .adjoint_gradient(observable=Observable.X(), target=1, parameters=["p_0"]),
            2,
            {"p_0": 0.432},
            [
                {
                    "type": {
                        "observable": "x()",
                        "targets": [[0]],
                        "parameters": ["p_0", "p_1"],
                        "type": "adjoint_gradient",
                    },
                    "value": {
                        "gradient": {"p_0": -0.194399, "p_1": 0.9316158},
                        "expectation": 0.0,
                    },
                },
            ],
            [np.tensor([0]), np.tensor([-0.194399, 0.9316158])],
        ),
        (
            CIRCUIT_4,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, 0.432)
            .ry(0, 0.543)
            .expectation(observable=Observable.X(), target=1),
            2,
            {},
            [
                {
                    "type": {"observable": ["x"], "targets": [1], "type": "expectation"},
                    "value": 0.0,
                }
            ],
            [np.tensor([0]), np.tensor([])],
        ),
        (
            CIRCUIT_6,
            Circuit()
            .h(0)
            .unitary([0], 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))
            .rx(0, FreeParameter("p_1"))
            .unitary([0], 1 / np.sqrt(2) * anp.array([[1, 1], [1, -1]]))
            .cnot(0, 1)
            .adjoint_gradient(observable=Observable.X(), target=1, parameters=["p_0"]),
            2,
            {"p_1": 0.432},
            [
                {
                    "type": {
                        "observable": "x()",
                        "targets": [[0]],
                        "parameters": ["p_0", "p_1"],
                        "type": "adjoint_gradient",
                    },
                    "value": {
                        "gradient": {"p_0": -0.194399, "p_1": 0.9316158},
                        "expectation": 0.0,
                    },
                },
            ],
            [np.tensor([0]), np.tensor([-0.194399, 0.9316158])],
        ),
    ],
)
def test_execute_and_gradients(
    mock_run,
    pl_circ,
    expected_braket_circ,
    wires,
    expected_inputs,
    result_types,
    expected_pl_result,
    old_return_type,
):
    if old_return_type:
        qml.disable_return()
    task = Mock()
    type(task).id = PropertyMock(return_value="task_arn")
    task.state.return_value = "COMPLETED"
    task.result.return_value = get_test_result_object(rts=result_types)
    mock_run.return_value = task
    dev = _aws_device(
        wires=wires,
        foo="bar",
        shots=0,
        device_type=AwsDeviceType.SIMULATOR,
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    )

    results, jacs = dev.execute_and_gradients([pl_circ])
    qml.enable_return()

    assert dev.task == task
    mock_run.assert_called_with(
        expected_braket_circ,
        s3_destination_folder=("foo", "bar"),
        shots=0,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
        inputs=expected_inputs,
    )

    # assert results & jacs are right
    assert (results == expected_pl_result[0]).all()
    assert (jacs == expected_pl_result[1]).all()


@patch("braket.pennylane_plugin.braket_device.param_shift")
@patch.object(AwsDevice, "run")
@pytest.mark.parametrize(
    "pl_circ, expected_braket_circ, wires, expected_inputs, result_types, expected_pl_result",
    [
        (
            CIRCUIT_5,
            Circuit()
            .h(0)
            .cnot(0, 1)
            .rx(0, 0.432)
            .ry(0, 0.543)
            .variance(observable=Observable.X() @ Observable.Y(), target=[0, 1]),
            2,
            {"p_1": 0.543},
            [
                {
                    "type": {"observable": ["x", "y"], "targets": [0, 1], "type": "variance"},
                    "value": 0.0,
                }
            ],
            [np.tensor([0]), np.tensor([0])],
        ),
    ],
)
def test_execute_and_gradients_non_adjoint(
    mock_run,
    mock_param_shift,
    pl_circ,
    expected_braket_circ,
    wires,
    expected_inputs,
    result_types,
    expected_pl_result,
):
    task = Mock()
    type(task).id = PropertyMock(return_value="task_arn")
    task.state.return_value = "COMPLETED"
    task.result.return_value = get_test_result_object(rts=result_types)
    mock_run.return_value = task

    grad = [1, 2]
    mock_param_shift.return_value = [pl_circ, pl_circ], lambda x: grad

    dev = _aws_device(
        wires=wires,
        foo="bar",
        shots=0,
        device_type=AwsDeviceType.SIMULATOR,
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    )

    results, jacs = dev.execute_and_gradients([pl_circ])
    assert dev.task == task
    mock_run.assert_called_with(
        expected_braket_circ,
        s3_destination_folder=("foo", "bar"),
        shots=0,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
        inputs={},
    )

    # assert results & jacs are right
    assert (results == expected_pl_result[0]).all()
    assert np.allclose(jacs[0][0], grad[0])
    assert np.allclose(jacs[0][1], grad[1])
    assert len(jacs[0]) == len(grad)


def test_capabilities_class_and_instance_method():
    class_caps = BraketAwsQubitDevice.capabilities()
    instance_caps = _aws_device(wires=2).capabilities()
    expected_caps = {
        "model": "qubit",
        "supports_broadcasting": False,
        "supports_finite_shots": True,
        "supports_tensor_observables": True,
        "returns_probs": True,
    }
    assert class_caps == expected_caps
    # the instance should not have provides_jacobian, even though AdjointGradient is in the
    # supported result types, because shots != 0
    assert instance_caps == expected_caps


def test_capabilities_adjoint_shots_0():
    instance_caps = _aws_device(
        wires=2, device_type=AwsDeviceType.SIMULATOR, shots=0
    ).capabilities()
    expected_caps = {
        "model": "qubit",
        "supports_broadcasting": False,
        "supports_finite_shots": True,
        "supports_tensor_observables": True,
        "returns_probs": True,
        # the instance should have provides_jacobian because AdjointGradient is in the
        # supported result types and shots == 0
        "provides_jacobian": True,
    }
    assert instance_caps == expected_caps


class DummyLocalQubitDevice(BraketQubitDevice):
    short_name = "dummy"


class DummyCircuitSimulator(BraketSimulator):
    def run(
        self, program: ir.openqasm.Program, qubits: int, shots: Optional[int], *args, **kwargs
    ) -> Dict[str, Any]:
        self._shots = shots
        self._qubits = qubits
        return GATE_MODEL_RESULT

    @property
    def properties(self) -> GateModelSimulatorDeviceCapabilities:
        name = "braket.device_schema.simulators.gate_model_simulator_paradigm_properties"
        input_json = {
            "braketSchemaHeader": {
                "name": "braket.device_schema.simulators.gate_model_simulator_device_capabilities",
                "version": "1",
            },
            "service": {
                "braketSchemaHeader": {
                    "name": "braket.device_schema.device_service_properties",
                    "version": "1",
                },
                "executionWindows": [
                    {
                        "executionDay": "Everyday",
                        "windowStartHour": "09:00",
                        "windowEndHour": "11:00",
                    }
                ],
                "shotsRange": [1, 10],
                "deviceCost": {"price": 0.25, "unit": "minute"},
                "deviceDocumentation": {
                    "imageUrl": "image_url",
                    "summary": "Summary on the device",
                    "externalDocumentationUrl": "external doc link",
                },
                "deviceLocation": "us-east-1",
                "updatedAt": "2020-06-16T19:28:02.869136",
            },
            "action": {
                "braket.ir.openqasm.program": {
                    "actionType": "braket.ir.openqasm.program",
                    "version": ["1"],
                    "supportedOperations": ["x", "y", "h", "cnot"],
                    "supportedResultTypes": [
                        {
                            "name": "resultType1",
                            "observables": ["observable1"],
                            "minShots": 2,
                            "maxShots": 4,
                        }
                    ],
                }
            },
            "paradigm": {
                "braketSchemaHeader": {
                    "name": name,
                    "version": "1",
                },
                "qubitCount": 31,
            },
            "deviceParameters": {},
        }
        return GateModelSimulatorDeviceCapabilities.parse_obj(input_json)


@patch.object(AwsDevice, "__init__", _aws_device_mock_init)
@patch.object(AwsDevice, "aws_session", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "type", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "properties")
def _aws_device(
    properties_mock,
    type_mock,
    session_mock,
    wires,
    device_type=AwsDeviceType.QPU,
    shots=SHOTS,
    device_arn="baz",
    action_properties=ACTION_PROPERTIES,
    **kwargs,
):
    properties_mock.action = {DeviceActionType.OPENQASM: action_properties}
    properties_mock.return_value.action.return_value = {
        DeviceActionType.OPENQASM: action_properties
    }
    type_mock.return_value = device_type
    dev = BraketAwsQubitDevice(
        wires=wires,
        s3_destination_folder=("foo", "bar"),
        device_arn=device_arn,
        aws_session=Mock(),
        shots=shots,
        **kwargs,
    )
    return dev


@patch.object(AwsDevice, "__init__", _aws_device_mock_init)
@patch.object(AwsDevice, "aws_session", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "properties")
def _bad_aws_device(properties_mock, session_mock, wires, **kwargs):
    properties_mock.action = {DeviceActionType.ANNEALING: ACTION_PROPERTIES}
    properties_mock.type = AwsDeviceType.QPU
    return BraketAwsQubitDevice(
        wires=wires,
        s3_destination_folder=("foo", "bar"),
        device_arn=DEVICE_ARN,
        aws_session=Mock(),
        shots=SHOTS,
        **kwargs,
    )


def get_test_result_object(rts=[], source="qubit[2] q; cnot q[0], q[1]; measure q;"):
    json_str = json.dumps(
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.gate_model_task_result",
                "version": "1",
            },
            "measurements": [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            "resultTypes": rts,
            "measuredQubits": [0, 1, 2, 3],
            "taskMetadata": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.task_metadata",
                    "version": "1",
                },
                "id": "task_arn",
                "shots": 0,
                "deviceId": "default",
            },
            "additionalMetadata": {
                "action": {
                    "braketSchemaHeader": {
                        "name": "braket.ir.openqasm.program",
                        "version": "1",
                    },
                    "source": source,
                },
            },
        }
    )
    return GateModelQuantumTaskResult.from_string(json_str)


@pytest.fixture
def noise_model():
    return (
        NoiseModel()
        .add_noise(Noise.BitFlip(0.05), GateCriteria(Gate.H))
        .add_noise(Noise.TwoQubitDepolarizing(0.10), GateCriteria(Gate.CNot))
    )


@pytest.mark.parametrize("backend", ["braket_dm"])
def test_valid_local_device_for_noise_model(backend, noise_model):
    dev = BraketLocalQubitDevice(wires=2, backend=backend, noise_model=noise_model)
    assert dev._noise_model.instructions == [
        NoiseModelInstruction(Noise.BitFlip(0.05), GateCriteria(Gate.H)),
        NoiseModelInstruction(Noise.TwoQubitDepolarizing(0.10), GateCriteria(Gate.CNot)),
    ]


@pytest.mark.parametrize(
    "backend, device_name",
    [("default", "StateVectorSimulator"), ("braket_sv", "StateVectorSimulator")],
)
def test_invalid_local_device_for_noise_model(backend, device_name, noise_model):
    with pytest.raises(
        ValueError,
        match=(
            f"{device_name} does not support noise or the noise model "
            + f"includes noise that is not supported by {device_name}."
        ),
    ):
        BraketLocalQubitDevice(wires=2, backend=backend, noise_model=noise_model)


@pytest.mark.parametrize("device_name", ["dm1"])
@patch.object(AwsDevice, "name", new_callable=mock.PropertyMock)
def test_valide_aws_device_for_noise_model(name_mock, device_name, noise_model):
    name_mock.return_value = device_name
    dev = _aws_device(
        wires=2,
        device_type=AwsDeviceType.SIMULATOR,
        noise_model=noise_model,
        action_properties=ACTION_PROPERTIES_DM_DEVICE,
    )

    assert dev._noise_model.instructions == [
        NoiseModelInstruction(Noise.BitFlip(0.05), GateCriteria(Gate.H)),
        NoiseModelInstruction(Noise.TwoQubitDepolarizing(0.10), GateCriteria(Gate.CNot)),
    ]


@pytest.mark.parametrize("device_name", ["sv1", "tn1", "foo", "bar"])
@patch.object(AwsDevice, "name", new_callable=mock.PropertyMock)
def test_invalide_aws_device_for_noise_model(name_mock, device_name, noise_model):
    name_mock.return_value = device_name
    with pytest.raises(
        ValueError,
        match=(
            f"{device_name} does not support noise or the noise model "
            + f"includes noise that is not supported by {device_name}."
        ),
    ):
        _aws_device(wires=2, device_type=AwsDeviceType.SIMULATOR, noise_model=noise_model)


@patch.object(AwsDevice, "run")
@patch.object(AwsDevice, "name", new_callable=mock.PropertyMock)
def test_execute_with_noise_model(mock_name, mock_run, noise_model):
    mock_run.return_value = TASK
    mock_name.return_value = "dm1"
    dev = _aws_device(
        wires=4,
        device_type=AwsDeviceType.SIMULATOR,
        noise_model=noise_model,
        action_properties=ACTION_PROPERTIES_DM_DEVICE,
    )

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.QubitUnitary(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), wires=0)
        qml.RX(0.432, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))
    circuit.trainable_params = []

    _ = dev.execute(circuit)

    assert dev.task == TASK

    EXPECTED_CIRC = (
        Circuit()
        .h(0)
        .bit_flip(0, 0.05)
        .unitary([0], 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))
        .rx(0, 0.432)
        .cnot(0, 1)
        .two_qubit_depolarizing(0, 1, 0.10)
        .i(2)
        .i(3)
        .probability(target=[0])
        .expectation(observable=Observable.X(), target=1)
        .variance(observable=Observable.Y(), target=2)
        .sample(observable=Observable.Z(), target=3)
    )
    mock_run.assert_called_with(
        EXPECTED_CIRC,
        s3_destination_folder=("foo", "bar"),
        shots=SHOTS,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        inputs={},
    )


OQC_PULSE_PROPERTIES = json.dumps(
    {
        "braketSchemaHeader": {
            "name": "braket.device_schema.pulse.pulse_device_action_properties",
            "version": "1",
        },
        "supportedQhpTemplateWaveforms": {},
        "ports": {},
        "supportedFunctions": {},
        "frames": {
            "q0_drive": {
                "frameId": "q0_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q0_second_state": {
                "frameId": "q0_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
        },
        "supportsLocalPulseElements": False,
        "supportsDynamicFrames": True,
        "supportsNonNativeGatesWithPulses": True,
        "validationParameters": {
            "MAX_SCALE": 1.0,
            "MAX_AMPLITUDE": 1.0,
            "PERMITTED_FREQUENCY_DIFFERENCE": 1.0,
            "MIN_PULSE_LENGTH": 8e-09,
            "MAX_PULSE_LENGTH": 0.00012,
        },
    }
)

OQC_PARADIGM_PROPERTIES = json.dumps(
    {
        "braketSchemaHeader": {
            "name": "braket.device_schema.gate_model_qpu_paradigm_properties",
            "version": "1",
        },
        "connectivity": {
            "fullyConnected": False,
            "connectivityGraph": {
                "0": ["1", "7"],
                "1": ["2"],
                "2": ["3"],
                "4": ["3", "5"],
                "6": ["5"],
                "7": ["6"],
            },
        },
        "qubitCount": 8,
        "nativeGateSet": ["ecr", "i", "rz", "v", "x"],
    }
)


class TestPulseFunctionality:
    """Test the functions specific to supporting pulse programming via Pennylane"""

    @pytest.mark.parametrize(
        "frameId, expected_result", [("q0_second_state", False), ("q0_drive", True)]
    )
    def test_single_frame_filter_oqc_lucy(self, frameId, expected_result):
        dev = _aws_device(wires=2, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
        assert dev._is_single_qubit_01_frame(frameId) == expected_result

    @pytest.mark.parametrize(
        "frameId, expected_result", [("q0_second_state", True), ("q0_drive", False)]
    )
    def test_single_frame_filter_oqc_lucy_12(self, frameId, expected_result):
        dev = _aws_device(wires=2, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
        assert dev._is_single_qubit_12_frame(frameId) == expected_result

    def test_frame_filters_raise_error_if_not_oqc_lucy(self):
        dev = _aws_device(wires=2, device_arn="baz")

        with pytest.raises(
            RuntimeError, match="Single-qubit drive frame for pulse control not defined for device"
        ):
            dev._is_single_qubit_01_frame("q0_drive")

        with pytest.raises(
            RuntimeError, match="Single-qubit drive frame for pulse control not defined for device"
        ):
            dev._is_single_qubit_12_frame("q0_second_state")

    def test_get_frames_01(self):
        dev = _aws_device(wires=2, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")

        class DummyProperties:
            def __init__(self):
                self.pulse = PulseDeviceActionProperties.parse_raw(OQC_PULSE_PROPERTIES)

        dev._device._properties = DummyProperties()

        frames = dev._get_frames(filter=dev._is_single_qubit_01_frame)
        frames_12 = dev._get_frames(filter=dev._is_single_qubit_12_frame)

        assert len(frames) == len(frames_12) == 1
        assert "q0_drive" in frames.keys()
        assert "q0_second_state" in frames_12.keys()

    def test_settings(self):
        dev = _aws_device(wires=2, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")

        class DummyProperties:
            def __init__(self):
                self.pulse = PulseDeviceActionProperties.parse_raw(OQC_PULSE_PROPERTIES)
                self.paradigm = GateModelQpuParadigmProperties.parse_raw(OQC_PARADIGM_PROPERTIES)

        dev._device._properties = DummyProperties()

        settings = dev.settings
        assert settings["connections"] == [
            (0, 1),
            (0, 7),
            (1, 2),
            (2, 3),
            (4, 3),
            (4, 5),
            (6, 5),
            (7, 6),
        ]
        assert settings["wires"] == [0, 1, 2, 3, 4, 5, 6, 7]
        assert np.allclose(settings["qubit_freq"], 4.6)
        assert np.allclose(settings["anharmonicity"], 0.1)
