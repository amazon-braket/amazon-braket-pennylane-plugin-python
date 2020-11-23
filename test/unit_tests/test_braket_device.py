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

import json
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pennylane as qml
import pytest
from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask, AwsQuantumTaskBatch
from braket.circuits import Circuit, Instruction, Observable, gates, result_types
from braket.device_schema import DeviceActionType
from braket.tasks import GateModelQuantumTaskResult
from pennylane import QubitDevice
from pennylane.qnodes import QuantumFunctionError
from pennylane.tape import QuantumTape
from pennylane.wires import Wires

from braket.pennylane_plugin import (
    CY,
    ISWAP,
    PSWAP,
    XX,
    XY,
    YY,
    ZZ,
    BraketAwsQubitDevice,
    BraketLocalQubitDevice,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    V,
)

SHOTS = 10000

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
                    "braketSchemaHeader": {"name": "braket.ir.jaqcd.program", "version": "1"},
                    "instructions": [{"control": 0, "target": 1, "type": "cnot"}],
                },
            },
        }
    )
)
TASK = Mock()
TASK.result.return_value = RESULT
TASK_BATCH = Mock()
TASK_BATCH.results.return_value = [RESULT, RESULT]
type(TASK_BATCH).tasks = PropertyMock(return_value=[TASK, TASK])
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

testdata = [
    (qml.Hadamard, gates.H, [0], []),
    (qml.PauliX, gates.X, [0], []),
    (qml.PauliY, gates.Y, [0], []),
    (qml.PauliZ, gates.Z, [0], []),
    (qml.S, gates.S, [0], []),
    (qml.T, gates.T, [0], []),
    (qml.CNOT, gates.CNot, [0, 1], []),
    (qml.CZ, gates.CZ, [0, 1], []),
    (qml.PhaseShift, gates.PhaseShift, [0], [np.pi]),
    (qml.RX, gates.Rx, [0], [np.pi]),
    (qml.RY, gates.Ry, [0], [np.pi]),
    (qml.RZ, gates.Rz, [0], [np.pi]),
    (qml.SWAP, gates.Swap, [0, 1], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], []),
    (qml.QubitUnitary, gates.Unitary, [0], [np.array([[0, 1], [1, 0]])]),
    (V, gates.V, [0], []),
    (CY, gates.CY, [0, 1], []),
    (CPhaseShift, gates.CPhaseShift, [0, 1], [np.pi]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [np.pi]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [np.pi]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [np.pi]),
    (ISWAP, gates.ISwap, [0, 1], []),
    (PSWAP, gates.PSwap, [0, 1], [np.pi]),
    (XY, gates.XY, [0, 1], [np.pi]),
    (XX, gates.XX, [0, 1], [np.pi]),
    (YY, gates.YY, [0, 1], [np.pi]),
    (ZZ, gates.ZZ, [0, 1], [np.pi]),
]

testdata_inverses = [
    (qml.S, gates.Si),
    (qml.T, gates.Ti),
    (V, gates.Vi),
]


def test_reset():
    """Tests that the members of the device are cleared on reset."""
    dev = _device(wires=2)
    dev._circuit = CIRCUIT
    dev._task = TASK

    dev.reset()
    assert dev.circuit is None
    assert dev.task is None


@pytest.mark.parametrize("pl_op, braket_gate, qubits, params", testdata)
def test_apply(pl_op, braket_gate, qubits, params):
    """Tests that the correct Braket gate is applied for each PennyLane operation."""
    dev = _device(wires=len(qubits))
    circuit = dev.apply([pl_op(*params, wires=qubits)])
    assert circuit == Circuit().add_instruction(Instruction(braket_gate(*params), qubits))


@pytest.mark.xfail
@pytest.mark.parametrize("pl_op, braket_gate", testdata_inverses)
def test_apply_inverse_gates(pl_op, braket_gate):
    """
    Tests that the correct Braket gate is applied for the inverse of each PennyLane operations
    where the inverse is defined.
    """
    dev = _device(wires=1)
    circuit = dev.apply([pl_op(wires=0).inv()])
    assert circuit == Circuit().add_instruction(Instruction(braket_gate(), 0))


def test_apply_unused_qubits():
    """Tests that the correct circuit is created when not all qires in the dievce are used."""
    dev = _device(wires=4)
    operations = [qml.Hadamard(wires=1), qml.CNOT(wires=[1, 2]), qml.RX(np.pi / 2, wires=2)]
    rotations = [qml.RY(np.pi, wires=1)]
    circuit = dev.apply(operations, rotations)

    assert circuit == Circuit().h(1).cnot(1, 2).rx(2, np.pi / 2).ry(1, np.pi).i(0).i(3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_apply_unsupported():
    """Tests that apply() throws NotImplementedError when it encounters an unknown gate."""
    dev = _device(wires=2)
    mock_op = Mock()
    mock_op.name = "foo"

    operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), mock_op]
    dev.apply(operations)


@patch.object(AwsQuantumTask, "create")
def test_execute(mock_create):
    mock_create.return_value = TASK
    dev = _device(wires=4, foo="bar")

    circuit = qml.CircuitGraph(
        [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.probs(wires=[0]),
            qml.expval(qml.PauliX(1)),
            qml.var(qml.PauliY(2)),
            qml.sample(qml.PauliZ(3)),
        ],
        {},
        wires=Wires([0, 1, 2, 3]),
    )
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

    mock_create.assert_called_with(
        mock.ANY,
        DEVICE_ARN,
        CIRCUIT,
        ("foo", "bar"),
        SHOTS,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
    )


def test_pl_to_braket_circuit():
    """Tests that a PennyLane circuit is correctly converted into a Braket circuit"""
    dev = _device(wires=2, foo="bar")

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


def test_bad_statistics():
    """Test if a QuantumFunctionError is raised for an invalid return type"""
    dev = _device(wires=1, foo="bar")
    observable = qml.Identity(wires=0, do_queue=False)
    observable.return_type = None

    with pytest.raises(QuantumFunctionError, match="Unsupported return type specified"):
        dev.statistics(None, [observable])


def test_batch_execute_non_parallel(monkeypatch):
    """Test if the batch_execute() method simply calls the inherited method if parallel=False"""
    dev = _device(wires=2, foo="bar", parallel=False)
    assert dev.parallel is False

    with monkeypatch.context() as m:
        m.setattr(QubitDevice, "batch_execute", lambda self, circuits: 1967)
        res = dev.batch_execute([])
        assert res == 1967


@patch.object(AwsDevice, "run_batch")
def test_batch_execute_parallel(mock_run_batch):
    """Test batch_execute(parallel=True) correctly calls batch execution methods in Braket SDK"""
    mock_run_batch.return_value = TASK_BATCH
    dev = _device(wires=4, foo="bar", parallel=True)
    assert dev.parallel is True

    circuit = qml.CircuitGraph(
        [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.probs(wires=[0]),
            qml.expval(qml.PauliX(1)),
            qml.var(qml.PauliY(2)),
            qml.sample(qml.PauliZ(3)),
        ],
        {},
        wires=Wires([0, 1, 2, 3]),
    )
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
        max_parallel=AwsQuantumTaskBatch.MAX_PARALLEL_DEFAULT,
        max_connections=AwsQuantumTaskBatch.MAX_CONNECTIONS_DEFAULT,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
    )


@patch.object(AwsQuantumTask, "create")
def test_execute_all_samples(mock_create):
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
                        "braketSchemaHeader": {"name": "braket.ir.jaqcd.program", "version": "1"},
                        "instructions": [{"control": 0, "target": 1, "type": "cnot"}],
                    },
                },
            }
        )
    )
    task = Mock()
    task.result.return_value = result
    mock_create.return_value = task
    dev = _device(wires=3)

    circuit = qml.CircuitGraph(
        [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.sample(qml.Hadamard(0) @ qml.Identity(1)),
            qml.sample(qml.Hermitian(np.array([[0, 1], [1, 0]]), wires=[2])),
        ],
        {},
        wires=Wires([0, 1, 2]),
    )
    assert dev.execute(circuit).shape == (2, 4)


@pytest.mark.xfail(raises=ValueError)
def test_non_jaqcd_device():
    """Tests that BraketDevice cannot be instantiated with a non-JAQCD AwsDevice"""
    _bad_device(wires=2)


def test_simulator_default_shots():
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = _device(wires=2, device_type=AwsDeviceType.SIMULATOR, shots=None)
    assert dev.shots == 1
    assert dev.analytic


def test_simulator_0_shots():
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = _device(wires=2, device_type=AwsDeviceType.SIMULATOR, shots=0)
    assert dev.shots == 1
    assert dev.analytic


def test_local_default_shots():
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = BraketLocalQubitDevice(wires=2)
    assert dev.shots == 1
    assert dev.analytic


def test_local_0_shots():
    """Tests that simulator devices are analytic if ``shots`` is not supplied"""
    dev = BraketLocalQubitDevice(wires=2, shots=0)
    assert dev.shots == 1
    assert dev.analytic


def test_qpu_default_shots():
    """Tests that QPU devices have the right default value for ``shots``"""
    dev = _device(wires=2, shots=None)
    assert dev.shots == AwsDevice.DEFAULT_SHOTS_QPU
    assert not dev.analytic


@pytest.mark.xfail(raises=ValueError)
def test_qpu_0_shots():
    """Tests that QPUs can not be instantiated with 0 shots"""
    _device(wires=2, shots=0)


@pytest.mark.xfail(raises=ValueError)
def test_invalid_device_type():
    """Tests that BraketDevice cannot be instantiated with an unknown device type"""
    _device(wires=2, device_type="foo", shots=None)


@patch.object(AwsDevice, "type", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "properties")
@patch.object(AwsDevice, "refresh_metadata", return_value=None)
def _device(
    refresh_metadata_mock,
    properties_mock,
    type_mock,
    wires,
    device_type=AwsDeviceType.QPU,
    shots=SHOTS,
    **kwargs
):
    properties_mock.action = {DeviceActionType.JAQCD: "foo"}
    type_mock.return_value = device_type
    return BraketAwsQubitDevice(
        wires=wires,
        s3_destination_folder=("foo", "bar"),
        device_arn="baz",
        aws_session=Mock(),
        shots=shots,
        **kwargs
    )


@patch.object(AwsDevice, "type")
@patch.object(AwsDevice, "properties")
@patch.object(AwsDevice, "refresh_metadata", return_value=None)
def _bad_device(refresh_metadata_mock, properties_mock, type_mock, wires, **kwargs):
    properties_mock.action = {DeviceActionType.ANNEALING: "foo"}
    properties_mock.type = AwsDeviceType.QPU
    return BraketAwsQubitDevice(
        wires=wires,
        s3_destination_folder=("foo", "bar"),
        device_arn=DEVICE_ARN,
        aws_session=Mock(),
        shots=SHOTS,
        **kwargs
    )
