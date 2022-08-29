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
from braket.circuits import Circuit, Observable, result_types
from braket.device_schema import DeviceActionType
from braket.device_schema.openqasm_device_action_properties import OpenQASMDeviceActionProperties
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.simulator import BraketSimulator
from braket.task_result import GateModelTaskResult
from braket.tasks import GateModelQuantumTaskResult
from pennylane import QuantumFunctionError, QubitDevice
from pennylane import numpy as np
from pennylane.tape import QuantumTape

import braket.pennylane_plugin.braket_device
from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice, __version__
from braket.pennylane_plugin.braket_device import BraketQubitDevice, Shots

SHOTS = 10000

ACTION_PROPERTIES = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot"],
            "supportedResultTypes": [
                {"name": "StateVector", "observables": None, "minShots": 0, "maxShots": 0}
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
        qml.CNOT(wires=[0, 1])
        qml.probs(wires=[0])
        qml.expval(qml.PauliX(1))
        qml.var(qml.PauliY(2))
        qml.sample(qml.PauliZ(3))

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

    mock_run.assert_called_with(
        CIRCUIT,
        s3_destination_folder=("foo", "bar"),
        shots=SHOTS,
        poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        foo="bar",
    )


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


@patch.object(AwsDevice, "run")
def test_execute_all_samples(mock_run):
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
                        "braketSchemaHeader": {"name": "braket.ir.openqasm.program", "version": "1"},
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

    assert dev.execute(circuit).shape == (2, 4)


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
    mock_run.assert_called_with(
        CIRCUIT,
        shots=shots,
        foo="bar",
    )


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

    measure_types = ["expval", "var", "sample"]
    projector_01 = qml.Projector([0, 1], wires=range(wires))
    projector_10 = qml.Projector([1, 0], wires=range(wires))

    # 01 case
    fs = [qml.map(f, [projector_01], dev, measure=m) for m in measure_types]
    assert np.allclose(fs[0](thetas), p_01)
    assert np.allclose(fs[1](thetas), p_01 - p_01**2)

    samples = fs[2](thetas, shots=100)[0].tolist()
    assert set(samples) == {0, 1}

    # 10 case
    fs = [qml.map(f, [projector_10], dev, measure=m) for m in measure_types]
    assert np.allclose(fs[0](thetas), p_10)
    assert np.allclose(fs[1](thetas), p_10 - p_10**2)

    samples = fs[2](thetas, shots=100)[0].tolist()
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


def _noop(*args, **kwargs):
    return None


@patch.object(AwsDevice, "__init__", _noop)
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
    **kwargs,
):
    properties_mock.action = {DeviceActionType.OPENQASM: ACTION_PROPERTIES}
    properties_mock.return_value.action.return_value = {DeviceActionType.OPENQASM: ACTION_PROPERTIES}
    type_mock.return_value = device_type
    return BraketAwsQubitDevice(
        wires=wires,
        s3_destination_folder=("foo", "bar"),
        device_arn="baz",
        aws_session=Mock(),
        shots=shots,
        **kwargs,
    )


@patch.object(AwsDevice, "__init__", _noop)
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
