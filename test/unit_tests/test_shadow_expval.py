import json
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

from typing import Any, Dict, Optional
import pennylane as qml
import pytest
import braket.ir as ir
from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask
from braket.circuits import Circuit
from braket.device_schema import DeviceActionType
from braket.device_schema.openqasm_device_action_properties import OpenQASMDeviceActionProperties
from braket.simulator import BraketSimulator
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.task_result import GateModelTaskResult
from braket.tasks import GateModelQuantumTaskResult
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.measurements import MeasurementTransform
from pennylane.wires import Wires

from braket.pennylane_plugin import BraketAwsQubitDevice, BraketLocalQubitDevice
from braket.pennylane_plugin.braket_device import BraketQubitDevice

SHOTS = 2
DEVICE_ARN = "baz"
SEED = 42

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

SNAPSHOTS = [[0, 0], [1, 1]]

GATE_MODEL_RESULT = GateModelTaskResult(
    **{
        "measurements": SNAPSHOTS,
        "measuredQubits": [0, 1],
        "taskMetadata": {
            "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
            "id": "task_arn",
            "shots": SHOTS,
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
            "measurements": [SNAPSHOTS[0]],
            "resultTypes": [],
            "measuredQubits": [0, 1],
            "taskMetadata": {
                "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
                "id": "task_arn",
                "shots": 1,
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

BATCH_RESULT_1 = GateModelQuantumTaskResult.from_string(
    json.dumps(
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.gate_model_task_result",
                "version": "1",
            },
            "measurements": [SNAPSHOTS[0]],
            "resultTypes": [],
            "measuredQubits": [0, 1],
            "taskMetadata": {
                "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
                "id": "task_arn",
                "shots": 1,
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
BATCH_RESULT_2 = GateModelQuantumTaskResult.from_string(
    json.dumps(
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.gate_model_task_result",
                "version": "1",
            },
            "measurements": [SNAPSHOTS[1]],
            "resultTypes": [],
            "measuredQubits": [0, 1],
            "taskMetadata": {
                "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
                "id": "task_arn",
                "shots": 1,
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
TASK_BATCH.results.return_value = [BATCH_RESULT_1, BATCH_RESULT_2]
type(TASK_BATCH).tasks = PropertyMock(return_value=[TASK, TASK])


@pytest.mark.xfail(raises=ValueError)
def test_only_one_operator_in_shadow_expval():
    """Tests that the correct circuit is created when not all wires in the device are used."""
    dev = _aws_device(wires=2)
    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.RX(0.432, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.shadow_expval(qml.PauliX(1))
        qml.probs(wires=[0, 1])

    dev.execute(circuit)


CIRCUIT_1 = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
        qml.RX(0.432, wires=0),
        qml.RY(0.543, wires=0),
    ],
    measurements=[qml.shadow_expval(qml.PauliX(1), seed=SEED)],
)
CIRCUIT_1.trainable_params = [0]

circs = [Circuit().h(0).cnot(0, 1).rx(0, 0.432).ry(0, 0.543) for s in range(SHOTS)]
# basis rotation with seed
circs[0].h(1)


@patch.object(AwsDevice, "run")
@patch.object(AwsDevice, "run_batch")
@pytest.mark.parametrize(
    "pl_circ, wires, result_types, expected_pl_result",
    [
        (
            CIRCUIT_1,
            2,
            [
                {
                    "type": {
                        "observable": "x()",
                        "targets": [[0]],
                        "parameters": ["p_0", "p_1"],
                        "type": "expectation_value",
                    },
                    "value": {
                        "expectation": 1.5,
                    },
                },
            ],
            [[1.5]],
        ),
    ],
)
@pytest.mark.parametrize(
    "parallel, expected_braket_circ, return_val, call_count, max_para, max_conn",
    [
        (False, circs, TASK, SHOTS, None, None),
        (True, circs, TASK_BATCH, 1, 10, 10),
    ],
)
def test_shadow_expval_aws_device(
    mock_run_batch,
    mock_run,
    pl_circ,
    wires,
    result_types,
    expected_pl_result,
    parallel,
    expected_braket_circ,
    return_val,
    call_count,
    max_para,
    max_conn,
):
    dev = _aws_device(
        wires=2, foo="bar", parallel=parallel, max_parallel=max_para, max_connections=max_conn
    )
    mock_runner = mock_run_batch if parallel else mock_run
    mock_runner.return_value = return_val
    res = dev.execute(pl_circ)
    kwargs = {"max_parallel": max_para, "max_connections": max_conn} if parallel else {}
    assert mock_runner.call_count == call_count
    # assert results are right
    assert res == expected_pl_result[0]
    if parallel:
        mock_runner.assert_called_with(
            expected_braket_circ,
            s3_destination_folder=("foo", "bar"),
            shots=1,
            **kwargs,
            poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
            poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
            foo="bar",
        )
    else:
        for c in expected_braket_circ:
            mock_runner.assert_any_call(
                c,
                s3_destination_folder=("foo", "bar"),
                shots=1,
                **kwargs,
                poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
                poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
                foo="bar",
            )


@patch.object(LocalSimulator, "run")
@pytest.mark.parametrize(
    "pl_circ, expected_braket_circ, wires, result_types, expected_pl_result",
    [
        (
            CIRCUIT_1,
            Circuit().h(0).cnot(0, 1).rx(0, 0.432).ry(0, 0.543),
            2,
            [
                {
                    "type": {
                        "observable": "x()",
                        "targets": [[0]],
                        "parameters": ["p_0", "p_1"],
                        "type": "expectation_value",
                    },
                    "value": {
                        "expectation": 1.5,
                    },
                },
            ],
            [[1.5]],
        ),
    ],
)
@pytest.mark.parametrize("backend", ["default", "braket_sv", "braket_dm"])
def test_shadow_expval_local(
    mock_run,
    pl_circ,
    expected_braket_circ,
    wires,
    result_types,
    expected_pl_result,
    backend,
):
    dev = BraketLocalQubitDevice(wires=2, backend=backend, shots=SHOTS, foo="bar")
    mock_run.return_value = TASK
    res = dev.execute(pl_circ)
    assert mock_run.call_count == SHOTS
    # assert results are right
    assert res == expected_pl_result[0]
    mock_run.assert_called_with(
        expected_braket_circ,
        shots=1,
        foo="bar",
    )


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
    # needed by the BraketAwsQubitDevice.capabilities function
    dev._device._arn = device_arn
    return dev


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


@pytest.mark.xfail(raises=NotImplementedError)
def test_run_snapshots_not_implemented():
    """Tests that an error is thrown when the device doesn't have a _run_snapshots method"""
    dummy = DummyCircuitSimulator()
    dev = DummyLocalQubitDevice(wires=2, device=dummy, shots=1000)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.shadow_expval(qml.PauliX(1))

    dev.execute(circuit)


@patch.object(AwsDevice, "run_batch")
def test_shadows_parallel_tracker(mock_run_batch):
    """Asserts tracker updates during parallel shadows computation"""

    mock_run_batch.return_value = TASK_BATCH
    type(TASK_BATCH).unsuccessful = PropertyMock(return_value={})
    dev = _aws_device(wires=2, foo="bar", parallel=True, shots=SHOTS)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.shadow_expval(qml.PauliX(1))

    callback = Mock()
    with qml.Tracker(dev, callback=callback) as tracker:
        dev.execute(circuit)
    dev.execute(circuit)

    latest = {"batches": 1, "executions": SHOTS, "shots": SHOTS}
    history = {
        "batches": [1],
        "executions": [SHOTS],
        "shots": [SHOTS],
        "braket_task_id": ["task_arn", "task_arn"],
    }
    totals = {"batches": 1, "executions": SHOTS, "shots": SHOTS}
    assert tracker.latest == latest
    assert tracker.history == history
    assert tracker.totals == totals

    callback.assert_called_with(latest=latest, history=history, totals=totals)


class DummyMeasurementTransform(MeasurementTransform):
    def __init__(
        self, wires: Optional[Wires] = None, seed: Optional[int] = None, id: Optional[str] = None
    ):
        self.seed = seed
        super().__init__(wires=wires, id=id)

    def process(self, tape, device):
        pass


def dummy_measurement_transform():
    return DummyMeasurementTransform()


@pytest.mark.xfail(raises=RuntimeError)
def test_non_shadow_expval_transform():
    """Tests that an error is thrown when the circuit has an unsupported MeasurementTransform"""
    dummy = DummyCircuitSimulator()
    dev = DummyLocalQubitDevice(wires=2, device=dummy, shots=1000)

    with QuantumTape() as circuit:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        dummy_measurement_transform()

    dev.execute(circuit)
