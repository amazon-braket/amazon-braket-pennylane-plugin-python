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
from typing import Any
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

import braket.ir as ir
import pennylane as qml
import pytest
from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask
from braket.circuits import Circuit
from braket.device_schema import DeviceActionType
from braket.device_schema import (
    OpenQASMDeviceActionProperties,
    OpenQASMProgramSetDeviceActionProperties,
)
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.devices import LocalSimulator
from braket.program_sets import ProgramSet
from braket.simulator import BraketSimulator
from braket.task_result import GateModelTaskResult, ProgramSetTaskResult
from braket.tasks import GateModelQuantumTaskResult, ProgramSetQuantumTaskResult
from pennylane.measurements import MeasurementTransform
from pennylane.tape import QuantumScript, QuantumTape
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
)

ACTION_PROPERTIES_PROGRAM_SET = OpenQASMProgramSetDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "version": ["1"],
            "actionType": "braket.ir.openqasm.program_set",
            "maximumExecutables": 10,
            "maximumTotalShots": 200000,
        }
    )
)

SNAPSHOTS = [[0, 0], [1, 1]]

GATE_MODEL_RESULT = GateModelTaskResult(
    **{
        "measurements": SNAPSHOTS,
        "measuredQubits": [0, 1],
        "taskMetadata": {
            "braketSchemaHeader": {
                "name": "braket.task_result.task_metadata",
                "version": "1",
            },
            "id": "task_arn",
            "shots": SHOTS,
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
                "braketSchemaHeader": {
                    "name": "braket.task_result.task_metadata",
                    "version": "1",
                },
                "id": "task_arn",
                "shots": 1,
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
                "braketSchemaHeader": {
                    "name": "braket.task_result.task_metadata",
                    "version": "1",
                },
                "id": "task_arn",
                "shots": 1,
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
                "braketSchemaHeader": {
                    "name": "braket.task_result.task_metadata",
                    "version": "1",
                },
                "id": "task_arn",
                "shots": 1,
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
PROGRAM_RESULT_1 = {
    "braketSchemaHeader": {
        "name": "braket.task_result.program_result",
        "version": "1",
    },
    "executableResults": [
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.program_set_executable_result",
                "version": "1",
            },
            "measurements": [SNAPSHOTS[0]],
            "measuredQubits": [0, 1],
            "inputsIndex": 0,
        }
    ],
    "source": {
        "braketSchemaHeader": {
            "name": "braket.ir.openqasm.program",
            "version": "1",
        },
        "source": "OPENQASM 3.0;\nbit[2] b;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];\nb[0] = measure q[0];\nb[1] = measure q[1];",  # noqa
        "inputs": {"theta": [0.12, 2.1]},
    },
    "additionalMetadata": {
        "simulatorMetadata": {
            "braketSchemaHeader": {
                "name": "braket.task_result.simulator_metadata",
                "version": "1",
            },
            "executionDuration": 50,
        }
    },
}
PROGRAM_RESULT_2 = {
    "braketSchemaHeader": {
        "name": "braket.task_result.program_result",
        "version": "1",
    },
    "executableResults": [
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.program_set_executable_result",
                "version": "1",
            },
            "measurements": [SNAPSHOTS[1]],
            "measuredQubits": [0, 1],
            "inputsIndex": 0,
        }
    ],
    "source": {
        "braketSchemaHeader": {
            "name": "braket.ir.openqasm.program",
            "version": "1",
        },
        "source": "OPENQASM 3.0;\nbit[2] b;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];\nb[0] = measure q[0];\nb[1] = measure q[1];",  # noqa
        "inputs": {"theta": [0.12, 2.1]},
    },
    "additionalMetadata": {
        "simulatorMetadata": {
            "braketSchemaHeader": {
                "name": "braket.task_result.simulator_metadata",
                "version": "1",
            },
            "executionDuration": 50,
        }
    },
}
PROGRAM_SET_RESULT = ProgramSetQuantumTaskResult.from_object(
    ProgramSetTaskResult(
        **{
            "braketSchemaHeader": {
                "name": "braket.task_result.program_set_task_result",
                "version": "1",
            },
            "programResults": [PROGRAM_RESULT_1, PROGRAM_RESULT_2],
            "taskMetadata": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.program_set_task_metadata",
                    "version": "1",
                },
                "id": "arn:aws:braket:us-west-2:667256736152:quantum-task/bfebc86f-e4ed-4d6f-8131-addd1a49d6dc",  # noqa
                "deviceId": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                "requestedShots": 2,
                "successfulShots": 2,
                "programMetadata": [{"executables": [{}]}],
                "deviceParameters": {
                    "braketSchemaHeader": {
                        "name": "braket.device_schema.simulators.gate_model_simulator_device_parameters",
                        "version": "1",
                    },
                    "paradigmParameters": {
                        "braketSchemaHeader": {
                            "name": "braket.device_schema.gate_model_parameters",
                            "version": "1",
                        },
                        "qubitCount": 2,
                        "disableQubitRewiring": False,
                    },
                },
                "createdAt": "2024-10-15T19:06:58.986Z",
                "endedAt": "2024-10-15T19:07:00.382Z",
                "status": "COMPLETED",
                "totalFailedExecutables": 0,
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
TASK_PROGRAM_SET = Mock()
TASK_PROGRAM_SET.result.return_value = PROGRAM_SET_RESULT


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

# Circuits with measurements for program set (required for program sets to work)
circs_with_measurements = [
    Circuit().h(0).cnot(0, 1).rx(0, 0.432).ry(0, 0.543).h(1).measure(0).measure(1),
    Circuit().h(0).cnot(0, 1).rx(0, 0.432).ry(0, 0.543).measure(0).measure(1),
]


@patch.object(AwsDevice, "properties", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "run")
@patch.object(AwsDevice, "run_batch")
@pytest.mark.parametrize(
    "pl_circ, wires, expected_pl_result",
    [
        (
            CIRCUIT_1,
            2,
            [[1.5]],
        ),
    ],
)
@pytest.mark.parametrize(
    "parallel, expected_braket_task_spec, return_val, call_count, max_para, max_conn, supports_program_sets",
    [
        (False, circs, TASK, SHOTS, None, None, False),
        (True, circs, TASK_BATCH, 1, 10, 10, False),
        (False, ProgramSet(circs), TASK_PROGRAM_SET, 1, None, None, True),
    ],
)
def test_shadow_expval_aws_device(
    mock_run_batch,
    mock_run,
    mock_properties,
    pl_circ,
    wires,
    expected_pl_result,
    parallel,
    expected_braket_task_spec,
    return_val,
    call_count,
    max_para,
    max_conn,
    supports_program_sets,
):
    mock_action = Mock()
    if supports_program_sets:
        mock_action.action = {
            DeviceActionType.OPENQASM: ACTION_PROPERTIES,
            DeviceActionType.OPENQASM_PROGRAM_SET: ACTION_PROPERTIES_PROGRAM_SET,
        }
    else:
        mock_action.action = {DeviceActionType.OPENQASM: ACTION_PROPERTIES}
    mock_properties.return_value = mock_action
    dev = _aws_device(
        wires=2,
        foo="bar",
        parallel=parallel,
        max_parallel=max_para,
        max_connections=max_conn,
        supports_program_sets=supports_program_sets,
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
            expected_braket_task_spec,
            s3_destination_folder=("foo", "bar"),
            shots=1,
            **kwargs,
            poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
            poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
            foo="bar",
        )
    elif supports_program_sets:
        mock_runner.assert_called_with(
            expected_braket_task_spec,
            s3_destination_folder=("foo", "bar"),
            shots=2,
            **kwargs,
            poll_timeout_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
            poll_interval_seconds=AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
            foo="bar",
        )
    else:
        for c in expected_braket_task_spec:
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
    "pl_circ, expected_braket_circ, wires, expected_pl_result",
    [
        (
            CIRCUIT_1,
            Circuit().h(0).cnot(0, 1).rx(0, 0.432).ry(0, 0.543),
            2,
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


def _aws_device_mock_init(self, arn, aws_session):
    self._arn = arn


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
    supports_program_sets=False,
    **kwargs,
):
    properties_mock.action = {DeviceActionType.OPENQASM: action_properties}
    if supports_program_sets:
        properties_mock.action[DeviceActionType.OPENQASM_PROGRAM_SET] = (
            ACTION_PROPERTIES_PROGRAM_SET
        )
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
        self,
        program: ir.openqasm.Program,
        qubits: int,
        shots: int | None,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
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
                            "observables": ["z"],
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


@patch.object(AwsDevice, "properties", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "run_batch")
def test_shadows_parallel_tracker(mock_run_batch, mock_properties):
    """Asserts tracker updates during parallel shadows computation"""

    mock_run_batch.return_value = TASK_BATCH
    type(TASK_BATCH).unsuccessful = PropertyMock(return_value={})
    mock_action = Mock()
    mock_action.action = {"braket.ir.openqasm.program": None}
    mock_properties.return_value = mock_action
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
        self,
        wires: Wires | None = None,
        seed: int | None = None,
        id: str | None = None,
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


# Test for classical_shadow with program sets
CIRCUIT_CLASSICAL_SHADOW = QuantumScript(
    ops=[
        qml.Hadamard(wires=0),
        qml.CNOT(wires=[0, 1]),
    ],
    measurements=[qml.classical_shadow(wires=range(2), seed=SEED)],
)


@patch.object(AwsDevice, "properties", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "run")
def test_batch_execute_classical_shadow_single_circuit(mock_run, mock_properties):
    """Test that batch_execute handles ClassicalShadowMP with a single circuit"""
    mock_action = Mock()
    mock_action.action = {
        DeviceActionType.OPENQASM: ACTION_PROPERTIES,
        DeviceActionType.OPENQASM_PROGRAM_SET: ACTION_PROPERTIES_PROGRAM_SET,
    }
    mock_properties.return_value = mock_action

    dev = _aws_device(
        wires=2,
        foo="bar",
        supports_program_sets=True,
    )

    mock_run.return_value = TASK_PROGRAM_SET
    circuits = [CIRCUIT_CLASSICAL_SHADOW]
    results = dev.batch_execute(circuits)

    assert len(results) == 1
    bits, recipes = results[0]
    assert bits.shape == (SHOTS, 2)
    assert recipes.shape == (SHOTS, 2)


@patch.object(AwsDevice, "properties", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "run")
@pytest.mark.parametrize(
    "circuits, error_message",
    [
        (
            [
                QuantumScript(
                    ops=[qml.Hadamard(wires=0)],
                    measurements=[
                        qml.classical_shadow(wires=[0], seed=SEED),
                        qml.expval(qml.PauliZ(0)),
                    ],
                )
            ],
            "A circuit with a ClassicalShadowMP observable must have that as its only result type",
        ),
        (
            [CIRCUIT_CLASSICAL_SHADOW, CIRCUIT_CLASSICAL_SHADOW],
            "Classical shadow must be called with a single circuit",
        ),
    ],
)
def test_batch_execute_classical_shadow_errors(mock_run, mock_properties, circuits, error_message):
    """Test that batch_execute raises appropriate errors for invalid ClassicalShadowMP usage"""
    mock_action = Mock()
    mock_action.action = {
        DeviceActionType.OPENQASM: ACTION_PROPERTIES,
        DeviceActionType.OPENQASM_PROGRAM_SET: ACTION_PROPERTIES_PROGRAM_SET,
    }
    mock_properties.return_value = mock_action

    dev = _aws_device(
        wires=2,
        foo="bar",
        supports_program_sets=True,
    )

    with pytest.raises(ValueError, match=error_message):
        dev.batch_execute(circuits)
