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
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest

from braket.aws import (
    AwsQpu,
    AwsQpuArns,
    AwsQuantumSimulator,
    AwsQuantumSimulatorArns,
    AwsQuantumTask,
)
from braket.circuits import Circuit, Instruction, gates
from braket.pennylane_plugin import (
    CY,
    ISWAP,
    PSWAP,
    XX,
    XY,
    YY,
    ZZ,
    AWSIonQDevice,
    AWSRigettiDevice,
    AWSSimulatorDevice,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    V,
)
from braket.tasks import GateModelQuantumTaskResult

SHOTS = 10000
RESULT = GateModelQuantumTaskResult.from_string(
    json.dumps(
        {
            "measurements": [[0, 0], [1, 1], [1, 1], [1, 1]],
            "measuredQubits": [0, 1],
            "taskMetadata": {
                "braketSchemaHeader": {"name": "braket.task_result.task_metadata", "version": "1"},
                "id": "task_arn",
                "shots": SHOTS,
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
BELL_STATE = Circuit().h(0).cnot(0, 1)

testdata = [
    (qml.Identity, gates.I, [0], []),
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
    """ Tests that the members of the device are cleared on reset.
    """
    dev = _device(2)
    dev._circuit = BELL_STATE
    dev._task = TASK

    dev.reset()
    assert dev.circuit is None
    assert dev.task is None


@pytest.mark.parametrize("pl_op, braket_gate, qubits, params", testdata)
def test_apply(pl_op, braket_gate, qubits, params):
    """ Tests that the correct Braket gate is applied for each PennyLane operation.
    """
    dev = _device(wires=len(qubits))
    dev.apply([pl_op(*params, wires=qubits)])
    assert dev.circuit == Circuit().add_instruction(Instruction(braket_gate(*params), qubits))


@pytest.mark.parametrize("pl_op, braket_gate", testdata_inverses)
def test_apply_inverse_gates(pl_op, braket_gate):
    """
    Tests that the correct Braket gate is applied for the inverse of each PennyLane operations
    where the inverse is defined.
    """
    dev = _device(1)
    dev.apply([pl_op(wires=0).inv()])
    assert dev.circuit == Circuit().add_instruction(Instruction(braket_gate(), 0))


def test_apply_with_rotations():
    """ Tests that the device's circuit member is set properly after apply is called.
    """
    dev = _device(2)
    operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), qml.RX(np.pi / 2, wires=1)]
    rotations = [qml.RY(np.pi, wires=0)]
    dev.apply(operations, rotations)

    assert dev.circuit == Circuit().h(0).cnot(0, 1).rx(1, np.pi / 2).ry(0, np.pi)


def test_apply_unused_qubits():
    """ Tests that the correct circuit is created when not all qires in the dievce are used.
    """
    dev = _device(4)
    operations = [qml.Hadamard(wires=1), qml.CNOT(wires=[1, 2]), qml.RX(np.pi / 2, wires=2)]
    rotations = [qml.RY(np.pi, wires=1)]
    dev.apply(operations, rotations)

    assert dev.circuit == Circuit().h(1).cnot(1, 2).rx(2, np.pi / 2).ry(1, np.pi).i(0).i(3)


@pytest.mark.xfail(raises=NotImplementedError)
def test_apply_unsupported():
    """ Tests that apply() throws NotImplementedError when it encounters an unknown gate.
    """
    dev = _device(2)
    mock_op = Mock()
    mock_op.name = "foo"

    operations = [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1]), mock_op]
    dev.apply(operations)


# The next five tests ensure that for all of the Braket devices, the samples
# generated by generate_samples match those in the results
# and AwsQuantumTask.create is called with the right arguments.


@patch.object(AwsQuantumTask, "create")
def test_generate_samples_ionq(mock_create):
    mock_create.return_value = TASK
    dev = _device(2, AWSIonQDevice)
    dev.apply([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])])

    assert (dev.generate_samples() == RESULT.measurements).all()
    assert dev.task == TASK
    mock_create.assert_called_with(
        mock.ANY,
        AwsQpuArns.IONQ,
        BELL_STATE,
        ("foo", "bar"),
        SHOTS,
        poll_timeout_seconds=AwsQpu.DEFAULT_RESULTS_POLL_TIMEOUT_QPU,
        poll_interval_seconds=AwsQpu.DEFAULT_RESULTS_POLL_INTERVAL_QPU,
    )


@patch.object(AwsQuantumTask, "create")
def test_generate_samples_rigetti(mock_create):
    mock_create.return_value = TASK
    dev = _device(2, AWSRigettiDevice)
    dev.apply([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])])

    assert (dev.generate_samples() == RESULT.measurements).all()
    assert dev.task == TASK
    mock_create.assert_called_with(
        mock.ANY,
        AwsQpuArns.RIGETTI,
        BELL_STATE,
        ("foo", "bar"),
        SHOTS,
        poll_timeout_seconds=AwsQpu.DEFAULT_RESULTS_POLL_TIMEOUT_QPU,
        poll_interval_seconds=AwsQpu.DEFAULT_RESULTS_POLL_INTERVAL_QPU,
    )


@patch.object(AwsQuantumTask, "create")
def test_generate_samples_qs1(mock_create):
    mock_create.return_value = TASK
    dev = _device(2, AWSSimulatorDevice, arn="QS1")
    dev.apply([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])])

    assert (dev.generate_samples() == RESULT.measurements).all()
    assert dev.task == TASK
    mock_create.assert_called_with(
        mock.ANY,
        AwsQuantumSimulatorArns.QS1,
        BELL_STATE,
        ("foo", "bar"),
        SHOTS,
        poll_timeout_seconds=AwsQuantumSimulator.DEFAULT_RESULTS_POLL_TIMEOUT_SIMULATOR,
        poll_interval_seconds=AwsQuantumSimulator.DEFAULT_RESULTS_POLL_INTERVAL_SIMULATOR,
    )


def test_probability():
    """ Tests that the right probabilities are passed into marginal_prob.
    """
    dev = _device(2)
    dev._task = TASK
    probs = np.array([0.25, 0, 0, 0.75])
    assert (dev.probability() == dev.marginal_prob(probs)).all()


def _device(wires, device_class=AWSSimulatorDevice, **kwargs):
    return device_class(wires=wires, s3_destination_folder=("foo", "bar"), shots=SHOTS, **kwargs)
