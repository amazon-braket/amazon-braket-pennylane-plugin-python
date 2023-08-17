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

"""Tests that pulse uploads work via PennyLane for the OQC Lucy device"""

from unittest.mock import patch
from boto3.errorfactory import DeviceOfflineException
import braket.aws.aws_quantum_task
import numpy as np
import pennylane as qml
import pytest
from pennylane.pulse import ParametrizedEvolution, transmon_drive

lucy = qml.device(
    "braket.aws.qubit",
    device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
    wires=8,
    shots=1000,
)


# define drive hamiltonians
def amplitude(max_amp, t):
    b = 10
    c = 3
    return max_amp * np.exp(-((t - b) ** 2) / (2 * c**2))


qubit_freq = lucy.pulse_settings["qubit_freq"]

H_const = transmon_drive(
    qml.pulse.constant, 10, qubit_freq[1], wires=[1]
)  # drive with ConstantWaveform
H_arbitrary = transmon_drive(
    amplitude, 20, qubit_freq[7], wires=[7]
)  # drive with ArbitraryWaveform

H_multi_qubit = H_const + H_arbitrary  # drive on two qubits


@pytest.mark.parametrize(
    "drive_hamiltonian, params",
    [(H_const, [0.5]), (H_arbitrary, [0.7]), (H_multi_qubit, [0.5, 0.7])],
)
@patch.object(braket.aws.aws_quantum_task.AwsQuantumTask, "result")
@patch.object(braket.pennylane_plugin.braket_device.BraketAwsQubitDevice, "_braket_to_pl_result")
def test_pulse_upload_device_unavailable(result_mock, process_mock, drive_hamiltonian, params):
    """Check that pulse upload works when the device is online but unavailable.
    The results processing is mocked (so the qnode doesn't wait for results to be
    returned, but just returns None), and the task is cancelled after creation"""

    # if device is available or fully offline, skip this test
    if lucy._device.is_available or lucy._device.status == "OFFLINE":
        return

    @qml.qnode(lucy)
    def pl_circuit(p):
        ParametrizedEvolution(drive_hamiltonian, p, 20.0)
        return qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(7))

    try:
        _ = pl_circuit(params)
        task = lucy._task
        assert task is not None
    finally:
        if lucy._task:
            lucy._task.cancel()


@pytest.mark.parametrize(
    "drive_hamiltonian, params",
    [(H_const, [0.5]), (H_arbitrary, [0.7]), (H_multi_qubit, [0.5, 0.7])],
)
def test_pulse_upload_device_available(drive_hamiltonian, params):
    """Check that pulse upload works when the device is available - uploads task
    and checks that a result is returned."""

    # if device is unavailable, skip this test
    if not lucy._device.is_available:
        return

    @qml.qnode(lucy)
    def pl_circuit(p):
        ParametrizedEvolution(drive_hamiltonian, p, 20.0)
        return qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(7))

    result = pl_circuit(params)
    assert result is not None
