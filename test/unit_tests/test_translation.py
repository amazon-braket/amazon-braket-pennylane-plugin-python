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
import re
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest
from braket.aws import AwsDevice, AwsDeviceType
from braket.circuits import FreeParameter, gates, noises, observables
from braket.circuits.result_types import (
    AdjointGradient,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from braket.circuits.serialization import IRType
from braket.device_schema import DeviceActionType
from braket.device_schema.gate_model_qpu_paradigm_properties_v1 import (
    GateModelQpuParadigmProperties,
)
from braket.device_schema.pulse.pulse_device_action_properties_v1 import (
    PulseDeviceActionProperties,
)
from braket.pulse import ArbitraryWaveform, ConstantWaveform
from braket.tasks import GateModelQuantumTaskResult
from device_property_jsons import (
    ACTION_PROPERTIES,
    OQC_PARADIGM_PROPERTIES,
    OQC_PULSE_PROPERTIES_WITH_PORTS,
)
from pennylane import measurements
from pennylane import numpy as pnp
from pennylane.measurements import ObservableReturnTypes
from pennylane.pulse import ParametrizedEvolution, transmon_drive
from pennylane.wires import Wires

from braket.pennylane_plugin import (
    PSWAP,
    BraketAwsQubitDevice,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
)
from braket.pennylane_plugin.ops import AAMS, MS, GPi, GPi2, PRx
from braket.pennylane_plugin.translation import (
    _BRAKET_TO_PENNYLANE_OPERATIONS,
    _translate_observable,
    get_adjoint_gradient_result_type,
    supported_operations,
    translate_operation,
    translate_result,
    translate_result_type,
)


def mock_aws_init(self, arn, aws_session):
    self._arn = arn
    self._frames = None
    self._ports = None


class DummyProperties:
    def __init__(self):
        self.pulse = PulseDeviceActionProperties.parse_raw(OQC_PULSE_PROPERTIES_WITH_PORTS)
        self.paradigm = GateModelQpuParadigmProperties.parse_raw(OQC_PARADIGM_PROPERTIES)


@patch.object(AwsDevice, "__init__", mock_aws_init)
@patch.object(AwsDevice, "aws_session", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "type", new_callable=mock.PropertyMock)
@patch.object(AwsDevice, "properties")
def _aws_device(
    properties_mock,
    type_mock,
    session_mock,
    wires,
    device_type=AwsDeviceType.QPU,
    shots=10000,
    device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
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

    dev._device._properties = DummyProperties()
    return dev


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
    (qml.SX, gates.V, [0], []),
    (qml.CY, gates.CY, [0, 1], []),
    (qml.ControlledPhaseShift, gates.CPhaseShift, [0, 1], [np.pi]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [np.pi]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [np.pi]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [np.pi]),
    (GPi, gates.GPi, [0], [2]),
    (GPi2, gates.GPi2, [0], [2]),
    (MS, gates.MS, [0, 1], [2, 3]),
    (PRx, gates.PRx, [0], [2, 3]),
    (AAMS, gates.MS, [0, 1], [2, 3, 0.5]),
    (qml.ECR, gates.ECR, [0, 1], []),
    (qml.ISWAP, gates.ISwap, [0, 1], []),
    (PSWAP, gates.PSwap, [0, 1], [np.pi]),
    (qml.IsingXY, gates.XY, [0, 1], [np.pi]),
    (qml.IsingXX, gates.XX, [0, 1], [np.pi]),
    (qml.IsingYY, gates.YY, [0, 1], [np.pi]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [np.pi]),
    (qml.AmplitudeDamping, noises.AmplitudeDamping, [0], [0.1]),
    (
        qml.GeneralizedAmplitudeDamping,
        noises.GeneralizedAmplitudeDamping,
        [0],
        [0.1, 0.15],
    ),
    (qml.PhaseDamping, noises.PhaseDamping, [0], [0.1]),
    (qml.DepolarizingChannel, noises.Depolarizing, [0], [0.1]),
    (qml.BitFlip, noises.BitFlip, [0], [0.1]),
    (qml.PhaseFlip, noises.PhaseFlip, [0], [0.1]),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
    ),
]

testdata_inverses = [
    (qml.Identity, gates.I, [0], [], []),
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.PauliX, gates.X, [0], [], []),
    (qml.PauliY, gates.Y, [0], [], []),
    (qml.PauliZ, gates.Z, [0], [], []),
    (qml.Hadamard, gates.H, [0], [], []),
    (qml.CNOT, gates.CNot, [0, 1], [], []),
    (qml.CZ, gates.CZ, [0, 1], [], []),
    (qml.CY, gates.CY, [0, 1], [], []),
    (qml.SWAP, gates.Swap, [0, 1], [], []),
    (qml.ECR, gates.ECR, [0, 1], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], []),
    (qml.RX, gates.Rx, [0], [0.15], [-0.15]),
    (qml.RY, gates.Ry, [0], [0.15], [-0.15]),
    (qml.RZ, gates.Rz, [0], [0.15], [-0.15]),
    (qml.PhaseShift, gates.PhaseShift, [0], [0.15], [-0.15]),
    (
        qml.QubitUnitary,
        gates.Unitary,
        [0, 1],
        [
            1
            / np.sqrt(2)
            * np.array(
                [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]],
                dtype=complex,
            )
        ],
        [
            1
            / np.sqrt(2)
            * np.array(
                [[1, 0, 0, 1], [0, -1j, -1j, 0], [0, 1, -1, 0], [-1j, 0, 0, 1j]],
                dtype=complex,
            )
        ],
    ),
    (qml.ControlledPhaseShift, gates.CPhaseShift, [0, 1], [0.15], [-0.15]),
    (CPhaseShift00, gates.CPhaseShift00, [0, 1], [0.15], [-0.15]),
    (CPhaseShift01, gates.CPhaseShift01, [0, 1], [0.15], [-0.15]),
    (CPhaseShift10, gates.CPhaseShift10, [0, 1], [0.15], [-0.15]),
    (GPi, gates.GPi, [0], [2], [2]),
    (GPi2, gates.GPi2, [0], [2], [2 + np.pi]),
    (MS, gates.MS, [0, 1], [2, 3], [2 + np.pi, 3]),
    (AAMS, gates.MS, [0, 1], [2, 3, 0.5], [2 + np.pi, 3, 0.5]),
    (PSWAP, gates.PSwap, [0, 1], [0.15], [-0.15]),
    (qml.IsingXX, gates.XX, [0, 1], [0.15], [-0.15]),
    (qml.IsingXY, gates.XY, [0, 1], [0.15], [-0.15]),
    (qml.IsingYY, gates.YY, [0, 1], [0.15], [-0.15]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [0.15], [-0.15]),
]

testdata_named_inverses = [
    (qml.S, gates.Si, 0),
    (qml.T, gates.Ti, 0),
    (qml.SX, gates.Vi, 0),
]

testdata_with_params = [
    (qml.Identity, gates.I, [0], [], [], []),
    (qml.Hadamard, gates.H, [0], [], [], []),
    (qml.PauliX, gates.X, [0], [], [], []),
    (qml.PauliY, gates.Y, [0], [], [], []),
    (qml.PauliZ, gates.Z, [0], [], [], []),
    (qml.Hadamard, gates.H, [0], [], [], []),
    (qml.CNOT, gates.CNot, [0, 1], [], [], []),
    (qml.CZ, gates.CZ, [0, 1], [], [], []),
    (qml.CY, gates.CY, [0, 1], [], [], []),
    (qml.SWAP, gates.Swap, [0, 1], [], [], []),
    (qml.ECR, gates.ECR, [0, 1], [], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], [], []),
    (qml.PhaseShift, gates.PhaseShift, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RX, gates.Rx, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RY, gates.Ry, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.RZ, gates.Rz, [0], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.SWAP, gates.Swap, [0, 1], [], [], []),
    (qml.CSWAP, gates.CSwap, [0, 1, 2], [], [], []),
    (qml.Toffoli, gates.CCNot, [0, 1, 2], [], [], []),
    (
        qml.ControlledPhaseShift,
        gates.CPhaseShift,
        [0, 1],
        [np.pi],
        ["pi"],
        [FreeParameter("pi")],
    ),
    (
        CPhaseShift00,
        gates.CPhaseShift00,
        [0, 1],
        [np.pi],
        ["pi"],
        [FreeParameter("pi")],
    ),
    (
        CPhaseShift01,
        gates.CPhaseShift01,
        [0, 1],
        [np.pi],
        ["pi"],
        [FreeParameter("pi")],
    ),
    (
        CPhaseShift10,
        gates.CPhaseShift10,
        [0, 1],
        [np.pi],
        ["pi"],
        [FreeParameter("pi")],
    ),
    (GPi, gates.GPi, [0], [2], ["a"], [FreeParameter("a")]),
    (GPi2, gates.GPi2, [0], [2], ["a"], [FreeParameter("a")]),
    (
        MS,
        gates.MS,
        [0, 1],
        [2, 3],
        ["a", "c"],
        [FreeParameter("a"), FreeParameter("c")],
    ),
    (
        AAMS,
        gates.MS,
        [0, 1],
        [2, 3, 0.5],
        ["a", "c", "d"],
        [FreeParameter("a"), FreeParameter("c"), FreeParameter("d")],
    ),
    (PSWAP, gates.PSwap, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.ECR, gates.ECR, [0, 1], [], [], []),
    (qml.ISWAP, gates.ISwap, [0, 1], [], [], []),
    (qml.IsingXY, gates.XY, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingXX, gates.XX, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingYY, gates.YY, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (qml.IsingZZ, gates.ZZ, [0, 1], [np.pi], ["pi"], [FreeParameter("pi")]),
    (
        qml.AmplitudeDamping,
        noises.AmplitudeDamping,
        [0],
        [0.1],
        ["alpha"],
        [FreeParameter("alpha")],
    ),
    (
        qml.GeneralizedAmplitudeDamping,
        noises.GeneralizedAmplitudeDamping,
        [0],
        [0.1, 0.15],
        ["p_000", "p_001"],
        [FreeParameter("p_000"), FreeParameter("p_001")],
    ),
    (qml.PhaseDamping, noises.PhaseDamping, [0], [0.1], ["a"], [FreeParameter("a")]),
    (
        qml.DepolarizingChannel,
        noises.Depolarizing,
        [0],
        [0.1],
        ["a"],
        [FreeParameter("a")],
    ),
    (qml.BitFlip, noises.BitFlip, [0], [0.1], ["a"], [FreeParameter("a")]),
    (qml.PhaseFlip, noises.PhaseFlip, [0], [0.1], ["a"], [FreeParameter("a")]),
    (
        qml.QubitUnitary,
        gates.Unitary,
        [0],
        [np.array([[0, 1], [1, 0]])],
        [],
        [np.array([[0, 1], [1, 0]])],
    ),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
        [],
        [[np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])]],
    ),
    (
        qml.QubitChannel,
        noises.Kraus,
        [0],
        [pnp.tensor([np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])])],
        [],
        [pnp.tensor([np.array([[0, 0.8], [0.8, 0]]), np.array([[0.6, 0], [0, 0.6]])])],
    ),
]

_braket_to_pl = {
    op.lower().replace("_", ""): _BRAKET_TO_PENNYLANE_OPERATIONS[op]
    for op in _BRAKET_TO_PENNYLANE_OPERATIONS
}

pl_return_types = [
    ObservableReturnTypes.Expectation,
    ObservableReturnTypes.Variance,
    ObservableReturnTypes.Sample,
]

braket_result_types = [
    Expectation(observables.H(), [0]),
    Variance(observables.H(), [0]),
    Sample(observables.H(), [0]),
]

_braket_to_pl_result_types = {
    Expectation: measurements.ExpectationMP,
    Variance: measurements.VarianceMP,
    Sample: measurements.SampleMP,
}


@pytest.mark.parametrize("pl_cls, braket_cls, qubits, params", testdata)
def test_translate_operation(pl_cls, braket_cls, qubits, params):
    """Tests that Braket operations are translated correctly"""
    pl_op = pl_cls(*params, wires=qubits)
    braket_gate = braket_cls(*params)
    assert translate_operation(pl_op) == braket_gate
    if isinstance(pl_op, (GPi, GPi2, MS, AAMS, PRx)):
        translated_back = _braket_to_pl[
            re.match("^[a-z0-2]+", braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM)).group(0)
        ]
        assert (
            translated_back == pl_op.name
            if pl_op.name != "MS"
            # PL MS and AAMS both get translated to Braket MS.
            # Braket MS gets translated to PL AAMS.
            else translated_back == "AAMS"
        )
    else:
        assert (
            _braket_to_pl[braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")]
            == pl_op.name
        )


@pytest.mark.parametrize(
    "pl_gate_fn, braket_gate_fn, qubits, pl_params, pl_param_names, expected_params",
    testdata_with_params,
)
def test_translate_operation_with_unique_params(
    pl_gate_fn, braket_gate_fn, qubits, pl_params, pl_param_names, expected_params
):
    """Tests that Braket operations are translated correctly"""
    pl_op = pl_gate_fn(*pl_params, wires=qubits)
    braket_gate = braket_gate_fn(*expected_params)
    assert (
        translate_operation(pl_op, use_unique_params=True, param_names=pl_param_names)
        == braket_gate
    )
    if isinstance(pl_op, (GPi, GPi2, MS, AAMS)):
        translated_back = _braket_to_pl[
            re.match("^[a-z0-2]+", braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM)).group(0)
        ]
        assert (
            translated_back == pl_op.name
            if pl_op.name != "MS"
            # PL MS and AAMS both get translated to Braket MS.
            # Braket MS gets translated to PL AAMS.
            else translated_back == "AAMS"
        )
    else:
        assert (
            _braket_to_pl[braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")]
            == pl_op.name
        )


def amplitude(p, t):
    return p * (np.sin(t) + 1)


def test_translate_parametrized_evolution_constant_amplitude():
    """Test that a ParametrizedEvolution with constant amplitude, phase, and frequency
    is translated to a PulseGate correctly."""
    n_wires = 4
    dev = _aws_device(wires=n_wires)

    H = transmon_drive(0.02, np.pi, 0.5, [0])
    op = ParametrizedEvolution(H, [], t=50)

    braket_gate = translate_operation(op, device=dev._device)

    assert isinstance(braket_gate, gates.PulseGate)
    assert braket_gate.qubit_count == 1

    ps = braket_gate.pulse_sequence
    expected_frame = dev._device.frames["q0_drive"]

    frames = list(ps._frames.values())
    assert len(frames) == 1
    assert frames[0] == expected_frame

    waveforms = list(ps._waveforms.values())
    assert len(waveforms) == 1
    assert isinstance(waveforms[0], ConstantWaveform)
    assert np.isclose(waveforms[0].length, 50e-9)
    assert np.isclose(waveforms[0].iq, 0.02)


def test_translate_parametrized_evolution_constant_callable_amplitude():
    """Test that a ParametrizedEvolution with constant amplitude, phase, and frequency
    is translated to a PulseGate correctly."""
    n_wires = 4
    dev = _aws_device(wires=n_wires)

    H = transmon_drive(qml.pulse.constant, np.pi, 0.5, [0])
    op = ParametrizedEvolution(H, [0.2], t=50)

    braket_gate = translate_operation(op, device=dev._device)

    assert isinstance(braket_gate, gates.PulseGate)
    assert braket_gate.qubit_count == 1

    ps = braket_gate.pulse_sequence
    expected_frame = dev._device.frames["q0_drive"]

    frames = list(ps._frames.values())
    assert len(frames) == 1
    assert frames[0] == expected_frame

    waveforms = list(ps._waveforms.values())
    assert len(waveforms) == 1
    assert isinstance(waveforms[0], ConstantWaveform)
    assert np.isclose(waveforms[0].length, 50e-9)
    assert np.isclose(waveforms[0].iq, 0.2)


def test_translate_parametrized_evolution_callable():
    """Test that a ParametrizedEvolution with callable amplitude, phase, and frequency
    is translated to a PulseGate correctly."""
    n_wires = 4
    dev = _aws_device(wires=n_wires)

    H = transmon_drive(amplitude, qml.pulse.constant, qml.pulse.constant, [0])

    amplitude_param = 0.1
    phase_param = np.pi
    frequency_param = 3.5
    op = ParametrizedEvolution(H, [amplitude_param, phase_param, frequency_param], t=50)

    braket_gate = translate_operation(op, device=dev._device)
    assert isinstance(braket_gate, gates.PulseGate)
    assert braket_gate.qubit_count == 1

    ps = braket_gate.pulse_sequence
    expected_frame = dev._device.frames["q0_drive"]

    frames = list(ps._frames.values())
    assert len(frames) == 1
    assert frames[0] == expected_frame

    dt = expected_frame.port.dt * 1e9
    amplitudes = [amplitude(amplitude_param, t) for t in np.arange(0, 50 + dt, dt)]

    waveforms = list(ps._waveforms.values())
    assert len(waveforms) == 1
    assert isinstance(waveforms[0], ArbitraryWaveform)
    assert np.allclose(waveforms[0].amplitudes, amplitudes)


def test_translate_parametrized_evolution_mixed():
    """Test that a ParametrizedEvolution with one constant and one callable amplitude pulse
    along with mixed constant and qml.pulse.constant phase and frequenciesis translated to
    a PulseGate correctly."""
    n_wires = 4
    dev = _aws_device(wires=n_wires)

    H = transmon_drive(0.02, qml.pulse.constant, 0.5, [0])
    H += transmon_drive(amplitude, -np.pi / 2, qml.pulse.constant, [1])

    phase_param = 1.5
    amplitude_param = 0.1
    frequency_param = 4.0
    op = ParametrizedEvolution(H, [phase_param, amplitude_param, frequency_param], t=50)

    braket_gate = translate_operation(op, device=dev._device)

    assert isinstance(braket_gate, gates.PulseGate)
    assert braket_gate.qubit_count == 2

    ps = braket_gate.pulse_sequence
    expected_frames = [dev._device.frames[f"q{w}_drive"] for w in range(2)]

    frames = list(ps._frames.values())
    assert len(frames) == 2
    assert all(actual == expected for actual, expected in zip(frames, expected_frames))

    dt = expected_frames[0].port.dt * 1e9
    amplitudes = [amplitude(amplitude_param, t) for t in np.arange(0, 50 + dt, dt)]

    waveforms = list(ps._waveforms.values())
    assert len(waveforms) == 2
    assert isinstance(waveforms[0], ConstantWaveform)
    assert np.isclose(waveforms[0].length, 50e-9)
    assert np.isclose(waveforms[0].iq, 0.02)

    assert isinstance(waveforms[1], ArbitraryWaveform)
    assert np.allclose(waveforms[1].amplitudes, amplitudes)


def test_translate_parametrized_evolution_multi_callable_amplitudes():
    """Test that a ParametrizedEvolution with multiple callable amplitude pulses is translated to
    a PulseGate correctly."""
    n_wires = 4
    dev = _aws_device(wires=n_wires)

    def second_amplitude(p, t):
        return p[0] * np.sin(p[1] * t) ** p[2] + p[0] * 1.1

    H = transmon_drive(amplitude, np.pi, qml.pulse.constant, [0])
    H += transmon_drive(second_amplitude, -np.pi / 2, 0.42, [1])

    first_amp_param = 0.1
    second_amp_param = [0.5, np.pi, 3]
    op = ParametrizedEvolution(H, [first_amp_param, 3.5, second_amp_param], t=50)

    braket_gate = translate_operation(op, device=dev._device)
    assert braket_gate.qubit_count == 2

    ps = braket_gate.pulse_sequence
    expected_frames = [dev._device.frames[f"q{w}_drive"] for w in range(2)]

    frames = list(ps._frames.values())
    assert len(frames) == 2
    assert all(actual == expected for actual, expected in zip(frames, expected_frames))

    dt = expected_frames[0].port.dt * 1e9
    amplitudes = [
        [amplitude(first_amp_param, t) for t in np.arange(0, 50 + dt, dt)],
        [second_amplitude(second_amp_param, t) for t in np.arange(0, 50 + dt, dt)],
    ]

    waveforms = list(ps._waveforms.values())
    assert len(waveforms) == 2
    assert all(isinstance(w, ArbitraryWaveform) for w in waveforms)
    assert all(np.allclose(w.amplitudes, a) for w, a in zip(waveforms, amplitudes))


@pytest.mark.parametrize("pl_cls, braket_cls, qubits, params, inv_params", testdata_inverses)
def test_translate_operation_inverse(pl_cls, braket_cls, qubits, params, inv_params):
    """Tests that inverse gates are translated correctly"""
    pl_op = qml.adjoint(pl_cls(*params, wires=qubits))
    braket_gate = braket_cls(*inv_params)
    assert translate_operation(pl_op) == braket_gate
    if isinstance(pl_op.base, (GPi, GPi2, MS, AAMS)):
        op_name = _braket_to_pl[
            re.match(
                "^[a-z0-2]+",
                braket_gate.to_ir(qubits, ir_type=IRType.OPENQASM),
            )[0]
        ]
    else:
        op_name = _braket_to_pl[
            braket_gate.to_ir(qubits).__class__.__name__.lower().replace("_", "")
        ]

    assert (
        f"Adjoint({op_name})" == pl_op.name
        if pl_op.name != "Adjoint(MS)"
        # PL MS and AAMS both get translated to Braket MS.
        # Braket MS gets translated to PL AAMS.
        else f"Adjoint({op_name})" == "Adjoint(AAMS)"
    )
    # assert f"Adjoint({op_name})" == pl_op.name


@patch("braket.circuits.gates.X.adjoint")
def test_translate_operation_multiple_inverses_unsupported(adjoint):
    """Test that an error is raised when translating a Braket operation which adjoint contains
    multiple operations."""
    # Mock ``gates.X.adjoint()`` to return two gates
    adjoint.return_value = [gates.X(), gates.I()]
    pl_op = qml.adjoint(qml.PauliX(0))
    with pytest.raises(
        NotImplementedError,
        match="The adjoint of the Braket operation X",
    ):
        translate_operation(pl_op)


@pytest.mark.parametrize("pl_cls, braket_cls, qubit", testdata_named_inverses)
def test_translate_operation_named_inverse(pl_cls, braket_cls, qubit):
    """Tests that operations whose inverses are named Braket gates are inverted correctly"""
    pl_op = qml.adjoint(pl_cls(wires=[qubit]))
    braket_gate = braket_cls()
    assert translate_operation(pl_op) == braket_gate
    assert (
        _braket_to_pl[braket_gate.to_ir([qubit]).__class__.__name__.lower().replace("_", "")]
        == pl_op.name
    )


def test_translate_operation_iswap_inverse():
    """Tests that the iSwap gate is inverted correctly"""
    assert translate_operation(qml.adjoint(qml.ISWAP(wires=[0, 1]))) == gates.PSwap(3 * np.pi / 2)


def test_translate_operation_param_names_wrong_length():
    """Tests that translation fails if provided param_names list is the wrong length"""
    with pytest.raises(
        ValueError,
        match="Parameter names list must be equal to number of operation parameters",
    ):
        translate_operation(qml.RX(0.432, wires=0), use_unique_params=True, param_names=["a", "b"])


@pytest.mark.parametrize(
    "return_type, braket_result_type", zip(pl_return_types, braket_result_types)
)
def test_translate_result_type_observable(return_type, braket_result_type):
    """Tests if a PennyLane return type that involves an observable is successfully converted into a
    Braket result using translate_result_type"""
    res_type = braket_result_type.__class__
    pl_res_type = _braket_to_pl_result_types[res_type]
    tape = qml.tape.QuantumTape(measurements=[pl_res_type(qml.Hadamard(0))])
    braket_result_type_calculated = translate_result_type(tape.measurements[0], [0], frozenset())

    assert braket_result_type == braket_result_type_calculated


@pytest.mark.parametrize(
    "pl_obs, braket_obs, targets, param_names",
    [
        (qml.Hadamard(0), observables.H(), [0], []),
        (qml.PauliX(0), observables.X(), [0], ["p_000"]),
        (
            qml.PauliX(0) @ qml.PauliY(1),
            observables.X() @ observables.Y(),
            [0, 1],
            ["p_000", "p_001", "p_003"],
        ),
    ],
)
def test_get_adjoint_gradient_result_type(pl_obs, braket_obs, targets, param_names):
    """Tests that an AdjointGradient result type is returned correctly"""
    braket_result_type_calculated = get_adjoint_gradient_result_type(
        pl_obs,
        targets,
        frozenset(["AdjointGradient"]),
        param_names,
    )
    braket_result_type = AdjointGradient(
        observable=braket_obs, target=targets, parameters=param_names
    )
    assert braket_result_type == braket_result_type_calculated


def test_get_adjoint_gradient_result_type_unsupported():
    """Tests if a NotImplementedError is raised by translate_result_type when a PennyLane state
    return type is converted while not supported by the device"""
    pl_obs = qml.Hadamard(0)
    targets = [0]
    param_names = ["p_000", "p_001"]
    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        get_adjoint_gradient_result_type(pl_obs, targets, frozenset(), param_names)


def test_translate_result_type_hamiltonian_expectation():
    """Tests that a Hamiltonian is translated correctly"""
    obs = qml.Hamiltonian((2, 3), (qml.PauliX(wires=0), qml.PauliY(wires=1)))
    tape = qml.tape.QuantumTape(measurements=[qml.expval(obs)])
    braket_result_type_calculated = translate_result_type(tape.measurements[0], [0], frozenset())
    braket_result_type = (
        Expectation(observables.X(), [0]),
        Expectation(observables.Y(), [1]),
    )
    assert braket_result_type == braket_result_type_calculated


@pytest.mark.parametrize("return_type", [Variance, Sample])
def test_translate_result_type_hamiltonian_unsupported_return(return_type):
    """Tests if a NotImplementedError is raised by translate_result_type
    with Hamiltonian observable and non-Expectation return type"""
    obs = qml.Hamiltonian((2, 3), (qml.PauliX(wires=0), qml.PauliY(wires=1)))
    tape = qml.tape.QuantumTape(measurements=[_braket_to_pl_result_types[return_type](obs)])
    with pytest.raises(NotImplementedError, match="unsupported for Hamiltonian"):
        translate_result_type(tape.measurements[0], [0], frozenset())


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = qml.probs(wires=Wires([0]))
    braket_result_type_calculated = translate_result_type(mp, [0], frozenset())

    braket_result_type = Probability([0])

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_state_vector():
    """Tests if a PennyLane state vector return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = qml.state()
    braket_result_type_calculated = translate_result_type(
        mp, [], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result_type = StateVector()

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_density_matrix():
    """Tests if a PennyLane density matrix return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = qml.state()
    braket_result_type_calculated = translate_result_type(mp, [], frozenset(["DensityMatrix"]))

    braket_result_type = DensityMatrix()

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_density_matrix_partial():
    """Tests if a PennyLane partial density matrix return type is successfully converted into a
    Braket result using translate_result_type"""
    mp = qml.density_matrix(wires=[0])
    braket_result_type_calculated = translate_result_type(
        mp, [0], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result_type = DensityMatrix([0])

    assert braket_result_type == braket_result_type_calculated


def test_translate_result_type_state_unimplemented():
    """Tests if a NotImplementedError is raised by translate_result_type when a PennyLane state
    return type is converted while not supported by the device"""
    mp = qml.state()
    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(mp, [0], frozenset())


def test_translate_result_type_unsupported_return():
    """Tests if a NotImplementedError is raised by translate_result_type for an unknown
    return_type"""
    tape = qml.tape.QuantumTape(measurements=[qml.purity(wires=[0])])

    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(tape.measurements[0], [0], frozenset())


def test_translate_result_type_unsupported_obs():
    """Tests if a DeviceError is raised by translate_result_type for an unknown observable"""
    tape = qml.tape.QuantumTape(measurements=[qml.expval(qml.S(wires=0))])

    with pytest.raises(qml.DeviceError, match="Unsupported observable"):
        translate_result_type(tape.measurements[0], [0], frozenset())


def test_translate_result():
    result_dict = _result_meta()
    result_dict["resultTypes"] = [
        {"type": {"targets": [0], "type": "probability"}, "value": [0.5, 0.5]}
    ]
    targets = [0]
    result_dict["measuredQubits"]: targets
    result = GateModelQuantumTaskResult.from_string(json.dumps(result_dict))
    mp = qml.probs(wires=Wires([0]))
    translated = translate_result(result, mp, targets, frozenset())
    assert (translated == result.result_types[0].value).all()


def test_translate_result_hamiltonian():
    result_dict = _result_meta()
    result_dict["resultTypes"] = [
        {
            "type": {
                "observable": ["x", "y"],
                "targets": [0, 1],
                "type": "expectation",
            },
            "value": 2.0,
        },
        {
            "type": {"observable": ["x"], "targets": [1], "type": "expectation"},
            "value": 3.0,
        },
    ]
    targets = [0, 1]
    result_dict["measuredQubits"]: targets
    result = GateModelQuantumTaskResult.from_string(json.dumps(result_dict))
    ham = qml.Hamiltonian((2, 1), (qml.PauliX(0) @ qml.PauliY(1), qml.PauliX(1)))
    tape = qml.tape.QuantumTape(measurements=[qml.expval(ham)])
    translated = translate_result(result, tape.measurements[0], targets, frozenset())
    expected = 2 * result.result_types[0].value + result.result_types[1].value
    assert translated == expected


def _result_meta() -> dict:
    return {
        "braketSchemaHeader": {
            "name": "braket.task_result.gate_model_task_result",
            "version": "1",
        },
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
                    "name": "braket.ir.jaqcd.program",
                    "version": "1",
                },
                "instructions": [{"control": 0, "target": 1, "type": "cnot"}],
            },
        },
    }


@pytest.mark.parametrize(
    "expected_braket_H, pl_H",
    [
        (
            2 * observables.X(0) @ observables.Y(1) @ observables.Z(2),
            2 * qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2),
        ),
        (
            2 * (observables.X(0) @ observables.Y(1) @ observables.Z(2)),
            2 * (qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2)),
        ),
        (
            2 * observables.X(0) @ observables.Y(1) @ observables.Z(2) + 0.75 * observables.X(0),
            2 * qml.PauliX(wires=0) @ qml.PauliY(wires=1) @ qml.PauliZ(wires=2)
            + 0.75 * qml.PauliX(0),
        ),
        (1.25 * observables.H(0), 1.25 * qml.Hadamard(wires=0)),
        (
            observables.X(0) @ observables.Y(1),
            qml.ops.Prod(qml.PauliX(0), qml.PauliY(1)),
        ),
        (
            observables.X(0) + observables.Y(1),
            qml.ops.Sum(qml.PauliX(0), qml.PauliY(1)),
        ),
        (observables.X(0), qml.ops.SProd(scalar=4, base=qml.PauliX(0))),
    ],
)
def test_translate_hamiltonian_observable(expected_braket_H, pl_H):
    translated_braket_H = _translate_observable(pl_H)
    assert expected_braket_H == translated_braket_H


def test_translate_result_type_adjoint_gradient():
    print("not implemented yet")


def test_translation_no_properties():
    class MockDevice:
        pass

    needs_props = "Device needs to have properties defined."
    with pytest.raises(AttributeError, match=needs_props):
        supported_operations(MockDevice())
