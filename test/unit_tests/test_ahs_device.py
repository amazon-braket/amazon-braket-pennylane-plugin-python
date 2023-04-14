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


import pytest
import warnings
import json
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch

import pennylane as qml
import numpy as np

from braket.pennylane_plugin.ahs_device import BraketAwsAhsDevice, BraketLocalAhsDevice, BraketAhsDevice

from braket.aws import AwsDevice, AwsQuantumTask
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.ahs.atom_arrangement import AtomArrangement
from braket.device_schema import DeviceActionType, DeviceActionProperties
from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import QueraAhsParadigmProperties
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import ShotResult
from braket.timings.time_series import TimeSeries

from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.pulse.rydberg import rydberg_interaction, rydberg_drive
from pennylane.pulse.hardware_hamiltonian import HardwareHamiltonian, HardwarePulse

from dataclasses import dataclass
from functools import partial


coordinates1 = [[0, 0], [0, 5], [5, 0], [10, 5], [5, 10], [10, 10]]
wires1 = [1, 6, 0, 2, 4, 3]

coordinates2 = [[0, 0], [5.5, 0.0], [2.75, 4.763139720814412]]  # in µm
H_i = rydberg_interaction(coordinates2)


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p[0] * np.cos(p[1] * t**2)


def amp(p, t):
    return p[0] * np.exp(-(t-p[1])**2/(2*p[2]**2))


# functions of time to use as partially evaluated callable parameters in tests
def sin_fn(t):
    return np.sin(t)


def cos_fn(t):
    return np.cos(t)


def lin_fn(t):
    return 3.48 * t


def quad_fn(t):
    return 4.5 * t ** 2


params1 = 1.2
params2 = [3.4, 5.6]
params_amp = [2.5, 0.9, 0.3]

HAMILTONIANS_AND_PARAMS = [(H_i + rydberg_drive(amplitude=4, phase=1, detuning=3, wires=[0, 1, 2]), []),
                (H_i + rydberg_drive(amplitude=amp, phase=1, detuning=2, wires=[0, 1, 2]), [params_amp]),
                (H_i + rydberg_drive(amplitude=2, phase=f1, detuning=2, wires=[0, 1, 2]), [params1]),
                (H_i + rydberg_drive(amplitude=2, phase=2, detuning=f2, wires=[0, 1, 2]), [params2]),
                (H_i + rydberg_drive(amplitude=amp, phase=1, detuning=f2, wires=[0, 1, 2]), [params_amp, params2]),
                (H_i + rydberg_drive(amplitude=4, phase=f2, detuning=f1, wires=[0, 1, 2]), [params2, params1]),
                (H_i + rydberg_drive(amplitude=amp, phase=f2, detuning=4, wires=[0, 1, 2]), [params_amp, params2]),
                (H_i + rydberg_drive(amplitude=amp, phase=f2, detuning=f1, wires=[0, 1, 2]), [params_amp, params2, params1])
                ]


DEV_ATTRIBUTES = [(BraketAwsAhsDevice, "Aquila", "braket.aws.ahs")]

dev_sim = BraketLocalAhsDevice(wires=3, shots=17)


PARADIGM_PROPERTIES = QueraAhsParadigmProperties.parse_raw_schema(
    json.dumps(
        {
            "braketSchemaHeader": {"name": "braket.device_schema.quera.quera_ahs_paradigm_properties", "version": "1"},
            "qubitCount": 1,
            "lattice": {'area': {'width': 3, 'height': 3}, 'geometry': {'spacingRadialMin': 3,
                                                                        'spacingVerticalMin': 3,
                                                                        'positionResolution': 3,
                                                                        'numberSitesMax': 3}},
            "rydberg": {'c6Coefficient': 5.42e-24,
                        'rydbergGlobal': {'rabiFrequencyRange': (0, 3),
                                          'rabiFrequencyResolution': 3,
                                          'rabiFrequencySlewRateMax': 3,
                                          'detuningRange': (0, 3),
                                          'detuningResolution': 3,
                                          'detuningSlewRateMax': 3,
                                          'phaseRange': (0, 3),
                                          'phaseResolution': 3,
                                                                     'timeResolution': 3,
                                                                     'timeDeltaMin': 3,
                                                                     'timeMin': 3,
                                                                     'timeMax': 3}},
            "performance": {'lattice': {'positionErrorAbs': 3}, 'rydberg': {'rydbergGlobal': {'rabiFrequencyErrorRel': 3}}},
        }
    )
)


class MockAwsSession:
    @staticmethod
    def add_braket_user_agent(user_agent):
        pass


class MockDevProperties:
    paradigm = PARADIGM_PROPERTIES
    action = {DeviceActionType.AHS: DeviceActionProperties(version=['1'], actionType=DeviceActionType.AHS)}


@pytest.fixture(scope="function")
def mock_aws_device(monkeypatch, wires=3):
    """A function to create a mock device that mocks most of the methods except for e.g. probability()"""
    with monkeypatch.context() as m:
        m.setattr(AwsDevice, "__init__", lambda self, *args, **kwargs: None)
        m.setattr(AwsDevice, "aws_session", MockAwsSession)
        m.setattr(AwsDevice, "type", mock.PropertyMock)
        m.setattr(AwsDevice, "properties", MockDevProperties)

        def get_aws_device(
                wires=wires,
                shots=17,
                device_arn="baz",
                **kwargs,
        ):

            dev = BraketAwsAhsDevice(
                wires=wires,
                s3_destination_folder=("foo", "bar"),
                device_arn=device_arn,
                aws_session=Mock(),
                shots=shots,
                **kwargs,
            )
            # needed by BraketAwsAhsDevice functions
            dev._device._arn = device_arn
            dev._device._aws_session = Mock()
            return dev

        yield get_aws_device


def dummy_ahs_program():

    # amplutide 10 for full duration
    amplitude = TimeSeries()
    amplitude.put(0, 10)
    amplitude.put(4e-6, 10)

    # phase and detuning 0 for full duration
    phi = TimeSeries().put(0, 0).put(4e-6, 0)
    detuning = TimeSeries().put(0, 0).put(4e-6, 0)

    # Hamiltonian
    H = DrivingField(amplitude, phi, detuning)

    # register
    register = AtomArrangement()
    for [x, y] in coordinates2:
        register.add([x*1e-6, y*1e-6])

    ahs_program = AnalogHamiltonianSimulation(
        hamiltonian=H,
        register=register
    )

    return ahs_program


# dummy data classes for testing result processing
@dataclass
class Status:
    value: str


@dataclass
class DummyMeasurementResult:
    status: Status
    pre_sequence: np.array
    post_sequence: np.array


DUMMY_RESULTS = [(DummyMeasurementResult(Status('Success'), np.array([1]), np.array([1])), np.array([0])),
                 (DummyMeasurementResult(Status('Success'), np.array([1]), np.array([0])), np.array([1])),
                 (DummyMeasurementResult(Status('Success'), np.array([0]), np.array([0])), np.array([np.NaN])),
                 (DummyMeasurementResult(Status('Failure'), np.array([1]), np.array([1])), np.array([np.NaN])),
                 (DummyMeasurementResult(Status('Success'), np.array([1, 1, 0]), np.array([1, 0, 0])), np.array([0, 1, np.NaN])),
                 (DummyMeasurementResult(Status('Success'), np.array([1, 1]), np.array([0, 0])), np.array([1, 1])),
                 (DummyMeasurementResult(Status('Success'), np.array([0, 1]), np.array([0, 0])), np.array([np.NaN, 1])),
                 (DummyMeasurementResult(Status('Failure'), np.array([1, 1]), np.array([1, 1])), np.array([np.NaN, np.NaN])),
                 (DummyMeasurementResult(Status('Success'), np.array([0, 1]), np.array([1, 0])), np.array([np.NaN, 1])),
                 ]


class TestBraketAhsDevice:
    """Tests that behaviour defined for both the LocalSimulator and the
    Aquila hardware in the base device work as expected"""

    def test_initialization(self):
        """Test the device initializes with the expected attributes"""

        dev = BraketLocalAhsDevice(wires=3, shots=11)

        assert dev._device.name == "RydbergAtomSimulator"
        assert dev.short_name == "braket.local.ahs"
        assert dev.shots == 11
        assert dev.ahs_program is None
        assert dev.samples is None
        assert dev.pennylane_requires == ">=0.30.0"
        assert dev.operations == {"ParametrizedEvolution"}

    def test_settings(self):
        dev = dev_sim
        assert isinstance(dev.settings, dict)
        assert 'interaction_coeff' in dev.settings.keys()
        assert len(dev.settings.keys()) == 1
        assert dev.settings['interaction_coeff'] == 5420000

    @pytest.mark.parametrize("dev_cls, shots", [(BraketLocalAhsDevice, 1000),
                                                (BraketLocalAhsDevice, 2)])
    def test_setting_shots(self, dev_cls, shots):
        """Test that setting shots changes number of shots from default (100)"""
        dev = dev_cls(wires=3, shots=shots)
        assert dev.shots == shots

    @pytest.mark.parametrize("shots", [0, None])
    def test_no_shots_raises_error(self, shots):
        """Test that an error is raised if shots are set to 0 or None"""
        with pytest.raises(RuntimeError, match="This device requires shots"):
            BraketLocalAhsDevice(wires=3, shots=shots)

    @pytest.mark.parametrize("dev_cls, wires", [(BraketLocalAhsDevice, 2),
                                                (BraketLocalAhsDevice, [0, 2, 4]),
                                                (BraketLocalAhsDevice, [0, 'a', 7]),
                                                (BraketLocalAhsDevice, 7)])
    def test_setting_wires(self, dev_cls, wires):
        """Test setting wires"""
        dev = dev_cls(wires=wires)

        if isinstance(wires, int):
            assert len(dev.wires) == wires
            assert dev.wires.labels == tuple(i for i in range(wires))
        else:
            assert len(wires) == len(dev.wires)
            assert dev.wires.labels == tuple(wires)

    @pytest.mark.parametrize("hamiltonian, params", HAMILTONIANS_AND_PARAMS)
    def test_apply(self, hamiltonian, params):
        """Test that apply creates and saves an ahs_program and samples as expected"""
        t = 0.4
        operations = [ParametrizedEvolution(hamiltonian, params, t)]
        dev = BraketLocalAhsDevice(wires=operations[0].wires)

        assert dev.samples is None
        assert dev.ahs_program is None

        dev.apply(operations)

        assert dev.samples is not None
        assert len(dev.samples.measurements) == dev.shots
        assert len(dev.samples.measurements[0].pre_sequence) == len(dev.wires)

        assert isinstance(dev.ahs_program, AnalogHamiltonianSimulation)
        assert dev.ahs_program.register == dev.register
        assert dev.ahs_program.hamiltonian.amplitude.time_series.times()[-1] == t*1e-6

    def test_apply_unsupported(self):
        """Tests that apply() throws NotImplementedError when it encounters an unknown gate."""

        with pytest.raises(NotImplementedError):
            dev_sim.apply([qml.PauliX(0)])

    @pytest.mark.parametrize("hamiltonian, params", HAMILTONIANS_AND_PARAMS)
    def test_create_ahs_program(self, hamiltonian, params):
        """Test that we can create an AnalogueHamiltonianSimulation from an
        evolution operator and store it on the device"""

        evolution = ParametrizedEvolution(hamiltonian, params, 1.5)
        dev = BraketLocalAhsDevice(wires=3)

        assert dev.ahs_program is None

        ahs_program = dev.create_ahs_program(evolution)

        # AHS program is created and stored on the device
        assert isinstance(dev.ahs_program, AnalogHamiltonianSimulation)

        # compare evolution and ahs_program registers
        assert ahs_program.register.coordinate_list(0) == [c[0] * 1e-6 for c in evolution.H.settings.register]
        assert ahs_program.register.coordinate_list(1) == [c[1] * 1e-6 for c in evolution.H.settings.register]

        # elements of the hamiltonian have the expected shape
        h = ahs_program.hamiltonian
        amp_time, amp_vals = h.amplitude.time_series.times(), h.amplitude.time_series.values()
        phase_time, phase_vals = h.phase.time_series.times(), h.phase.time_series.values()
        det_time, det_vals = h.detuning.time_series.times(), h.detuning.time_series.values()

        assert amp_time == phase_time == det_time
        assert amp_time[0] == evolution.t[0] * 1e-6
        assert amp_time[-1] == evolution.t[1] * 1e-6

    def test_generate_samples(self):
        """Test that generate_samples creates a list of arrays with the expected shape for the task run"""
        ahs_program = dummy_ahs_program()

        # checked in _validate_operations in the full pipeline
        # since these are created manually for the unit test elsewhere in the file,
        # we confirm the values used for the test are valid here
        assert len(ahs_program.register.coordinate_list(0)) == len(dev_sim.wires)

        task = dev_sim._run_task(ahs_program)

        dev_sim.samples = task.result()

        samples = dev_sim.generate_samples()

        assert len(samples) == 17
        assert len(samples[0]) == len(dev_sim.wires)
        assert isinstance(samples[0], np.ndarray)

    def test_validate_operations_multiple_operators(self):
        """Test that an error is raised if there are multiple operators"""

        H1 = rydberg_drive(amp, f1, 2, wires=[0, 1, 2])
        op1 = qml.evolve(H_i + H1)
        op2 = qml.evolve(H_i + H1)

        with pytest.raises(NotImplementedError, match="Support for multiple ParametrizedEvolution operators"):
            dev_sim._validate_operations([op1, op2])

    def test_validate_operations_wires_match_device(self):
        """Test that an error is raised if the wires on the Hamiltonian
        don't match the wires on the device."""
        H = H_i + rydberg_drive(3, 2, 2, wires=[0, 1, 2])

        dev1 = BraketLocalAhsDevice(wires=len(H.wires)-1)
        dev2 = BraketLocalAhsDevice(wires=len(H.wires)+1)

        with pytest.raises(RuntimeError, match="Device wires must match wires of the evolution."):
            dev1._validate_operations([ParametrizedEvolution(H, [], 1)])

        with pytest.raises(RuntimeError, match="Device wires must match wires of the evolution."):
            dev2._validate_operations([ParametrizedEvolution(H, [], 1)])

    def test_validate_operations_register_matches_wires(self):
        """Test that en error is raised in the length of the register doesn't match
        the number of wires on the device"""

        # register has wires [0, 1, 2], drive has wire [3]
        # creating a Hamiltonian like this in PL will raise a warning, but not an error
        H = H_i + rydberg_drive(3, 2, 2, wires=3)

        # device wires [0, 1, 2, 3] match overall wires, but not length of register
        dev = BraketLocalAhsDevice(wires=4)

        with pytest.raises(RuntimeError, match="The defined interaction term has register"):
            dev._validate_operations([ParametrizedEvolution(H, [], 1)])

    def test_validate_operations_not_hardware_hamiltonian(self):
        """Test that an error is raised if the ParametrizedHamiltonian on the operator
        is not a HardwareHamiltonian and so does not contain pulse upload information"""

        H1 = 2 * qml.PauliX(0) + f1 * qml.PauliY(1) + f2 * qml.PauliZ(2)
        op1 = qml.evolve(H1)

        with pytest.raises(RuntimeError, match="Expected a HardwareHamiltonian instance"):
            dev_sim._validate_operations([op1])

    def test_validate_pulses_no_pulses(self):
        """Test that _validate_pulses raises an error if there are no pulses saved
        on the Hamiltonian"""

        with pytest.raises(RuntimeError, match="No pulses found"):
            dev_sim._validate_pulses(H_i.pulses)

    @pytest.mark.parametrize("coordinates", [coordinates1, coordinates2])
    def test_create_register(self, coordinates):
        """Test that an AtomArrangement with the expected coordinates is created
        and stored on the device"""

        dev = BraketLocalAhsDevice(wires=len(coordinates))
        
        assert dev.register is None

        dev._create_register(coordinates)
        coordinates_from_register = [[x*1e6, y*1e6] for x, y in zip(dev.register.coordinate_list(0),
                                                                    dev.register.coordinate_list(1))]

        assert isinstance(dev.register, AtomArrangement)
        assert coordinates_from_register == coordinates

    @pytest.mark.parametrize("hamiltonian, params", HAMILTONIANS_AND_PARAMS)
    def test_evaluate_pulses(self, hamiltonian, params):
        """Test that the callables describing pulses are partially evaluated as expected"""

        ev_op = ParametrizedEvolution(hamiltonian, params, 1.5)

        pulse = ev_op.H.pulses[0]
        params = ev_op.parameters
        idx = 0

        # check which of initial pulse parameters are callable
        callable_amp = callable(pulse.amplitude)
        callable_phase = callable(pulse.phase)
        callable_detuning = callable(pulse.frequency)

        # get an expected value for each pulse parameter at t=1.7
        if callable_amp:
            amp_sample = pulse.amplitude(params[idx], 1.7)
            idx += 1
        else:
            amp_sample = pulse.amplitude

        if callable_phase:
            phase_sample = pulse.phase(params[idx], 1.7)
            idx += 1
        else:
            phase_sample = pulse.phase

        if callable_detuning:
            detuning_sample = pulse.frequency(params[idx], 1.7)
            idx += 1
        else:
            detuning_sample = pulse.frequency

        # evaluate pulses
        dev_sim._evaluate_pulses(ev_op)

        # confirm that if initial pulse parameter was a callable, it is now a partial
        # confirm that post-evaluation value at t=1.7 seconds matches expectation
        if callable_amp:
            assert isinstance(dev_sim.pulses[0].amplitude, partial)
            assert amp_sample == dev_sim.pulses[0].amplitude(1.7)
        else:
            assert amp_sample == dev_sim.pulses[0].amplitude

        if callable_phase:
            assert isinstance(dev_sim.pulses[0].phase, partial)
            assert phase_sample == dev_sim.pulses[0].phase(1.7)
        else:
            assert phase_sample == dev_sim.pulses[0].phase

        if callable_detuning:
            assert isinstance(dev_sim.pulses[0].frequency, partial)
            assert detuning_sample == dev_sim.pulses[0].frequency(1.7)
        else:
            assert detuning_sample == dev_sim.pulses[0].frequency

    @pytest.mark.parametrize("time_interval", [[1.5, 2.3], [0, 1.2], [0.111, 3.789]])
    def test_get_sample_times(self, time_interval):
        """Tests turning an array of [start, end] times into time set-points"""

        times = dev_sim._get_sample_times(time_interval)

        num_points = len(times)
        diffs = [times[i]-times[i-1] for i in range(1, num_points)]

        # start and end times match but are in units of s and us respectively
        assert times[0]*1e6 == time_interval[0]
        assert times[-1]*1e6 == time_interval[1]

        # distances between points are close to but exceed 50ns
        assert all(d > 50e-9 for d in diffs)
        assert np.allclose(diffs, 50e-9, atol=5e-9)

    def test_convert_to_time_series_constant(self):
        """Test creating a TimeSeries when the pulse parameter is defined as a constant float"""

        times = [0, 1, 2, 3, 4, 5]
        ts = dev_sim._convert_to_time_series(pulse_parameter=4.3, time_points=times)

        assert ts.times() == times
        assert all(p == 4.3 for p in ts.values())

    def test_convert_to_time_series_callable(self):
        """Test creating a TimeSeries when the pulse parameter is defined as a function of time"""

        def f(t):
            return np.sin(t)

        times_us = [0, 1, 2, 3, 4, 5]  # microseconds
        times_s = [t * 1e-6 for t in times_us]  # seconds

        ts = dev_sim._convert_to_time_series(pulse_parameter=f, time_points=times_s)
        expected_vals = [np.sin(t) for t in times_us]

        assert ts.times() == times_s
        assert np.all(ts.values() == expected_vals)

    def test_convert_to_time_series_scaling_factor(self):
        """Test creating a TimeSeries from pulse information and time set-points"""
        def f(t):
            return np.sin(t)

        times_us = [0, 1, 2, 3, 4, 5]  # microseconds
        times_s = [t * 1e-6 for t in times_us]  # seconds

        ts = dev_sim._convert_to_time_series(pulse_parameter=f, time_points=times_s, scaling_factor=1.7)
        expected_vals = [np.sin(t)*1.7 for t in times_us]

        assert ts.times() == times_s
        assert ts.values() == expected_vals

    @pytest.mark.parametrize("pulse", [HardwarePulse(1, 2, sin_fn, wires=[0, 1, 2]),
                                       HardwarePulse(cos_fn, 1.7, 2.3, wires=[0, 1, 2]),
                                       HardwarePulse(3.8, lin_fn, 1.9, wires=[0, 1, 2]),
                                       HardwarePulse(lin_fn, sin_fn, quad_fn, wires=[0, 1, 2])])
    def test_convert_pulse_to_driving_field(self, pulse):
        """Test that a time interval in microseconds (as passed to the qnode in PennyLane)
        and a Pulse object containing constant or time-dependent pulse parameters (floats
        and/or callables that have been evaluated to be a function only of time)
        and can be converted into a DrivingField
        """

        drive = dev_sim._convert_pulse_to_driving_field(pulse, [0, 1.5])

        assert isinstance(drive, DrivingField)

    @pytest.mark.parametrize("res, expected_output", DUMMY_RESULTS)
    def test_result_to_sample_output(self, res, expected_output):
        """Test function for converting the task results as returned by the
        device into sample measurement results for PennyLane"""

        output = dev_sim._result_to_sample_output(res)

        assert isinstance(output, np.ndarray)
        assert len(output) == len(res.post_sequence)
        assert np.allclose(output, expected_output, equal_nan=True)


class TestLocalAquilaDevice:
    """Test functionality specific to the local simulator device"""

    def test_validate_operations_multiple_drive_terms(self):
        """Test that an error is raised if there are multiple drive terms on
        the Hamiltonian"""
        pulses = [HardwarePulse(3, 4, 5, [0, 1]), HardwarePulse(4, 6, 7, [1, 2])]

        with pytest.raises(NotImplementedError, match="Multiple pulses in a Hamiltonian are not currently supported"):
            dev_sim._validate_pulses(pulses)

    @pytest.mark.parametrize("pulse_wires, dev_wires, res", [([0, 1, 2], [0, 1, 2, 3], 'error'),  # subset
                                                             ([5, 6, 7, 8, 9], [4, 5, 6, 7, 8], 'error'),  # mismatch
                                                             ([0, 1, 2, 3, 6], [1, 2, 3], 'error'),
                                                             ([0, 1, 2], [0, 1, 2], 'success')])
    def test_validate_pulse_is_global_drive(self, pulse_wires, dev_wires, res):
        """Test that an error is raised if the pulse does not describe a global drive"""

        dev = BraketLocalAhsDevice(wires=dev_wires)
        pulse = HardwarePulse(3, 4, 5, pulse_wires)

        if res == 'error':
            with pytest.raises(NotImplementedError, match="Only global drive is currently supported"):
                dev._validate_pulses([pulse])
        else:
            dev._validate_pulses([pulse])

    def test_run_task(self):
        ahs_program = dummy_ahs_program()

        task = dev_sim._run_task(ahs_program)

        assert isinstance(task, LocalQuantumTask)
        assert len(task.result().measurements) == 17  # dev_sim takes 17 shots
        assert isinstance(task.result().measurements[0], ShotResult)


class TestBraketAwsAhsDevice:
    """Test functionality specific to the hardware device"""

    def test_initialize(self, mock_aws_device):
        """Test the device initializes with the expected attributes"""
        dev = mock_aws_device()

        assert dev._s3_folder == ("foo", "bar")
        assert dev.shots == 17
        assert dev.ahs_program is None
        assert dev.samples is None
        assert dev.pennylane_requires == ">=0.30.0"
        assert dev.operations == {"ParametrizedEvolution"}
        assert dev.short_name == "braket.aws.ahs"

    def test_hardware_capabilities(self, mock_aws_device):
        """Test hardware capabilities can be retrieved"""

        dev = mock_aws_device()

        assert isinstance(dev.hardware_capabilities, dict)
        assert dev.hardware_capabilities == dict(PARADIGM_PROPERTIES)

    def test_settings(self, mock_aws_device):
        dev = mock_aws_device()
        assert list(dev.settings.keys()) == ['interaction_coeff']
        assert np.isclose(dev.settings['interaction_coeff'], 5420000)

    def test_validate_operations_multiple_drive_terms(self, mock_aws_device):
        """Test that an error is raised if there are multiple drive terms on
        the Hamiltonian"""
        dev = mock_aws_device()
        pulses = [HardwarePulse(3, 4, 5, [0, 1]), HardwarePulse(4, 6, 7, [1, 2])]

        with pytest.raises(NotImplementedError, match="Multiple pulses in a Hamiltonian are not currently supported"):
            dev._validate_pulses(pulses)

    @pytest.mark.parametrize("pulse_wires, dev_wires, res", [([0, 1, 2], [0, 1, 2, 3], 'error'),
                                                             ([5, 6, 7, 8, 9], [4, 5, 6, 7, 8], 'error'),
                                                             ([0, 1, 2, 3, 6], [1, 2, 3], 'error'),
                                                             ([0, 1, 2], [0, 1, 2], 'success')])
    def test_validate_pulse_is_global_drive(self, mock_aws_device, pulse_wires, dev_wires, res):
        """Test that an error is raised if the pulse does not describe a global drive"""

        dev = mock_aws_device(wires=dev_wires)
        pulse = HardwarePulse(3, 4, 5, pulse_wires)

        if res == 'error':
            with pytest.raises(NotImplementedError, match="Only global drive is currently supported"):
                dev._validate_pulses([pulse])
        else:
            dev._validate_pulses([pulse])

    def test_get_rydberg_c6(self, mock_aws_device):
        """Test that _get_rydberg_c6 retrieves the c6 coefficient from the hardware properties
        and converts to expected PennyLane units"""

        dev = mock_aws_device()
        c6 = dev._get_rydberg_c6()

        assert np.isclose(float(2 * np.pi * c6 / 1e30),  float(dev._device.properties.paradigm.rydberg.c6Coefficient))

    @pytest.mark.parametrize("hamiltonian, params", HAMILTONIANS_AND_PARAMS)
    def test_create_ahs_program(self, hamiltonian, params, mock_aws_device):
        """Test that we can create an AnalogueHamiltonianSimulation from an
        evolution operator and store it on the device"""

        evolution = ParametrizedEvolution(hamiltonian, params, 1.5)
        dev = mock_aws_device()

        assert dev.ahs_program is None

        ahs_program = dev.create_ahs_program(evolution)

        # AHS program is created and stored on the device
        assert isinstance(dev.ahs_program, AnalogHamiltonianSimulation)

        # all zeros because the ahs.discretize function without an actual hardware connection returns 0s
        assert ahs_program.register.coordinate_list(0) == [0 for c in evolution.H.settings.register]
        assert ahs_program.register.coordinate_list(1) == [0 for c in evolution.H.settings.register]

        # elements of the hamiltonian have the expected shape
        h = ahs_program.hamiltonian
        amp_time, amp_vals = h.amplitude.time_series.times(), h.amplitude.time_series.values()
        phase_time, phase_vals = h.phase.time_series.times(), h.phase.time_series.values()
        det_time, det_vals = h.detuning.time_series.times(), h.detuning.time_series.values()

        assert amp_time == phase_time == det_time

    def test_run_task(self, mock_aws_device):
        """Tests that a (mock) task can be created"""
        dev = mock_aws_device()
        ahs_program = dummy_ahs_program()

        task = dev._run_task(ahs_program)

        assert isinstance(task, AwsQuantumTask)
