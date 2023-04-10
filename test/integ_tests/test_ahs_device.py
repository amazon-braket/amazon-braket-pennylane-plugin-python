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

"""Tests that gates are correctly applied in the plugin device"""

import numpy as np
import pennylane as qml
import pkg_resources
import pytest
from conftest import shortname_and_backends

from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.pulse.rydberg import rydberg_interaction
from pennylane.pulse.hardware_hamiltonian import HardwarePulse, drive

from braket.pennylane_plugin.ahs_device import BraketAwsAhsDevice, BraketLocalAhsDevice

shortname_and_backendname = [("braket.local.ahs", "RydbergAtomSimulator"),
                             ("braket.aws.ahs", "Aquila")]

# =========================================================
coordinates = [[0, 0], [0, 5], [5, 0]]  # in micrometers


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p[0] * np.cos(p[1] * t**2)


def amp(p, t):
    return p[0] * np.exp(-(t-p[1])**2/(2*p[2]**2))


params1 = 1.2
params2 = [3.4, 5.6]
params_amp = [2.5, 0.9, 0.3]

# Hamiltonians to be tested
H_i = rydberg_interaction(coordinates)

HAMILTONIANS_AND_PARAMS = [(H_i + drive(1, 2, 3, wires=[0, 1, 2]), []),
                           (H_i + drive(amp, 1, 2, wires=[0, 1, 2]), [params_amp]),
                           (H_i + drive(2, f1, 2, wires=[0, 1, 2]), [params1]),
                           (H_i + drive(2, 2, f2, wires=[0, 1, 2]), [params2]),
                           (H_i + drive(amp, 1, f2, wires=[0, 1, 2]), [params_amp, params2]),
                           (H_i + drive(4, f2, f1, wires=[0, 1, 2]), [params2, params1]),
                           (H_i + drive(amp, f2, 4, wires=[0, 1, 2]), [params_amp, params2]),
                           (H_i + drive(amp, f2, f1, wires=[0, 1, 2]), [params_amp, params2, params1])
                           ]


class TestBraketAwsAhsDevice:
    """Test functionality specific to the hardware device"""

    def test_hardware_capabilities(self):
        """Test hardware capabilities can be retrieved"""

        dev = BraketAwsAhsDevice(wires=3)

        assert isinstance(dev.hardware_capabilities, dict)
        assert 'rydberg' in dev.hardware_capabilities.keys()
        assert 'lattice' in dev.hardware_capabilities.keys()

    def test_validate_operations_multiple_drive_terms(self):
        """Test that an error is raised if there are multiple drive terms on
        the Hamiltonian"""
        dev = BraketAwsAhsDevice(wires=3)
        pulses = [HardwarePulse(3, 4, 5, [0, 1]), HardwarePulse(4, 6, 7, [1, 2])]

        with pytest.raises(NotImplementedError, match="Multiple pulses in a Hamiltonian are not currently supported"):
            dev._validate_pulses(pulses)

    @pytest.mark.parametrize("pulse_wires, dev_wires, res", [([0, 1, 2], [0, 1, 2, 3], 'error'),
                                                             ([5, 6, 7, 8, 9], [4, 5, 6, 7, 8], 'error'),
                                                             ([0, 1, 2, 3, 6], [1, 2, 3], 'error'),
                                                             ([0, 1, 2], [0, 1, 2], 'success')])
    def test_validate_pulse_is_global_drive(self, pulse_wires, dev_wires, res):
        """Test that an error is raised if the pulse does not describe a global drive"""

        dev = BraketAwsAhsDevice(wires=dev_wires)
        pulse = HardwarePulse(3, 4, 5, pulse_wires)

        if res == 'error':
            with pytest.raises(NotImplementedError, match="Only global drive is currently supported"):
                dev._validate_pulses([pulse])
        else:
            dev._validate_pulses([pulse])


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("shortname, backend_name", shortname_and_backendname)
    def test_load_device(self, shortname, backend_name):
        """Test that the device loads correctly"""
        dev = TestDeviceIntegration._device(shortname, wires=2)
        assert dev.num_wires == 2
        assert dev.shots is 100
        assert dev.short_name == shortname
        assert dev._device.name == backend_name

    def test_args_hardware(self):
        """Test that BraketAwsDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("braket.aws.ahs")

    def test_args_local(self):
        """Test that BraketLocalDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("braket.local.ahs")

    @staticmethod
    def _device(shortname, wires, shots=100):
        return qml.device(shortname, wires=wires, shots=shots)


class TestDeviceAttributes:
    """Test application of PennyLane operations on hardware simulators."""

    @pytest.mark.parametrize("shots", [1003, 2])
    def test_setting_shots(self, shots):
        """Test that setting shots changes number of shots from default (100)"""
        dev = BraketLocalAhsDevice(wires=3, shots=shots)
        assert dev.shots == shots

        global_drive = drive(2, 1, 2, wires=[0, 1, 2])
        ts = [0.0, 1.75]

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H_i + global_drive, [], ts)
            return qml.sample()

        res = circuit()

        assert len(res) == shots


class TestQnodeIntegration:
    """Test integration with the qnode"""

    @pytest.mark.parametrize("H, params", HAMILTONIANS_AND_PARAMS)
    def test_circuit_can_be_called(self, H, params):
        """Test that the circuit consisting of a ParametrizedEvolution with a single, global pulse
        runs successfully for all combinations of amplitude, phase and detuning being constants or callables"""

        dev = qml.device('braket.local.ahs', wires=3)

        t = 1.13

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H, params, t)
            return qml.sample()

        circuit()

