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
import json
from unittest import mock
from unittest.mock import Mock, PropertyMock, patch
from conftest import shortname_and_backends

from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.pulse.rydberg import rydberg_interaction, rydberg_drive
from pennylane.pulse.hardware_hamiltonian import HardwarePulse

from braket.pennylane_plugin.ahs_device import BraketAwsAhsDevice, BraketLocalAhsDevice
from braket.aws import AwsDevice
from braket.device_schema import DeviceActionType, DeviceActionProperties
from braket.device_schema.quera.quera_ahs_paradigm_properties_v1 import QueraAhsParadigmProperties

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

HAMILTONIANS_AND_PARAMS = [(H_i + rydberg_drive(1, 2, 3, wires=[0, 1, 2]), []),
                           (H_i + rydberg_drive(amp, 1, 2, wires=[0, 1, 2]), [params_amp]),
                           (H_i + rydberg_drive(2, f1, 2, wires=[0, 1, 2]), [params1]),
                           (H_i + rydberg_drive(2, 2, f2, wires=[0, 1, 2]), [params2]),
                           (H_i + rydberg_drive(amp, 1, f2, wires=[0, 1, 2]), [params_amp, params2]),
                           (H_i + rydberg_drive(4, f2, f1, wires=[0, 1, 2]), [params2, params1]),
                           (H_i + rydberg_drive(amp, f2, 4, wires=[0, 1, 2]), [params_amp, params2]),
                           (H_i + rydberg_drive(amp, f2, f1, wires=[0, 1, 2]), [params_amp, params2, params1])
                           ]

HARDWARE_ARN_NRS = ["arn:aws:braket:us-east-1::device/qpu/quera/Aquila"]

ALL_DEVICES = [("braket.local.ahs", None, "RydbergAtomSimulator"),
               ("braket.aws.ahs", "arn:aws:braket:us-east-1::device/qpu/quera/Aquila", "Aquila")]


DEVS = [("arn:aws:braket:us-east-1::device/qpu/quera/Aquila", "Aquila")]


@pytest.mark.parametrize("arn_nr, name", DEVS)
def test_initialization(arn_nr, name):
    """Test the device initializes with the expected attributes"""

    dev = BraketAwsAhsDevice(wires=3, shots=11, device_arn=arn_nr)

    assert dev._device.name == name
    assert dev.short_name == "braket.aws.ahs"
    assert dev.shots == 11
    assert dev.ahs_program is None
    assert dev.samples is None
    assert dev.pennylane_requires == ">=0.30.0"
    assert dev.operations == {"ParametrizedEvolution"}


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("shortname, arn_nr, backend_name", ALL_DEVICES)
    def test_load_device(self, shortname, arn_nr, backend_name):
        """Test that the device loads correctly"""
        dev = TestDeviceIntegration._device(shortname, arn_nr, wires=2)
        assert dev.num_wires == 2
        assert dev.shots is 100
        assert dev.short_name == shortname
        assert dev._device.name == backend_name

    def test_args_hardware(self):
        """Test that BraketAwsDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 2 required positional argument"):
            qml.device("braket.aws.ahs")

    def test_args_local(self):
        """Test that BraketLocalDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("braket.local.ahs")

    @staticmethod
    def _device(shortname, arn_nr, wires, shots=100):
        if arn_nr:
            return qml.device(shortname, wires=wires, device_arn=arn_nr, shots=shots)
        return qml.device(shortname, wires=wires, shots=shots)


class TestDeviceAttributes:
    """Test application of PennyLane operations on hardware simulators."""

    @pytest.mark.parametrize("shots", [1003, 2])
    def test_setting_shots(self, shots):
        """Test that setting shots changes number of shots from default (100)"""
        dev = BraketLocalAhsDevice(wires=3, shots=shots)
        assert dev.shots == shots

        global_drive = rydberg_drive(2, 1, 2, wires=[0, 1, 2])
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
