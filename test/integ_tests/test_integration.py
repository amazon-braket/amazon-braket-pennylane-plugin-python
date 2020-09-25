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

"""Tests that plugin devices are accessible and integrate with PennyLane"""
import numpy as np
import pennylane as qml
import pkg_resources
import pytest
from conftest import shortnames

ENTRY_POINTS = {entry.name: entry for entry in pkg_resources.iter_entry_points("pennylane.plugins")}


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortnames)
    def test_load_device(self, d, extra_kwargs):
        """Test that the device loads correctly"""
        dev = TestDeviceIntegration._device(d, 2, extra_kwargs)
        assert dev.num_wires == 2
        assert dev.shots == 1
        assert dev.short_name == d

    def test_args_aws(self):
        """Test that BraketAwsDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 3 required positional arguments"):
            qml.device("braket.awsqubit")

    def test_args_local(self):
        """Test that BraketLocalDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("braket.localqubit")

    @pytest.mark.parametrize("d", shortnames)
    @pytest.mark.parametrize("shots", [0, 8192])
    def test_one_qubit_circuit(self, shots, d, tol, extra_kwargs):
        """Test that devices provide correct result for a simple circuit"""
        dev = TestDeviceIntegration._device(d, 1, extra_kwargs)

        a = 0.543
        b = 0.123
        c = 0.987

        @qml.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qml.BasisState(np.array([1]), wires=0)
            qml.Hadamard(wires=0)
            qml.Rot(x, y, z, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    @staticmethod
    def _device(device_name, wires, extra_kwargs):
        device_class = ENTRY_POINTS[device_name].load()
        return qml.device(device_name, wires=wires, **extra_kwargs(device_class))
