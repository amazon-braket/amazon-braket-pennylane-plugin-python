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

"""Tests that plugin devices are accessible and integrate with PennyLane"""

from importlib.metadata import entry_points

import numpy as np
import pennylane as qp
import pytest
from conftest import shortname_and_backends

ENTRY_POINTS = {entry.name: entry for entry in entry_points(group="pennylane.plugins")}


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortname_and_backends)
    def test_load_device(self, d, extra_kwargs):
        """Test that the device loads correctly"""
        dev = TestDeviceIntegration._device(d, 2, extra_kwargs)
        # PennyLane 0.38+ wraps the device in a `LegacyDeviceFacade`
        # TODO: Remove else branch once minimum PennyLane is >=0.38
        dev = dev.target_device if hasattr(dev, "target_device") else dev
        assert dev.num_wires == 2
        assert dev.shots is None
        assert dev.short_name == d[0]

    def test_args_aws(self):
        """Test that BraketAwsDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            qp.device("braket.aws.qubit")

    def test_args_local(self):
        """Test that BraketLocalDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qp.device("braket.local.qubit")

    @pytest.mark.parametrize("d", shortname_and_backends)
    @pytest.mark.parametrize("shots", [None, 8192])
    def test_one_qubit_circuit(self, shots, d, tol, extra_kwargs):
        """Test that devices provide correct result for a simple circuit"""
        dev = TestDeviceIntegration._device(d, 1, extra_kwargs)

        a = 0.543
        b = 0.123
        c = 0.987

        @qp.qnode(dev)
        def circuit(x, y, z):
            """Reference QNode"""
            qp.BasisState(np.array([1]), wires=0)
            qp.Hadamard(wires=0)
            qp.Rot(x, y, z, wires=0)
            return qp.expval(qp.PauliZ(0))

        assert np.allclose(circuit(a, b, c), np.cos(a) * np.sin(b), **tol)

    @staticmethod
    def _device(shortname_and_backend, wires, extra_kwargs):
        device_name, backend = shortname_and_backend
        device_class = ENTRY_POINTS[device_name].load()
        return qp.device(device_name, wires=wires, **extra_kwargs(device_class, backend))
