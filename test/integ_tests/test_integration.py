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
import pytest
from conftest import shortnames


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("d", shortnames)
    def test_load_device(self, d, s3):
        """Test that the device loads correctly"""
        dev = qml.device(d, wires=2, shots=1024, s3_destination_folder=s3)
        assert dev.num_wires == 2
        assert dev.shots == 1024
        assert dev.short_name == d

    def test_args(self):
        """Test that the device requires correct arguments"""
        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            qml.device("braket.simulator")

    @pytest.mark.parametrize("d", shortnames)
    @pytest.mark.parametrize("shots", [8192])
    def test_one_qubit_circuit(self, shots, d, tol, s3):
        """Test that devices provide correct result for a simple circuit"""
        dev = qml.device(d, wires=1, shots=shots, s3_destination_folder=s3)

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
