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

"""Tests that counts are correctly computed in the plugin device"""

import numpy as np
import pennylane as qml
import pytest

np.random.seed(42)


@pytest.mark.parametrize("shots", [8192])
class TestCounts:
    """Tests for the count return type"""

    def test_counts_values(self, device, shots, tol):
        """Tests if the result returned by counts have
        the correct values
        """
        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, wires=0)
            return qml.counts()

        result = circuit().item()

        # The sample should only contain 00 and 10
        assert "00" in result
        assert "10" in result
        assert "11" not in result
        assert "01" not in result
        assert result["00"] + result["10"] == shots
        assert np.allclose(result["00"] / shots, 0.75, **tol)
        assert np.allclose(result["10"] / shots, 0.25, **tol)

    def test_counts_values_specify_target(self, device, shots, tol):
        """Tests if the result returned by counts have
        the correct values when specifying a target
        """
        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, wires=0)
            return qml.counts(wires=[0])

        result = circuit().item()

        # The sample should only contain 00 and 10
        assert "0" in result
        assert "1" in result
        assert result["0"] + result["1"] == shots
        assert np.allclose(result["0"] / shots, 0.75, **tol)
        assert np.allclose(result["1"] / shots, 0.25, **tol)

    def test_counts_values_with_observable(self, device, shots, tol):
        """Tests if the result returned by counts have
        the correct values when specifying an observable
        """
        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(np.pi / 3, wires=0)
            return qml.counts(op=qml.PauliZ(wires=[0]))

        result = circuit().item()

        # The sample should only contain 00 and 10
        assert -1 in result
        assert 1 in result
        assert result[-1] + result[1] == shots
        assert np.allclose(result[1] / shots, 0.75, **tol)
        assert np.allclose(result[-1] / shots, 0.25, **tol)
