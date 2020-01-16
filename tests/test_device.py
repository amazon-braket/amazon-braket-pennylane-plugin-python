# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for any plugin- or framework-specific behaviour of the plugin devices"""
import pytest

import numpy as np

from plugin_name.qiskit_device import z_eigs
from plugin_name import Device1


Z = np.diag([1, -1])


class TestZEigs:
    r"""Test that eigenvalues of Z^{\otimes n} are correctly generated"""

    def test_one(self):
        """Test that eigs(Z) = [1, -1]"""
        assert np.all(z_eigs(1) == np.array([1, -1]))

    @pytest.mark.parametrize("n", [2, 3, 6])
    def test_multiple(self, n):
        r"""Test that eigs(Z^{\otimes n}) is correct"""
        res = z_eigs(n)
        Zn = np.kron(Z, Z)

        for _ in range(n - 2):
            Zn = np.kron(Zn, Z)

        expected = np.diag(Zn)
        assert np.all(res == expected)


class TestProbabilities:
    """Tests for the probability function"""

    def test_probability_no_results(self):
        """Test that the probabilities function returns
        None if no job has yet been run."""
        dev = Device1(backend="statevector_simulator", wires=1, shots=0)
        assert dev.probabilities() is None
