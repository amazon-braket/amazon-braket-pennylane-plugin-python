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
"""
Device 1
========

**Module name:** :mod:`plugin_name.device1`

.. currentmodule:: plugin_name.device1

This Device implements all the :class:`~pennylane.device.Device` methods,
for using Target Framework device/simulator as a PennyLane device.

It can inherit from the abstract FrameworkDevice to reduce
code duplication if needed.


See https://pennylane.readthedocs.io/en/latest/API/overview.html
for an overview of Device methods available.

Classes
-------

.. autosummary::
   Device1

----
"""

# we always import NumPy directly
import numpy as np

import braket as bk

from .braket_device import BraketDevice
from braket.aws import AwsQuantumSimulator
from braket.aws.aws_quantum_simulator_arns import AwsQuantumSimulatorArns

from braket.circuits import Instruction


class AWSSimulatorDevice(BraketDevice):
    r"""AWSSimulatorDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
            For simulator devices, 0 means the exact EV is returned.
    """
    name = "Braket AWSSimulatorDevice for PennyLane"
    short_name = "braket.simulator"

    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    _circuits = {}

    def __init__(self, wires, *, shots=1000):
        super().__init__(wires, aws_arn=AwsQuantumSimulator(AwsQuantumSimulatorArns.QS1), shots=shots)

    def apply(self, operation, wires, par):
        """Add operations to Braket Circuit object."""
        op = self._operation_map[operation]
        ins = Instruction(op(*par), wires)
        self._circuit.add_instruction(ins)

    def expval(self, observable, wires, par):
        pass

    def var(self, observable, wires, par):
        pass

    def sample(self, observable, wires, par, n=None):
        pass
