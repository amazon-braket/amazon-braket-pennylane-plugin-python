# Copyright 2019 Xanadu Quantum Technologies Inc.

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
AWS PennyLane plugin
"""
import numpy as np

from braket.circuits import Circuit, gates
from braket.circuits import Instruction

from braket.aws import AwsQuantumSimulator
from braket.aws.aws_quantum_simulator_arns import AwsQuantumSimulatorArns

from braket.aws import AwsQpu
from braket.aws.aws_qpu_arns import AwsQpuArns

from pennylane import QubitDevice

from ._version import __version__


class BraketDevice(QubitDevice):
    r"""Abstract Braket device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """
    name = "Braket PennyLane plugin"
    pennylane_requires = ">=0.7.0"
    version = __version__
    author = "Xanadu"

    short_name = "braket"
    _operation_map = {
        "Hadamard": gates.H,
        "CNOT": gates.CNot,
        "RX": gates.Rx,
    }
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

    def __init__(self, wires, aws_device, *, shots=1000):  # Note: `shots` currently has no effect
        super().__init__(wires, shots)
        self._capabilities.update({"model": "qubit"})
        self._aws_device = aws_device
        self._s3_folder = (
            "braket-output-355401967590",
            "test1",
        )  # likely should be argument, or pulled from config

        self.circuit = None
        self.result = None

    def reset(self):
        super().reset()
        self.circuit = None
        self.result = None

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operations, rotations=None, **kwargs):
        """Instantiate Braket Circuit object."""
        rotations = rotations or []
        self.circuit = Circuit()

        # Add operations to Braket Circuit object
        for operation in operations + rotations:
            op = self._operation_map[operation.name]
            ins = Instruction(op(*operation.parameters), operation.wires)
            self.circuit.add_instruction(ins)

    def generate_samples(self):
        ret = self._aws_device.run(self.circuit, self._s3_folder)
        self.result = ret.result()
        self._samples = self.result.measurements

    def probability(self, wires=None):
        probs = self.result.measurement_probabilities
        probs = {tuple(int(i) for i in s): p for s, p in probs.items()}
        probs = np.array([p[1] for p in sorted(probs.items())])
        return self.marginal_prob(probs, wires=wires)


class AWSSimulatorDevice(BraketDevice):
    r"""AWSSimulatorDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
        backend (str): the simulator backend to target.
    """
    name = "Braket AWSSimulatorDevice for PennyLane"
    short_name = "braket.simulator"

    backends = {
        "QS1": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS1),
        "QS2": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS2),
        "QS3": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS3),
    }

    def __init__(self, wires, *, backend="QS1", shots=1000):
        super().__init__(wires, aws_device=self.backends[backend], shots=shots)


class AWSIonQDevice(BraketDevice):
    r"""AWSIonQDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """
    name = "Braket AWSIonQDevice for PennyLane"
    short_name = "braket.ionq"

    def __init__(self, wires, *, shots=1000):
        super().__init__(wires, aws_device=AwsQpu(AwsQpuArns.IONQ), shots=shots)


class AWSRigettiDevice(BraketDevice):
    r"""AWSRigettiDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.
    """
    name = "Braket AWSRigettiDevice for PennyLane"
    short_name = "braket.rigetti"

    def __init__(self, wires, *, shots=1000):
        super().__init__(wires, aws_device=AwsQpu(AwsQpuArns.RIGETTI), shots=shots)
