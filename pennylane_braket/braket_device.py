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
AWS PennyLane plugin devices
"""
# pylint: disable=invalid-name
import numpy as np

from braket.circuits import Circuit, gates
from braket.circuits import Instruction

from braket.aws import AwsQuantumSimulator, AwsQpu
from braket.aws.aws_qpu_arns import AwsQpuArns
from braket.aws.aws_quantum_simulator_arns import AwsQuantumSimulatorArns

from pennylane import QubitDevice

from ._version import __version__


class BraketDevice(QubitDevice):
    r"""Abstract Braket device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.

    Keyword Args:
        s3 (tuple[str, str[]): Name of the S3 bucket and folder as a tuple
    """
    name = "Braket PennyLane plugin"
    pennylane_requires = ">=0.8.0"
    version = __version__
    author = "Xanadu"

    short_name = "braket"
    _operation_map = {
        "Hadamard": gates.H,
        "PauliX": gates.X,
        "PauliY": gates.Y,
        "PauliZ": gates.Z,
        "S": gates.S,
        "S.inv": gates.Si,
        "T": gates.T,
        "T.inv": gates.Ti,
        "CNOT": gates.CNot,
        "CZ": gates.CZ,
        "RX": gates.Rx,
        "RY": gates.Ry,
        "RZ": gates.Rz,
        "PhaseShift": gates.PhaseShift,
        "SWAP": gates.Swap,
        "CSWAP": gates.CSwap,
        "PSWAP": gates.PSwap,
        "ISWAP": gates.ISwap,
        "Toffoli": gates.CCNot,
    }

    def __init__(self, wires, aws_device, *, shots=1000, **kwargs):
        super().__init__(wires, shots, analytic=False)
        self._aws_device = aws_device
        self._s3_folder = kwargs.get("s3", None)

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
        ret = self._aws_device.run(
            task_specification=self.circuit, s3_destination_folder=self._s3_folder, shots=self.shots
        )
        self.result = ret.result()
        return self.result.measurements

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

    Keyword Args:
        s3 (tuple[str, str[]): Name of the S3 bucket and folder as a tuple
    """
    name = "Braket AWSSimulatorDevice for PennyLane"
    short_name = "braket.simulator"

    backends = {
        "QS1": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS1),
        "QS2": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS2),
        "QS3": AwsQuantumSimulator(AwsQuantumSimulatorArns.QS3),
    }

    def __init__(self, wires, *, backend="QS1", shots=1000, **kwargs):
        super().__init__(wires, aws_device=self.backends[backend], shots=shots, **kwargs)


class AWSIonQDevice(BraketDevice):
    r"""AWSIonQDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.

    Keyword Args:
        s3 (tuple[str, str[]): Name of the S3 bucket and folder as a tuple
    """
    name = "Braket AWSIonQDevice for PennyLane"
    short_name = "braket.ionq"

    def __init__(self, wires, *, shots=1000, **kwargs):
        super().__init__(wires, aws_device=AwsQpu(AwsQpuArns.IONQ), shots=shots, **kwargs)


class AWSRigettiDevice(BraketDevice):
    r"""AWSRigettiDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables.

    Keyword Args:
        s3 (tuple[str, str[]): Name of the S3 bucket and folder as a tuple
    """
    name = "Braket AWSRigettiDevice for PennyLane"
    short_name = "braket.rigetti"

    def __init__(self, wires, *, shots=1000, **kwargs):
        super().__init__(wires, aws_device=AwsQpu(AwsQpuArns.RIGETTI), shots=shots, **kwargs)
