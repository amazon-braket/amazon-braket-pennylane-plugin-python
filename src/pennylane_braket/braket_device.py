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

"""
Devices
=======

**Module name:** :mod:`pennylane_braket.braket_device`

.. currentmodule:: pennylane_braket.braket_device

Braket devices to be used with PennyLane

Classes
-------

.. autosummary::
   AWSSimulatorDevice
   AWSIonQDevice
   AWSRigettiDevice

Code details
~~~~~~~~~~~~
"""
# pylint: disable=invalid-name
from typing import Optional, Set, Tuple

import numpy as np
from braket.aws import AwsQpu, AwsQpuArns, AwsQuantumSimulator, AwsQuantumSimulatorArns, AwsSession
from braket.circuits import Circuit, Instruction, gates
from braket.devices import Device
from braket.tasks import QuantumTask, GateModelQuantumTaskResult
from pennylane import QubitDevice

from ._version import __version__


class BraketDevice(QubitDevice):
    r"""Abstract Braket device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        s3_destination_folder (Tuple[str, str]): Name of the S3 bucket
            and folder as a tuple
        poll_timeout_seconds (int): Time in seconds to poll for results
            before timing out
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Default: 1000
    """
    name = "Braket PennyLane plugin"
    pennylane_requires = ">=0.8.0"
    version = __version__
    author = "Xanadu"

    _operation_map = {
        "Hadamard": gates.H,
        "PauliX": gates.X,
        "PauliY": gates.Y,
        "PauliZ": gates.Z,
        "S": gates.S,
        "S.inv": gates.Si,
        "T": gates.T,
        "T.inv": gates.Ti,
        "CZ": gates.CZ,
        "CNOT": gates.CNot,
        "SWAP": gates.Swap,
        "PhaseShift": gates.PhaseShift,
        "RX": gates.Rx,
        "RY": gates.Ry,
        "RZ": gates.Rz,
        "CSWAP": gates.CSwap,
        "PSWAP": gates.PSwap,
        "ISWAP": gates.ISwap,
        "Toffoli": gates.CCNot,
    }

    def __init__(
            self,
            wires: int,
            aws_device: Device,
            s3_destination_folder: Tuple[str, str],
            *,
            poll_timeout_seconds: int,
            shots: int = 1000,
            **kwargs):
        super().__init__(wires, shots, analytic=False)
        self._aws_device = aws_device
        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds

        self._circuit = None
        self._task = None
        self._result = None

    def reset(self):
        super().reset()
        self._circuit = None
        self._task = None
        self._result = None

    @property
    def operations(self) -> Set[str]:
        """ Set[str]: The set of PennyLane operation names the device supports.
        """
        return set(self._operation_map.keys())

    @property
    def circuit(self) -> Circuit:
        """ Circuit: The last circuit run on this device.
        """
        return self._circuit

    @property
    def task(self) -> QuantumTask:
        """ QuantumTask: The task corresponding to the last run circuit.
        """
        return self._task

    @property
    def result(self) -> GateModelQuantumTaskResult:
        """ GateModelQuantumTaskResult: The result of the last run task.
        """
        return self._result

    def apply(self, operations, rotations=None, **kwargs):
        """Instantiate Braket Circuit object."""
        rotations = rotations or []
        circuit = Circuit()

        # Add operations to Braket Circuit object
        for operation in operations + rotations:
            try:
                op = self._operation_map[operation.name]
            except KeyError:
                raise NotImplementedError(
                    f"PennyLane-Braket does not support operation {operation.name}."
                )
            ins = Instruction(op(*operation.parameters), operation.wires)
            circuit.add_instruction(ins)

        unused = set(range(self.num_wires)) - {int(qubit) for qubit in circuit.qubits}
        for qubit in sorted(unused):
            circuit.i(qubit)

        self._circuit = circuit

    def generate_samples(self):
        self._task = self._aws_device.run(
            self.circuit,
            self._s3_folder,
            shots=self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds
        )
        self._result = self._task.result()
        return self._result.measurements

    def probability(self, wires=None):
        probs = {int(s, 2): p for s, p in self._result.measurement_probabilities.items()}
        probs_list = np.array([probs[i] if i in probs else 0 for i in range(2 ** self.num_wires)])
        return self.marginal_prob(probs_list, wires=wires)

class AWSSimulatorDevice(BraketDevice):
    r"""AWSSimulatorDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        s3_destination_folder (Tuple[str, str]): Name of the S3 bucket
            and folder as a tuple
        poll_timeout_seconds (int): Time in seconds to poll for results
            before timing out. Default: 120
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Default: 1000
        backend (str): The simulator backend to target;
            can be one of "QS1", "QS2" or "QS3". Default: "QS3"
        aws_session (Optional[AwsSession]): An AwsSession object to managed
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
    """
    name = "Braket AWSSimulatorDevice for PennyLane"
    short_name = "braket.simulator"

    simulator_arns = {
        "QS1": AwsQuantumSimulatorArns.QS1,
        "QS2": AwsQuantumSimulatorArns.QS2,
        "QS3": AwsQuantumSimulatorArns.QS3,
    }

    def __init__(
            self,
            wires,
            s3_destination_folder: Tuple[str, str],
            *,
            poll_timeout_seconds: int = 120,
            shots: int = 1000,
            backend: str = "QS3",
            aws_session: Optional[AwsSession] = None,
            **kwargs):
        super().__init__(
            wires,
            aws_device=AwsQuantumSimulator(self.simulator_arns[backend], aws_session=aws_session),
            s3_destination_folder=s3_destination_folder,
            poll_timeout_seconds=poll_timeout_seconds,
            shots=shots,
            **kwargs)


class AWSIonQDevice(BraketDevice):
    r"""AWSIonQDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        s3_destination_folder (Tuple[str, str]): Name of the S3 bucket
            and folder as a tuple
        poll_timeout_seconds (int): Time in seconds to poll for results
            before timing out. Default: 86400
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Default: 1000
        aws_session (Optional[AwsSession]): An AwsSession object to managed
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
    """
    name = "Braket AWSIonQDevice for PennyLane"
    short_name = "braket.ionq"

    def __init__(
            self,
            wires,
            s3_destination_folder: Tuple[str, str],
            *,
            poll_timeout_seconds: int = 86400,
            shots: int = 1000,
            aws_session: Optional[AwsSession] = None,
            **kwargs):
        super().__init__(
            wires,
            aws_device=AwsQpu(AwsQpuArns.IONQ, aws_session=aws_session),
            s3_destination_folder=s3_destination_folder,
            poll_timeout_seconds=poll_timeout_seconds,
            shots=shots,
            **kwargs)


class AWSRigettiDevice(BraketDevice):
    r"""AWSRigettiDevice for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in
        s3_destination_folder (Tuple[str, str]): Name of the S3 bucket
            and folder as a tuple
        poll_timeout_seconds (int): Time in seconds to poll for results
            before timing out. Default: 86400
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Default: 1000
        aws_session (Optional[AwsSession]): An AwsSession object to managed
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
    """
    name = "Braket AWSRigettiDevice for PennyLane"
    short_name = "braket.rigetti"

    def __init__(
            self,
            wires,
            s3_destination_folder: Tuple[str, str],
            *,
            poll_timeout_seconds: int = 86400,
            shots: int = 1000,
            aws_session: Optional[AwsSession] = None,
            **kwargs):
        super().__init__(
            wires,
            aws_device=AwsQpu(AwsQpuArns.RIGETTI, aws_session=aws_session),
            s3_destination_folder=s3_destination_folder,
            poll_timeout_seconds=poll_timeout_seconds,
            shots=shots,
            **kwargs)
