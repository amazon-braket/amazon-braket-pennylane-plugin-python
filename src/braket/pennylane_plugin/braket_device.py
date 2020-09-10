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

**Module name:** :mod:`braket.pennylane_braket.braket_device`

.. currentmodule:: braket.pennylane_braket.braket_device

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
from typing import Optional, Set

import numpy as np
from pennylane import QubitDevice

from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit, Instruction, gates
from braket.device_schema import DeviceActionType
from braket.tasks import QuantumTask

from ._version import __version__


class BraketDevice(QubitDevice):
    r"""AWS Braket device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in.
        device_arn (str): The ARN identifying the ``AwsDevice`` to be used to
            run circuits; The corresponding AwsDevice must support quantum
            circuits via JAQCD. You can get device ARNs using ``AwsDevice.get_devices``,
            from the Amazon Braket console or from the Amazon Braket Developer Guide.
        s3_destination_folder (AwsSession.S3DestinationFolder): Name of the S3 bucket
            and folder as a tuple.
        poll_timeout_seconds (int): Total time in seconds to wait for
            results before timing out.
        shots (int): Number of circuit evaluations/random samples used
            to estimate expectation values of observables. Default: 1000
        aws_session (Optional[AwsSession]): An AwsSession object to managed
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
    """
    name = "Braket PennyLane plugin"
    short_name = "braket.device"
    pennylane_requires = ">=0.11.0"
    version = __version__
    author = "Amazon Web Services"

    _operation_map = {
        "Identity": gates.I,
        "Hadamard": gates.H,
        "PauliX": gates.X,
        "PauliY": gates.Y,
        "PauliZ": gates.Z,
        "CNOT": gates.CNot,
        "CY": gates.CY,
        "CZ": gates.CZ,
        "S": gates.S,
        "S.inv": gates.Si,
        "T": gates.T,
        "T.inv": gates.Ti,
        "V": gates.V,
        "V.inv": gates.Vi,
        "PhaseShift": gates.PhaseShift,
        "CPhaseShift": gates.CPhaseShift,
        "CPhaseShift00": gates.CPhaseShift00,
        "CPhaseShift01": gates.CPhaseShift01,
        "CPhaseShift10": gates.CPhaseShift10,
        "RX": gates.Rx,
        "RY": gates.Ry,
        "RZ": gates.Rz,
        "SWAP": gates.Swap,
        "CSWAP": gates.CSwap,
        "ISWAP": gates.ISwap,
        "PSWAP": gates.PSwap,
        "XY": gates.XY,
        "XX": gates.XX,
        "YY": gates.YY,
        "ZZ": gates.ZZ,
        "Toffoli": gates.CCNot,
        "QubitUnitary": gates.Unitary,
    }

    def __init__(
        self,
        wires: int,
        device_arn: str,
        s3_destination_folder: AwsSession.S3DestinationFolder,
        *,
        shots: int = 1000,
        poll_timeout_seconds: int = AwsDevice.DEFAULT_RESULTS_POLL_TIMEOUT,
        aws_session: Optional[AwsSession] = None,
        **kwargs,
    ):
        super().__init__(wires, shots, analytic=False)
        self._aws_device = AwsDevice(device_arn, aws_session=aws_session)
        if DeviceActionType.JAQCD not in self._aws_device.properties.action:
            raise ValueError(
                f"Device {self._aws_device.name} does not support running quantum circuits"
            )

        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds

        self._circuit = None
        self._task = None

    def reset(self):
        super().reset()
        self._circuit = None
        self._task = None

    @property
    def operations(self) -> Set[str]:
        """Set[str]: The set of PennyLane operation names the device supports."""
        return set(self._operation_map.keys())

    @property
    def circuit(self) -> Circuit:
        """Circuit: The last circuit run on this device."""
        return self._circuit

    @property
    def task(self) -> QuantumTask:
        """QuantumTask: The task corresponding to the last run circuit."""
        return self._task

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
                    f"Braket PennyLane plugin does not support operation {operation.name}."
                )
            ins = Instruction(op(*operation.parameters), operation.wires.toarray())
            circuit.add_instruction(ins)

        unused = set(range(self.num_wires)) - {int(qubit) for qubit in circuit.qubits}

        # To ensure the results have the right number of qubits
        for qubit in sorted(unused):
            circuit.i(qubit)

        self._circuit = circuit

    def generate_samples(self):
        self._task = self._aws_device.run(
            self.circuit,
            s3_destination_folder=self._s3_folder,
            shots=self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
        )
        return self._task.result().measurements

    def probability(self, wires=None):
        probs = {int(s, 2): p for s, p in self._task.result().measurement_probabilities.items()}
        probs_list = np.array([probs[i] if i in probs else 0 for i in range(2 ** self.num_wires)])
        return self.marginal_prob(probs_list, wires=wires)
