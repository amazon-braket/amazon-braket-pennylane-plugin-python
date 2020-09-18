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

Braket device to be used with PennyLane

Classes
-------

.. autosummary::
   BraketDevice

Code details
~~~~~~~~~~~~
"""
# pylint: disable=invalid-name
from typing import FrozenSet, Optional

import numpy as np
from pennylane import CircuitGraph, Identity, QubitDevice
from pennylane.operation import Probability, Sample

from braket.aws import AwsDevice, AwsDeviceType, AwsSession
from braket.circuits import Circuit, Instruction
from braket.device_schema import DeviceActionType
from braket.pennylane_plugin.translation import (
    supported_operations,
    translate_operation,
    translate_result_type,
)
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
            to estimate expectation values of observables. If this is set to 0
            and the device ARN points to a simulator, then the device will run
            in analytic mode (calculations will be exact), and the device's
            ``shots`` property will be set to 1 and ignored; trying use 0 shots
            with QPUs will fail.
            Defaults to 1000 for QPUs and 1 with analytic mode for simulators.
        aws_session (Optional[AwsSession]): An AwsSession object to managed
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
    """
    name = "Braket PennyLane plugin"
    short_name = "braket.device"
    pennylane_requires = ">=0.11.0"
    version = __version__
    author = "Amazon Web Services"

    def __init__(
        self,
        wires: int,
        device_arn: str,
        s3_destination_folder: AwsSession.S3DestinationFolder,
        *,
        shots: Optional[int] = None,
        poll_timeout_seconds: int = AwsDevice.DEFAULT_RESULTS_POLL_TIMEOUT,
        aws_session: Optional[AwsSession] = None,
        **kwargs,
    ):
        device = AwsDevice(device_arn, aws_session=aws_session)
        if DeviceActionType.JAQCD not in device.properties.action:
            raise ValueError(f"Device {device.name} does not support quantum circuits")

        device_type = device.type
        if shots is not None:
            if shots == 0 and device_type == AwsDeviceType.QPU:
                raise ValueError("QPU devices do not support 0 shots")
            num_shots = shots
        elif device_type == AwsDeviceType.SIMULATOR:
            num_shots = AwsDevice.DEFAULT_SHOTS_SIMULATOR
        elif device_type == AwsDeviceType.QPU:
            num_shots = AwsDevice.DEFAULT_SHOTS_QPU
        else:
            raise ValueError(f"Invalid device type: {device_type}")

        # `shots` cannot be set to 0, but is ignored anyways
        super().__init__(wires, num_shots or 1, analytic=num_shots == 0)
        self._aws_device = device
        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds

        self._circuit = None
        self._task = None

    def reset(self):
        super().reset()
        self._circuit = None
        self._task = None

    @property
    def operations(self) -> FrozenSet[str]:
        """FrozenSet[str]: The set of PennyLane operation names the device supports."""
        return supported_operations()

    @property
    def circuit(self) -> Circuit:
        """Circuit: The last circuit run on this device."""
        return self._circuit

    @property
    def task(self) -> QuantumTask:
        """QuantumTask: The task corresponding to the last run circuit."""
        return self._task

    def execute(self, circuit: CircuitGraph, **kwargs):
        self.check_validity(circuit.operations, circuit.observables)

        # Apply all circuit operations
        self.apply(
            circuit.operations,
            rotations=None,  # Diagonalizing gates are applied in Braket SDK
            **kwargs,
        )

        for observable in circuit.observables:
            self._circuit.add_result_type(translate_result_type(observable))

        self._task = self._aws_device.run(
            self._circuit,
            s3_destination_folder=self._s3_folder,
            shots=0 if self.analytic else self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
        )

        # Compute the required statistics
        results = self.statistics(circuit.observables)

        # Ensures that a combination with sample does not put
        # single-number results in superfluous arrays
        all_sampled = all(obs.return_type is Sample for obs in circuit.observables)
        if circuit.is_sampled and not all_sampled:
            return np.asarray(results, dtype="object")

        return np.asarray(results)

    def apply(self, operations, rotations=None, **kwargs):
        """Instantiate Braket Circuit object."""
        rotations = rotations or []
        circuit = Circuit()

        # Add operations to Braket Circuit object
        for operation in operations + rotations:
            try:
                op = translate_operation(operation)
            except KeyError:
                raise NotImplementedError(
                    f"Braket PennyLane plugin does not support operation {operation.name}."
                )
            ins = Instruction(op(*operation.parameters), operation.wires.tolist())
            circuit.add_instruction(ins)

        unused = set(range(self.num_wires)) - {int(qubit) for qubit in circuit.qubits}

        # To ensure the results have the right number of qubits
        for qubit in sorted(unused):
            circuit.i(qubit)

        self._circuit = circuit

    def expval(self, observable):
        return BraketDevice._get_statistic(self._task, observable)

    def var(self, observable):
        return BraketDevice._get_statistic(self._task, observable)

    def sample(self, observable):
        return BraketDevice._get_statistic(self._task, observable)

    def probability(self, wires=None):
        return self._probability(wires)

    def analytic_probability(self, wires=None):
        return self._probability(wires)

    def _probability(self, wires):
        observable = Identity(wires=wires or self.wires, do_queue=False)
        observable.return_type = Probability
        return BraketDevice._get_statistic(self._task, observable)

    @staticmethod
    def _get_statistic(task, observable):
        return task.result().get_value_by_result_type(translate_result_type(observable))
