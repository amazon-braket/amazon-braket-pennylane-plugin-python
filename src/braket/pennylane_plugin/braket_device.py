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
   BraketAwsQubitDevice
   BraketLocalQubitDevice

Code details
~~~~~~~~~~~~
"""

# pylint: disable=invalid-name
from typing import FrozenSet, List, Optional, Sequence, Union

from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask, AwsQuantumTaskBatch, AwsSession
from braket.circuits import Circuit, Instruction
from braket.device_schema import DeviceActionType
from braket.devices import Device, LocalSimulator
from braket.simulator import BraketSimulator
from braket.tasks import GateModelQuantumTaskResult, QuantumTask
from pennylane import CircuitGraph, QubitDevice
from pennylane import numpy as np
from pennylane.operation import Expectation, Observable, Operation, Probability, Sample, Variance
from pennylane.qnodes import QuantumFunctionError

from braket.pennylane_plugin.translation import (
    supported_operations,
    translate_operation,
    translate_result_type,
)

from ._version import __version__

RETURN_TYPES = [Expectation, Variance, Sample, Probability]


class BraketQubitDevice(QubitDevice):
    r"""Abstract Amazon Braket qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in.
        device (Device): The Amazon Braket device to use with PennyLane.
        shots (int): Number of circuit evaluations or random samples included,
            to estimate expectation values of observables. If this value is set to 0,
            the device runs in analytic mode (calculations will be exact).
            The device's ``shots`` property is set to 1 and ignored.
        **run_kwargs: Variable length keyword arguments for ``braket.devices.Device.run()`.
    """
    name = "Braket PennyLane plugin"
    pennylane_requires = ">=0.11.0"
    version = __version__
    author = "Amazon Web Services"

    def __init__(
        self,
        wires: int,
        device: Device,
        *,
        shots: int,
        **run_kwargs,
    ):
        # `shots` cannot be set to 0, but is ignored anyways
        super().__init__(wires, shots or 1, analytic=shots == 0)
        self._device = device
        self._circuit = None
        self._task = None
        self._run_kwargs = run_kwargs

    def reset(self):
        super().reset()
        self._circuit = None
        self._task = None

    @property
    def operations(self) -> FrozenSet[str]:
        """FrozenSet[str]: The set of names of PennyLane operations that the device supports."""
        return supported_operations()

    @property
    def circuit(self) -> Circuit:
        """Circuit: The last circuit run on this device."""
        return self._circuit

    @property
    def task(self) -> QuantumTask:
        """QuantumTask: The task corresponding to the last run circuit."""
        return self._task

    def _pl_to_braket_circuit(self, circuit, **run_kwargs):
        """Converts a PennyLane circuit to a Braket circuit"""
        braket_circuit = self.apply(
            circuit.operations,
            rotations=None,  # Diagonalizing gates are applied in Braket SDK
            **run_kwargs,
        )
        for observable in circuit.observables:
            braket_circuit.add_result_type(translate_result_type(observable))
        return braket_circuit

    def statistics(
        self, braket_result: GateModelQuantumTaskResult, observables: Sequence[Observable]
    ) -> Union[float, List[float]]:
        """Processes measurement results from a Braket task result and returns statistics.

        Args:
            braket_result (GateModelQuantumTaskResult): the Braket task result
            observables (List[Observable]): the observables to be measured

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            Union[float, List[float]]: the corresponding statistics
        """
        results = []
        for obs in observables:
            if obs.return_type not in RETURN_TYPES:
                raise QuantumFunctionError(
                    "Unsupported return type specified for observable {}".format(obs.name)
                )
            results.append(self._get_statistic(braket_result, obs))

        return results

    def _braket_to_pl_result(self, braket_result, circuit):
        """Calculates the PennyLane results from a Braket task result. A PennyLane circuit
        also determines the output observables."""
        # Compute the required statistics
        results = self.statistics(braket_result, circuit.observables)

        # Ensures that a combination with sample does not put
        # single-number results in superfluous arrays
        all_sampled = all(obs.return_type is Sample for obs in circuit.observables)
        if circuit.is_sampled and not all_sampled:
            return np.asarray(results, dtype="object")

        return np.asarray(results)

    def execute(self, circuit: CircuitGraph, **run_kwargs) -> np.ndarray:
        self.check_validity(circuit.operations, circuit.observables)
        self._circuit = self._pl_to_braket_circuit(circuit, **run_kwargs)
        self._task = self._run_task(self._circuit)
        return self._braket_to_pl_result(self._task.result(), circuit)

    def apply(
        self, operations: Sequence[Operation], rotations: Sequence[Operation] = None, **run_kwargs
    ) -> Circuit:
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

            params = [p.numpy() if isinstance(p, np.tensor) else p for p in operation.parameters]

            ins = Instruction(op(*params), operation.wires.tolist())
            circuit.add_instruction(ins)

        unused = set(range(self.num_wires)) - {int(qubit) for qubit in circuit.qubits}

        # To ensure the results have the right number of qubits
        for qubit in sorted(unused):
            circuit.i(qubit)

        return circuit

    def _run_task(self, circuit):
        raise NotImplementedError("Need to implement task runner")

    @staticmethod
    def _get_statistic(braket_result, observable):
        return braket_result.get_value_by_result_type(translate_result_type(observable))


class BraketAwsQubitDevice(BraketQubitDevice):
    r"""Amazon Braket AwsDevice qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in.
        device_arn (str): The ARN identifying the ``AwsDevice`` to be used to
            run circuits; The corresponding AwsDevice must support quantum
            circuits via JAQCD. You can get device ARNs using ``AwsDevice.get_devices``,
            from the Amazon Braket console or from the Amazon Braket Developer Guide.
        s3_destination_folder (AwsSession.S3DestinationFolder): Name of the S3 bucket
            and folder, specified as a tuple.
        poll_timeout_seconds (float): Total time in seconds to wait for
            results before timing out.
        poll_interval_seconds (float): The polling interval for results in seconds.
        shots (int): Number of circuit evaluations or random samples included,
            to estimate expectation values of observables. If this value is set to 0
            and the device ARN points to a simulator, the device runs
            in analytic mode (calculations will be exact). The device's
            ``shots`` property is set to 1 and ignored; trying to use 0 shots
            with QPUs will fail.
            Defaults to 1000 for QPUs and 0 for simulators.
        aws_session (Optional[AwsSession]): An AwsSession object created to manage
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
        parallel (bool): Indicates whether to use parallel execution for gradient calculations.
            Default: False
        max_parallel (int, optional): Maximum number of tasks to run on AWS in parallel.
            Batch creation will fail if this value is greater than the maximum allowed concurrent
            tasks on the device. If unspecified, uses defaults defined in ``AwsDevice``.
            Ignored if ``parallel=False``.
        max_connections (int): The maximum number of connections in the Boto3 connection pool.
            Also the maximum number of thread pool workers for the batch.
            Ignored if ``parallel=False``.
        max_retries (int): The maximum number of retries to use for batch execution.
            When executing tasks in parallel, failed tasks will be retried up to ``max_retries``
            times. Ignored if ``parallel=False``.
        **run_kwargs: Variable length keyword arguments for ``braket.devices.Device.run()``.
    """
    name = "Braket AwsDevice for PennyLane"
    short_name = "braket.aws.qubit"

    def __init__(
        self,
        wires: int,
        device_arn: str,
        s3_destination_folder: AwsSession.S3DestinationFolder,
        *,
        shots: Optional[int] = None,
        poll_timeout_seconds: float = AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds: float = AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        aws_session: Optional[AwsSession] = None,
        parallel: bool = False,
        max_parallel: Optional[int] = None,
        max_connections: int = AwsQuantumTaskBatch.MAX_CONNECTIONS_DEFAULT,
        max_retries: int = AwsQuantumTaskBatch.MAX_RETRIES,
        **run_kwargs,
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

        super().__init__(wires, device, shots=num_shots, **run_kwargs)
        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._parallel = parallel
        self._max_parallel = max_parallel
        self._max_connections = max_connections
        self._max_retries = max_retries

    @property
    def parallel(self):
        """bool: True if gradient calculations are evaluated in parallel."""
        return self._parallel

    def batch_execute(self, circuits, **run_kwargs):
        if not self._parallel:
            return super().batch_execute(circuits)

        for circuit in circuits:
            self.check_validity(circuit.operations, circuit.observables)
        braket_circuits = [
            self._pl_to_braket_circuit(circuit, **run_kwargs) for circuit in circuits
        ]

        task_batch = self._device.run_batch(
            braket_circuits,
            s3_destination_folder=self._s3_folder,
            shots=0 if self.analytic else self.shots,
            max_parallel=self._max_parallel,
            max_connections=self._max_connections,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
            **self._run_kwargs,
        )
        # Call results() to retrieve the Braket results in parallel.
        braket_results_batch = task_batch.results(
            fail_unsuccessful=True, max_retries=self._max_retries
        )
        return [
            self._braket_to_pl_result(braket_result, circuit)
            for braket_result, circuit in zip(braket_results_batch, circuits)
        ]

    def _run_task(self, circuit):
        return self._device.run(
            circuit,
            s3_destination_folder=self._s3_folder,
            shots=0 if self.analytic else self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
            **self._run_kwargs,
        )


class BraketLocalQubitDevice(BraketQubitDevice):
    r"""Amazon Braket LocalSimulator qubit device for PennyLane.

    Args:
        wires (int): the number of modes to initialize the device in.
        backend (Union[str, BraketSimulator]): The name of the simulator backend or
            the actual simulator instance to use for simulation. Defaults to the
            ``default`` simulator backend name.
        shots (int): Number of circuit evaluations or random samples included,
            to estimate expectation values of observables. If this value is set to 0,
            then the device runs in analytic mode (calculations will be exact);
            the device's ``shots`` property is set to 1 and ignored.
            Default: 0
        **run_kwargs: Variable length keyword arguments for ``braket.devices.Device.run()``.
    """
    name = "Braket LocalSimulator for PennyLane"
    short_name = "braket.local.qubit"

    def __init__(
        self,
        wires: int,
        backend: Union[str, BraketSimulator] = "default",
        *,
        shots: int = 0,
        **run_kwargs,
    ):
        device = LocalSimulator(backend)
        super().__init__(wires, device, shots=shots, **run_kwargs)

    def _run_task(self, circuit):
        return self._device.run(
            circuit, shots=0 if self.analytic else self.shots, **self._run_kwargs
        )
