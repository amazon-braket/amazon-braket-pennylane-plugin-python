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

"""
=======
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

import collections
import numbers
import warnings
from collections.abc import Iterable, Sequence

# pylint: disable=invalid-name
from enum import Enum, auto
from typing import Optional, Union

import numpy as onp
import pennylane as qml
from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask, AwsQuantumTaskBatch, AwsSession
from braket.circuits import Circuit, Instruction
from braket.circuits.noise_model import NoiseModel
from braket.device_schema import DeviceActionType
from braket.devices import Device, LocalSimulator
from braket.simulator import BraketSimulator
from braket.tasks import GateModelQuantumTaskResult, QuantumTask
from pennylane import QuantumFunctionError, QubitDevice
from pennylane import numpy as np
from pennylane.gradients import param_shift
from pennylane.measurements import (
    Expectation,
    MeasurementProcess,
    MeasurementTransform,
    Probability,
    Sample,
    ShadowExpvalMP,
    State,
    Variance,
)
from pennylane.operation import Operation
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from pennylane.tape import QuantumTape

from braket.pennylane_plugin.translation import (
    get_adjoint_gradient_result_type,
    supported_operations,
    translate_operation,
    translate_result,
    translate_result_type,
)

from ._version import __version__

RETURN_TYPES = [Expectation, Variance, Sample, Probability, State]
MIN_SIMULATOR_BILLED_MS = 3000
OBS_LIST = (qml.PauliX, qml.PauliY, qml.PauliZ)


class Shots(Enum):
    """Used to specify the default number of shots in BraketAwsQubitDevice"""

    DEFAULT = auto()


class BraketQubitDevice(QubitDevice):
    r"""Abstract Amazon Braket qubit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device (Device): The Amazon Braket device to use with PennyLane.
        shots (int or None): Number of circuit evaluations or random samples included,
            to estimate expectation values of observables. If this value is set to ``None`` or
            ``0``, the device runs in analytic mode (calculations will be exact).
        noise_model (NoiseModel or None): The Braket noise model to apply to the circuit before
            execution.
        verbatim (bool): Whether to run tasks in verbatim mode. Note that verbatim mode only
            supports the native gate set of the device. Default False.
        parametrize_differentiable (bool): Whether to bind differentiable parameters (parameters
            marked with ``required_grad=True``) on the Braket device rather than in PennyLane.
            Default: True.
        `**run_kwargs`: Variable length keyword arguments for ``braket.devices.Device.run()``.
    """

    name = "Braket PennyLane plugin"
    pennylane_requires = ">=0.30.0"
    version = __version__
    author = "Amazon Web Services"

    def __init__(
        self,
        wires: Union[int, Iterable],
        device: Device,
        *,
        shots: Union[int, None],
        noise_model: Optional[NoiseModel] = None,
        verbatim: bool = False,
        parametrize_differentiable: bool = True,
        **run_kwargs,
    ):
        if DeviceActionType.OPENQASM not in device.properties.action:
            raise ValueError(f"Device {device.name} does not support quantum circuits")

        if (
            verbatim
            and "verbatim"
            not in device.properties.action[DeviceActionType.OPENQASM].supportedPragmas
        ):
            raise ValueError(f"Device {device.name} does not support verbatim circuits")

        super().__init__(wires, shots=shots or None)
        self._device = device
        self._circuit = None
        self._task = None
        self._noise_model = noise_model
        self._parametrize_differentiable = parametrize_differentiable
        self._run_kwargs = run_kwargs
        self._supported_ops = supported_operations(self._device, verbatim=verbatim)
        self._check_supported_result_types()
        self._verbatim = verbatim

        if noise_model:
            self._validate_noise_model_support()

    def reset(self):
        super().reset()
        self._circuit = None
        self._task = None

    @property
    def operations(self) -> frozenset[str]:
        """frozenset[str]: The set of names of PennyLane operations that the device supports."""
        return self._supported_ops

    @property
    def observables(self) -> frozenset[str]:
        base_observables = frozenset(super().observables)
        # This needs to be here bc expectation(ax+by)== a*expectation(x)+b*expectation(y)
        # is only true when shots=0
        if not self.shots:
            return base_observables.union({"Hamiltonian"})
        return base_observables

    @property
    def circuit(self) -> Circuit:
        """Circuit: The last circuit run on this device."""
        return self._circuit

    @property
    def task(self) -> QuantumTask:
        """QuantumTask: The task corresponding to the last run circuit."""
        return self._task

    def _pl_to_braket_circuit(
        self,
        circuit: QuantumTape,
        compute_gradient: bool = False,
        trainable_indices: frozenset[int] = None,
        **run_kwargs,
    ):
        """Converts a PennyLane circuit to a Braket circuit"""
        braket_circuit = self.apply(
            circuit.operations,
            rotations=None,  # Diagonalizing gates are applied in Braket SDK
            use_unique_params=False,
            trainable_indices=trainable_indices,
            **run_kwargs,
        )
        if self._verbatim:
            braket_circuit = Circuit().add_verbatim_box(braket_circuit)
        if compute_gradient:
            braket_circuit = self._apply_gradient_result_type(circuit, braket_circuit)
        elif not isinstance(circuit.measurements[0], MeasurementTransform):
            for measurement in circuit.measurements:
                dev_wires = self.map_wires(measurement.wires).tolist()
                translated = translate_result_type(
                    measurement, dev_wires, self._braket_result_types
                )
                if isinstance(translated, tuple):
                    for result_type in translated:
                        braket_circuit.add_result_type(result_type)
                else:
                    braket_circuit.add_result_type(translated)

        return braket_circuit

    def _apply_gradient_result_type(self, circuit, braket_circuit):
        """Adds the AdjointGradient result type to the braket_circuit with the first observable in
        circuit.measurements. This fails for circuits with multiple observables"""
        if len(circuit.observables) != 1:
            raise ValueError(
                f"Braket can only compute gradients for circuits with a single expectation"
                f" observable, not {len(circuit.observables)} observables."
            )
        pl_measurements = circuit.measurements[0]
        pl_observable = pl_measurements.obs
        if pl_measurements.return_type != Expectation:
            raise ValueError(
                f"Braket can only compute gradients for circuits with a single expectation"
                f" observable, not a {pl_measurements.return_type} observable."
            )
        if isinstance(pl_observable, Hamiltonian):
            targets = [self.map_wires(op.wires) for op in pl_observable.ops]
        else:
            targets = self.map_wires(pl_observable.wires).tolist()

        braket_circuit.add_result_type(
            get_adjoint_gradient_result_type(
                pl_observable,
                targets,
                self._braket_result_types,
                [f"p_{param_index}" for param_index in circuit.trainable_params],
            )
        )
        return braket_circuit

    def statistics(
        self, braket_result: GateModelQuantumTaskResult, measurements: Sequence[MeasurementProcess]
    ) -> list[float]:
        """Processes measurement results from a Braket task result and returns statistics.

        Args:
            braket_result (GateModelQuantumTaskResult): the Braket task result
            measurements (Sequence[MeasurementProcess]): the list of measurements

        Raises:
            QuantumFunctionError: if the value of :attr:`~.MeasurementProcess.return_type` is
                not supported.

        Returns:
            list[float]: the corresponding statistics
        """
        results = []
        for mp in measurements:
            if mp.return_type not in RETURN_TYPES:
                raise QuantumFunctionError(
                    "Unsupported return type specified for observable {}".format(mp.obs.name)
                )
            results.append(self._get_statistic(braket_result, mp))
        return results

    def _braket_to_pl_result(self, braket_result, circuit):
        """Calculates the PennyLane results from a Braket task result. A PennyLane circuit
        also determines the output observables."""
        # Compute the required statistics
        results = self.statistics(braket_result, circuit.measurements)
        ag_results = [
            result
            for result in braket_result.result_types
            if result.type.type == "adjoint_gradient"
        ]

        if ag_results:
            # adjoint gradient results are a "ragged nested sequences (which is a list-or-tuple of
            # lists-or-tuples-or ndarrays with different lengths or shapes)", so we have to set
            # dtype="object", otherwise numpy will throw a warning

            # whenever the adjoint gradient result type is present, it should be the only result
            # type, which is why this changing of dtype works. If we ever change this plugin
            # to submit another result type alongside adjoint gradient, this logic will need to
            # change.
            results_list = [
                (
                    np.asarray(result, dtype="object")
                    if isinstance(result, collections.abc.Sequence)
                    else result
                )
                for result in results
            ]
            return results_list[0]

        # Assuming that the braket device doesn't have native parameter broadcasting
        # Assuming that the braket device doesn't support shot vectors.
        # Otherwise, we may need additional nesting
        if len(circuit.measurements) == 1:
            return onp.array(results).squeeze()
        return tuple(onp.array(result).squeeze() for result in results)

    @staticmethod
    def _tracking_data(task):
        if task.state() == "COMPLETED":
            tracking_data = {"braket_task_id": task.id}
            try:
                simulation_ms = (
                    task.result().additional_metadata.simulatorMetadata.executionDuration
                )
                tracking_data["braket_simulator_ms"] = simulation_ms
                tracking_data["braket_simulator_billed_ms"] = max(
                    simulation_ms, MIN_SIMULATOR_BILLED_MS
                )
            except AttributeError:
                pass
            return tracking_data
        else:
            return {"braket_failed_task_id": task.id}

    def classical_shadow(self, obs, circuit):
        if circuit is None:  # pragma: no cover
            raise ValueError("Circuit must be provided when measuring classical shadows")

        wires = obs.wires
        n_snapshots = self.shots
        seed = obs.seed
        n_qubits = len(wires)
        mapped_wires = np.array(self.map_wires(wires))
        # seed the random measurement generation so that recipes
        # are the same for different executions with the same seed
        rng = np.random.default_rng(seed)
        recipes = rng.integers(0, 3, size=(n_snapshots, n_qubits))

        outcomes = np.zeros((n_snapshots, n_qubits))

        snapshot_rotations = [
            [
                rot
                for wire_idx, wire in enumerate(wires)
                for rot in OBS_LIST[recipes[t][wire_idx]].compute_diagonalizing_gates(wires=wire)
            ]
            for t in range(n_snapshots)
        ]

        snapshot_circuits = [
            self.apply(
                circuit.operations,
                rotations=circuit.diagonalizing_gates + snapshot_rotation,
                use_unique_params=False,
            )
            for snapshot_rotation in snapshot_rotations
        ]

        outcomes = self._run_snapshots(snapshot_circuits, n_qubits, mapped_wires)

        return self._cast(self._stack([outcomes, recipes]), dtype=np.int8)

    def shadow_expval(self, obs, circuit):
        bits, recipes = self.classical_shadow(obs, circuit)
        shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=obs.wires.tolist())
        return shadow.expval(obs.H, obs.k)

    def execute(self, circuit: QuantumTape, compute_gradient=False, **run_kwargs) -> np.ndarray:
        self.check_validity(circuit.operations, circuit.observables)
        trainable = (
            BraketQubitDevice._get_trainable_parameters(circuit)
            if compute_gradient or self._parametrize_differentiable
            else {}
        )
        self._circuit = self._pl_to_braket_circuit(
            circuit,
            compute_gradient=compute_gradient,
            trainable_indices=frozenset(trainable.keys()),
            **run_kwargs,
        )
        if not isinstance(circuit.observables[0], MeasurementTransform):
            self._task = self._run_task(
                self._circuit, inputs={f"p_{k}": v for k, v in trainable.items()}
            )
            braket_result = self._task.result()

            if self.tracker.active:
                tracking_data = self._tracking_data(self._task)
                self.tracker.update(executions=1, shots=self.shots, **tracking_data)
                self.tracker.record()
            return self._braket_to_pl_result(braket_result, circuit)
        elif isinstance(circuit.observables[0], ShadowExpvalMP):
            if len(circuit.observables) > 1:
                raise ValueError(
                    "A circuit with a ShadowExpvalMP observable must "
                    "have that as its only result type."
                )
            return [self.shadow_expval(circuit.observables[0], circuit)]
        raise RuntimeError("The circuit has an unsupported MeasurementTransform.")

    def _execute_legacy(
        self, circuit: QuantumTape, compute_gradient=False, **run_kwargs
    ) -> np.ndarray:
        return self.execute(circuit, compute_gradient=compute_gradient, **run_kwargs)

    def apply(
        self,
        operations: Sequence[Operation],
        rotations: Sequence[Operation] = None,
        use_unique_params: bool = False,
        *,
        trainable_indices: Optional[frozenset[int]] = None,
        **run_kwargs,
    ) -> Circuit:
        """Instantiate Braket Circuit object."""
        rotations = rotations or []
        circuit = Circuit()
        trainable_indices = trainable_indices or frozenset()

        # Add operations to Braket Circuit object
        param_index = 0
        for operation in operations + rotations:
            param_names = []
            for _ in operation.parameters:
                if not isinstance(operation, qml.operation.Channel) and (
                    param_index in trainable_indices or use_unique_params
                ):
                    param_names.append(f"p_{param_index}")
                else:
                    param_names.append(None)
                param_index += 1

            dev_wires = self.map_wires(operation.wires).tolist()
            gate = translate_operation(
                operation,
                use_unique_params=bool(trainable_indices) or use_unique_params,
                param_names=param_names,
                device=self._device,
            )

            ins = Instruction(gate, dev_wires)
            circuit.add_instruction(ins)

        unused = set(range(self.num_wires)) - {int(qubit) for qubit in circuit.qubits}

        # To ensure the results have the right number of qubits
        for qubit in sorted(unused):
            circuit.i(qubit)

        if self._noise_model:
            circuit = self._noise_model.apply(circuit)

        return circuit

    def _check_supported_result_types(self):
        supported_result_types = self._device.properties.action[
            "braket.ir.openqasm.program"
        ].supportedResultTypes
        self._braket_result_types = frozenset(
            result_type.name for result_type in supported_result_types
        )

    def _validate_noise_model_support(self):
        supported_pragmas = [
            ops.lower().replace("_", "")
            for ops in (self._device.properties.action[DeviceActionType.OPENQASM].supportedPragmas)
        ]
        noise_pragmas = [
            ("braket_noise_" + noise_instr.noise.name).lower().replace("_", "")
            for noise_instr in self._noise_model._instructions
        ]
        if not all([noise in supported_pragmas for noise in noise_pragmas]):
            raise ValueError(
                f"{self._device.name} does not support noise or the noise model includes noise "
                + f"that is not supported by {self._device.name}."
            )

    def _run_task(self, circuit, inputs=None):
        raise NotImplementedError("Need to implement task runner")

    def _run_snapshots(self, snapshot_circuits, n_qubits, mapped_wires):
        raise NotImplementedError("Need to implement snapshots runner")

    def _get_statistic(self, braket_result, observable):
        dev_wires = self.map_wires(observable.wires).tolist()
        return translate_result(braket_result, observable, dev_wires, self._braket_result_types)

    @staticmethod
    def _get_trainable_parameters(tape: QuantumTape) -> dict[int, numbers.Number]:
        trainable_indices = sorted(tape.trainable_params)
        params = tape.get_parameters()
        trainable = {}
        for i in range(len(trainable_indices)):
            param = params[i]
            if isinstance(param, numbers.Number):
                trainable[trainable_indices[i]] = param
            elif isinstance(param, np.tensor):
                param_np = param.numpy()
                if isinstance(param_np, numbers.Number):
                    trainable[trainable_indices[i]] = param_np
        return trainable


class BraketAwsQubitDevice(BraketQubitDevice):
    r"""Amazon Braket AwsDevice qubit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device_arn (str): The ARN identifying the ``AwsDevice`` to be used to
            run circuits; The corresponding AwsDevice must support quantum
            circuits via OpenQASM. You can get device ARNs using ``AwsDevice.get_devices``,
            from the Amazon Braket console or from the Amazon Braket Developer Guide.
        s3_destination_folder (AwsSession.S3DestinationFolder): Name of the S3 bucket
            and folder, specified as a tuple.
        poll_timeout_seconds (float): Total time in seconds to wait for
            results before timing out.
        poll_interval_seconds (float): The polling interval for results in seconds.
        shots (int, None or Shots.DEFAULT): Number of circuit evaluations or random samples
            included, to estimate expectation values of observables. If set to Shots.DEFAULT,
            uses the default number of shots specified by the remote device. If ``shots`` is set
            to ``0`` or ``None``, the device runs in analytic mode (calculations will be exact).
            Analytic mode is not available on QPU and hence an error will be raised.
            Default: Shots.DEFAULT
        aws_session (Optional[AwsSession]): An AwsSession object created to manage
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None
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
        verbatim (bool): Whether to verbatim mode for the device. Note that verbatim mode only
            supports the native gate set of the device. Default False.
        `**run_kwargs`: Variable length keyword arguments for ``braket.devices.Device.run()``.
    """

    name = "Braket AwsDevice for PennyLane"
    short_name = "braket.aws.qubit"

    def __init__(
        self,
        wires: Union[int, Iterable],
        device_arn: str,
        s3_destination_folder: AwsSession.S3DestinationFolder = None,
        *,
        shots: Union[int, None, Shots] = Shots.DEFAULT,
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
        user_agent = f"BraketPennylanePlugin/{__version__}"
        device.aws_session.add_braket_user_agent(user_agent)
        device_type = device.type
        if device_type not in (AwsDeviceType.SIMULATOR, AwsDeviceType.QPU):
            raise ValueError(f"Invalid device type: {device_type}")

        if shots == Shots.DEFAULT and device_type == AwsDeviceType.SIMULATOR:
            num_shots = AwsDevice.DEFAULT_SHOTS_SIMULATOR
        elif shots == Shots.DEFAULT and device_type == AwsDeviceType.QPU:
            num_shots = AwsDevice.DEFAULT_SHOTS_QPU
        elif (shots is None or shots == 0) and device_type == AwsDeviceType.QPU:
            raise ValueError("QPU devices do not support 0 shots")
        else:
            num_shots = shots

        super().__init__(wires, device, shots=num_shots, **run_kwargs)
        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._parallel = parallel
        self._max_parallel = max_parallel
        self._max_connections = max_connections
        self._max_retries = max_retries

    @property
    def use_grouping(self) -> bool:
        # We *need* to do this because AdjointGradient doesn't support multiple
        # observables and grouping converts single Hamiltonian observables into
        # multiple tensor product observables, which breaks AG.
        # We *can* do this without fear because grouping is only beneficial when
        # shots!=0, and (conveniently) AG is only supported when shots=0
        caps = self.capabilities()
        return not ("provides_jacobian" in caps and caps["provides_jacobian"])

    @property
    def parallel(self):
        return self._parallel

    def batch_execute(self, circuits, **run_kwargs):
        if not self._parallel:
            return super().batch_execute(circuits)

        for circuit in circuits:
            self.check_validity(circuit.operations, circuit.observables)
        all_trainable = []
        braket_circuits = []
        for circuit in circuits:
            trainable = (
                BraketQubitDevice._get_trainable_parameters(circuit)
                if self._parametrize_differentiable
                else {}
            )
            all_trainable.append(trainable)
            braket_circuits.append(
                self._pl_to_braket_circuit(
                    circuit,
                    trainable_indices=frozenset(trainable.keys()),
                    **run_kwargs,
                )
            )

        batch_shots = 0 if self.analytic else self.shots

        task_batch = self._device.run_batch(
            braket_circuits,
            s3_destination_folder=self._s3_folder,
            shots=batch_shots,
            max_parallel=self._max_parallel,
            max_connections=self._max_connections,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
            inputs=(
                [{f"p_{k}": v for k, v in trainable.items()} for trainable in all_trainable]
                if self._parametrize_differentiable
                else []
            ),
            **self._run_kwargs,
        )
        # Call results() to retrieve the Braket results in parallel.
        try:
            braket_results_batch = task_batch.results(
                fail_unsuccessful=True, max_retries=self._max_retries
            )

        # Update the tracker before raising an exception further if some circuits do not complete.
        finally:
            if self.tracker.active:
                for task in task_batch.tasks:
                    tracking_data = self._tracking_data(task)
                    self.tracker.update(**tracking_data)
                total_executions = len(task_batch.tasks) - len(task_batch.unsuccessful)
                total_shots = total_executions * batch_shots
                self.tracker.update(batches=1, executions=total_executions, shots=total_shots)
                self.tracker.record()

        return [
            self._braket_to_pl_result(braket_result, circuit)
            for braket_result, circuit in zip(braket_results_batch, circuits)
        ]

    def _run_task(self, circuit, inputs=None):
        return self._device.run(
            circuit,
            s3_destination_folder=self._s3_folder,
            shots=0 if self.analytic else self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
            inputs=inputs or {},
            **self._run_kwargs,
        )

    def _run_snapshots(self, snapshot_circuits, n_qubits, mapped_wires):
        n_snapshots = len(snapshot_circuits)
        outcomes = np.zeros((n_snapshots, n_qubits))
        if self._parallel:
            task_batch = self._device.run_batch(
                snapshot_circuits,
                s3_destination_folder=self._s3_folder,
                shots=1,
                max_parallel=self._max_parallel,
                max_connections=self._max_connections,
                poll_timeout_seconds=self._poll_timeout_seconds,
                poll_interval_seconds=self._poll_interval_seconds,
                **self._run_kwargs,
            )

            # Call results() to retrieve the Braket results in parallel.
            try:
                braket_results_batch = task_batch.results(
                    fail_unsuccessful=True, max_retries=self._max_retries
                )

            # Update the tracker before raising an exception further
            # if some circuits do not complete.
            finally:
                if self.tracker.active:
                    for task in task_batch.tasks:
                        tracking_data = self._tracking_data(task)
                        self.tracker.update(**tracking_data)
                    total_executions = len(task_batch.tasks) - len(task_batch.unsuccessful)
                    total_shots = total_executions
                    self.tracker.update(batches=1, executions=total_executions, shots=total_shots)
                    self.tracker.record()

            for t in range(n_snapshots):
                outcomes[t] = np.array(braket_results_batch[t].measurements[0])[mapped_wires]
        else:
            for t in range(n_snapshots):
                task = self._device.run(
                    snapshot_circuits[t],
                    shots=1,
                    s3_destination_folder=self._s3_folder,
                    poll_timeout_seconds=self._poll_timeout_seconds,
                    poll_interval_seconds=self._poll_interval_seconds,
                    **self._run_kwargs,
                )
                res = task.result()
                outcomes[t] = np.array(res.measurements[0])[mapped_wires]

        return outcomes

    def _check_pulse_frequency_validity(self, ev):
        """Confirm that, for each waveform on the ParametrizedEvolution operator, the frequency
        setting is a constant, and the value is within the frequency range for the relevant frame;
        if not, raise an error"""

        # confirm all frequency values are constant (or the qml.pulse.constant function)
        callable_freqs = [
            pulse.frequency
            for pulse in ev.H.pulses
            if (callable(pulse.frequency) and pulse.frequency != qml.pulse.constant)
        ]

        if callable_freqs:
            raise RuntimeError(
                "Expected all frequencies to be constants or qml.pulse.constant, "
                "but received callable(s)"
            )

        # confirm all frequencies are within permitted difference from center frequency
        param_idx = 0
        for pulse in ev.H.pulses:
            freq = pulse.frequency
            # track the index for parameters in case we need to evaluate qml.pulse.constant
            if callable(pulse.amplitude):
                param_idx += 1
            if callable(pulse.phase):
                param_idx += 1
            if callable(pulse.frequency):
                # if frequency is callable, its qml.pulse.constant and equal to its parameter
                freq = ev.parameters[param_idx]
                param_idx += 1

            wires = self.map_wires(pulse.wires).tolist()
            freq_min = 3  # GHz
            freq_max = 8

            if not (freq_min < freq < freq_max):
                raise RuntimeError(
                    f"Frequency range for wire(s) {wires} is between {freq_min} "
                    f"and {freq_max} GHz, but received {freq} GHz."
                )

    def _validate_pulse_parameters(self, ev):
        """Validates pulse input (ParametrizedEvolution) before converting to a PulseGate"""

        # note: the pulse upload on the AWS service checks at task creation that the max amplitude
        # is not exceeded, so that check has not been included here

        # confirm frequencies are constant and within the permitted frequency range for the channel
        self._check_pulse_frequency_validity(ev)

        # confirm all phase values are constant (or the qml.pulse.constant function)
        callable_phase = [
            pulse.phase
            for pulse in ev.H.pulses
            if (callable(pulse.phase) and pulse.phase != qml.pulse.constant)
        ]

        if callable_phase:
            raise RuntimeError(
                "Expected all phases to be constants or qml.pulse.constant, "
                "but received callable(s)"
            )

        # ensure each ParametrizedEvolution/PulseGate contains at most one waveform per frame/wire
        wires_used = []
        for pulse in ev.H.pulses:
            for wire in pulse.wires:
                if wire in wires_used:
                    raise RuntimeError(
                        f"Multiple waveforms assigned to wire {wire} in the same "
                        f"ParametrizedEvolution gate"
                    )
                wires_used.append(wire)

    def check_validity(self, queue, observables):
        """Check validity of pulse operations before running the standard check_validity function

        Checks whether the operations and observables in queue are all supported by the device. Runs
        the standard check_validity function for a PennyLane device, and an additional check to
        validate any pulse-operations in the form of a ParametrizedEvolution operation.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device

        Raises:
            DeviceError: if there are operations in the queue or observables that the device does
                not support
            RuntimeError: if there are ParametrizedEvolution operations in the queue that are not
                supported because of invalid pulse parameters
        """

        super().check_validity(queue, observables)

        for op in queue:
            if isinstance(op, qml.pulse.ParametrizedEvolution):
                self._validate_pulse_parameters(op)

    def capabilities(self=None):
        """Add support for AG on sv1"""
        # normally, we'd just call super().capabilities() here, but super()
        # resolution doesn't work when you override a classmethod with an instance method
        capabilities = BraketQubitDevice.capabilities().copy()
        # if this method is called as a class method, don't add provides_jacobian since
        # we don't know if the device is sv1
        if self and "AdjointGradient" in self._braket_result_types and not self.shots:
            capabilities.update(provides_jacobian=True)
        return capabilities

    def execute_and_gradients(self, circuits, **kwargs):
        """Execute a list of circuits and calculate their gradients.
        Returns a list of circuit results and a list of gradients/jacobians, one of each
        for each circuit in circuits.

        The gradient is returned as a list of floats, 1 float for every instance
        of a trainable parameter in a gate in the circuit. Functions like qml.grad or qml.jacobian
        then use that format to generate a per-parameter format.
        """
        res = []
        jacs = []
        for circuit in circuits:
            measurements = circuit.measurements
            observables = circuit.observables
            if not circuit.trainable_params:
                new_res = self.execute(circuit, compute_gradient=False)
                # don't bother computing a gradient when there aren't any trainable parameters.
                new_jac = np.tensor([])
            elif len(observables) != 1 or measurements[0].return_type != Expectation:
                gradient_circuits, post_processing_fn = param_shift(circuit)
                warnings.warn(
                    "This circuit cannot be differentiated with the adjoint method. "
                    "Falling back to the parameter-shift method, which will execute "
                    f"{len(gradient_circuits)} circuits. "
                    "To use the adjoint gradient method, make sure the circuit's only measurement "
                    "is an expectation of one observable. "
                    "To avoid gradient calculation, mark parameters with `requires_grad=False`."
                )
                grad_circuit_results = [
                    self.execute(c, compute_gradient=False) for c in gradient_circuits
                ]
                new_jac = post_processing_fn(grad_circuit_results)
                new_res = self.execute(circuit, compute_gradient=False)
            else:
                results = self.execute(circuit, compute_gradient=True)
                new_res, new_jac = results
                # PennyLane expects the forward execution result to be a scalar
                # when it is accompanied by an adjoint gradient calculation
                new_res = new_res[0]
                new_jac = self._adjoint_jacobian_processing(new_jac)
            res.append(new_res)
            jacs.append(new_jac)
        return res, jacs

    def _is_single_qubit_01_frame(self, f_string, wire=None):
        """Defines the condition for selecting frames addressing the qubit (01)
        drive based on frame name"""
        if self._device.arn == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            if wire is not None:
                return f_string == f"q{wire}_drive"
            return "drive" in f_string
        else:
            raise NotImplementedError(
                f"Single-qubit drive frame for pulse control not defined for "
                f"device {self._device.arn}"
            )

    def _is_single_qubit_12_frame(self, f_string, wire=None):
        """Defines the condition for selecting frames addressing excitation to
        the second excited state based on frame name"""
        if self._device.arn == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            if wire is not None:
                return f_string == f"q{wire}_second_state"
            return "second_state" in f_string
        else:
            raise NotImplementedError(
                f"Second excitation drive frame for pulse control not defined for "
                f"device {self._device.arn}"
            )

    def _get_frames(self, filter, wires):
        """Takes a filter defining how the relevant frames are labelled, and returns all the frames
        that fit, i.e.:

        cond = lambda frame_id, wire: f"q{wire}_drive" == frame_id
        frames = self._get_frames(cond, wires=[0, 1, 2])

        would return all the frames with ids "q0_drive" "q1_drive", and "q2_drive", stored
        in a dictionary with keys [0, 1, 2] identifying the qubit number.
        """
        if not self._device.arn == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            raise NotImplementedError(
                f"Accessing drive frame for pulse control is not defined for "
                f"device {self._device.arn}"
            )

        frames = {}
        for wire in wires:
            for frame, info in self._device.properties.pulse.dict()["frames"].items():
                if filter(frame, wire):
                    frames[wire] = info

        return frames

    @property
    def pulse_settings(self):
        """Dictionary of constants set by the hardware (qubit resonant frequencies,
        inter-qubit connection graph, wires and anharmonicities).

        Used to enable initializing hardware-consistent Hamiltonians by returning
        values that would need to be passed, i.e.:

            >>> dev_remote = qml.device('braket.aws.qubit',
            >>>                          wires=8,
            >>>                          arn='arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy')
            >>> pulse_settings = dev_remote.pulse_settings
            >>> H_int = qml.pulse.transmon_interaction(**pulse_settings, coupling=0.02)

        By passing the ``pulse_settings`` from the remote device to ``transmon_interaction``, an
        ``H_int`` Hamiltonian term is created using the constants specific to the hardware.
        This is relevant for simulating the hardware in PennyLane on the ``default.qubit`` device.

        Note that the user must supply coupling coefficients, as these are not available from the
        hardware backend.
        """
        if not self._device.arn == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy":
            raise NotImplementedError(
                f"The pulse_settings property for pulse control is not defined for "
                f"device {self._device.arn}"
            )

        device_info = self._device.properties.paradigm
        wires = [i for i in range(device_info.qubitCount)]

        drive_frames_01 = self._get_frames(filter=self._is_single_qubit_01_frame, wires=wires)
        drive_frames_12 = self._get_frames(filter=self._is_single_qubit_12_frame, wires=wires)

        qubit_freq = [drive_frames_01[wire]["frequency"] * 1e-9 for wire in wires]  # Hz to GHz

        connections = []
        for q1, connected_qubits in device_info.connectivity.connectivityGraph.items():
            for q2 in connected_qubits:
                connection = (int(q1), int(q2))
                connections.append(connection)

        anharmonicity = [
            (drive_frames_01[wire]["frequency"] - drive_frames_12[wire]["frequency"]) * 1e-9
            for wire in wires
        ]

        return {
            "qubit_freq": qubit_freq,
            "connections": connections,
            "wires": wires,
            "anharmonicity": anharmonicity,
        }


class BraketLocalQubitDevice(BraketQubitDevice):
    r"""Amazon Braket LocalSimulator qubit device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        backend (Union[str, BraketSimulator]): The name of the simulator backend or
            the actual simulator instance to use for simulation. Defaults to the
            ``default`` simulator backend name.
        shots (int or None): Number of circuit evaluations or random samples included,
            to estimate expectation values of observables. If this value is set to ``None`` or
            ``0``, then the device runs in analytic mode (calculations will be exact).
            Default: None
        `**run_kwargs`: Variable length keyword arguments for ``braket.devices.Device.run()``.
    """

    name = "Braket LocalSimulator for PennyLane"
    short_name = "braket.local.qubit"

    def __init__(
        self,
        wires: Union[int, Iterable],
        backend: Union[str, BraketSimulator] = "default",
        *,
        shots: Union[int, None] = None,
        **run_kwargs,
    ):
        device = LocalSimulator(backend)
        super().__init__(wires, device, shots=shots, **run_kwargs)

    def _run_task(self, circuit, inputs=None):
        return self._device.run(
            circuit,
            shots=0 if self.analytic else self.shots,
            inputs=inputs or {},
            **self._run_kwargs,
        )

    def _run_snapshots(self, snapshot_circuits, n_qubits, mapped_wires):
        n_snapshots = len(snapshot_circuits)
        outcomes = np.zeros((n_snapshots, n_qubits))
        for t in range(n_snapshots):
            task = self._device.run(snapshot_circuits[t], shots=1, **self._run_kwargs)
            res = task.result()
            outcomes[t] = np.array(res.measurements[0])[mapped_wires]
        return outcomes
