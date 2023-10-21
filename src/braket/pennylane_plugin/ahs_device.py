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

**Module name:** :mod:`braket.pennylane_braket.ahs_device`

.. currentmodule:: braket.pennylane_braket.ahs_device

Braket analog Hamiltonian simulation (AHS) devices to be used with PennyLane

Classes
-------

.. autosummary::
   BraketAwsAhsDevice
   BraketLocalAhsDevice

Code details
~~~~~~~~~~~~
"""
from collections.abc import Iterable
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.aws import AwsDevice, AwsQuantumTask, AwsSession
from braket.devices import Device, LocalSimulator
from pennylane import QubitDevice
from pennylane._version import __version__
from pennylane.measurements import MeasurementProcess, SampleMeasurement
from pennylane.ops import CompositeOp, Hamiltonian
from pennylane.pulse import ParametrizedEvolution
from pennylane.pulse.hardware_hamiltonian import HardwareHamiltonian, HardwarePulse

from .ahs_translation import (
    _create_register,
    _create_valid_local_detunings,
    _evaluate_pulses,
    _get_sample_times,
    translate_ahs_shot_result,
    translate_pulse_to_driving_field,
    translate_pulses_to_shifting_field,
)


class Shots(Enum):
    """Used to specify the default number of shots in BraketAwsQubitDevice"""

    DEFAULT = auto()


class BraketAhsDevice(QubitDevice):
    """Abstract Amazon Braket device for analog Hamiltonian simulation with PennyLane.

    Args:
        wires (int or Iterable[int, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device (Device): The Amazon Braket device to use with PennyLane.
        shots (int or Shots.DEFAULT): Number of executions to run to aquire measurements.
            Default: Shots.DEFAULT
    """

    name = "Braket AHS PennyLane plugin"
    pennylane_requires = ">=0.30.0"
    version = __version__
    author = "Xanadu Inc."
    short_name = "braket_ahs_device"

    operations = {"ParametrizedEvolution"}

    def __init__(
        self,
        wires: Union[int, Iterable],
        device: Device,
        *,
        shots: Union[int, Shots] = Shots.DEFAULT,
    ):
        if not shots:
            raise RuntimeError(f"This device requires shots. Received shots={shots}")
        # Simulator default of 0 not suitable for AHS simulator, use DEFAULT_SHOTS_QPU for both
        elif shots == Shots.DEFAULT:
            num_shots = AwsDevice.DEFAULT_SHOTS_QPU
        else:
            num_shots = shots

        super().__init__(wires=wires, shots=num_shots)

        self._device = device
        self._register = None
        self._pulses = None
        self._ahs_program = None
        self._task = None

    def apply(self, operations: list[ParametrizedEvolution], **kwargs):
        """Convert the pulse operation to an AHS program and run on the connected device

        Args:
            operations(list[ParametrizedEvolution]): a list containing a single
                ParametrizedEvolution operator
        """
        ev_op = operations[0]  # only one!

        ahs_program = self.create_ahs_program(ev_op)
        self._task = self._run_task(ahs_program)

    def expval(self, observable, shot_range=None, bin_size=None):
        # estimate the ev
        samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)

        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2

        # use nanmean to ignore failed measurements in taking the average
        return np.nanmean(samples, axis=axis)

    @property
    def task(self):
        return self._task

    @property
    def ahs_program(self):
        return self._ahs_program

    @property
    def register(self):
        return self._register

    @property
    def result(self):
        if self._task:
            return self._task.result()
        return None

    def _run_task(self, ahs_program: AnalogHamiltonianSimulation):
        """Run and return a task executing the AnalogHamiltonianSimulation program on
        the device"""
        raise NotImplementedError("Running a task not implemented for the base class")

    def _ahs_program_from_evolution(self, evolution: ParametrizedEvolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an AnalogHamiltonianSimulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        # sets self._pulses to be the evaluated pulses (now only a function of time)
        self._pulses = _evaluate_pulses(evolution)
        self._register = _create_register(evolution.H.settings.register)

        time_interval = evolution.t
        time_points = _get_sample_times(time_interval)

        # no gurarentee that global drive is index 0 once we start allowing more just global drive
        drive = translate_pulse_to_driving_field(self._pulses[0], time_points)

        return AnalogHamiltonianSimulation(register=self._register, hamiltonian=drive)

    def create_ahs_program(self, evolution: ParametrizedEvolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an AnalogHamiltonianSimulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        ahs_program = self._ahs_program_from_evolution(evolution)

        self._ahs_program = ahs_program

        return ahs_program

    def generate_samples(self):
        r"""Returns the computational basis samples measured for all wires.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        return np.array([translate_ahs_shot_result(res) for res in self.result.measurements])

    def check_validity(self, queue, observables):
        """Checks whether the operations and observables in queue are all supported by the device.

        Args:
            queue (Iterable[~.operation.Operation]): quantum operation objects which are intended
                to be applied on the device
            observables (Iterable[~.operation.Observable]): observables which are intended
                to be evaluated on the device

        Raises:
            Exception: if there are operations in the queue or observables that the device does
                not support
        """
        # Validate operations
        self._validate_operations(queue)

        # Validate pulses
        pulses = queue[0].H.pulses
        self._validate_pulses(pulses)

        # Validate observables
        for o in observables:
            if isinstance(o, MeasurementProcess):
                # state-based measurements not supported
                if not isinstance(o, SampleMeasurement):
                    raise RuntimeError(
                        f"Device only support sample-based measurement, but received observable {o}"
                    )
                continue
            self._validate_measurement_basis(o)

    def _validate_operations(self, operations: list[ParametrizedEvolution]):
        """Confirms that the list of operations provided contains a single ParametrizedEvolution
        from a HardwareHamiltonian with only a single, global pulse

        Args:
            operations(list[ParametrizedEvolution]): a list containing a single
                ParametrizedEvolution operator
        """

        if not np.all([op.name in self.operations for op in operations]):
            raise NotImplementedError(
                f"Device {self.short_name} expected only operations "
                f"{self.operations} but received {operations}."
            )

        if len(operations) > 1:
            raise NotImplementedError(
                f"Support for multiple ParametrizedEvolution operators in a single circuit is "
                f"not yet implemented. Received {len(operations)} operators."
            )

        ev_op = operations[0]  # only one!

        if not isinstance(ev_op.H, HardwareHamiltonian):
            raise RuntimeError(
                f"Expected a HardwareHamiltonian instance for interfacing with the device, but "
                f"recieved {type(ev_op.H)}."
            )

        if not set(ev_op.wires) == set(self.wires):
            raise RuntimeError(
                f"Device contains wires {self.wires}, but received a `ParametrizedEvolution` "
                f"operator working on wires {ev_op.wires}. Device wires must match wires of "
                f"the evolution."
            )

        if len(ev_op.H.settings.register) != len(self.wires):
            raise RuntimeError(
                f"The defined interaction term has register {ev_op.H.settings.register} of length "
                f"{len(ev_op.H.settings.register)}, which does not match the number of wires on "
                f"the device ({len(self.wires)})"
            )

    def _validate_pulses(self, pulses: list[HardwarePulse]):
        """Confirms that the list of HardwarePulses describes a single, global pulse

        Args:
            pulses: List of HardwarePulses

        Raises:
            RuntimeError, NotImplementedError"""

        if not pulses:
            raise RuntimeError("No pulses found in the ParametrizedEvolution")

        if len(pulses) > 1:
            raise NotImplementedError(
                f"Multiple pulses in a Hamiltonian are not currently supported. "
                f"Received {len(pulses)} pulses."
            )

        if pulses[0].wires != self.wires:
            raise NotImplementedError(
                f"Only global drive is currently supported. Found drive defined for subset "
                f"{[pulses[0].wires]} of all wires [{self.wires}]"
            )

    def _validate_measurement_basis(self, observable):
        """Confirm that all elements of the observable are in the measurement basis,
        and otherwise raise an error"""

        # if the observable is a composite of other operations,
        # loop through those and evaluate individually
        if isinstance(observable, CompositeOp):
            for op in observable.operands:
                self._validate_measurement_basis(op)
        elif isinstance(observable, Hamiltonian):
            for op in observable.ops:
                self._validate_measurement_basis(op)

        elif not observable.has_diagonalizing_gates:
            raise RuntimeError(
                f"Received observable {observable} with no diagonalizing gates; "
                f"cannot determine basis"
            )
        elif observable.diagonalizing_gates():
            # if diagonalizing gates are not empty (i.e. `[]`), raise an error
            raise RuntimeError(
                f"{self.short_name} can only measure in the Z basis, "
                f"but received observable {observable}"
            )


class BraketAwsAhsDevice(BraketAhsDevice):
    r"""Amazon Braket AHS device for hardware in PennyLane.

    More information about AHS and the capabilities of the hardware can be found in the
    `Amazon Braket Developer Guide
    <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.

    Args:
        wires (int or Iterable[int, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device_arn (str): The ARN identifying the ``AwsDevice`` to be used to
            run circuits; The corresponding AwsDevice must support analog Hamiltonian simulation.
            You can get device ARNs from the Amazon Braket console or from the Amazon Braket
            Developer Guide.
        s3_destination_folder (AwsSession.S3DestinationFolder): Name of the S3 bucket
            and folder, specified as a tuple.
        poll_timeout_seconds (float): Total time in seconds to wait for
            results before timing out.
        poll_interval_seconds (float): The polling interval for results in seconds.
        shots (int or Shots.DEFAULT): Number of executions to run to aquire measurements.
            Default: Shots.DEFAULT
        aws_session (Optional[AwsSession]): An AwsSession object created to manage
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None

    .. note::
        It is important to keep track of units when specifying electromagnetic pulses for hardware
        control. The frequency and amplitude provided in PennyLane for Rydberg atom systems are
        expected to be in units of MHz, time in microseconds, phase in radians, and distance in
        micrometers. All of these will be converted to SI units internally as needed for upload to
        the hardware, and frequency will be converted to angular frequency (multiplied by
        :math:`2 \pi`).

        When reading hardware specifications from the Braket backend, bear in mind that all units
        are SI and frequencies are in rad/s. This conversion is done when creating a pulse program
        for upload, and units in the PennyLane functions should follow the conventions specified
        in the PennyLane docs to ensure correct unit conversion. See
        `rydberg_interaction
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_interaction.html>`_
        and `rydberg_drive
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_drive.html>`_
        in Pennylane for specification of expected input units, and examples for creating hardware
        compatible `ParametrizedEvolution
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html>`_
        operators in PennyLane.
    """

    name = "Braket Device for AHS in PennyLane"
    short_name = "braket.aws.ahs"

    def __init__(
        self,
        wires: Union[int, Iterable],
        device_arn: str,
        s3_destination_folder: AwsSession.S3DestinationFolder = None,
        *,
        poll_timeout_seconds: float = AwsQuantumTask.DEFAULT_RESULTS_POLL_TIMEOUT,
        poll_interval_seconds: float = AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
        shots: Union[int, Shots] = Shots.DEFAULT,
        aws_session: Optional[AwsSession] = None,
    ):
        device = AwsDevice(device_arn, aws_session=aws_session)
        user_agent = f"BraketPennylanePlugin/{__version__}"
        device.aws_session.add_braket_user_agent(user_agent)

        super().__init__(wires=wires, device=device, shots=shots)

        self._s3_folder = s3_destination_folder
        self._poll_timeout_seconds = poll_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    @property
    def hardware_capabilities(self):
        """Dictionary of hardware capabilities for the hardware device"""
        return dict(self._device.properties.paradigm)

    @property
    def settings(self):
        """Dictionary of constants set by the hardware.

        Used to enable initializing hardware-consistent Hamiltonians by saving
        all the values that would need to be passed, i.e.:

            >>> dev_remote = qml.device('braket.aws.ahs', wires=3)
            >>> dev_pl = qml.device('default.qubit', wires=3)
            >>> settings = dev_remote.settings
            >>> H_int = qml.pulse.rydberg.rydberg_interaction(coordinates, **settings)

        By passing the ``settings`` from the remote device to ``rydberg_interaction``, an
        ``H_int`` Hamiltonian term is created using the constants specific to the hardware.
        This is relevant for simulating the hardware in PennyLane on the ``default.qubit`` device.
        """
        return {"interaction_coeff": self._get_rydberg_c6()}

    def _get_rydberg_c6(self):
        """Get rydberg C6 and convert from rad/s m^6 (AWS units) to MHz um^6
        (PL simulation units)"""
        c6 = float(self._device.properties.paradigm.rydberg.c6Coefficient)  # rad/s x m^6
        c6 = 1e-6 * c6 / (2 * np.pi)  # rad/s --> MHz
        c6 = c6 * 1e36  # m^6 --> um^6
        return c6

    def create_ahs_program(self, evolution: ParametrizedEvolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an AnalogHamiltonianSimulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        ahs_program = self._ahs_program_from_evolution(evolution)
        ahs_program_discretized = ahs_program.discretize(self._device)

        self._ahs_program = ahs_program_discretized

        return ahs_program_discretized

    def _run_task(self, ahs_program: AnalogHamiltonianSimulation):
        """Run and return a task executing the AnalogHamiltonianSimulation program on
        the device"""
        task = self._device.run(
            ahs_program,
            s3_destination_folder=self._s3_folder,
            shots=self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
        )
        return task


class BraketLocalAhsDevice(BraketAhsDevice):
    r"""Amazon Braket LocalSimulator AHS device for PennyLane.

    Runs programs on `Braket's local AHS simulator
    <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html#braket-simulator-ahs-local>`_.
    Can be used to emulate the :class:`~.BraketAwsAhsDevice`.

    Args:
        wires (int or Iterable[int, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int or Shots.DEFAULT): Number of executions to run to aquire measurements.
            Default: Shots.DEFAULT

    .. note::
        It is important to keep track of units when specifying electromagnetic pulses for hardware
        control. The frequency and amplitude provided in PennyLane for Rydberg atom systems are
        expected to be in units of MHz, time in microseconds, phase in radians, and distance in
        micrometers. All of these will be converted to SI units internally as needed for upload to
        the hardware, and frequency will be converted to angular frequency (multiplied by
        :math:`2 \pi`).

        When reading hardware specifications from the Braket backend, bear in mind that all units
        are SI and frequencies are in rad/s. This conversion is done when creating a pulse program
        for upload, and units in the PennyLane functions should follow the conventions specified in
        the PennyLane docs to ensure correct unit conversion. See `rydberg_interaction
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_interaction.html>`_
        and `rydberg_drive
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_drive.html>`_
        in Pennylane for specification of expected input units, and examples for creating hardware
        compatible `ParametrizedEvolution
        <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html>`_
        operators in PennyLane.
    """

    name = "Braket LocalSimulator for AHS in PennyLane"
    short_name = "braket.local.ahs"

    def __init__(
        self,
        wires: Union[int, Iterable],
        *,
        shots: Union[int, Shots] = Shots.DEFAULT,
    ):
        device = LocalSimulator("braket_ahs")
        super().__init__(wires=wires, device=device, shots=shots)

    @property
    def settings(self) -> dict:
        """Dictionary of constants set by the hardware.

        Used to enable initializing hardware-consistent Hamiltonians by saving
        all the values that would need to be passed, i.e.:

            >>> dev_remote = qml.device('braket.aws.ahs', wires=3)
            >>> dev_pl = qml.device('default.qubit', wires=3)
            >>> settings = dev_remote.settings
            >>> H_int = qml.pulse.rydberg.rydberg_interaction(coordinates, **settings)

        By passing the ``settings`` from the remote device to ``rydberg_interaction``, an
        ``H_int`` Hamiltonian term is created using the constants specific to the hardware.
        This is relevant for simulating the remote device in PennyLane on the ``default.qubit``
        device.
        """
        # C6 for the Rubidium transition used by the simulator, converted to MHz x um^6
        return {"interaction_coeff": 862620}

    def _ahs_program_from_evolution(
        self, evolution: ParametrizedEvolution
    ) -> AnalogHamiltonianSimulation:
        """Create AHS program for simulation from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an AnalogHamiltonianSimulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation."""

        # sets self.pulses to be the evaluated pulses (now only a function of time)
        self._pulses = _evaluate_pulses(evolution)
        self._register = _create_register(evolution.H.settings.register)

        time_interval = evolution.t
        time_points = _get_sample_times(time_interval)

        H = translate_pulse_to_driving_field(self._pulses[self._global_pulse_idx], time_points)

        # Create local detunings
        local_pulses = self._pulses.copy()
        local_pulses.pop(self._global_pulse_idx)

        local_detunings = _create_valid_local_detunings(local_pulses, self.wires)
        if local_detunings is not None:
            shift = translate_pulses_to_shifting_field(local_detunings, time_points)
            H = H + shift

        ahs_program = AnalogHamiltonianSimulation(register=self.register, hamiltonian=H)

        return ahs_program

    def _run_task(self, ahs_program: AnalogHamiltonianSimulation) -> AwsQuantumTask:
        """Run and return a task executing the AnalogHamiltonianSimulation program on the
        device"""
        task = self._device.run(ahs_program, shots=self.shots, steps=100)
        return task

    def _validate_pulses(self, pulses: list[HardwarePulse]):  # noqa: C901
        """Validate that all pulses are defined as expected by the device. This validation includes:

        * Verifying that a global drive is present
        * Verifying that all local pulses have zero amplitude and phase
        * Verifying that there are no overlapping wires among the local drives
        * Verifying that all local detunings are of the same type (float or callable)

        Args:
            pulses (list[HardwarePulse]): List containing all pulses

        Raises:
            ValueError: if pulses are invalid
        """

        # Iterate through pulses to find global drive
        global_index = None
        for i, pulse in enumerate(pulses):
            if set(pulse.wires) == set(self.wires):
                if global_index is not None:
                    raise ValueError(
                        "Cannot execute a ParametrizedEvolution with multiple global drives."
                    )
                global_index = i
            elif not self.wires.contains_wires(pulse.wires):
                raise ValueError(
                    f"ParametrizedEvolution contains wires {pulse.wires} which are not a subset "
                    f"of device wires {self.wires}."
                )

        # Validate that global drive covers all wires
        if global_index is None:
            raise ValueError(
                "ParametrizedEvolution doesn't apply a global driving field to all wires."
            )

        self._global_pulse_idx = global_index

        local_pulses = pulses.copy()
        local_pulses.pop(global_index)

        if len(local_pulses) == 0:
            return

        self._validate_local_pulses(local_pulses)

    def _validate_local_pulses(self, local_pulses: list[HardwarePulse]):
        """Validate that local drives don't have amplitude or phase, and that various detunings
        aren't inconsistent The detunings are stored in the `frequency` attribute of
        0`HardwarePulse`."""
        callable_detunings = callable(local_pulses[0].frequency)
        local_wires = set()

        for pulse in local_pulses:
            if pulse.amplitude is not None and (
                callable(pulse.amplitude) or not np.isclose(pulse.amplitude, 0.0)
            ):
                raise ValueError(
                    "Shifting field only allows specification of detuning. Amplitude must be zero."
                )
            if callable(pulse.frequency) ^ callable_detunings:
                raise ValueError(
                    "Found local pulses with both `float` and `callable` detunings. Pulses for "
                    "local detunings must all have only `float` or `callable` detuning (frequency)."
                )
            if set(pulse.wires).intersection(local_wires):
                raise ValueError("Local drives must not have overlapping wires.")

            local_wires.update(set(pulse.wires))
