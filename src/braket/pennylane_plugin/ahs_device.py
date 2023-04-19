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
Devices
=======

**Module name:** :mod:`braket.pennylane_braket.ahs_device`

.. currentmodule:: braket.pennylane_braket.ahs_device

Braket Analogue Hamiltonian Simulation (AHS) devices to be used with PennyLane

Classes
-------

.. autosummary::
   BraketAwsAhsDevice
   BraketLocalAhsDevice

Code details
~~~~~~~~~~~~
"""
from functools import partial
from typing import Iterable, Union, Optional
import numpy as np

from pennylane import QubitDevice
from pennylane._version import __version__
from pennylane.pulse.hardware_hamiltonian import HardwarePulse, HardwareHamiltonian

from braket.aws import AwsDevice, AwsSession, AwsQuantumTask
from braket.devices import Device, LocalSimulator
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries


class BraketAhsDevice(QubitDevice):
    """Abstract Amazon Braket device for analogue hamiltonian simulation with PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device (Device): The Amazon Braket device to use with PennyLane.
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket AHS PennyLane plugin"
    pennylane_requires = ">=0.30.0"
    version = __version__
    author = "Xanadu Inc."

    operations = {"ParametrizedEvolution"}

    def __init__(
            self,
            wires: Union[int, Iterable],
            device: Device, *,
            shots: int = AwsDevice.DEFAULT_SHOTS_QPU
    ):
        if not shots:
            raise RuntimeError(f"This device requires shots. Received shots={shots}")
        self._device = device

        super().__init__(wires=wires, shots=shots)

        self.register = None
        self.pulses = None
        self.ahs_program = None
        self.samples = None

    def apply(self, operations, **kwargs):
        """Convert the pulse operation to an AHS program and run on the connected device"""

        if not np.all([op.name in self.operations for op in operations]):
            raise NotImplementedError(
                "Device {self.short_name} expected only operations "
                "{self.operations} but recieved {operations}"
            )

        self._validate_operations(operations)

        ev_op = operations[0]  # only one!

        self._validate_pulses(ev_op.H.pulses)

        ahs_program = self.create_ahs_program(ev_op)

        task = self._run_task(ahs_program)

        self.samples = task.result()

    def _run_task(self, ahs_program):
        raise NotImplementedError("Running a task not implemented for the base class")

    def _ahs_program_from_evolution(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an Analogue Hamiltonian Simulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        # sets self.pulses to be the evaluated pulses (now only a function of time)
        self._evaluate_pulses(evolution)
        self._create_register(evolution.H.settings.register)

        time_interval = evolution.t

        # no gurarentee that global drive is index 0 once we start allowing more just global drive
        drive = self._convert_pulse_to_driving_field(self.pulses[0], time_interval)

        return AnalogHamiltonianSimulation(register=self.register, hamiltonian=drive)

    def create_ahs_program(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an Analogue Hamiltonian Simulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        ahs_program = self._ahs_program_from_evolution(evolution)

        self.ahs_program = ahs_program

        return ahs_program

    def generate_samples(self):
        r"""Returns the computational basis samples measured for all wires.

        Returns:
             array[complex]: array of samples in the shape ``(dev.shots, dev.num_wires)``
        """
        return [self._result_to_sample_output(res) for res in self.samples.measurements]

    def _validate_operations(self, operations):
        """Confirms that the list of operations provided contains a single ParametrizedEvolution
        from a HardwareHamiltonian with only a single, global pulse"""

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
                f"{len(ev_op.H.settings.register)}, which does not match the number of wires on the device "
                f"({len(self.wires)})"
            )

    def _validate_pulses(self, pulses):
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

    def _create_register(self, coordinates):
        """Create an AtomArrangement to describe the atom layout from the coordinates in the
        ParametrizedEvolution, and saves it as self.register"""

        register = AtomArrangement()
        for [x, y] in coordinates:
            # PL asks users to specify in um, Braket expects SI units
            register.add([x * 1e-6, y * 1e-6])

        self.register = register

    def _evaluate_pulses(self, ev_op):
        """Feeds in the parameters in order to partially evaluate the callables (amplitude,
        phase and/or frequency detuning) describing the pulses, so they are only a function of time.
        Saves the pulses on the device as `dev.pulses`.

        Args:
            ev_op(ParametrizedEvolution): the operator containing the pulses to be evaluated
            params(list): a list of the parameters to be passed to the respective callables
        """

        params = ev_op.parameters
        pulses = ev_op.H.pulses

        evaluated_pulses = []
        idx = 0

        for pulse in pulses:
            amplitude = pulse.amplitude
            if callable(pulse.amplitude):
                amplitude = partial(pulse.amplitude, params[idx])
                idx += 1

            phase = pulse.phase
            if callable(pulse.phase):
                phase = partial(pulse.phase, params[idx])
                idx += 1

            detuning = pulse.frequency
            if callable(pulse.frequency):
                detuning = partial(pulse.frequency, params[idx])
                idx += 1

            evaluated_pulses.append(HardwarePulse(amplitude=amplitude,
                                                  phase=phase,
                                                  frequency=detuning,
                                                  wires=pulse.wires))

        self.pulses = evaluated_pulses

    def _get_sample_times(self, time_interval):
        """Takes a time interval and returns an array of times with a minimum of 50ns spacing"""
        # time_interval from PL is in microseconds, we convert to ns
        interval_ns = np.array(time_interval) * 1e3
        start = interval_ns[0]
        end = interval_ns[1]

        # number of points must ensure at least 50ns between sample points
        num_points = int((end - start) // 50)

        # we want an integer number of nanoseconds
        times = np.linspace(start, end, num_points, dtype=int)

        # we return time in seconds
        return times / 1e9

    def _convert_to_time_series(self, pulse_parameter, time_points, scaling_factor=1):
        """Converts pulse information into a TimeSeries

        Args:
            pulse_parameter(Union[float, Callable]): a physical parameter (pulse, amplitude
                or frequency detuning) of the pulse. If this is a callalbe, it has already been partially
                evaluated, such that it is only a function of time.
            time_points(array): the times where parameters will be set in the TimeSeries, specified
                in seconds scaling_factor(float): A multiplication factor for the pulse_parameter
                where relevant to convert between units. Defaults to 1.

        Returns:
            TimeSeries: a description of setpoints and corresponding times
        """

        ts = TimeSeries()

        if callable(pulse_parameter):
            # convert time to microseconds to evaluate (expected unit for the PL functions)
            vals = [float(pulse_parameter(t * 1e6)) * scaling_factor for t in time_points]
        else:
            vals = [pulse_parameter for t in time_points]

        for t, v in zip(time_points, vals):
            ts.put(t, v)

        return ts

    def _convert_pulse_to_driving_field(self, pulse, time_interval):
        """Converts a ``HardwarePulse`` from PennyLane describing a global drive to a 
        ``DrivingField`` from Braket AHS

        Args:
            pulse[HardwarePulse]: a dataclass object containing amplitude, phase and frequency detuning
                information
            time_interval(array[Number, Number]]): The start and end time for the applied pulse

        Returns:
            DrivingField: the object representing the global drive for the
                AnalogueHamiltonianSimulation object
        """

        time_points = self._get_sample_times(time_interval)

        # scaling factor for amp and frequency detuning converts MHz (PL input) to rad/s (upload units)
        amplitude = self._convert_to_time_series(
            pulse.amplitude, time_points, scaling_factor=2 * np.pi * 1e6
        )
        detuning = self._convert_to_time_series(
            pulse.frequency, time_points, scaling_factor=2 * np.pi * 1e6
        )
        phase = self._convert_to_time_series(pulse.phase, time_points)

        drive = DrivingField(amplitude=amplitude, detuning=detuning, phase=phase)

        return drive

    @staticmethod
    def _result_to_sample_output(res):
        """This function converts a single shot of the QuEra measurement results to
        0 (ground), 1 (excited) and NaN (failed to measure) for all atoms in the result.

        The QuEra results are summarized via 3 values: status, pre_sequence, and post_sequence.

        Status is success or fail. The pre_sequence is 1 if an atom in the ground state was
        successfully initialized, and 0 otherwise. The post_sequence is 1 if an atom in the
        ground state was measured, and 0 otherwise. Comparison of pre_sequence and post_sequence
        reveals one of 4 possible outcomes. The first two (initial measurement of 0) indicate a
        failure to initialize correctly, and will yeild a NaN result. The second two are measurements
        of the excited and ground state repsectively, and yield 1 and 0.

        0 --> 0: NaN - Atom failed to be placed (no atom detected in the ground state either before or after)
        0 --> 1: NaN - Atom failed to be placed (but was recaptured, or something else weird happened)
        1 --> 0: 1 (Rydberg state) - atom measured in ground state before, but not after
        1 --> 1: 0 (ground state) - atom measured in ground state both before and after
        """

        # if entire measurement failed, all NaN
        if not res.status.value.lower() == "success":
            return np.array([np.NaN for i in res.pre_sequence])

        # if a single atom failed to initialize, NaN for that individual measurement
        pre_sequence = [i if i else np.NaN for i in res.pre_sequence]

        # set entry to 0 if ground state is measured, 1 if excited state is measured, NaN if measurement failed
        return np.array(pre_sequence - res.post_sequence)


class BraketAwsAhsDevice(BraketAhsDevice):
    """Amazon Braket AHS device for hardware in PennyLane.

More information about AHS and the capabilities of the hardware can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        device_arn (str): The ARN identifying the ``AwsDevice`` to be used to
            run circuits; The corresponding AwsDevice must support Analogue Hamiltonian Simulation.
            You can get device ARNs from the Amazon Braket console or from the Amazon Braket Developer Guide.
        s3_destination_folder (AwsSession.S3DestinationFolder): Name of the S3 bucket
            and folder, specified as a tuple.
        poll_timeout_seconds (float): Total time in seconds to wait for
            results before timing out.
        poll_interval_seconds (float): The polling interval for results in seconds.
        shots (int): Number of executions to run to acquire measurements. Defaults to 100.
        aws_session (Optional[AwsSession]): An AwsSession object created to manage
            interactions with AWS services, to be supplied if extra control
            is desired. Default: None

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
            poll_interval_seconds : float = AwsQuantumTask.DEFAULT_RESULTS_POLL_INTERVAL,
            shots: int = AwsDevice.DEFAULT_SHOTS_QPU,
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

        By passing the ``settings`` from the remote device to ``rydberg_interaction``, an ``H_int`` Hamiltonian
        term is created using the constants specific to the hardware. This is relevant for simulating the hardware
        in PennyLane on the ``default.qubit`` device.
        """
        return {"interaction_coeff": self._get_rydberg_c6()}

    def _get_rydberg_c6(self):
        """Get rydberg C6 and convert from rad/s m^6 (AWS units) to MHz um^6 (PL simulation units)"""
        c6 = float(self._device.properties.paradigm.rydberg.c6Coefficient)  # rad/s x m^6
        c6 = 1e-6 * c6/(2*np.pi)  # rad/s --> MHz
        c6 = c6 * 1e36  # m^6 --> um^6
        return c6

    def create_ahs_program(self, evolution):
        """Create AHS program for upload to hardware from a ParametrizedEvolution

        Args:
            evolution (ParametrizedEvolution): the PennyLane operator describing the pulse
                to be converted into an Analogue Hamiltonian Simulation program

        Returns:
            AnalogHamiltonianSimulation: a program containing the register and drive
                information for running an AHS task on simulation or hardware"""

        ahs_program = self._ahs_program_from_evolution(evolution)
        ahs_program_discretized = ahs_program.discretize(self._device)

        self.ahs_program = ahs_program_discretized

        return ahs_program_discretized

    def _run_task(self, ahs_program):
        task = self._device.run(
            ahs_program,
            s3_destination_folder=self._s3_folder,
            shots=self.shots,
            poll_timeout_seconds=self._poll_timeout_seconds,
            poll_interval_seconds=self._poll_interval_seconds,
        )
        return task


class BraketLocalAhsDevice(BraketAhsDevice):
    """Amazon Braket LocalSimulator AHS device for PennyLane.

    Emulates :class:`~.BraketAwsAhsDevice` by running <link to https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html#braket-simulator-ahs-local>

    Args:
        wires (int or Iterable[Number, str]]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers
            (i.e., ``[-1, 0, 2]``) or strings (``['ancilla', 'q1', 'q2']``).
        shots (int): Number of executions to run to aquire measurements. Defaults to 100.
    """

    name = "Braket LocalSimulator for AHS in PennyLane"
    short_name = "braket.local.ahs"

    def __init__(
            self,
            wires: Union[int, Iterable],
            *,
            shots: int = AwsDevice.DEFAULT_SHOTS_QPU  # Simulator default of 0 not suitable for AHS simulator
    ):
        device = LocalSimulator("braket_ahs")
        super().__init__(wires=wires, device=device, shots=shots)

    @property
    def settings(self):
        """Dictionary of constants set by the hardware.

        Used to enable initializing hardware-consistent Hamiltonians by saving
        all the values that would need to be passed, i.e.:

            >>> dev_remote = qml.device('braket.aws.ahs', wires=3)
            >>> dev_pl = qml.device('default.qubit', wires=3)
            >>> settings = dev_remote.settings
            >>> H_int = qml.pulse.rydberg.rydberg_interaction(coordinates, **settings)

        By passing the ``settings`` from the remote device to ``rydberg_interaction``, an ``H_int`` Hamiltonian
        term is created using the constants specific to the hardware. This is relevant for simulating the remote
        device in PennyLane on the ``default.qubit`` device.
        """
        return {"interaction_coeff": 862690}  # C6 for the Rubidium transition used by the simulator, MHz x um^6

    def _run_task(self, ahs_program):
        task = self._device.run(ahs_program, shots=self.shots, steps=100)
        return task
