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

from functools import partial
from typing import Callable, List, Tuple, Union

import numpy as np
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from braket.tasks.analog_hamiltonian_simulation_quantum_task_result import ShotResult
from braket.timings.time_series import TimeSeries
from numpy.typing import ArrayLike
from pennylane.pulse import ParametrizedEvolution
from pennylane.pulse.hardware_hamiltonian import HardwarePulse


def _convert_to_time_series(
    pulse_parameter: Union[float, Callable],
    time_points: ArrayLike,
    scaling_factor: float = 1,
):
    """Converts pulse information into a TimeSeries

    Args:
        pulse_parameter(Union[float, Callable]): a physical parameter (pulse, amplitude
            or frequency detuning) of the pulse. If this is a callalbe, it has already been
            partially evaluated, such that it is only a function of time.
        time_points(array): the times where parameters will be set in the TimeSeries, specified
            in seconds
        scaling_factor(float): A multiplication factor for the pulse_parameter
            where relevant to convert between units. Defaults to 1.

    Returns:
        TimeSeries: a description of setpoints and corresponding times
    """

    ts = TimeSeries()

    if callable(pulse_parameter):
        # convert time to microseconds to evaluate (expected unit for the PL functions)
        vals = [float(pulse_parameter(t * 1e6)) * scaling_factor for t in time_points]
    else:
        vals = [pulse_parameter * scaling_factor for t in time_points]

    for t, v in zip(time_points, vals):
        ts.put(t, v)

    return ts


def translate_pulse_to_driving_field(pulse: HardwarePulse, time_points: ArrayLike):
    """Converts a ``HardwarePulse`` from PennyLane describing a global drive to a
    ``DrivingField`` from Braket AHS

    Args:
        pulse[HardwarePulse]: a dataclass object containing amplitude, phase and frequency
            detuning information
        time_interval(array[float, float]): The start and end time for the applied pulse

    Returns:
        drive(DrivingField): the object representing the global drive for the
            AnalogHamiltonianSimulation object
    """

    # scaling factor for amp and frequency detuning converts from MHz to rad/s
    amplitude = _convert_to_time_series(pulse.amplitude, time_points, scaling_factor=2 * np.pi * 1e6)
    detuning = _convert_to_time_series(pulse.frequency, time_points, scaling_factor=2 * np.pi * 1e6)
    phase = _convert_to_time_series(pulse.phase, time_points)

    drive = DrivingField(amplitude=amplitude, detuning=detuning, phase=phase)

    return drive


def _evaluate_pulses(ev_op: ParametrizedEvolution):
    """Feeds in the parameters in order to partially evaluate the callables (amplitude,
    phase and/or frequency detuning) describing the pulses, so they are only a function of time.
    Saves the pulses on the device as `dev.pulses`.

    Args:
        ev_op(ParametrizedEvolution): the operator containing the pulses to be evaluated
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

        evaluated_pulses.append(
            HardwarePulse(amplitude=amplitude, phase=phase, frequency=detuning, wires=pulse.wires)
        )

    return evaluated_pulses


def _create_register(coordinates: List[Tuple[float, float]]):
    """Create an AtomArrangement to describe the atom layout from the coordinates in the
    ParametrizedEvolution, and saves it as self._register

    Args:
        coordinates(List): a list of pairs [x, y] of coordinates denoting atom locations, in um
    """

    register = AtomArrangement()
    for [x, y] in coordinates:
        # PL asks users to specify in um, Braket expects SI units
        register.add([x * 1e-6, y * 1e-6])

    return register


def translate_ahs_shot_result(res: ShotResult):
    """This function converts a single shot of the QuEra measurement results to
    0 (ground), 1 (excited) and NaN (failed to measure) for all atoms in the result.

    Args:
        res(ShotResult): the result of a single measurement shot

    The results are summarized via 3 values: status, pre_sequence, and post_sequence.

    Status is success or fail. The pre_sequence is 1 if an atom in the ground state was
    successfully initialized, and 0 otherwise. The post_sequence is 1 if an atom in the
    ground state was measured, and 0 otherwise. Comparison of pre_sequence and post_sequence
    reveals one of 4 possible outcomes. The first two (initial measurement of 0) indicate a
    failure to initialize correctly, and will yeild a NaN result. The second two are
    measurements of the excited and ground state repsectively, and yield 1 and 0.

    0 --> 0: NaN - Atom failed to be placed (no atom in the ground state either before or after)
    0 --> 1: NaN - Atom failed to be placed (but was recaptured, or something else odd happened)
    1 --> 0: 1 (Rydberg state) - atom measured in ground state before, but not after
    1 --> 1: 0 (ground state) - atom measured in ground state both before and after
    """

    # if entire measurement failed, all NaN
    if not res.status.value.lower() == "success":
        return np.array([np.NaN for i in res.pre_sequence])

    # if a single atom failed to initialize, NaN for that individual measurement
    pre_sequence = [i if i else np.NaN for i in res.pre_sequence]

    # set entry to 0 if ground state measured
    # 1 if excited state measured
    # and NaN if measurement failed
    return np.array(pre_sequence - res.post_sequence)


def _get_sample_times(time_interval: ArrayLike):
    """Takes a time interval and returns an array of times with a minimum of 50ns spacing

    Args:
        time_interval(array[float, float]): an array with start and end times for the
            pulse, in us

    Returns:
        times(array[float]): an array of times sampled at 1ns intervals between the start
            and end times, in SI units (seconds)
    """
    # time_interval from PL is in microseconds, we convert to ns
    interval_ns = np.array(time_interval) * 1e3
    start = interval_ns[0]
    end = interval_ns[1]

    # number of points must ensure at least 50ns between sample points
    num_points = int((end - start) // 50) + 1

    # we want an integer number of nanoseconds
    times = np.linspace(start, end, num_points, dtype=int)

    # we return time in seconds
    return times / 1e9


def _create_valid_local_detunings(local_pulses, dev_wires):
    """Return ordered list of local detunings for all wires in device.

    This function uses the local detunings of the pulses of the ``ParametrizedEvolution`` being executed
    to create a list of local detunings with the same length and order as the device wires. For wires that
    aren't locally detuned, the list is padded with zeros if the detunings are of type ``float``, or functions
    that return zero if the detunings are ``callable``.

    Args:
        local_pulses (List[HardwarePulse]): Partially evaluated list of local pulses
        dev_wires (~.Wires): Device wires

    Returns:
        List[Union[callable, float]]: List of detunings covering all device wires.
    """
    if len(local_pulses) == 0:
        return None

    callable_detunings = callable(local_pulses[0].frequency)
    device_detunings = (
        [lambda t: 0] * len(dev_wires) if callable_detunings else [0] * len(dev_wires)
    )

    for pulse in local_pulses:
        for wire in pulse.wires:
            device_detunings[dev_wires.index(wire)] = pulse.frequency

    return device_detunings


def _extract_pattern_from_detunings(detunings, time_points):
    """Use the detunings to find the pattern for the ``ShiftingField``.

    This function creates a time series for the local detunings and uses the values
    of the detunings at each time step to calculate and validate the pattern for the
    ``ShiftingField`` term of the driving Hamiltonian.

    Args:
        detunings (List[Union[float, callable]]): detunings to extract pattern from
        time_points (array[Number, Number]]): Array of sampled time steps

    Returns:
        Union[float, callable]: Maximum detuning to be used as the magnitude for ``ShiftingField``
        Pattern: object containing magnitude of detunings for individual atoms in the device

    Raises:
        ValueError: if the shape of all local detunings don't match or the detunings have negative values
    """
    # If a single item is not callable, no others should be callable. This validation happens in
    # ``_validate_pulses``.
    callable_detunings = callable(detunings[0])
    negative_detunings_error = (
        "Found negative value in local detunings. Make sure that all local detunings "
        + "take only positive values within the specified time interval."
    )

    if callable_detunings:
        evaluated_detunings = np.array([
            [float(detuning(t * 1e6)) for t in time_points] for detuning in detunings
        ])
        if not np.allclose(evaluated_detunings, np.abs(evaluated_detunings)):
            raise ValueError(negative_detunings_error)

        pattern = []

        # Find pattern if callable detuning
        for i in range(len(time_points)):
            time_slice = evaluated_detunings[:, i]

            if not np.allclose(time_slice, 0.0):
                max_index = np.argmax(time_slice)
                _max = time_slice[max_index]
                max_detuning = detunings[max_index]
                pattern = [det / _max for det in time_slice]
                break

        # If all time steps are zero, local detuning is assumed to be zero
        if pattern == []:
            max_detuning = 0
            pattern = [1.0] * len(detunings)
            return max_detuning, Pattern(pattern)

    else:
        max_index = np.argmax(np.abs(detunings))
        max_detuning = np.abs(detunings[max_index])

        if np.isclose(max_detuning, 0):
            return 0, Pattern([1] * len(detunings))

        pattern = [det / max_detuning for det in detunings]

        # Validate that all detunings are positive
        if not np.allclose(pattern, np.abs(pattern)):
            raise ValueError(negative_detunings_error)

        return max_detuning, Pattern(pattern)

    # Validate that detunings follow pattern along all time steps for callable detunings
    for i, t in enumerate(time_points):
        time_slice = evaluated_detunings[:, i]
        
        new_time_slice = [p * float(max_detuning(t * 1e6)) for p in pattern]
        if not np.allclose(time_slice, new_time_slice):
            raise ValueError(
                "Local detunings don't match. Make sure that all local detunings match "
                "in shape and only differ in magnitude."
            )

    return max_detuning, Pattern(pattern)


def translate_pulses_to_shifting_field(detunings, time_points):
    """Uses the overall detuning and pattern to create a ``ShiftingField`` object from
    AWS Braket.

    Args:
        detunings (List[Union[float, callable]]): Local detuning per wire
        time_points (array[Number, Number]]): Array of sampled time steps

    Returns:
        ShiftingField: the object representing the local drive for the AnalogueHamiltonianSimulation object
    """
    detuning, pattern = _extract_pattern_from_detunings(detunings, time_points)
    ts_detuning = _convert_to_time_series(
        detuning, time_points, scaling_factor=1e6
    )
    shift = ShiftingField(magnitude=Field(time_series=ts_detuning, pattern=pattern))

    return shift
