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

from functools import partial, reduce, singledispatch
from typing import Any, Optional, Union

import numpy as onp
import pennylane as qml
from braket.aws import AwsDevice
from braket.circuits import FreeParameter, Gate, ResultType, gates, noises, observables
from braket.circuits.result_types import (
    AdjointGradient,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from braket.devices import Device
from braket.pulse import ArbitraryWaveform, ConstantWaveform, PulseSequence
from braket.tasks import GateModelQuantumTaskResult
from pennylane import numpy as np
from pennylane.measurements import MeasurementProcess, ObservableReturnTypes
from pennylane.operation import Observable, Operation
from pennylane.ops import Adjoint
from pennylane.pulse import ParametrizedEvolution

from braket.pennylane_plugin.ops import (
    AAMS,
    MS,
    PSWAP,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    GPi,
    GPi2,
)

_BRAKET_TO_PENNYLANE_OPERATIONS = {
    "i": "Identity",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "h": "Hadamard",
    "ry": "RY",
    "rx": "RX",
    "rz": "RZ",
    "s": "S",
    "si": "Adjoint(S)",
    "v": "SX",
    "vi": "Adjoint(SX)",
    "t": "T",
    "ti": "Adjoint(T)",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
    "cswap": "CSWAP",
    "ccnot": "Toffoli",
    "phaseshift": "PhaseShift",
    "unitary": "QubitUnitary",
    "amplitude_damping": "AmplitudeDamping",
    "generalized_amplitude_damping": "GeneralizedAmplitudeDamping",
    "phase_damping": "PhaseDamping",
    "depolarizing": "DepolarizingChannel",
    "bit_flip": "BitFlip",
    "phase_flip": "PhaseFlip",
    "kraus": "QubitChannel",
    "cphaseshift": "ControlledPhaseShift",
    "cphaseshift00": "CPhaseShift00",
    "cphaseshift01": "CPhaseShift01",
    "cphaseshift10": "CPhaseShift10",
    "iswap": "ISWAP",
    "pswap": "PSWAP",
    "xy": "IsingXY",
    "xx": "IsingXX",
    "yy": "IsingYY",
    "zz": "IsingZZ",
    "ecr": "ECR",
    "gpi": "GPi",
    "gpi2": "GPi2",
    "ms": "AAMS",
}


def supported_operations(device: Device, verbatim: bool = False) -> frozenset[str]:
    """Returns the operations supported by the plugin based upon the device.

    Args:
        device (Device): The device to obtain the supported operations for
        verbatim (bool): Whether to return the operations supported in verbatim mode,
            the native gate set of the device. Default False

    Returns:
        frozenset[str]: The names of the supported operations
    """
    try:
        properties = (
            device.properties.paradigm
            if verbatim
            else device.properties.action["braket.ir.openqasm.program"]
        )
    except AttributeError:
        raise AttributeError("Device needs to have properties defined.")

    if verbatim:
        supported_ops = frozenset(op.lower() for op in properties.nativeGateSet)
        supported_pragmas = []
    else:
        supported_ops = frozenset(op.lower() for op in properties.supportedOperations)
        supported_pragmas = frozenset(op.lower() for op in properties.supportedPragmas)
    translated = frozenset(
        _BRAKET_TO_PENNYLANE_OPERATIONS[op]
        for op in _BRAKET_TO_PENNYLANE_OPERATIONS
        if op.lower() in supported_ops or f"braket_noise_{op.lower()}" in supported_pragmas
    )
    # both AAMS and MS map to ms
    if "AAMS" in translated:
        translated |= {"MS"}

    if (
        isinstance(device, AwsDevice)
        and device.arn == "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy"
    ):
        translated |= {"ParametrizedEvolution"}
    return translated


def translate_operation(
    operation: Operation,
    use_unique_params: bool = False,
    param_names: Optional[list[str]] = None,
    *args,
    **kwargs,
) -> Gate:
    """Translates a PennyLane ``Operation`` into the corresponding Braket ``Gate``.

    Args:
        operation (Operation): The PennyLane ``Operation`` to translate
        use_unique_params (bool): If true, numeric parameters in the resulting operation will be
        replaced with FreeParameter objects (with names corresponding to param_names). Non-numeric
        parameters will be skipped.
        param_names (Optional[list[str]]): A list of parameter names to be supplied
            to the new operation. The length of the list must match the number of
            the operator's parameters; if no named parameter is needed for the corresponding
            operation parameter, then the list entry should be `None`.

    Returns:
        Gate: The Braket gate corresponding to the given operation
    """
    if use_unique_params:
        parameters = []
        param_names = param_names or [None] * len(operation.parameters)
        if len(param_names) != len(operation.parameters):
            raise ValueError("Parameter names list must be equal to number of operation parameters")
        for param_name, param in zip(param_names, operation.parameters):
            # PennyLane passes any non-keyword argument in the operation.parameters list.
            # In some cases, like the unitary gate or qml.QubitChannel (Kraus noise), these
            # parameter can be matrices. Braket only supports parameterization of numeric parameters
            # (so far, these are all angle parameters), so non-numeric parameters are handled
            # separately.
            if param_name is not None:
                new_param = FreeParameter(param_name)
            elif isinstance(param, qml.numpy.tensor):
                new_param = param.numpy()
            else:
                new_param = param
            parameters.append(new_param)
    else:
        parameters = [
            p.numpy() if isinstance(p, qml.numpy.tensor) else p for p in operation.parameters
        ]
    device = kwargs.get("device", None)
    return _translate_operation(operation, parameters, device)


@singledispatch
def _translate_operation(operation: Operation, _parameters, device=None) -> Gate:
    raise NotImplementedError(
        f"Braket PennyLane plugin does not support operation {operation.name}."
    )


@_translate_operation.register
def _(_: qml.Identity, _parameters, device=None):
    return gates.I()


@_translate_operation.register
def _(_: qml.Hadamard, _parameters, device=None):
    return gates.H()


@_translate_operation.register
def _(_: qml.PauliX, _parameters, device=None):
    return gates.X()


@_translate_operation.register
def _(_: qml.PauliY, _parameters, device=None):
    return gates.Y()


@_translate_operation.register
def _(_: qml.PauliZ, _parameters, device=None):
    return gates.Z()


@_translate_operation.register
def _(_: qml.ECR, _parameters, device=None):
    return gates.ECR()


@_translate_operation.register
def _(s: qml.S, _parameters, device=None):
    return gates.S()


@_translate_operation.register
def _(sx: qml.SX, _parameters, device=None):
    return gates.V()


@_translate_operation.register
def _(t: qml.T, _parameters, device=None):
    return gates.T()


@_translate_operation.register
def _(_: qml.CNOT, _parameters, device=None):
    return gates.CNot()


@_translate_operation.register
def _(_: qml.CY, _parameters, device=None):
    return gates.CY()


@_translate_operation.register
def _(_: qml.CZ, _parameters, device=None):
    return gates.CZ()


@_translate_operation.register
def _(_: qml.SWAP, _parameters, device=None):
    return gates.Swap()


@_translate_operation.register
def _(_: qml.CSWAP, _parameters, device=None):
    return gates.CSwap()


@_translate_operation.register
def _(_: qml.Toffoli, _parameters, device=None):
    return gates.CCNot()


@_translate_operation.register
def _(rx: qml.RX, parameters, device=None):
    phi = parameters[0]
    return gates.Rx(phi)


@_translate_operation.register
def _(ry: qml.RY, parameters, device=None):
    phi = parameters[0]
    return gates.Ry(phi)


@_translate_operation.register
def _(rz: qml.RZ, parameters, device=None):
    phi = parameters[0]
    return gates.Rz(phi)


@_translate_operation.register
def _(phase_shift: qml.PhaseShift, parameters, device=None):
    phi = parameters[0]
    return gates.PhaseShift(phi)


@_translate_operation.register
def _(qubit_unitary: qml.QubitUnitary, parameters, device=None):
    U = np.asarray(parameters[0])
    return gates.Unitary(U)


@_translate_operation.register
def _(_: qml.AmplitudeDamping, parameters, device=None):
    gamma = parameters[0]
    return noises.AmplitudeDamping(gamma)


@_translate_operation.register
def _(_: qml.GeneralizedAmplitudeDamping, parameters, device=None):
    gamma = parameters[0]
    probability = parameters[1]
    return noises.GeneralizedAmplitudeDamping(probability=probability, gamma=gamma)


@_translate_operation.register
def _(_: qml.PhaseDamping, parameters, device=None):
    gamma = parameters[0]
    return noises.PhaseDamping(gamma)


@_translate_operation.register
def _(_: qml.DepolarizingChannel, parameters, device=None):
    probability = parameters[0]
    return noises.Depolarizing(probability)


@_translate_operation.register
def _(_: qml.BitFlip, parameters, device=None):
    probability = parameters[0]
    return noises.BitFlip(probability)


@_translate_operation.register
def _(_: qml.PhaseFlip, parameters, device=None):
    probability = parameters[0]
    return noises.PhaseFlip(probability)


@_translate_operation.register
def _(_: qml.QubitChannel, parameters, device=None):
    K_list = [np.asarray(matrix) for matrix in parameters]
    return noises.Kraus(K_list)


@_translate_operation.register
def _(c_phase_shift: qml.ControlledPhaseShift, parameters, device=None):
    phi = parameters[0]
    return gates.CPhaseShift(phi)


@_translate_operation.register
def _(c_phase_shift_00: CPhaseShift00, parameters, device=None):
    phi = parameters[0]
    return gates.CPhaseShift00(phi)


@_translate_operation.register
def _(c_phase_shift_01: CPhaseShift01, parameters, device=None):
    phi = parameters[0]
    return gates.CPhaseShift01(phi)


@_translate_operation.register
def _(c_phase_shift_10: CPhaseShift10, parameters, device=None):
    phi = parameters[0]
    return gates.CPhaseShift10(phi)


@_translate_operation.register
def _(iswap: qml.ISWAP, _parameters, device=None):
    return gates.ISwap()


@_translate_operation.register
def _(pswap: PSWAP, parameters, device=None):
    phi = parameters[0]
    return gates.PSwap(phi)


@_translate_operation.register
def _(xy: qml.IsingXY, parameters, device=None):
    phi = parameters[0]
    return gates.XY(phi)


@_translate_operation.register
def _(xx: qml.IsingXX, parameters, device=None):
    phi = parameters[0]
    return gates.XX(phi)


@_translate_operation.register
def _(yy: qml.IsingYY, parameters, device=None):
    phi = parameters[0]
    return gates.YY(phi)


@_translate_operation.register
def _(zz: qml.IsingZZ, parameters, device=None):
    phi = parameters[0]
    return gates.ZZ(phi)


@_translate_operation.register
def _(_gpi: GPi, parameters, device=None):
    phi = parameters[0]
    return gates.GPi(phi)


@_translate_operation.register
def _(gpi2: GPi2, parameters, device=None):
    phi = parameters[0]
    return gates.GPi2(phi)


@_translate_operation.register
def _(ms: MS, parameters, device=None):
    phi_0, phi_1 = parameters[:2]
    return gates.MS(phi_0, phi_1)


@_translate_operation.register
def _(ms: AAMS, parameters, device=None):
    phi_0, phi_1, theta = parameters[:3]
    return gates.MS(phi_0, phi_1, theta)


@_translate_operation.register
def _(adjoint: Adjoint, parameters, device=None):
    if isinstance(adjoint.base, qml.ISWAP):
        # gates.ISwap.adjoint() returns a different value
        return gates.PSwap(3 * np.pi / 2)
    base = _translate_operation(adjoint.base, parameters)
    if len(base.adjoint()) > 1:
        raise NotImplementedError(
            f"The adjoint of the Braket operation {base} contains more than one operation."
        )
    return base.adjoint()[0]


@_translate_operation.register
def _(op: ParametrizedEvolution, _parameters, device):
    start, end = op.t[0], op.t[1]
    pulse_length = (end - start) * 1e-9  # nanoseconds to seconds
    pulses = op.H.pulses

    # The driven wires aren't the same as `op.wires` as `op.wires` contains
    # all device wires due to interaction term.
    pulse_wires = qml.wires.Wires.all_wires([pulse.wires for pulse in pulses])
    frames = {w: device.frames[f"q{w}_drive"] for w in pulse_wires}

    # take dt from first frame (all frames have identical dt)
    time_step = {
        wire: frame.port.dt * 1e9 for wire, frame in frames.items()
    }  # seconds to nanoseconds

    pulse_sequence = PulseSequence().barrier(list(frames.values()))
    callable_index = 0

    for pulse in pulses:
        # Create waveform for each pulse in `ParametrizedEvolution`
        if callable(pulse.amplitude):
            if pulse.amplitude == qml.pulse.constant:
                amplitude = complex(op.parameters[callable_index])
                callable_index += 1

                def waveform(dt):
                    return ConstantWaveform(pulse_length, amplitude)

            else:
                amplitude = partial(pulse.amplitude, op.parameters[callable_index])
                callable_index += 1

                def waveform(dt):
                    # Calculate amplitude for each time step and normalize
                    amplitudes = onp.array([amplitude(t) for t in np.arange(start, end + dt, dt)])

                    return ArbitraryWaveform(amplitudes)

        else:

            def waveform(dt):
                return ConstantWaveform(pulse_length, pulse.amplitude)

        if callable(pulse.phase):
            phase = float(op.parameters[callable_index])
            callable_index += 1
        else:
            phase = pulse.phase

        if callable(pulse.frequency):
            frequency = float(op.parameters[callable_index])
            callable_index += 1
        else:
            frequency = pulse.frequency

        # Play pulse for each frame
        for w in pulse.wires:
            pulse_sequence = (
                pulse_sequence.set_frequency(frames[w], frequency * 1e9)  # GHz to Hz
                .shift_phase(frames[w], phase)
                .play(frames[w], waveform(time_step[w]))
                .shift_phase(frames[w], -phase)
            )

    pulse_sequence = pulse_sequence.barrier(list(frames.values()))
    return gates.PulseGate(pulse_sequence, qubit_count=len(op.wires))


def get_adjoint_gradient_result_type(
    observable: Observable,
    targets: Union[list[int], list[list[int]]],
    supported_result_types: frozenset[str],
    parameters: list[str],
):
    if "AdjointGradient" not in supported_result_types:
        raise NotImplementedError("Unsupported return type: AdjointGradient")
    braket_observable = _translate_observable(observable)

    braket_observable = (
        braket_observable.item() if hasattr(braket_observable, "item") else braket_observable
    )
    return AdjointGradient(observable=braket_observable, target=targets, parameters=parameters)


def translate_result_type(
    measurement: MeasurementProcess, targets: list[int], supported_result_types: frozenset[str]
) -> Union[ResultType, tuple[ResultType, ...]]:
    """Translates a PennyLane ``Observable`` into the corresponding Braket ``ResultType``.

    Args:
        observable (Observable): The PennyLane ``Observable`` to translate
        targets (list[int]): The target wires of the observable using a consecutive integer wire
            ordering
        supported_result_types (frozenset[str]): Braket result types supported by the Braket device

    Returns:
        Union[ResultType, tuple[ResultType]]: The Braket result type corresponding to
        the given observable; if the observable type has multiple terms, for example a Hamiltonian,
        then this will return a result type for each term.
    """
    return_type = measurement.return_type

    if return_type is ObservableReturnTypes.Probability:
        return Probability(targets)

    if return_type is ObservableReturnTypes.State:
        if not targets and "StateVector" in supported_result_types:
            return StateVector()
        elif "DensityMatrix" in supported_result_types:
            return DensityMatrix(targets)
        raise NotImplementedError(f"Unsupported return type: {return_type}")

    if isinstance(measurement.obs, qml.Hamiltonian):
        if return_type is ObservableReturnTypes.Expectation:
            return tuple(
                Expectation(_translate_observable(term), term.wires) for term in measurement.obs.ops
            )
        raise NotImplementedError(f"Return type {return_type} unsupported for Hamiltonian")

    braket_observable = _translate_observable(measurement.obs)
    if return_type is ObservableReturnTypes.Expectation:
        return Expectation(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Variance:
        return Variance(braket_observable, targets)
    elif return_type is ObservableReturnTypes.Sample:
        return Sample(braket_observable, targets)
    else:
        raise NotImplementedError(f"Unsupported return type: {return_type}")


@singledispatch
def _translate_observable(observable):
    raise TypeError(f"Unsupported observable: {type(observable)}")


@_translate_observable.register
def _(H: qml.Hamiltonian):
    # terms is structured like [C, O] where C is a tuple of all the coefficients, and O is
    # a tuple of all the corresponding observable terms (X, Y, Z, H, etc or a tensor product
    # of them)
    coefficents, pl_observables = H.terms()
    braket_observables = list(map(lambda obs: _translate_observable(obs), pl_observables))
    braket_hamiltonian = sum(
        (coef * obs for coef, obs in zip(coefficents[1:], braket_observables[1:])),
        coefficents[0] * braket_observables[0],
    )
    return braket_hamiltonian


@_translate_observable.register
def _(_: qml.PauliX):
    return observables.X()


@_translate_observable.register
def _(_: qml.PauliY):
    return observables.Y()


@_translate_observable.register
def _(_: qml.PauliZ):
    return observables.Z()


@_translate_observable.register
def _(_: qml.Hadamard):
    return observables.H()


@_translate_observable.register
def _(_: qml.Identity):
    return observables.I()


@_translate_observable.register
def _(h: qml.Hermitian):
    return observables.Hermitian(qml.matrix(h))


_zero = np.array([[1, 0], [0, 0]])
_one = np.array([[0, 0], [0, 1]])


@_translate_observable.register
def _(p: qml.Projector):
    state, wires = p.parameters[0], p.wires
    if len(state) == len(wires):  # state is a basis state
        products = [_one if b else _zero for b in state]
        hermitians = [observables.Hermitian(p) for p in products]
        return observables.TensorProduct(hermitians)

    # state is a state vector
    return observables.Hermitian(p.matrix())


@_translate_observable.register
def _(t: qml.operation.Tensor):
    return reduce(lambda x, y: x @ y, [_translate_observable(factor) for factor in t.obs])


def translate_result(
    braket_result: GateModelQuantumTaskResult,
    measurement: MeasurementProcess,
    targets: list[int],
    supported_result_types: frozenset[str],
) -> Any:
    """Translates a Braket result into the corresponding PennyLane return type value.

    Args:
        braket_result (GateModelQuantumTaskResult): The Braket result to translate.
        measurement (MeasurementProcess): The PennyLane measurement process associated with the result.
        targets (list[int]): The qubits in the result.
        supported_result_types (frozenset[str]): The result types supported by the device.

    Returns:
        Any: The translated return value.

    Note:
        Hamiltonian results will be summed over all terms.
    """

    # if braket result contains adjoint gradient, just return it since it should be the only
    # result type if it's there at all.
    observable = measurement.obs
    ag_results = [
        result for result in braket_result.result_types if result.type.type == "adjoint_gradient"
    ]
    if ag_results:
        ag_result = ag_results[0]
        key_indices = [int(param_name.split("p_")[1]) for param_name in ag_result.value["gradient"]]
        return [ag_result.value["expectation"]], [
            # we need to sort the keys by index since braket can return them in the wrong order
            ag_result.value["gradient"][f"p_{i}"]
            for i in sorted(key_indices)
        ]
    translated = translate_result_type(measurement, targets, supported_result_types)
    if isinstance(observable, qml.Hamiltonian):
        coeffs, _ = observable.terms()
        return sum(
            coeff * braket_result.get_value_by_result_type(result_type)
            for coeff, result_type in zip(coeffs, translated)
        )
    else:
        return braket_result.get_value_by_result_type(translated)
