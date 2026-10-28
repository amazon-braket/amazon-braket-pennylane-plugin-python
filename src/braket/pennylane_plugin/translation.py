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

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import partial, reduce, singledispatch
from typing import Any

import numpy as onp
import pennylane as qp
from pennylane import numpy as np
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation, Operator
from pennylane.pulse import ParametrizedEvolution
from sympy import Expr

from braket.aws import AwsDevice
from braket.circuits import (
    FreeParameter,
    FreeParameterExpression,
    Gate,
    ResultType,
    gates,
    noises,
    observables,
)
from braket.circuits.gates import format_complex
from braket.circuits.result_types import (
    AdjointGradient,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from braket.circuits.serialization import (
    OpenQASMSerializationProperties,
    QubitReferenceType,
)
from braket.device_schema import DeviceActionType
from braket.devices import Device
from braket.ir.openqasm import Program
from braket.pennylane_plugin.ops import (
    AAMS,
    MS,
    PSWAP,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    GPi,
    GPi2,
    PRx,
)
from braket.pulse import ArbitraryWaveform, ConstantWaveform, PulseSequence
from braket.tasks import GateModelQuantumTaskResult

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
    "prx": "PRx",
    "gpi": "GPi",
    "gpi2": "GPi2",
    "ms": "AAMS",
}


_BRAKET_TO_PENNYLANE_OBSERVABLES = {
    "x": frozenset({"PauliX"}),
    "y": frozenset({"PauliY"}),
    "z": frozenset({"PauliZ"}),
    "h": frozenset({"Hadamard"}),
    "hermitian": frozenset({"Hermitian", "Projector"}),
    "i": frozenset({"Identity"}),
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
            else device.properties.action[DeviceActionType.OPENQASM]
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


_PENNYLANE_TO_OPENQASM_OPERATIONS = {
    pennylane: braket for braket, pennylane in _BRAKET_TO_PENNYLANE_OPERATIONS.items()
} | {"MS": "ms"}


def to_openqasm(
    operations: Iterable[tuple[Operation, Sequence[str | None]]],
    measurements: Sequence[MeasurementProcess] = (),
    *,
    wire_map: Mapping[Hashable, int],
    supported_result_types: frozenset[str] = frozenset(),
    gradient_parameters: Sequence[str] | None = None,
    rotations: Iterable[tuple[Operation, Sequence[str | None]]] = (),
    verbatim: bool = False,
    inputs: Iterable[str] = (),
) -> Program:
    """Serialize PennyLane operations directly into an OpenQASM program, matching the program
    that the SDK serializes for the equivalent circuit.

    Pulse gates and noise models are handled by the device's SDK circuit path.

    Args:
        operations (Iterable[tuple[Operation, Sequence[str | None]]]): PennyLane operations,
            each with the free parameter name of each of its parameters, or None for
            parameters that keep their values.
        measurements (Sequence[MeasurementProcess]): Measurements to request as the result
            types that ``translate_result_type`` translates them to. Defaults to an empty
            sequence, in which case all used qubits are measured.
        wire_map (Mapping[Hashable, int]): The device qubit of each wire. Device qubits that
            operations do not act on receive identities, so that results include them.
        supported_result_types (frozenset[str]): Braket result types supported by the device.
            Defaults to an empty collection.
        gradient_parameters (Sequence[str] | None): If given, the adjoint gradient of the
            single expectation in ``measurements`` with respect to these free parameters is
            requested instead. Defaults to None.
        rotations (Iterable[tuple[Operation, Sequence[str | None]]]): Operations in the same
            form, applied after operations and outside any verbatim box.
            Defaults to an empty iterable.
        verbatim (bool): Whether to enclose operations in a verbatim box and use physical
            qubit references. Defaults to False.
        inputs (Iterable[str]): Names of free parameters that will be bound when the program
            is submitted. Defaults to an empty iterable.

    Returns:
        Program: The serialized OpenQASM program with an empty inputs dictionary.

    Raises:
        NotImplementedError: If an operation or measurement is not supported.
        ValueError: If no qubits are used, or a free parameter is absent from inputs.
    """
    properties = OpenQASMSerializationProperties(
        qubit_reference_type=(
            QubitReferenceType.PHYSICAL if verbatim else QubitReferenceType.VIRTUAL
        )
    )
    qubits = {}
    parameters = set()
    statements = ["#pragma braket verbatim", "box{"] if verbatim else []
    statements.extend(_operation_statements(operations, wire_map, properties, qubits, parameters))
    for qubit in sorted(set(wire_map.values()).difference(qubits)):
        statements.append(f"i {properties.format_target(qubit)};")
        qubits[qubit] = None
    if verbatim:
        statements.append("}")
    statements.extend(_operation_statements(rotations, wire_map, properties, qubits, parameters))
    if not qubits:
        raise ValueError("Circuit must have at least one non-zero-qubit gate to run on a device")
    if unbound := parameters - set(inputs):
        raise ValueError(f"Cannot execute circuit with unbound parameters: {unbound}")

    if gradient_parameters is not None:
        results = [
            _adjoint_gradient_to_openqasm(
                measurements, gradient_parameters, supported_result_types, wire_map, properties
            )
        ]
    else:
        # Ordered deduplication, as with Circuit.add_result_type
        results = list(
            dict.fromkeys(
                result
                for measurement in measurements
                for result in _result_types_to_openqasm(
                    measurement.map_wires(wire_map),
                    supported_result_types,
                    properties,
                )
            )
        )
    declarations = ["OPENQASM 3.0;"]
    declarations.extend(f"input float {parameter};" for parameter in sorted(parameters))
    if not results:
        declarations.append(f"bit[{len(qubits)}] b;")
    if not verbatim:
        declarations.append(f"qubit[{max(qubits) + 1}] q;")
    if results:
        statements.extend(results)
    else:
        measured_qubits = qubits if verbatim else sorted(qubits)
        statements.extend(
            f"b[{index}] = measure {properties.format_target(qubit)};"
            for index, qubit in enumerate(measured_qubits)
        )
    return Program.construct(source="\n".join(declarations + statements), inputs={})


def _result_types_to_openqasm(
    measurement: MeasurementProcess,
    supported_result_types: frozenset[str],
    properties: OpenQASMSerializationProperties,
) -> list[str]:
    qubits = [int(wire) for wire in measurement.wires]
    targets = ", ".join(properties.format_target(qubit) for qubit in qubits) or "all"
    observable = measurement.obs

    if isinstance(measurement, qp.measurements.ProbabilityMP):
        return [f"#pragma braket result probability {targets}"]

    if isinstance(measurement, qp.measurements.DensityMatrixMP):
        if "DensityMatrix" in supported_result_types:
            return [f"#pragma braket result density_matrix {targets}"]
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    if isinstance(measurement, qp.measurements.StateMP):
        if "StateVector" in supported_result_types:
            return ["#pragma braket result state_vector"]
        elif "DensityMatrix" in supported_result_types:
            return [f"#pragma braket result density_matrix {targets}"]
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    if observable is None:
        if isinstance(measurement, qp.measurements.CountsMP) and not measurement.all_outcomes:
            return [
                f"#pragma braket result sample z({properties.format_target(qubit)})"
                for qubit in qubits
            ]
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    observable = flatten_observable(observable)

    if isinstance(observable, qp.ops.LinearCombination):
        if isinstance(measurement, qp.measurements.ExpectationMP):
            return [
                f"#pragma braket result expectation {_observable_to_openqasm(op, int, properties)}"
                for op in observable.terms()[1]
            ]
        raise NotImplementedError(
            f"Return type {type(measurement)} unsupported for LinearCombination"
        )

    braket_observable = _observable_to_openqasm(observable, int, properties)
    if isinstance(measurement, qp.measurements.ExpectationMP):
        return [f"#pragma braket result expectation {braket_observable}"]
    if isinstance(measurement, qp.measurements.VarianceMP):
        return [f"#pragma braket result variance {braket_observable}"]
    if isinstance(measurement, qp.measurements.CountsMP) and not measurement.all_outcomes:
        return [f"#pragma braket result sample {braket_observable}"]
    if isinstance(measurement, qp.measurements.SampleMP):
        return [f"#pragma braket result sample {braket_observable}"]
    raise NotImplementedError(f"Unsupported return type: {type(measurement)}")


def _adjoint_gradient_to_openqasm(
    measurements: Sequence[MeasurementProcess],
    parameters: Sequence[str],
    supported_result_types: frozenset[str],
    wire_map: Mapping[Hashable, int],
    properties: OpenQASMSerializationProperties,
) -> str:
    if len(measurements) != 1:
        raise ValueError(
            f"Braket can only compute gradients for circuits with a single expectation"
            f" observable, not {len(measurements)} observables."
        )
    measurement = measurements[0]
    observable = flatten_observable(measurement.obs)
    if not isinstance(measurement, qp.measurements.ExpectationMP):
        raise TypeError(
            f"Braket can only compute gradients for circuits with a single expectation"
            f" observable, not a {type(measurement)} measurement."
        )
    if "AdjointGradient" not in supported_result_types:
        raise NotImplementedError("Unsupported return type: AdjointGradient")
    braket_observable = _observable_to_openqasm(observable, wire_map.__getitem__, properties)
    return (
        "#pragma braket result adjoint_gradient "
        f"expectation({braket_observable}) {', '.join(parameters) or 'all'}"
    )


def _observable_to_openqasm(
    observable: Operator,
    qubit: Callable[[Hashable], int],
    properties: OpenQASMSerializationProperties,
) -> str:
    terms = _observable_terms(observable, qubit, properties)
    serialized = " + ".join(
        f"{coefficient} * {' @ '.join(factors)}" if coefficient != 1 else " @ ".join(factors)
        for coefficient, factors in terms
    )
    return serialized.replace("+ -", "- ") if len(terms) > 1 else serialized


def _observable_terms(
    observable: Operator,
    qubit: Callable[[Hashable], int],
    properties: OpenQASMSerializationProperties,
) -> list[tuple[Any, list[str]]]:
    match observable:
        case qp.Identity(wires=wires):
            return [(1, [f"i({properties.format_target(qubit(wire))})" for wire in wires])]
        case qp.PauliX(wires=wires):
            return [(1, [f"x({properties.format_target(qubit(wires[0]))})"])]
        case qp.PauliY(wires=wires):
            return [(1, [f"y({properties.format_target(qubit(wires[0]))})"])]
        case qp.PauliZ(wires=wires):
            return [(1, [f"z({properties.format_target(qubit(wires[0]))})"])]
        case qp.Hadamard(wires=wires):
            return [(1, [f"h({properties.format_target(qubit(wires[0]))})"])]
        case qp.Hermitian(wires=wires):
            return [(1, [_hermitian_to_openqasm(qp.matrix(observable), wires, qubit, properties)])]
        case qp.Projector(parameters=parameters, wires=wires):
            state = parameters[0]
            if len(state) == len(wires):  # state is a basis state
                return [
                    (
                        1,
                        [
                            _hermitian_to_openqasm(_one if b else _zero, [w], qubit, properties)
                            for b, w in zip(state, wires)
                        ],
                    )
                ]
            # state is a state vector
            return [(1, [_hermitian_to_openqasm(observable.matrix(), wires, qubit, properties)])]
        case qp.ops.Prod(operands=operands):
            coefficient, factors = 1, []
            for operand in operands:
                [(operand_coefficient, operand_factors)] = _observable_terms(
                    operand, qubit, properties
                )
                coefficient *= operand_coefficient
                factors.extend(operand_factors)
            return [(coefficient, factors)]
        case qp.ops.SProd(base=base, scalar=scalar):
            return [
                (coefficient * scalar, factors)
                for coefficient, factors in _observable_terms(base, qubit, properties)
            ]
        case qp.ops.Sum(operands=operands):
            return [
                term
                for operand in operands
                for term in _observable_terms(operand, qubit, properties)
            ]
        case _:
            raise qp.exceptions.DeviceError(f"Unsupported observable: {type(observable)}")


def _hermitian_to_openqasm(
    matrix,
    wires: Sequence[Hashable],
    qubit: Callable[[Hashable], int],
    properties: OpenQASMSerializationProperties,
) -> str:
    rows = onp.asarray(matrix, dtype=complex).tolist()
    serialized = str([[f"{complex(element)}" for element in row] for row in rows])
    for old, new in (("(", ""), (")", ""), ("'", ""), ("j", "im")):
        serialized = serialized.replace(old, new)
    targets = ", ".join(properties.format_target(qubit(wire)) for wire in wires)
    return f"hermitian({serialized}) {targets}"


def _operation_statements(
    operations: Iterable[tuple[Operation, Sequence[str | None]]],
    wire_map: Mapping[Hashable, int],
    properties: OpenQASMSerializationProperties,
    qubits: dict[int, None],
    parameters: set[str],
):
    for operation, param_names in operations:
        targets = [wire_map[wire] for wire in operation.wires]
        qubits.update(dict.fromkeys(targets))
        parameters.update(_free_parameters(operation, param_names))
        yield _operation_to_openqasm(operation, param_names, targets, properties)


def _operation_to_openqasm(
    operation: Operation,
    param_names: Sequence[str | None],
    targets: Sequence[int],
    properties: OpenQASMSerializationProperties,
) -> str:
    try:
        name = _PENNYLANE_TO_OPENQASM_OPERATIONS[operation.name]
    except KeyError:
        raise NotImplementedError(
            f"Braket PennyLane plugin does not support operation {operation.name}."
        ) from None
    parameters = _parameters(operation, param_names)
    qubits = ", ".join(properties.format_target(int(target)) for target in targets)
    match operation:
        case qp.Identity():
            return "\n".join(f"i {properties.format_target(int(target))};" for target in targets)
        case qp.QubitUnitary():
            return f"#pragma braket unitary({_format_matrix(parameters[0])}) {qubits}"
        case qp.QubitChannel():
            matrices = ", ".join(_format_matrix(matrix) for matrix in parameters)
            return f"#pragma braket noise kraus({matrices}) {qubits}"
        case qp.GeneralizedAmplitudeDamping():
            parameters = [parameters[0], 1 - parameters[1]]
        case MS():
            parameters = [*parameters, np.pi / 2]
    if isinstance(operation, qp.operation.Channel):
        return f"#pragma braket noise {name}({', '.join(map(str, parameters))}) {qubits}"
    if parameters:
        return f"{name}({', '.join(map(_format_angle, parameters))}) {qubits};"
    return f"{name} {qubits};"


def _parameters(operation: Operation, param_names: Sequence[str | None]) -> list:
    return [
        name if name is not None else value.numpy() if isinstance(value, np.tensor) else value
        for name, value in zip(param_names, operation.parameters, strict=True)
    ]


def _free_parameters(operation: Operation, param_names: Sequence[str | None]):
    for name, value in zip(param_names, operation.parameters, strict=True):
        if name is not None:
            yield name
        elif isinstance(value, FreeParameterExpression) and isinstance(value.expression, Expr):
            yield from (symbol.name for symbol in value.expression.free_symbols)


def _format_angle(angle: str | FreeParameterExpression | float) -> str:
    if isinstance(angle, str):
        return angle
    return str(angle) if isinstance(angle, FreeParameterExpression) else str(float(angle))


def _format_matrix(matrix) -> str:
    return onp.array2string(
        onp.asarray(matrix, dtype=complex),
        separator=", ",
        formatter={"all": format_complex},
        threshold=float("inf"),
    ).replace("\n", "")


def translate_operation(
    operation: Operation,
    use_unique_params: bool = False,
    param_names: list[str] | None = None,
    *args,
    **kwargs,
) -> Gate:
    """Translates a PennyLane ``Operation`` into the corresponding Braket ``Gate``.

    Args:
        operation (Operation): The PennyLane ``Operation`` to translate
        use_unique_params (bool): If true, numeric parameters in the resulting operation will be
        replaced with FreeParameter objects (with names corresponding to param_names). Non-numeric
        parameters will be skipped.
        param_names (list[str] | None): A list of parameter names to be supplied
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
            # In some cases, like the unitary gate or qp.QubitChannel (Kraus noise), these
            # parameter can be matrices. Braket only supports parameterization of numeric parameters
            # (so far, these are all angle parameters), so non-numeric parameters are handled
            # separately.
            if param_name is not None:
                new_param = FreeParameter(param_name)
            elif isinstance(param, qp.numpy.tensor):
                new_param = param.numpy()
            else:
                new_param = param
            parameters.append(new_param)
    else:
        parameters = [
            p.numpy() if isinstance(p, qp.numpy.tensor) else p for p in operation.parameters
        ]
    device = kwargs.get("device", None)
    return _translate_operation(operation, parameters, device)


@singledispatch
def _translate_operation(operation: Operation, _parameters, device=None) -> Gate:
    raise NotImplementedError(
        f"Braket PennyLane plugin does not support operation {operation.name}."
    )


@_translate_operation.register
def _(_: qp.Identity, _parameters, device=None):
    return gates.I()


@_translate_operation.register
def _(_: qp.Hadamard, _parameters, device=None):
    return gates.H()


@_translate_operation.register
def _(_: qp.PauliX, _parameters, device=None):
    return gates.X()


@_translate_operation.register
def _(_: qp.PauliY, _parameters, device=None):
    return gates.Y()


@_translate_operation.register
def _(_: qp.PauliZ, _parameters, device=None):
    return gates.Z()


@_translate_operation.register
def _(_: qp.ECR, _parameters, device=None):
    return gates.ECR()


@_translate_operation.register
def _(s: qp.S, _parameters, device=None):
    return gates.S()


@_translate_operation.register
def _(sx: qp.SX, _parameters, device=None):
    return gates.V()


@_translate_operation.register
def _(t: qp.T, _parameters, device=None):
    return gates.T()


@_translate_operation.register
def _(_: qp.CNOT, _parameters, device=None):
    return gates.CNot()


@_translate_operation.register
def _(_: qp.CY, _parameters, device=None):
    return gates.CY()


@_translate_operation.register
def _(_: qp.CZ, _parameters, device=None):
    return gates.CZ()


@_translate_operation.register
def _(_: qp.SWAP, _parameters, device=None):
    return gates.Swap()


@_translate_operation.register
def _(_: qp.CSWAP, _parameters, device=None):
    return gates.CSwap()


@_translate_operation.register
def _(_: qp.Toffoli, _parameters, device=None):
    return gates.CCNot()


@_translate_operation.register
def _(rx: qp.RX, parameters, device=None):
    phi = parameters[0]
    return gates.Rx(phi)


@_translate_operation.register
def _(ry: qp.RY, parameters, device=None):
    phi = parameters[0]
    return gates.Ry(phi)


@_translate_operation.register
def _(rz: qp.RZ, parameters, device=None):
    phi = parameters[0]
    return gates.Rz(phi)


@_translate_operation.register
def _(phase_shift: qp.PhaseShift, parameters, device=None):
    phi = parameters[0]
    return gates.PhaseShift(phi)


@_translate_operation.register
def _(qubit_unitary: qp.QubitUnitary, parameters, device=None):
    U = np.asarray(parameters[0])
    return gates.Unitary(U)


@_translate_operation.register
def _(_: qp.AmplitudeDamping, parameters, device=None):
    gamma = parameters[0]
    return noises.AmplitudeDamping(gamma)


@_translate_operation.register
def _(_: qp.GeneralizedAmplitudeDamping, parameters, device=None):
    gamma = parameters[0]
    probability = 1 - parameters[1]
    return noises.GeneralizedAmplitudeDamping(probability=probability, gamma=gamma)


@_translate_operation.register
def _(_: qp.PhaseDamping, parameters, device=None):
    gamma = parameters[0]
    return noises.PhaseDamping(gamma)


@_translate_operation.register
def _(_: qp.DepolarizingChannel, parameters, device=None):
    probability = parameters[0]
    return noises.Depolarizing(probability)


@_translate_operation.register
def _(_: qp.BitFlip, parameters, device=None):
    probability = parameters[0]
    return noises.BitFlip(probability)


@_translate_operation.register
def _(_: qp.PhaseFlip, parameters, device=None):
    probability = parameters[0]
    return noises.PhaseFlip(probability)


@_translate_operation.register
def _(_: qp.QubitChannel, parameters, device=None):
    K_list = [np.asarray(matrix) for matrix in parameters]
    return noises.Kraus(K_list)


@_translate_operation.register
def _(c_phase_shift: qp.ControlledPhaseShift, parameters, device=None):
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
def _(iswap: qp.ISWAP, _parameters, device=None):
    return gates.ISwap()


@_translate_operation.register
def _(pswap: PSWAP, parameters, device=None):
    phi = parameters[0]
    return gates.PSwap(phi)


@_translate_operation.register
def _(xy: qp.IsingXY, parameters, device=None):
    phi = parameters[0]
    return gates.XY(phi)


@_translate_operation.register
def _(xx: qp.IsingXX, parameters, device=None):
    phi = parameters[0]
    return gates.XX(phi)


@_translate_operation.register
def _(yy: qp.IsingYY, parameters, device=None):
    phi = parameters[0]
    return gates.YY(phi)


@_translate_operation.register
def _(zz: qp.IsingZZ, parameters, device=None):
    phi = parameters[0]
    return gates.ZZ(phi)


@_translate_operation.register
def _(_prx: PRx, parameters, device=None):
    theta = parameters[0]
    phi = parameters[1]
    return gates.PRx(theta, phi)


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
def _(adjoint: qp.ops.Adjoint, parameters, device=None):
    if isinstance(adjoint.base, qp.ISWAP):
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
    pulse_wires = qp.wires.Wires.all_wires([pulse.wires for pulse in pulses])
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
            if pulse.amplitude == qp.pulse.constant:
                amplitude = complex(op.parameters[callable_index])
                callable_index += 1
                waveform = lambda dt: ConstantWaveform(pulse_length, amplitude)  # noqa: B023
            else:
                amplitude = partial(pulse.amplitude, op.parameters[callable_index])
                callable_index += 1
                waveform = lambda dt: ArbitraryWaveform(
                    # Calculate amplitude for each time step and normalize
                    onp.array([amplitude(t) for t in np.arange(start, end + dt, dt)])  # noqa: B023
                )
        else:
            waveform = lambda dt: ConstantWaveform(pulse_length, pulse.amplitude)  # noqa: B023

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


def supported_observables(device: Device, shots: int) -> frozenset[str]:
    action = device.properties.action[DeviceActionType.OPENQASM]
    braket_observables = set.union(
        *[set(r.observables) for r in action.supportedResultTypes if r.observables]
    )
    supported = frozenset.union(
        *[_BRAKET_TO_PENNYLANE_OBSERVABLES[braket_obs] for braket_obs in braket_observables],
    )
    supported |= {"Prod", "SProd"}
    return supported if shots else supported | {"Sum", "LinearCombination"}


def get_adjoint_gradient_result_type(
    observable: Operator,
    targets: list[int] | list[list[int]],
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
    measurement: MeasurementProcess,
    targets: list[int] | None,
    supported_result_types: frozenset[str],
) -> ResultType | tuple[ResultType, ...]:
    """Translates a PennyLane ``MeasurementProcess`` into the corresponding Braket ``ResultType``.

    Args:
        measurement (MeasurementProcess): The PennyLane ``MeasurementProcess`` to translate
        targets (list[int] | None): The target wires of the observable using a consecutive
            integer wire ordering
        supported_result_types (frozenset[str]): Braket result types supported by the Braket device

    Returns:
        ResultType | tuple[ResultType, ...]: The Braket result type corresponding to
        the given observable; if the observable type has multiple terms, for example a Sum,
        then this will return a result type for each term.
    """
    targets = targets or measurement.wires.tolist()
    observable = measurement.obs

    if isinstance(measurement, qp.measurements.ProbabilityMP):
        return Probability(targets)

    if isinstance(measurement, qp.measurements.DensityMatrixMP):
        if "DensityMatrix" in supported_result_types:
            return DensityMatrix(targets)
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    if isinstance(measurement, qp.measurements.StateMP):
        if "StateVector" in supported_result_types:
            return StateVector()
        elif "DensityMatrix" in supported_result_types:
            return DensityMatrix(targets)
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    if observable is None:
        if isinstance(measurement, qp.measurements.CountsMP) and not measurement.all_outcomes:
            return tuple(Sample(observables.Z(target)) for target in targets or measurement.wires)
        raise NotImplementedError(f"Unsupported return type: {type(measurement)}")

    observable = flatten_observable(observable)

    if isinstance(observable, qp.ops.LinearCombination):
        if isinstance(measurement, qp.measurements.ExpectationMP):
            return tuple(Expectation(_translate_observable(op)) for op in observable.terms()[1])
        raise NotImplementedError(
            f"Return type {type(measurement)} unsupported for LinearCombination"
        )

    braket_observable = _translate_observable(observable)
    if isinstance(measurement, qp.measurements.ExpectationMP):
        return Expectation(braket_observable)
    if isinstance(measurement, qp.measurements.VarianceMP):
        return Variance(braket_observable)
    if isinstance(measurement, qp.measurements.CountsMP) and not measurement.all_outcomes:
        return Sample(braket_observable)
    if isinstance(measurement, qp.measurements.SampleMP):
        return Sample(braket_observable)
    raise NotImplementedError(f"Unsupported return type: {type(measurement)}")


def flatten_observable(observable):
    if isinstance(observable, (qp.ops.CompositeOp, qp.ops.SProd)):
        simplified = qp.ops.LinearCombination(*observable.terms()).simplify()
        coeffs, _ = simplified.terms()
        if len(coeffs) > 1 or coeffs[0] != 1:
            return simplified
    return observable


_zero = np.array([[1, 0], [0, 0]])
_one = np.array([[0, 0], [0, 1]])


@singledispatch
def _translate_observable(observable):
    match observable:
        case qp.Identity(wires=wires):
            return (
                observables.I(wires[0])
                if len(wires) == 1
                else observables.TensorProduct([observables.I(wire) for wire in wires])
            )
        case qp.PauliX(wires=wires):
            return observables.X(wires[0])
        case qp.PauliY(wires=wires):
            return observables.Y(wires[0])
        case qp.PauliZ(wires=wires):
            return observables.Z(wires[0])
        case qp.Hadamard(wires=wires):
            return observables.H(wires[0])
        case qp.Hermitian(wires=wires):
            return observables.Hermitian(qp.matrix(observable), targets=wires)
        case qp.Projector(parameters=parameters, wires=wires):
            state = parameters[0]
            if len(state) == len(wires):  # state is a basis state
                products = [_one if b else _zero for b in state]
                hermitians = [
                    observables.Hermitian(p, targets=[w]) for p, w in zip(products, wires)
                ]
                return observables.TensorProduct(hermitians)
            # state is a state vector
            return observables.Hermitian(observable.matrix(), targets=wires)
        case qp.ops.Prod(operands=operands):
            return reduce(
                lambda x, y: x @ y, [_translate_observable(factor) for factor in operands]
            )
        case qp.ops.SProd(base=base, scalar=scalar):
            return scalar * _translate_observable(base)
        case qp.ops.Sum(operands=operands):
            return reduce(
                lambda x, y: x + y, [_translate_observable(operator) for operator in operands]
            )
        case _:
            raise qp.exceptions.DeviceError(f"Unsupported observable: {type(observable)}")


def translate_result(
    braket_result: GateModelQuantumTaskResult,
    measurement: MeasurementProcess,
    targets: list[int] | None,
    supported_result_types: frozenset[str],
) -> Any:
    """Translates a Braket result into the corresponding PennyLane return type value.

    Args:
        braket_result (GateModelQuantumTaskResult): The Braket result to translate.
        measurement (MeasurementProcess): The PennyLane measurement process associated with the
            result.
        targets (list[int] | None): The qubits in the result.
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

    targets = targets or measurement.wires.tolist()
    if (
        isinstance(measurement, qp.measurements.CountsMP)
        and not measurement.all_outcomes
        and observable is None
    ):
        if targets:
            new_dict = {}
            for key, value in braket_result.measurement_counts.items():
                new_key = "".join(key[i] for i in targets)
                if new_key not in new_dict:
                    new_dict[new_key] = 0
                new_dict[new_key] += value
            return new_dict

        return dict(braket_result.measurement_counts)

    translated = translate_result_type(measurement, targets, supported_result_types)
    observable = flatten_observable(observable)
    if isinstance(observable, qp.ops.LinearCombination):
        coeffs, _ = observable.terms()
        return sum(
            coeff * braket_result.get_value_by_result_type(result_type)
            for coeff, result_type in zip(coeffs, translated)
        )
    elif isinstance(measurement, qp.measurements.CountsMP) and not measurement.all_outcomes:
        return dict(Counter(braket_result.get_value_by_result_type(translated)))
    else:
        return braket_result.get_value_by_result_type(translated)
