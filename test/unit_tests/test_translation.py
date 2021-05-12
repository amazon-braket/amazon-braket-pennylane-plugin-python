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
import pennylane as qml
import pytest
from braket.circuits import observables
from braket.circuits.result_types import (
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from pennylane.measure import MeasurementProcess
from pennylane.operation import ObservableReturnTypes
from pennylane.wires import Wires

from braket.pennylane_plugin.translation import translate_result_type

pl_return_types = [
    ObservableReturnTypes.Expectation,
    ObservableReturnTypes.Variance,
    ObservableReturnTypes.Sample,
]
braket_results = [
    Expectation(observables.H(), [0]),
    Variance(observables.H(), [0]),
    Sample(observables.H(), [0]),
]


@pytest.mark.parametrize("return_type, braket_result", zip(pl_return_types, braket_results))
def test_translate_result_type_observable(return_type, braket_result):
    """Tests if a PennyLane return type that involves an observable is successfully converted into a
    Braket result using translate_result_type"""
    obs = qml.Hadamard(0)
    obs.return_type = return_type
    braket_result_calculated = translate_result_type(obs, [0], frozenset())

    assert braket_result == braket_result_calculated


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    braket_result_calculated = translate_result_type(mp, [0], frozenset())

    braket_result = Probability([0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_state_vector():
    """Tests if a PennyLane state vector return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_calculated = translate_result_type(
        mp, [], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result = StateVector()

    assert braket_result == braket_result_calculated


def test_translate_result_type_density_matrix():
    """Tests if a PennyLane density matrix return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    braket_result_calculated = translate_result_type(mp, [], frozenset(["DensityMatrix"]))

    braket_result = DensityMatrix()

    assert braket_result == braket_result_calculated


def test_translate_result_type_density_matrix_partial():
    """Tests if a PennyLane partial density matrix return type is successfully converted into a
    Braket result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.State, wires=[0])
    braket_result_calculated = translate_result_type(
        mp, [0], frozenset(["StateVector", "DensityMatrix"])
    )

    braket_result = DensityMatrix([0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_state_unimplemented():
    """Tests if a NotImplementedError is raised by translate_result_type when a PennyLane state
    return type is converted while not supported by the device"""
    mp = MeasurementProcess(ObservableReturnTypes.State)
    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(mp, [0], frozenset())


def test_translate_result_type_unsupported_return():
    """Tests if a NotImplementedError is raised by translate_result_type for an unknown
    return_type"""
    obs = qml.Hadamard(0)
    obs.return_type = None

    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(obs, [0], frozenset())


def test_translate_result_type_unsupported_obs():
    """Tests if a TypeError is raised by translate_result_type for an unknown observable"""
    obs = qml.S(wires=0)
    obs.return_type = None

    with pytest.raises(TypeError, match="Unsupported observable"):
        translate_result_type(obs, [0], frozenset())
