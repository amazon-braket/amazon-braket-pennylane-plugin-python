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
from braket.circuits.result_types import Expectation, Probability, Sample, Variance
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
    braket_result_calculated = translate_result_type(obs, [0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result using translate_result_type"""
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    braket_result_calculated = translate_result_type(mp, [0])

    braket_result = Probability([0])

    assert braket_result == braket_result_calculated


def test_translate_result_type_unsupported_return():
    """Tests if a NotImplementedError is raised by translate_result_type for an unknown
    return_type"""
    obs = qml.Hadamard(0)
    obs.return_type = None

    with pytest.raises(NotImplementedError, match="Unsupported return type"):
        translate_result_type(obs, [0])


def test_translate_result_type_unsupported_obs():
    """Tests if a TypeError is raised by translate_result_type for an unknown observable"""
    obs = qml.S(wires=0)
    obs.return_type = None

    with pytest.raises(TypeError, match="Unsupported observable"):
        translate_result_type(obs, [0])
