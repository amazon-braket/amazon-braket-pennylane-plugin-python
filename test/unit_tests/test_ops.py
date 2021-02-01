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

import math

import numpy as np
import pennylane as qml
import pytest
from braket.circuits import gates, observables
from braket.circuits.result_types import Expectation, Probability, Sample, Variance
from pennylane.operation import ObservableReturnTypes
from pennylane.tape.measure import MeasurementProcess
from pennylane.wires import Wires

from braket.pennylane_plugin import (
    ISWAP,
    PSWAP,
    XX,
    XY,
    YY,
    ZZ,
    CPhaseShift,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
)
from braket.pennylane_plugin.translation import translate_result_type

testdata = [
    (CPhaseShift, gates.CPhaseShift, [math.pi]),
    (CPhaseShift00, gates.CPhaseShift00, [math.pi]),
    (CPhaseShift01, gates.CPhaseShift01, [math.pi]),
    (CPhaseShift10, gates.CPhaseShift10, [math.pi]),
    (ISWAP, gates.ISwap, []),
    (PSWAP, gates.PSwap, [math.pi]),
    (XY, gates.XY, [math.pi]),
    (XX, gates.XX, [math.pi]),
    (YY, gates.YY, [math.pi]),
    (ZZ, gates.ZZ, [math.pi]),
]


@pytest.mark.parametrize("pl_op, braket_gate, params", testdata)
def test_matrices(pl_op, braket_gate, params):
    """Tests that the matrices of the custom operations are correct."""
    assert np.allclose(pl_op._matrix(*params), braket_gate(*params).to_matrix())


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
    Braket result"""
    obs = qml.Hadamard(0)
    obs.return_type = return_type
    braket_result_calculated = translate_result_type(obs)

    assert braket_result == braket_result_calculated


def test_translate_result_type_probs():
    """Tests if a PennyLane probability return type is successfully converted into a Braket
    result."""
    mp = MeasurementProcess(ObservableReturnTypes.Probability, wires=Wires([0]))
    braket_result_calculated = translate_result_type(mp)

    braket_result = Probability([0])

    assert braket_result == braket_result_calculated
