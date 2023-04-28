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

"""Tests that gates are correctly applied in the plugin device"""

import numpy as np
import pennylane as qml
import pytest
from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.pulse.rydberg import rydberg_drive, rydberg_interaction

from braket.pennylane_plugin.ahs_device import BraketAwsAhsDevice, BraketLocalAhsDevice

# =========================================================
coordinates = [[0, 0], [0, 5], [5, 0]]  # in micrometers


def f1(p, t):
    return p * np.sin(t) * (t - 1)


def f2(p, t):
    return p[0] * np.cos(p[1] * t**2)


def amp(p, t):
    return p[0] * np.exp(-((t - p[1]) ** 2) / (2 * p[2] ** 2))


params1 = 1.2
params2 = [3.4, 5.6]
params_amp = [2.5, 0.9, 0.3]

# Hamiltonians to be tested
H_i = rydberg_interaction(coordinates)

HAMILTONIANS_AND_PARAMS = [
    (H_i + rydberg_drive(1, 2, 3, wires=[0, 1, 2]), []),
    (H_i + rydberg_drive(amp, 1, 2, wires=[0, 1, 2]), [params_amp]),
    (H_i + rydberg_drive(2, f1, 2, wires=[0, 1, 2]), [params1]),
    (H_i + rydberg_drive(2, 2, f2, wires=[0, 1, 2]), [params2]),
    (H_i + rydberg_drive(amp, 1, f2, wires=[0, 1, 2]), [params_amp, params2]),
    (H_i + rydberg_drive(4, f2, f1, wires=[0, 1, 2]), [params2, params1]),
    (H_i + rydberg_drive(amp, f2, 4, wires=[0, 1, 2]), [params_amp, params2]),
    (H_i + rydberg_drive(amp, f2, f1, wires=[0, 1, 2]), [params_amp, params2, params1]),
]

HARDWARE_ARN_NRS = ["arn:aws:braket:us-east-1::device/qpu/quera/Aquila"]

ALL_DEVICES = [
    ("braket.local.ahs", None, "RydbergAtomSimulator"),
    ("braket.aws.ahs", "arn:aws:braket:us-east-1::device/qpu/quera/Aquila", "Aquila"),
]


DEVS = [("arn:aws:braket:us-east-1::device/qpu/quera/Aquila", "Aquila")]


@pytest.mark.parametrize("arn_nr, name", DEVS)
def test_initialization(arn_nr, name):
    """Test the device initializes with the expected attributes"""

    dev = BraketAwsAhsDevice(wires=3, shots=11, device_arn=arn_nr)

    assert dev._device.name == name
    assert dev.short_name == "braket.aws.ahs"
    assert dev.shots == 11
    assert dev.ahs_program is None
    assert dev.result is None
    assert dev.pennylane_requires == ">=0.30.0"
    assert dev.operations == {"ParametrizedEvolution"}


class TestDeviceIntegration:
    """Test the devices work correctly from the PennyLane frontend."""

    @pytest.mark.parametrize("shortname, arn_nr, backend_name", ALL_DEVICES)
    def test_load_device(self, shortname, arn_nr, backend_name):
        """Test that the device loads correctly"""
        dev = TestDeviceIntegration._device(shortname, arn_nr, wires=2)
        assert dev.num_wires == 2
        assert dev.shots == 100
        assert dev.short_name == shortname
        assert dev._device.name == backend_name

    def test_args_hardware(self):
        """Test that BraketAwsDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 2 required positional argument"):
            qml.device("braket.aws.ahs")

    def test_args_local(self):
        """Test that BraketLocalDevice requires correct arguments"""
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            qml.device("braket.local.ahs")

    @staticmethod
    def _device(shortname, arn_nr, wires, shots=100):
        if arn_nr:
            return qml.device(shortname, wires=wires, device_arn=arn_nr, shots=shots)
        return qml.device(shortname, wires=wires, shots=shots)


class TestDeviceAttributes:
    """Test application of PennyLane operations on hardware simulators."""

    @pytest.mark.parametrize("shots", [1003, 2])
    def test_setting_shots(self, shots):
        """Test that setting shots changes number of shots from default (100)"""
        dev = BraketLocalAhsDevice(wires=3, shots=shots)
        assert dev.shots == shots

        global_drive = rydberg_drive(2, 1, 2, wires=[0, 1, 2])
        ts = [0.0, 1.75]

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H_i + global_drive, [], ts)
            return qml.sample()

        res = circuit()

        assert len(res) == shots

    def test_local_device_settings(self):
        """Test that device settings dictionary stores the correct keys and values."""
        dev = qml.device("braket.local.ahs", wires=2)
        assert dev.settings == {"interaction_coeff": 862620}


class TestQnodeIntegration:
    """Test integration with the qnode"""

    @pytest.mark.parametrize("H, params", HAMILTONIANS_AND_PARAMS)
    def test_circuit_can_be_called_global_drive(self, H, params):
        """Test that the circuit consisting of a ParametrizedEvolution with a single, global pulse
        runs successfully for all combinations of amplitude, phase and detuning being constants
        or callables
        """

        dev = qml.device("braket.local.ahs", wires=3)

        t = 1.13

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H, params, t)
            return qml.sample()

        circuit()

    def test_qnode_shape_multimeasure(self):
        """Test that a qnode with multiple measurements has the correct shape"""
        dev = qml.device("braket.local.ahs", wires=3)

        H = H_i + rydberg_drive(3, 2, 1, [0, 1, 2])
        measurements = (
            qml.sample(wires=1),
            qml.expval(qml.PauliZ(0)),
            qml.var(qml.PauliZ(0)),
            qml.probs(wires=[1, 2]),
            qml.probs(op=qml.PauliZ(1)),
            qml.sample(qml.PauliZ(2)),
        )

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H, [], 1.8)
            return (
                qml.sample(wires=1),
                qml.expval(qml.PauliZ(0)),
                qml.var(qml.PauliZ(0)),
                qml.probs(wires=[1, 2]),
                qml.probs(op=qml.PauliZ(1)),
                qml.sample(qml.PauliZ(2)),
            )

        res = circuit()
        expected_shape = (mp.shape(dev, qml.measurements.Shots(dev.shots)) for mp in measurements)

        assert len(res) == len(measurements)
        assert all(r.shape == es for r, es in zip(res, expected_shape))

    def test_observable_not_in_z_basis_raises_error(self):
        """Test that asking for the expectation value of an observable not in
        the computational basis raises an error"""

        dev = qml.device("braket.local.ahs", wires=3)

        H = H_i + rydberg_drive(3, 2, 1, [0, 1, 2])

        @qml.qnode(dev)
        def circuit():
            ParametrizedEvolution(H, [], 1.8)
            return qml.expval(qml.PauliX(0))

        with pytest.raises(RuntimeError, match="can only measure in the Z basis"):
            circuit()

    def test_uploaded_ahs_program_and_exact_solution_match(self):
        """Test that the results of a created AHS program run on the AWS Local simulator roughly
        matches the expected exact solution for the operator"""

        # code run to generate the array of exact results:
        # def exact(H, H_obj, t):
        #     psi0 = np.eye(2 ** len(H.wires))[0]
        #     U_exact = jax.scipy.linalg.expm(-1j * t * qml.matrix(H([], 1)))
        #     return psi0 @ U_exact.conj().T @ qml.matrix(H_obj, wire_order=[0, 1, 2]) @ U_exact @ psi0

        # exact_result = np.array([exact(H, H_obj, _t) for _t in t])

        exact_result = np.array(
            [
                0.69484478 - 2.7223859e-09j,
                0.63014996 + 3.5694474e-09j,
                0.59220868 - 3.2304570e-09j,
                0.58607256 - 6.6702283e-10j,
                0.61215657 - 5.6307230e-09j,
                0.66467023 + 1.6380476e-10j,
                0.73498142 - 7.1622348e-09j,
                0.81132668 + 1.4453147e-08j,
                0.88302624 - 8.8402761e-09j,
                0.94071472 - 1.3392033e-09j,
                0.979002 + 5.2728411e-10j,
                0.99656868 + 6.3435528e-09j,
                0.99565047 + 8.7986229e-10j,
                0.98224276 - 5.4241514e-09j,
                0.96249902 - 1.0663380e-11j,
                0.94417256 + 8.6651948e-09j,
                0.9313342 + 8.0441032e-09j,
                0.92794114 + 4.3532098e-09j,
                0.93276292 - 6.2324839e-09j,
                0.94468606 - 6.7148527e-09j,
                0.95891714 - 8.4284810e-09j,
                0.97237718 - 9.5171107e-09j,
                0.98131871 - 4.7603135e-09j,
                0.98455709 + 5.0439941e-10j,
                0.98252237 + 6.6446431e-09j,
                0.97686452 + 1.9499213e-09j,
                0.97114307 + 4.4445185e-09j,
                0.96698064 - 5.5634519e-09j,
                0.96721721 - 4.4257931e-09j,
                0.97005159 - 8.8042862e-09j,
                0.97473019 + 5.1543099e-09j,
                0.97567934 - 5.1634803e-09j,
                0.96977592 + 7.5646946e-09j,
                0.95152903 - 1.4419745e-08j,
                0.91982687 + 1.1928541e-08j,
                0.87425292 - 1.0108283e-08j,
                0.81908154 + 6.1293313e-09j,
                0.76075166 - 1.0051525e-09j,
                0.70762545 + 8.3687919e-09j,
                0.66914564 - 7.9337276e-10j,
                0.65190136 + 6.0308087e-09j,
                0.66135764 + 1.6767658e-09j,
                0.69625801 + 6.2405386e-10j,
                0.75339168 + 7.1651696e-10j,
                0.82255602 - 1.8565143e-09j,
                0.89352876 + 6.2863292e-09j,
                0.95282573 - 2.7223049e-10j,
                0.99048758 - 6.6763408e-09j,
                0.99859977 + 6.6213778e-11j,
                0.97512466 - 1.5285744e-08j,
                0.92333919 - 5.3908789e-09j,
                0.85111868 + 4.5347437e-09j,
                0.77111971 + 4.9047411e-09j,
                0.69567895 + 2.0746311e-09j,
                0.63849753 - 6.3125931e-09j,
                0.60749799 + 5.1294333e-09j,
                0.6085363 - 4.3690260e-09j,
                0.63933951 + 7.7354040e-11j,
                0.69485724 + 9.8120256e-10j,
                0.76429355 + 9.9224595e-10j,
                0.8363694 - 1.9486686e-09j,
                0.89944118 - 7.3339730e-11j,
                0.94451547 + 2.7164773e-10j,
                0.96716231 - 2.0633208e-09j,
                0.96644425 - 3.2646743e-09j,
                0.94764745 - 8.2543077e-11j,
                0.9169814 + 1.6294729e-09j,
                0.88490927 + 1.4793545e-09j,
                0.85850471 + 4.5424886e-09j,
                0.84566396 + 8.6566448e-11j,
                0.8481769 - 6.0154086e-09j,
                0.86651379 + 3.1183298e-09j,
                0.89572513 + 7.1632438e-09j,
                0.93021941 + 2.8748175e-09j,
                0.96245629 + 6.5750343e-09j,
                0.98634839 - 1.4078984e-09j,
                0.99789649 - 8.9403057e-10j,
                0.99549782 - 1.5278900e-09j,
                0.98128837 + 1.2225526e-09j,
                0.95850629 + 1.8635344e-09j,
                0.93322575 - 4.7416080e-09j,
                0.90951031 - 2.3704783e-09j,
                0.89196914 + 1.2524523e-09j,
                0.8813718 - 5.5856391e-09j,
                0.87759912 - 5.6646265e-09j,
                0.87732202 + 3.4028935e-09j,
                0.87696642 - 6.4964927e-09j,
                0.87250876 - 3.0387901e-09j,
                0.86158818 + 1.1833108e-09j,
                0.84383672 + 3.3963752e-09j,
                0.82133746 + 2.2564153e-09j,
                0.79832089 - 3.7487480e-09j,
                0.78037554 - 2.7689142e-09j,
                0.77284342 + 2.2521942e-09j,
                0.77984041 - 6.0057914e-09j,
                0.80240369 - 1.6370114e-09j,
                0.83891118 - 2.4027758e-09j,
                0.88371545 + 5.1036917e-09j,
                0.92971784 - 3.0863305e-09j,
                0.9677825 - 1.4565015e-08j,
                0.99052942 + 8.2257117e-09j,
                0.99202585 + 8.4473717e-09j,
                0.97089159 + 1.4851971e-08j,
                0.92939377 - 8.3187235e-09j,
                0.8739289 + 5.2713118e-09j,
                0.81400841 - 3.0034004e-09j,
                0.75967193 - 4.3109725e-09j,
                0.72110808 - 2.2081752e-09j,
                0.70413637 - 1.9961541e-09j,
                0.71246171 + 1.5180364e-10j,
                0.74274796 - 5.9781702e-10j,
                0.78988826 - 2.1006050e-09j,
                0.84319186 - 2.9964908e-09j,
                0.89320791 - 8.0247664e-10j,
                0.92920345 + 2.9292844e-09j,
                0.94537973 - 2.4045901e-09j,
                0.9385829 - 2.3936622e-09j,
                0.91133964 + 5.3401494e-10j,
                0.87008107 - 3.8251411e-09j,
                0.82389098 + 1.5311378e-09j,
                0.78364879 + 1.9405910e-09j,
                0.75757021 - 8.9238267e-10j,
                0.75296265 + 3.6073555e-09j,
                0.77024126 - 3.6427025e-10j,
                0.80805039 - 3.3535443e-09j,
                0.85804266 - 5.7137992e-09j,
                0.9120273 + 1.1001765e-08j,
                0.95863116 - 1.3252232e-09j,
                0.98974508 + 9.1662442e-11j,
                0.99891704 - 5.8767085e-09j,
                0.98457325 - 1.4554583e-08j,
                0.94961661 - 3.0085552e-09j,
                0.89984506 - 1.6171344e-09j,
                0.84497941 + 5.2012075e-09j,
                0.7932319 - 1.6518183e-09j,
                0.75427103 - 2.6276541e-09j,
                0.73198724 + 2.3092044e-09j,
                0.72974527 - 1.6644990e-09j,
                0.7444284 + 1.3268451e-09j,
                0.77248812 + 4.6409304e-10j,
                0.80631143 - 1.4426076e-10j,
                0.84001815 + 6.9762390e-12j,
                0.86764532 - 1.0151566e-10j,
                0.88610148 - 8.6556667e-10j,
                0.89530641 + 6.8558764e-10j,
                0.89680982 + 3.9257153e-09j,
                0.89541805 - 4.6812015e-10j,
                0.89414483 - 7.5314882e-10j,
                0.89825672 - 4.4961079e-09j,
                0.90795296 + 8.7674685e-09j,
                0.92436141 - 9.9437694e-09j,
            ]
        )

        ahs_local = qml.device("braket.local.ahs", wires=3)

        coordinates = [[0, 0], [0, 5], [5, 0]]

        H_i = qml.pulse.rydberg_interaction(coordinates)

        H = H_i + qml.pulse.rydberg_drive(3, 2, 4, [0, 1, 2])

        H_obj = qml.PauliZ(0)

        @qml.qnode(ahs_local)
        def circuit(t):
            ParametrizedEvolution(H, [], t)
            return qml.expval(H_obj)

        t = np.linspace(0.05, 1.55, 151)

        circuit_result = np.array([circuit(_t) for _t in t])

        # all results are approximately the same
        np.allclose(circuit_result, exact_result, atol=0.08)
