import pennylane as qml
import pennylane.numpy as np
import pytest


seed = 42
np.random.seed(seed)
H = qml.PauliZ(0) @ qml.PauliZ(1)
shots = 10000
wires = 2

class TestShadowExpval:
    """Test shadow_expval computation of expectation values."""

    @pytest.mark.parametrize("dev", [qml.device("default.qubit", wires=wires, shots=shots), qml.device("braket.local.qubit", wires=wires, shots=shots)])
    def test_shadow_expval(self, dev):
        @qml.qnode(dev)
        def shadow_circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.shadow_expval(H, seed=seed)
        
        @qml.qnode(dev)
        def circuit(x):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x, wires=0)
            return qml.expval(H)

        x = np.array(1.2)
        shadow_e = shadow_circuit(x) 
        e = circuit(x)
        assert np.allclose([shadow_e], [e], rtol=0.5) 
