import pennylane as qml

bucket = ("braket-output-355401967590", "test1")

dev1 = qml.device("braket.simulator", wires=2, s3=bucket)
dev2 = qml.device("braket.simulator", wires=2, backend="QS3", s3=bucket)

@qml.qnode(dev1)
def circuit(a):
	qml.Hadamard(wires=0)
	qml.CNOT(wires=[0, 1])
	qml.RX(a, wires=1)
	return qml.expval(qml.PauliZ(1))

print(circuit(0.543))


@qml.qnode(dev2)
def circuit(a):
	qml.Hadamard(wires=0)
	qml.CNOT(wires=[0, 1])
	qml.RX(a, wires=1)
	return qml.expval(qml.PauliZ(1))

print(circuit(0.543))