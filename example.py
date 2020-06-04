import pennylane as qml

s3 = ("my-bucket", "my-prefix")

dev_qs2 = qml.device(
	"braket.simulator",
	s3_destination_folder=s3,
	backend="QS1",
	wires=2
)
dev_qpu = qml.device(
	"braket.rigetti",
	s3_destination_folder=s3,
	poll_timeout_seconds=1800,
	shots=10000,
	wires=2
)


@qml.qnode(dev_qs2)
def circuit(a):
	qml.Hadamard(wires=0)
	qml.CNOT(wires=[0, 1])
	qml.RX(a, wires=1)
	return qml.expval(qml.PauliZ(1))


print(circuit(0.543))


@qml.qnode(dev_qpu)
def circuit(a):
	qml.Hadamard(wires=0)
	qml.CNOT(wires=[0, 1])
	qml.RX(a, wires=1)
	return qml.expval(qml.PauliZ(1))


print(circuit(0.543))
