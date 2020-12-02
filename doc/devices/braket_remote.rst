The remote Braket device
========================

The remote device of the PennyLane-Braket plugin runs quantum computations on Amazon Braket's remote service.

The remote service provides access to hardware providers and a high-performance simulator backend.

A list of hardware providers can be found `here <https://aws.amazon.com/braket/hardware-providers/>`_.

The `simulator <https://aws.amazon.com/braket/features/>`_ is particularly suited for executing circuits with high qubit numbers,
where parallelization is needed to handle the exponential scaling notorious for simulating quantum computations on classical hardware.

Usage
~~~~~

After the Braket SDK and the plugin are installed, you immediately have access to the Braket devices in PennyLane.

To instantiate an AWS device that communicates with the Braket service:

>>> import pennylane as qml
>>> s3 = ("my-bucket", "my-prefix")
>>> remote_device = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=2)

In this example, the string ``arn:aws:braket:::device/quantum-simulator/amazon/sv1`` is the ARN that identifies the SV1 device. Other supported devices and their ARNs can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_. Note that the plugin works with digital (qubit) circuit-based devices only.

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

A simple quantum function that returns the expectation value and variance of a measurement and
depends on three classical input parameters would look like:

.. code-block:: python

    @qml.qnode(remote_device)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), var(qml.PauliZ(1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value and variance:

>>> circuit(0.2, 0.1, 0.3)
array([0.97517033, 0.04904283])

Enabling the parallel execution of multiple circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where supported by the backend of the Amazon Braket service, the remote device can be used to execute batches
of quantum circuits in parallel. To unlock this feature, instantiate the device using the ``parralel=True`` argument:

>>> remote_device = qml.device('braket.aws.qubit', device_arn=..., wires=..., s3_destination_folder=..., parallel=True)

The details of the parallelization scheme depend on the PennyLane version you use, as well as your AWS account details.

For example, since PennyLane v0.13 the parallel execution of circuits created during the computation of gradients is supported.
Gradients are usually required for the optimization of quantum circuits.

In other words, just by creating the remote device with the ``parallel=True`` option, your optimization pipelines
should run a lot faster.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The device support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_,
with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``Hermitian`` observable.

The plugin provides the following framework-specific operations for PennyLane, which can be imported
from :mod:`braket.pennylane_plugin.ops <.ops>`:

.. autosummary::
    braket.pennylane_plugin.V
    braket.pennylane_plugin.CY
    braket.pennylane_plugin.CPhaseShift
    braket.pennylane_plugin.CPhaseShift00
    braket.pennylane_plugin.CPhaseShift01
    braket.pennylane_plugin.CPhaseShift10
    braket.pennylane_plugin.ISWAP
    braket.pennylane_plugin.PSWAP
    braket.pennylane_plugin.XY
    braket.pennylane_plugin.XX
    braket.pennylane_plugin.YY
    braket.pennylane_plugin.ZZ


