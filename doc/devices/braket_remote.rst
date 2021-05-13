The remote Braket device
========================

The remote device of the PennyLane-Braket plugin runs quantum computations on Amazon Braket's remote service.
The remote service provides access to hardware providers and a high-performance simulator backend.

A list of available quantum devices and their features can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.

Usage
~~~~~

After the Braket SDK and the plugin are installed, and once you
`sign up for Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`_,
you have access to the remote Braket device in PennyLane.

Instantiate an AWS device that communicates with the Braket service like this:

>>> import pennylane as qml
>>> s3 = ("my-bucket", "my-prefix")
>>> remote_device = qml.device("braket.aws.qubit", device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1", s3_destination_folder=s3, wires=2)

In this example, the string ``arn:aws:braket:::device/quantum-simulator/amazon/sv1`` is the ARN that identifies the SV1 device. Other supported devices and their ARNs can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.
Note that the plugin works with digital (qubit) gate-based devices only.

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.

For example:

.. code-block:: python

    @qml.qnode(remote_device)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0)), var(qml.PauliZ(1))

When executed, the circuit performs the computation on the Amazon Braket service.

>>> circuit(0.2, 0.1, 0.3)
array([0.97517033, 0.04904283])

Enabling the parallel execution of multiple circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where supported by the backend of the Amazon Braket service, the remote device can be used to execute multiple
quantum circuits in parallel. To unlock this feature, instantiate the device using the ``parallel=True`` argument:

>>> remote_device = qml.device('braket.aws.qubit', [... ,] parallel=True)

The details of the parallelization scheme depend on the PennyLane version you use, as well as your AWS account specifications.

For example, PennyLane 0.13.0 and higher supports the parallel execution of circuits created during the computation of gradients.
Just by instantiating the remote device with the ``parallel=True`` option, this feature is automatically used and can
lead to significant speedups of your optimization pipeline.

The maximum number of circuits that can be executed in parallel is specified by the ``max_parallel`` argument.

>>> remote_device = qml.device('braket.aws.qubit', [... ,] parallel=True, max_parallel=20)

Make sure that this number is not larger than the maximum number of concurrent tasks allowed for your account on the backend you choose. See the `Braket developer guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-quotas.html>`_ for more details.

The Braket remote device has the capability to retry failed circuit executions, up to 3 times per circuit by default.
You can set a timeout by using the ``poll_timeout_seconds`` argument;
the device will retry circuits that do not complete within the timeout.
A timeout of 30 to 60 seconds is recommended for circuits with fewer than 25 qubits.

Device options
~~~~~~~~~~~~~~

The default value of the ``shots`` argument is ``Shots.DEFAULT``, resulting in the default number of
shots specified by the remote device being used. For example, a simulator device may default to
analytic mode while a QPU must pick a finite number of shots.

Setting ``shots=0`` or ``shots=None`` will cause the device to run in analytic mode. If the device
ARN points to a QPU, analytic mode is not available and an error will be raised.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The device support all PennyLane `operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_,
with the exception of the PennyLane ``QubitUnitary`` and ``Rot`` gates and ``Hermitian`` observable.

The PennyLane-Braket plugin provides the following framework-specific operations for PennyLane, which can be imported
from :mod:`braket.pennylane_plugin.ops <.ops>`:

.. autosummary::
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
    braket.pennylane_plugin.AmplitudeDamping
    braket.pennylane_plugin.GeneralizedAmplitudeDamping
    braket.pennylane_plugin.PhaseDamping
    braket.pennylane_plugin.DepolarizingChannel
    braket.pennylane_plugin.BitFlip
    braket.pennylane_plugin.PhaseFlip
    braket.pennylane_plugin.QubitChannel
