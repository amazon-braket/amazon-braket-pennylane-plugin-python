The remote Braket device
========================

The remote qubit device of the PennyLane-Braket plugin runs gate-based quantum computations on Amazon Braket's remote service.
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

The operations supported by this device vary based on the operations supported by the underlying Braket device. To check
the device's supported operations, run

.. code-block:: python

    dev.operations

In addition to those `provided by PennyLane <https://pennylane.readthedocs.io/en/stable/introduction/operations.html#qubit-operations>`_,
the PennyLane-Braket plugin provides the following framework-specific operations, which can be imported
from :mod:`braket.pennylane_plugin.ops <.ops>`:

.. autosummary::
    braket.pennylane_plugin.CPhaseShift00
    braket.pennylane_plugin.CPhaseShift01
    braket.pennylane_plugin.CPhaseShift10
    braket.pennylane_plugin.PSWAP
    braket.pennylane_plugin.GPi
    braket.pennylane_plugin.GPi2
    braket.pennylane_plugin.MS

Pulse Programming
~~~~~~~~~~~~~~~~~

The PennyLane-Braket plugin provides pulse-level control for the OQC Lucy QPU through PennyLane's :class:`pennylane.pulse.ParametrizedEvolution`
operation. Compatible pulse Hamiltonians can be defined using the :func:`pennylane.pulse.transmon_drive` function and used to create
``ParametrizedEvolution``'s using :func:`pennylane.evolve`:

.. code-block:: python

    duration = 15
    def amp(p, t):
        return qml.pulse.pwc(duration)(p, t)

    dev = qml.device("braket.aws.qubit", wires=8, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")

    drive = qml.pulse.transmon.transmon_drive(amplitude=amp, phase=0, freq=4.8, wires=[0])

    @qml.qnode(dev)
    def circuit(params, t):
        qml.evolve(drive)(params, t)
        return qml.expval(qml.PauliZ(wires=0))

Note that the ``amplitude`` and ``freq`` arguments of ``qml.pulse.transmon_drive`` must be specified in :math:`\text{GHz}`. This will be internally
converted into :math:`\text{rad/s}` for use with the Braket API. The ``phase`` must be specified in radians.

The pulse settings for the device can be obtained using the ``pulse_settings`` property. These settings can be used to describe the transmon
interaction Hamiltonian using :func:`pennylane.pulse.transmon_interaction`:

    .. code-block:: python
        dev = qml.device("braket.aws.qubit", wires=8, device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")
        pulse_settings = dev.pulse_settings
        H = qml.pulse.transmon_interaction(**pulse_settings, coupling=0.02)

By passing ``pulse_settings`` from the remote device to :func:`pennylane.pulse.transmon_interaction`, an ``H`` Hamiltonian term is created using
the constants specific to the hardware. This is relevant for simulating the hardware in PennyLane on the ``default.qubit`` device.

Note that the user must supply coupling coefficients, as these are not available from the hardware backend.

Gradient computation on Braket with a QAOA Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, PennyLane will compute grouping indices for QAOA Hamiltonians and use them to split the Hamiltonian into multiple expectation values. If you wish to use `SV1’s adjoint differentiation capability<https://docs.aws.amazon.com/braket/latest/developerguide/hybrid.html>` when running QAOA from PennyLane, you will need reconstruct the cost Hamiltonian to remove the grouping indices from the cost Hamiltonian, like so:

.. code-block:: python

    cost_h, mixer_h = qml.qaoa.max_clique(g, constrained=False)
    cost_h = qml.Hamiltonian(cost_h.coeffs, cost_h.ops)
