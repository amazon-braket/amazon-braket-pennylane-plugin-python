The remote AHS device
=====================

The remote AHS device of the PennyLane-Braket plugin runs Analogue Hamiltonian Simulation (AHS) on
Amazon Braket's remote service. AHS is a quantum computing paradigm different from gate-based computing.
AHS uses a well-controlled quantum system and tunes its parameters to mimic the dynamics of another quantum
system, the one we aim to study.

The remote service provides access to running AHS on hardware. As AHS devices are not gate-based, they are not
compatible with the standard PennyLane operators. Instead, they are compatible with `pulse programming <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`_ in PennyLane.

Note that pulse programming in PennyLane requires the module ``jax``. You can install jax via: pip install jax==0.4.3 jaxlib==0.4.3

More information about AHS and the capabilities of the hardware can be found in the `Amazon Braket Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html#braket-qpu-partner-quera>`_.

Usage
~~~~~

After the Braket SDK and the plugin are installed, and once you
`sign up for Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`_,
you have access to the remote AHS device in PennyLane.

Instantiate an AWS device that communicates with the hardware like this:

>>> import pennylane as qml
>>> s3 = ("my-bucket", "my-prefix")
>>> device_arn = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
>>> remote_device = qml.device("braket.aws.ahs", device_arn=device_arn, s3_destination_folder=s3, wires=3)

This device can be used with a QNode within PennyLane. It accepts circuits with a single `ParametrizedEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html>`_
operator based on a hardware-compatible `ParametrizedHamiltonian <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedHamiltonian.html>`_.
More information about creating PennyLane operators for AHS can be
found in the `PennyLane docs <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`_.

.. note::
    It is important to keep track of units when specifying electromagnetic pulses for hardware control.
    The frequency and amplitude provided in PennyLane for Rydberg atom systems are expected to be in units of MHz,
    time in microseconds, phase in radians, and distance in micrometers. All of these will be converted to SI units
    internally as needed for upload to the hardware, and frequency will be converted to angular frequency
    (multiplied by :math:`2 \pi`).

    When reading hardware specifications from the Braket backend, bear in mind that all units are SI and frequencies
    are in rad/s. This conversion is done when creating a pulse program for upload, and units in the PennyLane
    functions should follow the conventions specified in the PennyLane docs to ensure correct unit conversion.
    See `rydberg_interaction <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_interaction.html>`_
    and `rydberg_drive <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_drive.html>`_ in
    Pennylane for specification of expected input units, and examples for creating hardware-compatible
    `ParametrizedEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html>`_
    operators in PennyLane.

Creating a register
^^^^^^^^^^^^^^^^^^^

The atom register defines where the atoms will be located, and determines the strength of the interaction
between the atoms. Here we define coordinates for the atoms to be placed at (in micrometers), and create a constant
interaction term for the Hamiltonian:

.. code-block:: python

    # number of coordinate pairs must match number of device wires
    coordinates = [[0, 0], [0, 5], [5, 0]]

    H_interaction = qml.pulse.rydberg_interaction(coordinates)

Creating a global drive
^^^^^^^^^^^^^^^^^^^^^^^

Hardware currently only supports a single global drive pulse applied to all atoms in the register.

Here we define a global drive with time dependent amplitude and detuning, with phase set to 0.

.. code-block:: python

    # gaussian amplitude function (qml.pulse.rect enforces 0 at start and end for hardware)
    def amp_fn(p, t):
        f = p[0] * jnp.exp(-(t-p[1])**2/(2*p[2]**2))
        return qml.pulse.rect(f, windows=[0.1, 1.7])(p, t)


    # defining a linear detuning
    def det_fn(p, t):
        return p * t

    # creating a global drive on all wires
    H_global = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=det_fn, wires=[0, 1, 2])


Creating and executing the circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have the terms describing the atomic interactions and the electromagnetic drive on the atoms, we can create
and execute a circuit to run the pulse program on the hardware:

.. code-block:: python

    @qml.qnode(remote_device)
    def circuit(amp_params, det_params):
        qml.evolve(H_interaction + H_global)([amp_params, det_params], t=1.75)
        return qml.sample()

When executed, the circuit performs the computation on the hardware.

>>> amp_params = [2.5, 1, 0.3]  # amp_fn expects p to contain 3 parameters
>>> det_params = 0.2  # det_fn expects p to be a single parameter
>>> circuit(amp_params, det_params)
array([0.97517033, 0.04904283])

Device options
~~~~~~~~~~~~~~

The default value of the ``shots`` argument is ``Shots.DEFAULT``, resulting in the default number of
shots specified by the remote device being used. For example, a simulator device may default to
analytic mode while a QPU must pick a finite number of shots.

This device is not compatible with analytic mode, so an error will be raised if ``shots=0`` or ``shots=None``.

Supported operations
~~~~~~~~~~~~~~~~~~~~

For Analogue Hamiltonian Simulation, the only supported operation is a `ParametrizedEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html>`_
describing a hardware-compatible electromagnetic pulse.

