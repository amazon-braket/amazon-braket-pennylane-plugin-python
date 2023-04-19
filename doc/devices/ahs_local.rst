The local AHS device
====================

The local analog hamiltonian simulation (AHS) device of the PennyLane-Braket plugin runs Analogue Hamiltonian Simulation on the local Braket SDK. This
could be either utilizing the processors of your own PC, or those of a `Braket notebook instance <https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-create-notebook.html>`_ hosted on AWS.

This device is useful for small-scale simulations in which the time of sending a job to a remote service would add
an unnecessary overhead. It can also be used for rapid prototyping before running a computation
on a paid-for remote service.


Usage
~~~~~

After the Braket SDK and the plugin are installed you immediately have access to the `local Braket AHS simulator <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html#braket-simulator-ahs-local>`_ in PennyLane.

The local AHS device is not gate-based. Instead, it is compatible with the ``ParametrizedEvolution`` operator from `pulse programming <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`_ in PennyLane.

To instantiate the local Braket simulator, simply use:

.. code-block:: python

    import pennylane as qml
    device_local = qml.device("braket.local.ahs", wires=2)

This device can be used with a QNode within PennyLane. It accepts circuits with a single ``ParametrizedEvolution``
operator based on a ``ParametrizedHamiltonian`` compatible with the simulated hardware. More information about creating
PennyLane operators for AHS can be found in the `PennyLane docs <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`_.

Creating a register
^^^^^^^^^^^^^^^^^^^

The atom register defines where the atoms will be located, which determines the strength of the interaction
between the atoms. Here we define coordinates for the atoms to be placed at (in micrometers), and create a constant
interaction term for the Hamiltonian:

.. code-block:: python

    # number of coordinate pairs must match number of device wires
    coordinates = [[0, 0], [0, 5]]  

    H_interaction = qml.pulse.rydberg_interaction(coordinates)

Creating a drive
^^^^^^^^^^^^^^^^^^^^^^^

We can create a drive with a global component and (positive) local detunings. If the local detunings are time-dependent,
they must all have the same time-dependent envelope, but can have different, positive scaling factors.

.. code-block:: python

    # gaussian amplitude function (qml.pulse.rect enforces 0 at start and end for hardware)
    def amp_fn(p, t):
        f = p[0] * jnp.exp(-(t-p[1])**2/(2*p[2]**2))
        return qml.pulse.rect(f, windows=[0.1, 1.7])(p, t)

    # defining a linear detuning
    def det_fn_global(p, t):
        return p * t

    def det_fn_local(p, t):
        return p * t**2

    # creating a global drive on all wires
    H_global = qml.pulse.rydberg_drive(amplitude=amp_fn, phase=0, detuning=det_fn_global, wires=[0, 1])

    # creating local drives
    # note only local detuning is currently supported, so amplitude and phase are set to 0
    H_local0 = qml.pulse.rydberg_drive(amplitude=0, phase=0, detuning = det_fn_local, wires=[0])
    H_local1 = qml.pulse.rydberg_drive(amplitude=0, phase=0, detuning = det_fn_local, wires=[1])

    # full hamiltonian
    H = H_interaction + H_global + H_local0 + H_local1


Executing an AHS program
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    @qml.qnode(device_local)
    def circuit(params):
        qml.evolve(H)(params, t=1.5)
        return qml.sample()

    # amp_fn expects p to contain 3 parameters
    amp_params = [2.5, 1, 0.3]
    # global_det_fn expects p to be a single parameter
    det_global_params = 0.2
    # each of the local drives take a single parameter for p
    # the detunings have the same shape, but vary by scaling factor p
    local_params1 = 0.5
    local_params2 = 1

When executed, the circuit will perform the computation on the local machine.

>>> circuit([amp_params, det_global_params, local_params1, local_params2])
array([[0, 0],
       [0, 0],
       [0, 0],
       ...,
       [1, 0],
       [1, 0],
       [1, 0]])




