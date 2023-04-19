PennyLane-Braket Plugin
#######################

:Release: |release|

.. image:: _static/pl-braket.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once the Pennylane-Braket plugin is installed, the provided Braket devices can be accessed straight
away in PennyLane, without the need to import any additional packages.

Devices
~~~~~~~

This plugin provides four Braket devices for use with PennyLane - two supporting gate-based computations,
and two supporting Analogue Hamiltonian Simulation (AHS) on a Rydberg atom system:

.. title-card::
    :name: 'braket.aws.qubit'
    :description: Runs gate-based circuits on the remote Amazon Braket service.
    :link: devices/braket_remote.html

.. title-card::
    :name: 'braket.local.qubit'
    :description: Runs gate-based circuits on the Braket SDK's local simulator.
    :link: devices/braket_local.html

.. title-card::
    :name: 'braket.aws.ahs'
    :description: Runs AHS on the remote Amazon Braket service.
    :link: devices/ahs_remote.html

.. title-card::
    :name: 'braket.local.ahs'
    :description: Runs AHS on the Braket SDK's local Rygberg atom simulator.
    :link: devices/ahs_local.html

.. raw:: html

        <div style='clear:both'></div>
        </br>

While the local device helps with small-scale simulations and rapid prototyping, the remote device allows you to run larger simulations or access quantum hardware via the Amazon Braket service.

Tutorials
~~~~~~~~~

To see the PennyLane-Braket plugin in action, you can use any of the qubit-based `demos
from the PennyLane documentation <https://pennylane.ai/qml/demonstrations.html>`_, for example
the tutorial on `qubit rotation <https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html>`_,
and simply replace ``'default.qubit'`` with the ``'braket.local.qubit'`` or the ``'braket.aws.qubit'`` device:

.. code-block:: python

    dev = qml.device('braket.XXX.qubit', [...])

Tutorials that showcase the Braket devices can be found on the  `PennyLane website <https://pennylane.ai/qml/demonstrations.html>`_
and the `Amazon Braket <https://github.com/aws/amazon-braket-examples>`_ examples GitHub repository.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices/braket_remote
   devices/braket_local
   devices/ahs_local
   devices/ahs_remote

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
