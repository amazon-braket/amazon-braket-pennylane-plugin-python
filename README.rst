PennyLane Braket plugin
#######################

.. image:: https://img.shields.io/badge/code_style-black-000000.svg
    :alt: Code Style: Black
    :target: https://github.com/psf/black

The Amazon Braket PennyLane plugin offers two Amazon Braket quantum devices to work with PennyLane:

* ``braket.aws.qubit`` for running with the Amazon Braket service's quantum devices, both QPUs and simulators
* ``braket.local.qubit`` for running with the Amazon Braket SDK's local simulator

.. header-start-inclusion-marker-do-not-remove

The `Amazon Braket Python SDK <https://github.com/aws/amazon-braket-sdk-python>`_ is an open source
library that provides a framework to interact with quantum computing hardware
devices and simulators through Amazon Braket.

`PennyLane <https://pennylane.readthedocs.io>`_ is a machine learning library for optimization and automatic
differentiation of hybrid quantum-classical computations.

.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `<https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.

Features
========

* Provides two devices to be used with PennyLane: ``braket.aws.qubit`` for running on the Braket service,
  and ``braket.local.qubit`` for running on Braket's local simulator.

* Both devices support most core qubit PennyLane operations.

* All PennyLane observables are supported.

* Provides custom PennyLane operations to cover additional Braket operations: ``ISWAP``, ``PSWAP``, and many more.
  Every custom operation supports analytic differentiation.

* Combines Amazon Braket with PennyLane's automatic differentiation and optimization.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

Before you begin working with the Amazon Braket PennyLane Plugin, make sure 
that you installed or configured the following prerequisites:


* Download and install `Python 3.7.2 <https://www.python.org/downloads/>`_ or greater.
  If you are using Windows, choose the option *Add Python to environment variables* before you begin the installation.

* Make sure that your AWS account is onboarded to Amazon Braket, as per the instructions
  `here <https://github.com/aws/amazon-braket-sdk-python#prerequisites>`_.

* Download and install `PennyLane <https://pennylane.ai/install.html>`_:

  .. code-block:: bash

      pip install pennylane


You can then install the latest release of the PennyLane-Braket plugin as follows:

.. code-block:: bash

    pip install amazon-braket-pennylane-plugin


You can also install the development version from source by cloning this repository and running a 
pip install command in the root directory of the repository:

.. code-block:: bash

    git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git
    cd amazon-braket-pennylane-plugin-python
    pip install .


You can check your currently installed version of ``amazon-braket-pennylane-plugin`` with ``pip show``:

.. code-block:: bash

    pip show amazon-braket-pennylane-plugin


or alternatively from within Python:

.. code-block:: python

    from braket import pennylane_plugin
    pennylane_plugin.__version__

Tests
~~~~~

Make sure to install test dependencies first:

.. code-block:: bash

    pip install -e "amazon-braket-pennylane-plugin-python[test]"

Unit tests
**********

Run the unit tests using:

.. code-block:: bash

    tox -e unit-tests


To run an individual test:

.. code-block:: bash

    tox -e unit-tests -- -k 'your_test'


To run linters and unit tests:

.. code-block:: bash

    tox

Integration tests
*****************

To run the integration tests, set the ``AWS_PROFILE`` as explained in the amazon-braket-sdk-python
`README <https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md>`_:

.. code-block:: bash

    export AWS_PROFILE=Your_Profile_Name


Running the integration tests creates an S3 bucket in the same account as the ``AWS_PROFILE``
with the following naming convention ``amazon-braket-pennylane-plugin-integ-tests-{account_id}``.

Run the integration tests with:

.. code-block:: bash

    tox -e integ-tests

To run an individual integration test:

.. code-block:: bash

    tox -e integ-tests -- -k 'your_test'

Documentation
~~~~~~~~~~~~~

To build the HTML documentation, run:

.. code-block:: bash

  tox -e docs

The documentation can then be found in the ``doc/build/documentation/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built with the plugin.

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/aws/amazon-braket-pennylane-plugin-python
- **Issue Tracker:** https://github.com/aws/amazon-braket-pennylane-plugin-python/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove

.. license-start-inclusion-marker-do-not-remove

License
=======

This project is licensed under the Apache-2.0 License.

.. license-end-inclusion-marker-do-not-remove
