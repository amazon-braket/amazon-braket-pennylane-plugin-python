.. _installation:

Setup
#####

Installation
============

The Amazon Braket PennyLane plugin requires both `PennyLane <https://pennylane.readthedocs.io>`_ and the `Amazon Braket Python SDK  <https://github.com/aws/amazon-braket-sdk-python>`_. Instructions for installing the Amazon Braket SDK are included in the README file for the repo.

After you install the Amazon Braket SDK, you can install the plugin from source by cloning this repository and running a pip install command in the root directory of the repository:

.. code-block:: bash

    git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git
    cd amazon-braket-pennylane-plugin-python
    pip install .


Software tests
==============

To ensure that the plugin is working correctly after installation, you can run the test suites. Before you can run them, though, you need to install the test dependencies:
::

    pip install -e "amazon-braket-pennylane-plugin-python[test]"

Run the unit tests with
::

    tox -e unit-tests

and the integ tests with
::

    tox -e integ-tests

To run the integ tests, the ``AWS_PROFILE`` environment variable has to be set to your desired `AWS profile <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html>`_.

.. warning::

    Running the integ tests will create an S3 bucket in your account.

Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

    tox -e docs

The documentation can then be found in the :file:`build/documentation/html/` directory.
