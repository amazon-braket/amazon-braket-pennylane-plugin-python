.. _installation:

Setup
#####

Installation
============

The Amazon Braket PennyLane plugin requires both `PennyLane <https://pennylane.readthedocs.io>`_ and the `Amazon Braket Python SDK  <https://github.com/aws/braket-python-sdk/tree/stable/latest>`_. You should use only the stable/latest branch of the braket-python-sdk repository. Instructions for installing the Amazon Braket SDK are included in the README file for the repo.

After you install the Amazon Braket SDK, either clone or download the amazon-braket-pennylane-plugin-python repo to your local environment. You must clone or download the repo into a folder in the same virtual environment where you are using the Amazon Braket SDK.

Use the following command to clone the repo.

.. code-block:: bash

    $ git clone https://github.com/aws/amazon-braket-pennylane-plugin-python.git

Note that you must have a valid SSH key created in your local environment that has been added to your GitHub account to clone the repo.

You can also download the repo as a .zip file by using the **Clone or download** button.

After you add the repo to your local environment, install the plugin with the following ``pip`` command:

.. code-block:: bash

    $ pip install -e amazon-braket-pennylane-plugin-python


Software tests
==============

To ensure that the plugin is working correctly after installation, you can run the test suites. Before you can run them, though, you need to install the test dependencies:
::

    $ pip install -e "amazon-braket-pennyLane-plugin-python[test]"

Run the unit tests with
::

    $ tox -e unit-tests

and the integ tests with
::

    $ tox -e integ-tests

To run the integ tests, the ``AWS_PROFILE`` environment variable has to be set to your desired `AWS profile <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html>`_.

.. warning::

    Running the integ tests will create an S3 bucket in your account.

Documentation
=============

To build the HTML documentation, go to the top-level directory and run
::

    $ tox -e docs

The documentation can then be found in the :file:`doc/_build/html/` directory.
