# Changelog

## v1.20.3.post0 (2023-09-14)

### Documentation Changes

 * Replace aws org with amazon-braket in badges

## v1.20.3 (2023-09-13)

### Bug Fixes and Other Changes

 * ops: update codeowner file to amazon-braket/braket-dev

## v1.20.2 (2023-09-12)

### Bug Fixes and Other Changes

 * Extract tf casting method

### Documentation Changes

 * update for amazon-braket org

## v1.20.1 (2023-08-29)

### Bug Fixes and Other Changes

 * build(deps): bump aws-actions/stale-issue-cleanup from 3 to 6

### Documentation Changes

 * Update PL requirement in `BraketQubitDevice`

## v1.20.0 (2023-08-21)

### Features

 * Enable `FreeParameter` creation for trainable parameters

### Bug Fixes and Other Changes

 * Don't create `FreeParameter`s for noise

## v1.19.1 (2023-07-27)

### Bug Fixes and Other Changes

 * docs: update README to recommend using Python 3.8

## v1.19.0 (2023-07-26)

### Features

 * Remove `do_queue` kwarg

### Bug Fixes and Other Changes

 * Replace Universal Analytics tag with GA4 tag in the docs
 * support noise model in batch_execute

## v1.18.0 (2023-07-20)

### Features

 * native mode

## v1.17.4 (2023-07-17)

### Bug Fixes and Other Changes

 * Fix compatibility with the new Projector class

## v1.17.3 (2023-07-13)

### Bug Fixes and Other Changes

 * set do_queue default to None

## v1.17.2 (2023-07-12)

### Bug Fixes and Other Changes

 * Update pennylane version constraint

## v1.17.1 (2023-07-06)

### Bug Fixes and Other Changes

 * constrain tensorflow version

## v1.17.0 (2023-06-30)

### Features

 * arbitrary angle MS gate

## v1.16.1 (2023-06-08)

### Bug Fixes and Other Changes

 * Lower shot count on shadow_expval integ test so it can finish

## v1.16.0 (2023-05-30)

### Features

 * implement support for classical shadows and computing expectation values with them

### Bug Fixes and Other Changes

 * temporarily disabling integ test.
 * Update expval validation

## v1.15.3 (2023-05-24)

### Bug Fixes and Other Changes

 * docs: add a linter to check proper rst formatting and fix up incorrect docs

## v1.15.2.post0 (2023-05-22)

### Testing and Release Infrastructure

 * twine check action

## v1.15.2 (2023-05-15)

### Bug Fixes and Other Changes

 * update the project name for coverage runs and clean the tox env

## v1.15.1 (2023-05-09)

### Bug Fixes and Other Changes

 * Adjoint gradient active return compatibility

## v1.15.0.post0 (2023-05-02)

### Documentation Changes

 * Correct README format for PyPI

## v1.15.0 (2023-05-02)

### Features

 * Add AHS devices

### Bug Fixes and Other Changes

 * Tempfix for failing tests
 * Postprocess jacobian result shape

## v1.14.1 (2023-04-26)

### Bug Fixes and Other Changes

 * test: parallelize test execution for pytest

## v1.14.0 (2023-04-25)

### Features

 * add Python 3.11 support

## v1.13.1 (2023-03-16)

### Bug Fixes and Other Changes

 * Merge pull request #144 from aws/readme-updates
 * documenation: Update README support info

### Testing and Release Infrastructure

 * add dependabot updates for GH actions

## v1.13.0 (2023-03-08)

### Features

 * Add support for Pennylane 0.29.1

## v1.12.2 (2023-03-07)

### Bug Fixes and Other Changes

 * Only include trainable parameters in input

### Documentation Changes

 * Correct supported operations

## v1.12.1 (2023-03-06)

### Bug Fixes and Other Changes

 * pin autoray

## v1.12.0 (2023-03-03)

### Deprecations and Removals

 * deprecate python 3.7

### Bug Fixes and Other Changes

 * Add support for new custom measurements
 * Remove use of in-place inversion. Replace it with `qml.adjoint`.

## v1.11.2.post0 (2023-02-14)

### Testing and Release Infrastructure

 * update github workflows for node12 retirement

## v1.11.2 (2023-02-11)

### Bug Fixes and Other Changes

 * `shots=0` measurements with trainable params

## v1.11.1 (2023-02-09)

### Bug Fixes and Other Changes

 * update: adding build for python 3.10
 * Merge branch 'main' into test310
 * testing python 3.10

## v1.11.0 (2023-02-01)

### Features

 * Braket noise model for density matrix simulators

## v1.10.4 (2023-01-03)

### Bug Fixes and Other Changes

 * Cap pennylane version at v0.27

### Testing and Release Infrastructure

 * Use the new Pennylane Sphinx Theme

## v1.10.3 (2022-12-13)

### Bug Fixes and Other Changes

 * Merge pull request #120 from aws/qaoa_bug_fix
 * fixing linters
 * removed qaoa integ test since it won't build on windows
 * integrated with pl's device.use_grouping to fix a bug with qaoa creating multiple observables

## v1.10.2 (2022-12-08)

### Bug Fixes and Other Changes

 * Only support adjoint gradient if shots=0

## v1.10.1 (2022-12-07)

### Bug Fixes and Other Changes

 * Workaround for np.tensor around observable

## v1.10.0 (2022-12-07)

### Features

 * adjoint gradient

### Bug Fixes and Other Changes

 * remove env variable from integ tests

## v1.9.0 (2022-09-21)

### Features

 * add identity operation

## v1.8.0 (2022-09-16)

### Features

 * IonQ native gates

## v1.7.0 (2022-08-30)

### Features

 * use OpenQASM for device property validation

## v1.6.9 (2022-07-01)

### Bug Fixes and Other Changes

 * Add user agent for Braket interactions

## v1.6.8 (2022-06-22)

### Bug Fixes and Other Changes

 * Update for PennyLane 0.24

### Testing and Release Infrastructure

 * Fix integ test imports

## v1.6.7 (2022-06-09)

### Bug Fixes and Other Changes

 * Merge pull request #100 from aws/grad_update
 * Update deprecated imports for version 0.23

## v1.6.6 (2022-04-26)

### Bug Fixes and Other Changes

 * Merge pull request #97 from aws/obs_types
 * pinning doc req to the same version as the build.
 * removing pinning from doc requirement.
 * updating pennylane req for docs.

## v1.6.5 (2022-04-20)

### Bug Fixes and Other Changes

 * unit test for ECR gate

## v1.6.4 (2022-04-18)

### Bug Fixes and Other Changes

 * update: fix ECR definition

## v1.6.3 (2022-04-06)

### Bug Fixes and Other Changes

 * Merge pull request #92 from antalszava/remove_qhack
 * Merge branch 'main' into remove_qhack
 * Merge branch 'main' into remove_qhack
 * Merge branch 'main' into remove_qhack

## v1.6.2 (2022-03-28)

### Bug Fixes and Other Changes

 * pennylane 0.22 breaking changes

## v1.6.1 (2022-03-24)

### Bug Fixes and Other Changes

 * integ tests for TF, Torch for PL>0.20

## v1.6.0 (2022-03-17)

### Features

 * add ECR gate

## v1.5.7 (2022-03-14)

### Bug Fixes and Other Changes

 * Pin pennylane version

## v1.5.6 (2022-02-15)

### Bug Fixes and Other Changes

 * Merge pull request #86 from aws/remove-pl-bucket
 * make s3_bucket optional

## v1.5.5 (2022-02-09)

### Bug Fixes and Other Changes

 * Fix device tracker integ test for PennyLane 0.21
 * Merge branch 'main' into braket-tracker
 * Merge pull request #85 from aws/update-format
 * fix formatting
 * Add check for older PL using batches differently
 * Merge branch 'main' into braket-tracker

## v1.5.4 (2022-01-28)

### Bug Fixes and Other Changes

 * Merge pull request #79 from aws/braket-tracker
 * Merge branch 'main' into braket-tracker
 * Add comment on which devices run tracking test
 * Remove extra print statements

## v1.5.3 (2021-12-16)

### Bug Fixes and Other Changes

 * integ tests for PL-0.20

## v1.5.2 (2021-09-30)

### Bug Fixes and Other Changes

 * remove minified version of jquery and bootstrap

## v1.5.1 (2021-09-27)

### Bug Fixes and Other Changes

 * Merge pull request #75 from albi3ro/add-tracker
 * Merge branch 'main' into add-tracker
 * fix format errors
 * Update test/unit_tests/test_braket_device.py

## v1.5.0 (2021-09-24)

### Features

 * Support for Hamiltonians

## v1.4.2 (2021-09-03)

### Bug Fixes and Other Changes

 * Remove YY gate import from integ tests
 * Add IsingYY support

## v1.4.1 (2021-08-06)

### Bug Fixes and Other Changes

 * correct operation names
 * update: Make supported operations device specific

## v1.4.0 (2021-07-13)

### Features

 * Support `Projector` observable

## v1.3.0 (2021-05-24)

### Features

 * Support for density matrix simulators

### Testing and Release Infrastructure

 * Use GitHub source for tox tests

## v1.2.1 (2021-05-13)

### Bug Fixes and Other Changes

 * cache supported ops in the device
 * Merge pull request #68 from antalszava/patch-2
 * Update setup.py to ease PennyLane pinning

### Documentation Changes

 * Correct supported gates

## v1.2.0 (2021-05-11)

### Features

 * Bring functionality in line with PennyLane v0.15.0 release

## v1.1.1 (2021-05-04)

### Bug Fixes and Other Changes

 * Correct parameter-shift rules

## v1.1.0.post2 (2021-04-26)

### Testing and Release Infrastructure

 * Crank up coverage to 100%

## v1.1.0.post1 (2021-03-11)

### Testing and Release Infrastructure

 * Add Python 3.9

## v1.1.0.post0 (2021-03-03)

### Testing and Release Infrastructure

 * Update tests for AwsDevice implementation

## v1.1.0 (2021-02-10)

### Features

 * Decompositions of XY and Ising gates

## v1.0.2 (2021-02-02)

### Bug Fixes and Other Changes

 * Improved wires support
 * Fix probability returns

## v1.0.1 (2021-01-28)

### Bug Fixes and Other Changes

 * Add inverses to integ tests
 * Merge pull request #53 from aws/inverse
 * integ test fix
 * small refactors

## v1.0.0.post3 (2021-01-12)

### Testing and Release Infrastructure

 * Enable Codecov

## v1.0.0.post2 (2020-12-30)

### Testing and Release Infrastructure

 * Add build badge
 * Use GitHub Actions for CI

## v1.0.0.post1 (2020-12-10)

### Testing and Release Infrastructure

 * Fix link formatting in README

## v1.0.0.post0 (2020-12-09)

### Testing and Release Infrastructure

 * Auto-publish to PyPi

## v1.0.0 (2020-12-07)

### Breaking Changes

 * Prime repo for public release

## v0.5.3 (2020-12-04)

### Bug Fixes and Other Changes

 * Merge pull request #46 from aws/infra/update-pl-min-ver
 * Merge pull request #45 from aws/documentation/enable-tape
 * Merge pull request #44 from aws/fix/batch-default
 * Minor wording fix
 * Use max_parallel=None as default for batching

### Documentation Changes

 * Mention tape mode and link to AWS quota details

### Testing and Release Infrastructure

 * Update minimum dependency versions
 * PennyLane docs theme

## v0.5.2 (2020-12-02)

### Bug Fixes and Other Changes

 * Unwrap `tensor`s into a NumPy `array`s in `apply()`

## v0.5.1.post0 (2020-12-01)

### Documentation Changes

 * Remove private beta instructions

## v0.5.1 (2020-11-26)

### Bug Fixes and Other Changes

 * Merge pull request #40 from aws/fix/batch-retry
 * Enable specification of max_retries for batch_execute()

## v0.5.0 (2020-11-23)

### Features

 * Use batching methods in the Braket SDK to run parallel batch executions.

### Bug Fixes and Other Changes

 * Merge pull request #39 from aws/feature/batching-from-sdk
 * Add minor comment about max_parallel and max_connections
 * Refactor translation from Braket to PennyLane results to read task results directly instead of Braket tasks
 * Merge pull request #37 from aws/fix/rename-package
 * Merge branch 'main' into fix/rename-package
 * Merge pull request #38 from aws/fix/default-polling-time
 * Merge branch 'fix/rename-package' of ssh://github.com/aws/amazon-braket-pennylane-plugin-python into fix/rename-package
 * Fix typo in package name
 * Fix typo in package name

## v0.4.4 (2020-11-20)

### Bug Fixes and Other Changes

 * Merge pull request #36 from aws/lundql-patch-1
 * Merge branch 'main' into lundql-patch-1
 * Update README.md

## v0.4.3 (2020-11-19)

### Bug Fixes and Other Changes

 * Merge pull request #35 from aws/lundql-patch-2
 * Merge branch 'main' into lundql-patch-2
 * Merge pull request #34 from aws/lundql-patch-1
 * Merge branch 'main' into lundql-patch-1
 * Merge branch 'main' into lundql-patch-2
 * Update index.rst
 * Update braket_device.py

## v0.4.2 (2020-11-17)

### Bug Fixes and Other Changes

 * Merge pull request #33 from aws/lundql-patch-1
 * Update doc/usage.rst
 * Update doc/usage.rst
 * Update usage.rst

## v0.4.1 (2020-10-30)

### Bug Fixes and Other Changes

 * Provide parallelized batch execution

### Testing and Release Infrastructure

 * updating codeowners

## v0.4.0.post0 (2020-10-06)

### Testing and Release Infrastructure

 * change s3 bucket exists check

## v0.4.0 (2020-09-30)

### Features

 * Add LocalSimulator support

## v0.3.0 (2020-09-18)

### Features

 * Enable shots=0

## v0.2.0 (2020-09-15)

### Features

 * One device to rule them all

## v0.1.4.post1 (2020-09-10)

### Testing and Release Infrastructure

 * fix black formatting

## v0.1.4.post0 (2020-09-09)

### Documentation Changes

 * Bring contribution instructions in line with other repos

### Testing and Release Infrastructure

 * Update formatting to follow new black rules
 * Update formatting to follow new black rules
