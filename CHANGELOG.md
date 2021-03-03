# Changelog

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
