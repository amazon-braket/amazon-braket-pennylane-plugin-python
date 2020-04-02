PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane_braket --cov-report term-missing --cov-report=html:coverage_html_report
UNIT := -m pytest tests/unit_tests
INTEG := -m pytest tests/integ_tests

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install PennyLane-Braket"
	@echo "  wheel              to build the PennyLane-Braket wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  coverage           to generate a coverage report"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PennyLane-Braket you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	rm -rf pennylane_braket/__pycache__
	rm -rf tests/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf .coverage coverage_html_report/

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean


unit-test:
	@echo "Generating coverage report for unit tests..."
	$(PYTHON) $(UNIT) $(COVERAGE)


integ-test:
	@echo "Generating coverage report for integ tests..."
	$(PYTHON) $(INTEG) $(COVERAGE)
