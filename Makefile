NAME := ny-taxi-web-service
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo " install: install packages and prepare environment"
	@echo " clean: remove all temporary files"
	@echo " lint: run the code linters"
	@echo " format:	reformat code"
	@echo " test: run all the tests"
	@echo " pre-commit-install:	install pre-commit hooks"
	@echo "  test-environment   test if the environment is set up correctly (Poetry and pre-commit)"
	@echo ""
	@echo "Check the Makefile to know exactly what each target is doing."

.PHONY: pre-commit-install
pre-commit-install:
	@if [ -z "$(POETRY)" ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
	$(POETRY) run pre-commit install

install: $(INSTALL_STAMP) test-python-version pre-commit-install
$(INSTALL_STAMP): pyproject.toml poetry.lock
		@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
		$(POETRY) install
		touch $(INSTALL_STAMP)
		$(POETRY) run pre-commit install

.PHONY: lint
lint: $(INSTALL_STAMP)
		$(POETRY) run ruff check . --fix

.PHONY: format
format: $(INSTALL_STAMP)
		$(POETRY) run ruff format .

.PHONY: test
test: $(INSTALL_STAMP)
		$(POETRY) run pytest ./tests/

.PHONY: clean
clean:
		find . -type d -name "__pycache__" | xargs rm -rf {};
		rm -rf $(INSTALL_STAMP) .coverage .mypy_cache

.PHONY: test-environment
test-environment: test-python-version poetry-check pre-commit-check

.PHONY: poetry-check
poetry-check:
	@echo "Checking if Poetry is installed..."
	@if [ -z "$(POETRY)" ]; then \
		echo "Poetry could not be found. See https://python-poetry.org/docs/"; \
		exit 2; \
	else \
		echo "Poetry is installed: $(POETRY)"; \
	fi

.PHONY: pre-commit-check
pre-commit-check:
	@echo "Checking if pre-commit is installed..."
	@if [ -z "$(shell command -v pre-commit 2> /dev/null)" ]; then \
		echo "pre-commit is not installed. Please install it from https://pre-commit.com/"; \
		exit 1; \
	else \
		echo "pre-commit is installed."; \
	fi

	@echo "Checking if pre-commit is configured..."
	@if [ -f ".pre-commit-config.yaml" ]; then \
		echo "pre-commit configuration file (.pre-commit-config.yaml) found."; \
	else \
		echo "pre-commit configuration file (.pre-commit-config.yaml) not found."; \
	fi

.PHONY: test-python-version
test-python-version:
	@echo "Testing Python version..."
	@python -c "import sys; assert (3, 10) <= sys.version_info < (3, 13), 'Python 3.10 to 3.12 is required.'"
	@echo "Python version is compatible."
