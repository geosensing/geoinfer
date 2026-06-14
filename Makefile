.PHONY: install lint format typecheck test docs build clean

install:        ## Sync all dependency groups into the uv environment
	uv sync --all-groups

lint:           ## Ruff lint
	uv run ruff check geoinfer/ examples/

format:         ## Ruff format (writes changes)
	uv run ruff format geoinfer/ examples/

typecheck:      ## Mypy on the package source
	uv run mypy geoinfer/

test:           ## Run the test suite
	uv run pytest

docs:           ## Build the HTML documentation
	cd docs && make html

build:          ## Build sdist + wheel
	uv build

clean:          ## Remove build/doc artifacts
	rm -rf dist build docs/_build
