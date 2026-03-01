.PHONY: test-local lint format

test-local:
	TRITON_INTERPRET=1 pytest tests/ -v

lint:
	ruff check kernels/ tests/

format:
	black kernels/ tests/ scripts/
	isort kernels/ tests/ scripts/
