lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

format:
	ruff check --fix src tests
	ruff format src tests
