lint:
	ruff check src
	ruff format --check src
	mypy src

format:
	ruff check --fix src
	ruff format src
