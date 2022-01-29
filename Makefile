.PHONY: lint
lint:
	python3 -m pylint --fail-under=9.0 *.py

.PHONY: test
test:
	python3 -m pytest

.PHONY: coverage
coverage:
	coverage run -m pytest
