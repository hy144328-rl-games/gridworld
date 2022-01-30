.PHONY: lint
lint:
	python3 -m $(git ls-files '*.py')

.PHONY: test
test:
	python3 -m pytest

.PHONY: coverage
coverage:
	coverage run -m pytest
	coverage report
