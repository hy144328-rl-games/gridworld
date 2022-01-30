.PHONY: lint
lint:
	python3 -m pylint $$(git ls-files '*.py')

.PHONY: test
test:
	python3 -m pytest

.PHONY: coverage
coverage:
	coverage run -m pytest
	coverage report
