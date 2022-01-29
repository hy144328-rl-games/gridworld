.PHONY: lint
lint:
	python3 -m pylint --fail-under=9.0 *.py
