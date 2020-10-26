PYTHONPATH=/usr/local/lib/python3.8/dist-packages/


all: pireqs mypy pytest

pireqs:
	pipreqs --force .	
mypy:
	python3 -m mypy snook --ignore-missing-imports
pytest:
	PYTHONPATH=${PYTHONPATH} python3 -m pytest .