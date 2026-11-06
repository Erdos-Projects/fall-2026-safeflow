PYTHON ?= python

.PHONY: report reproduce prepare test notebook verify
report:
	$(PYTHON) scripts/generate_report.py

reproduce:
	$(PYTHON) scripts/reproduce_results.py

prepare:
	$(PYTHON) scripts/reproduce_results.py --prepare-only --use-cache

verify:
	$(PYTHON) scripts/verify_release.py

test:
	$(PYTHON) -m unittest discover -s tests -v

notebook:
	$(PYTHON) scripts/execute_final_notebook.py
