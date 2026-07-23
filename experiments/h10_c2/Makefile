PYTHON ?= python
ROOT := $(abspath ../..)
ENV := PYTHONPATH=$(ROOT)/experiments/h10_c2/src:$(ROOT)/framework/fuzzyxai:$(ROOT)

.PHONY: bootstrap test power generate-development run-development generate-protocol-validation run-protocol-validation build-adjudication preconfirmatory-gate package

bootstrap:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 bootstrap

test:
	cd $(ROOT) && $(ENV) $(PYTHON) -m pytest -q experiments/h10_c2/tests
	cd $(ROOT) && $(ENV) $(PYTHON) -m ruff check experiments/h10_c2

power:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 power

generate-development:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 generate --split development

run-development:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 run --split development

generate-protocol-validation:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 generate --split protocol_validation

run-protocol-validation:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 run --split protocol_validation

build-adjudication:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 export-adjudication --sample-size 200

preconfirmatory-gate:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 preconfirmatory-gate

package:
	cd $(ROOT) && $(ENV) $(PYTHON) -m h10_c2 package
