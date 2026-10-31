PYTHON ?= python
export PYTHONPATH := src

.PHONY: predict serve train eval templates test

# make predict IN=tickets.jsonl OUT=pred.jsonl
predict:
	"$(PYTHON)" -X utf8 -m support_copilot.cli predict --input "$(IN)" --output "$(OUT)"

serve:
	"$(PYTHON)" -X utf8 -m support_copilot.cli serve --host 127.0.0.1 --port 8000

train:
	"$(PYTHON)" -X utf8 -m support_copilot.cli train

eval:
	"$(PYTHON)" -X utf8 scripts/eval_v3.py

templates:
	"$(PYTHON)" -X utf8 scripts/mine_templates_v3.py

test:
	"$(PYTHON)" -X utf8 -m pytest -q
