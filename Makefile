PYTHON ?= python

.PHONY: predict test serve
predict:
	"$(PYTHON)" -X utf8 -m support_copilot.cli predict --input "$(IN)" --output "$(OUT)"

test:
	"$(PYTHON)" -X utf8 -m pytest -q

serve:
	"$(PYTHON)" -X utf8 -m support_copilot.cli serve --host 127.0.0.1 --port 8000
