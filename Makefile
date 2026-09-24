# Precise but Uncoupled -- reproduction entry points.
#
# Written for GNU Make 3.81 (the version shipped with macOS): no .ONESHELL,
# no `!=`, no grouped targets, no pattern-specific conditionals.
# Every target is a thin wrapper around a script in scripts/, so Make is never
# required -- run the script directly if you prefer.

SHELL   := /bin/bash
PYTHON  ?= python3

.PHONY: help fetch-data verify-checksums reproduce-analysis reproduce-rebuttal \
        reproduce-figures smoke-test smoke-live validate test clean

help:
	@echo "Precise but Uncoupled -- make targets"
	@echo ""
	@echo "  TRACK A (offline, no model calls, deterministic):"
	@echo "    make fetch-data          download the public HF dataset into data/hf/"
	@echo "    make verify-checksums    validate data/manifests/*.sha256 against what is on disk"
	@echo "    make reproduce-analysis  rebuild the 2x5 table and CHECK its sha256"
	@echo "    make reproduce-rebuttal  rebuild the post-rebuttal robustness tables"
	@echo "    make reproduce-figures   regenerate figure input data"
	@echo "    make test                run the unit tests"
	@echo "    make validate            structural + anonymity self-check of this tree"
	@echo ""
	@echo "  TRACK B (calls YOUR model endpoint, outputs VARY run to run):"
	@echo "    make smoke-live          3-problem live run against \$$OPENAI_BASE_URL"
	@echo ""
	@echo "  Each target = scripts/<name>.sh . Make is optional."

fetch-data:
	@bash scripts/fetch_data.sh

verify-checksums:
	@bash scripts/verify_checksums.sh

reproduce-analysis:
	@bash scripts/reproduce_main_results.sh

reproduce-rebuttal:
	@bash scripts/reproduce_rebuttal_results.sh

reproduce-figures:
	@bash scripts/reproduce_figures.sh

smoke-test:
	@bash scripts/smoke_test.sh

smoke-live:
	@bash scripts/smoke_live.sh

validate:
	@$(PYTHON) scripts/validate_release.py

test:
	@$(PYTHON) -m pytest tests/ -q

clean:
	@rm -rf results/derived_tables/_regenerated results/figure_data/_regenerated
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
	@echo "cleaned regenerated outputs (released reference artifacts are untouched)"
