.PHONY: help setup download-model run-single run-isolation run-stacked run-accuracy run-server run-quant-prefill profile-16 profile-1024 clean-logs clean-results clean-profile clean-venv clean-all

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
VENV_DIR     := $(shell pwd)/venv
PYTHON       := $(VENV_DIR)/bin/python
PIP          := $(VENV_DIR)/bin/pip
REQUIREMENTS := env/requirements.txt

SUT_DIR      := llama3.1-8b
SCRIPTS_DIR  := scripts
MODEL_PATH   ?= $(SUT_DIR)/model
DATA_DIR     ?= $(SUT_DIR)/data
OUTPUT_DIR   ?= output-logs

MODEL_NAME   ?= meta-llama/Meta-Llama-3.1-8B-Instruct
HF_TOKEN     ?=

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup: ## Create venv and install dependencies
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

download-model: ## Download model weights from HuggingFace
	@bash $(SCRIPTS_DIR)/download_model.sh $(MODEL_NAME) $(MODEL_PATH) $(HF_TOKEN)

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
run-single: ## Run single/Offline baseline benchmark
	cd $(SUT_DIR) && $(PYTHON) main.py \
		--scenario Offline \
		--model-path $(MODEL_PATH) \
		--accuracy \
		--dataset-path $(DATA_DIR) \
		--output-log-dir $(OUTPUT_DIR)

run-isolation: ## Run one-knob isolation experiments
	bash $(SCRIPTS_DIR)/run_isolation.sh

run-stacked: ## Run stacked combination experiments
	bash $(SCRIPTS_DIR)/run_stacked.sh

run-accuracy: ## Run accuracy/ROUGE ablation (SLURM)
	sbatch $(SCRIPTS_DIR)/accuracy_test.sh

run-server: ## Run Server scenario experiments (SLURM)
	sbatch $(SCRIPTS_DIR)/server_run.sh

run-quant-prefill: ## Run quantization + chunked prefill study
	bash $(SCRIPTS_DIR)/quant_prefill.sh

# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
profile-16: ## Run Nsight profiling with batch_size=16
	nsys profile --stats=true --force-export=true \
		-o profiling/nsight/16_nsys_report \
		$(PYTHON) -m main --scenario Offline --batch-size 16 \
		--output-log-dir $(OUTPUT_DIR)

profile-1024: ## Run Nsight profiling with batch_size=1024
	nsys profile --stats=true --force-export=true \
		-o profiling/nsight/1024_nsys_report \
		$(PYTHON) -m main --scenario Offline --batch-size 1024 \
		--output-log-dir $(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean-logs: ## Remove benchmark output logs
	rm -rf $(OUTPUT_DIR)
	find $(SUT_DIR) -name 'mlperf_log_*' -delete

clean-results: ## Remove results directory
	rm -rf results

clean-profile: ## Remove profiling binary outputs
	rm -f profiling/nsight/*.nsys-rep
	rm -f profiling/nsight/*.sqlite

clean-venv: ## Remove Python virtual environment
	rm -rf $(VENV_DIR)

clean-all: clean-logs clean-results clean-profile clean-venv ## Remove all generated artifacts
