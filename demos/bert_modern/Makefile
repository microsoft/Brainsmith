# Makefile for BERT Modern Demo
# Provides convenient targets for common tasks

.PHONY: help quicktest configs clean test-parity gen-configs

# Default target
help:
	@echo "BERT Modern Demo - Available targets:"
	@echo "  make quicktest    - Run quick test with 1 layer"
	@echo "  make configs      - Generate example folding configurations"
	@echo "  make test-parity  - Test parity with old demo (requires old demo)"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make quicktest"
	@echo "  make configs LAYERS=3"

# Build directory
BUILD_DIR ?= ./build

# Quick test - 1 layer BERT
quicktest:
	@echo "Running quick test..."
	@bash scripts/quicktest.sh

# Generate example configurations
LAYERS ?= 3
configs:
	@echo "Generating folding configurations..."
	@mkdir -p configs
	# Small configuration
	python gen_folding_config.py \
		--simd 12 --pe 8 --num_layers $(LAYERS) -t 1 \
		-o configs/l$(LAYERS)_simd12_pe8.json
	# Medium configuration
	python gen_folding_config.py \
		--simd 24 --pe 16 --num_layers $(LAYERS) -t 2 \
		-o configs/l$(LAYERS)_simd24_pe16.json
	# Large configuration
	python gen_folding_config.py \
		--simd 48 --pe 32 --num_layers $(LAYERS) -t 4 \
		-o configs/l$(LAYERS)_simd48_pe32.json
	@echo "Generated configs in configs/"

# Test parity with old demo
test-parity:
	@echo "Testing parity with old demo..."
	@bash scripts/test_parity.sh

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/quicktest
	@rm -rf $(BUILD_DIR)/parity_test_*
	@rm -f configs/test_config.json
	@echo "Clean complete"

# Run specific configuration
run:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Usage: make run CONFIG=<config_name>"; \
		echo "Example: make run CONFIG=quicktest"; \
		exit 1; \
	fi
	@echo "Running configuration: $(CONFIG)"
	@bash scripts/run_config.sh $(CONFIG)