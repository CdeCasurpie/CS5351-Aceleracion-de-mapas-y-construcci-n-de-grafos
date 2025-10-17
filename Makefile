.PHONY: build shell shell-user test help benchmark example-acj test-acj clean-all

.DEFAULT_GOAL := help

help:
	@echo "ACJ Library - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Main commands:"
	@echo "  make build        - Build Docker image with all dependencies"
	@echo "  make test-acj     - Run ACJ library tests"
	@echo "  make example-acj  - Run ACJ library example"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make clean-all    - Clean everything including Docker cache"
	@echo ""
	@echo "Development commands:"
	@echo "  make shell        - Open Docker shell as root"
	@echo "  make shell-user   - Open Docker shell as user"
	@echo ""
	@echo "Legacy commands (from Alejandro's work):"
	@echo "  make test         - Run legacy matcher tests"
	@echo "  make example      - Run legacy matcher example"
	@echo "  make benchmark    - Run performance benchmark"

# Build Docker image
build:
	docker build -f Dockerfile -t ubuntu-acj:1 --build-arg uid="$(shell id -u)" --build-arg gid="$(shell id -g)" --build-arg user=dockeruser --build-arg group=dockergroup .

# Interactive shells
shell:
	docker run -v $(shell pwd):/workspace -w /workspace -it ubuntu-acj:1 /bin/bash

shell-user:
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -w /workspace -it ubuntu-acj:1 /bin/bash

# ACJ Library commands
test-acj: ## Run ACJ library test suite
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 -m pytest acj/tests/ -v"

example-acj: ## Run ACJ library example
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_acj.py"

# Legacy commands (from Alejandro's work)
test: ## Run legacy matcher test suite
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 -m pytest tests/ -v"

example: ## Run legacy matcher example
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 example.py"

benchmark: ## Run performance benchmark
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 benchmark.py"

# Cleanup
clean: ## Clean build artifacts
	rm -rf build/

clean-all: ## Clean everything including Docker cache
	rm -rf build/
	docker rmi ubuntu-acj:1 || true
