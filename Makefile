.PHONY: build shell shell-user test example clean clean-all help

.DEFAULT_GOAL := help

help:
	@echo "ACJ Library - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Main commands:"
	@echo "  make build        - Build Docker image with all dependencies"
	@echo "  make test         - Run ACJ library tests"
	@echo "  make example      - Run ACJ library example"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make clean-all    - Clean everything including Docker cache"
	@echo ""
	@echo "Development commands:"
	@echo "  make shell        - Open Docker shell as root"
	@echo "  make shell-user   - Open Docker shell as user"

# Build Docker image
build:
	docker build -f Dockerfile -t ubuntu-acj:1 --build-arg uid="$(shell id -u)" --build-arg gid="$(shell id -g)" --build-arg user=dockeruser --build-arg group=dockergroup .

# Interactive shells
shell:
	docker run -v $(shell pwd):/workspace -w /workspace -it ubuntu-acj:1 /bin/bash

shell-user:
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -w /workspace -it ubuntu-acj:1 /bin/bash

# ACJ Library commands
test: ## Run ACJ library test suite
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 -m pytest acj/tests/ -v"

example: ## Run ACJ library example
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_acj.py"

# Cleanup
clean: ## Clean build artifacts
	rm -rf build/

clean-all: ## Clean everything including Docker cache
	rm -rf build/
	docker rmi ubuntu-acj:1 || true
