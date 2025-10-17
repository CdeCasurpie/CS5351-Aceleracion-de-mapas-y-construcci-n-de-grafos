.PHONY: build shell shell-user test help benchmark

.DEFAULT_GOAL := help

help:
	@grep -hE '^[a-zA-Z_0-9-]+:.*? ## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*? ## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# Matcher Docker
build:
	docker build -f Dockerfile -t ubuntu-matcher:1 --build-arg uid="$(shell id -u)" --build-arg gid="$(shell id -g)" --build-arg user=dockeruser --build-arg group=dockergroup .

shell:
	docker run -v $(shell pwd):/workspace -w /workspace -it ubuntu-matcher:1 /bin/bash

shell-user:
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -w /workspace -it ubuntu-matcher:1 /bin/bash

test: ## Run the test suite
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-matcher:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 -m pytest tests/ -v"

example: ## Run the example script
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-matcher:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 example.py"

benchmark: ## Run performance benchmark comparing brute-force vs CGAL
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-matcher:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 benchmark.py"

clean: ## Clean build artifacts
	rm -rf build/
