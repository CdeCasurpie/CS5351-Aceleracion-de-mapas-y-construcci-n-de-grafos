.PHONY: build shell shell-user test example example-realtime example-simplification example-simplification-visual example-crime example-crime-osm example-simplification-osm clean clean-all help

.DEFAULT_GOAL := help

help:
	@echo "ACJ Library - Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Main commands:"
	@echo "  make build                           - Build Docker image with all dependencies"
	@echo "  make test                            - Run ACJ library tests"
	@echo ""
	@echo "Examples with synthetic data:"
	@echo "  make example                         - Basic ACJ example"
	@echo "  make example-simplification-visual   - Graph simplification demo (synthetic)"
	@echo "  make example-crime                   - Crime heatmap demo (synthetic)"
	@echo ""
	@echo "Examples with real OSM data (configurable city):"
	@echo "  make example-realtime                - Real-time crime heatmap"
	@echo "  make example-simplification          - Graph simplification comparison"
	@echo "  make example-crime-osm               - Crime heatmap (configurable location)"
	@echo "  make example-simplification-osm      - Simplification (configurable location)"
	@echo ""
	@echo "Cleanup commands:"
	@echo "  make clean                           - Clean build artifacts"
	@echo "  make clean-all                       - Clean everything including Docker cache"
	@echo ""
	@echo "Development commands:"
	@echo "  make shell                           - Open Docker shell as root"
	@echo "  make shell-user                      - Open Docker shell as user"

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

example: ## Run basic ACJ library example
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_acj.py"

example-realtime: ## Run real-time interactive visualization with VisPy
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_realtime.py"

example-simplification: ## Run graph simplification comparison
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_simplification.py"

example-simplification-visual: ## Run visual graph simplification demo with interactive comparison
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_simplification_visual.py"

example-crime: ## Run crime heatmap visualization demo with interactive comparison
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_crime_visualization.py"

example-crime-osm: ## Run crime heatmap with real OSM data (configurable location)
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_crime_osm.py"

example-simplification-osm: ## Run simplification comparison with real OSM data (configurable location)
	docker run --user $(shell id -u):$(shell id -g) -v $(shell pwd):/workspace -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$$DISPLAY ubuntu-acj:1 sh -c "cd /workspace && mkdir -p build && cd build && cmake .. && make -j\$$(nproc) && cd .. && PYTHONPATH=/workspace/build python3 examples/example_simplification_osm.py"

# Cleanup
clean: ## Clean build artifacts
	rm -rf build/ cache/ output/

clean-all: ## Clean everything including Docker cache
	rm -rf build/ cache/ output/
	docker rmi ubuntu-acj:1 || true
