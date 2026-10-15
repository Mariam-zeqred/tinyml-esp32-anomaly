# Host-side checks (no board, no TensorFlow Lite Micro). `make check` is what CI runs first.
PY ?= python3
CXX ?= g++
CXXFLAGS ?= -std=gnu++17 -Wall -Wextra -Werror -O1 -DNATIVE_TEST
CORE_DIR = lib/anomaly_core
CORE_SRC = $(CORE_DIR)/Window.cpp $(CORE_DIR)/RefModel.cpp $(CORE_DIR)/model_data.cc
CORE_HDR = $(wildcard $(CORE_DIR)/*.h)

.PHONY: check lint pytest test-core pio-test train evaluate export all build clean

check: lint pytest test-core

lint:
	ruff check src tests && ruff format --check src tests

pytest:
	$(PY) -m pytest -q

test-core: build/test_core
	./build/test_core

build/test_core: test/test_native/test_core.cpp test/minitest.h $(CORE_SRC) $(CORE_HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -I$(CORE_DIR) -Itest test/test_native/test_core.cpp $(CORE_SRC) -o $@

# Same host tests, run by PlatformIO (needs the registry reachable to fetch the native platform).
pio-test:
	pio test -e native

# The full pipeline (needs TensorFlow): train -> evaluate -> regenerate the C sources.
train:
	cd src && $(PY) -m tinyml.train
evaluate:
	cd src && $(PY) -m tinyml.evaluate
export:
	cd src && $(PY) -m tinyml.export_c
all: train evaluate export check

# Firmware (needs the espressif32 platform; CI does this, the cloud sandbox cannot flash).
build:
	pio run -e esp32dev

clean:
	rm -rf build .pio
