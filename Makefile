NATIVE_CC ?= cc
PPC64LE_CC ?= powerpc64le-linux-gnu-gcc
THREAD_FLAGS ?= -pthread
UNAME_MACHINE := $(shell uname -m)
UNAME_RELEASE := $(shell uname -r)

ifneq ($(findstring microsoft,$(UNAME_RELEASE)),)
TSAN_RUNNER ?= setarch $(UNAME_MACHINE) -R
else
TSAN_RUNNER ?=
endif

CPPFLAGS += -Iinclude
CFLAGS += -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow $(THREAD_FLAGS)
LDLIBS += -lm

LIBRARY_SOURCES := src/activation.c src/atomic_file.c src/backprop.c src/cli_options.c src/dense.c src/digest.c src/error.c src/gradient.c src/gradient_check.c src/init.c src/loss.c src/model.c src/parallel.c src/parse.c src/path.c src/persistence.c src/project.c src/random.c src/sha256.c src/tensor_ops.c src/training.c src/version.c
PROGRAM_SOURCES := src/main.c $(LIBRARY_SOURCES)
PUBLIC_HEADERS := $(wildcard include/neural/*.h)
INTERNAL_HEADERS := $(wildcard src/*.h)
CORE_TEST_SOURCES := tests/test_core.c $(LIBRARY_SOURCES)
MODEL_TEST_SOURCES := tests/test_model.c $(LIBRARY_SOURCES)
PERSISTENCE_TEST_SOURCES := tests/test_persistence.c $(LIBRARY_SOURCES)
MATH_TEST_SOURCES := tests/test_math.c $(LIBRARY_SOURCES)
PARALLEL_TEST_SOURCES := tests/test_parallel.c $(LIBRARY_SOURCES)
BACKPROP_TEST_SOURCES := tests/test_backprop.c $(LIBRARY_SOURCES)
GRADIENT_CHECK_TEST_SOURCES := tests/test_gradient_check.c $(LIBRARY_SOURCES)

.PHONY: all build build-native build-ppc64le test test-defaults test-sanitize test-thread-sanitize check verify-binaries clean

all: build-native

build: build-native build-ppc64le

build-native: build/x86_64/neural-c

build-ppc64le: build/ppc64le/neural-c

build/x86_64/neural-c: $(PROGRAM_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PROGRAM_SOURCES) -o $@ $(LDLIBS)

build/ppc64le/neural-c: $(PROGRAM_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(PPC64LE_CC) $(CPPFLAGS) $(CFLAGS) $(PROGRAM_SOURCES) -o $@ $(LDLIBS)

build/tests/test_core: $(CORE_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(CORE_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_model: $(MODEL_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(MODEL_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_persistence: $(PERSISTENCE_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PERSISTENCE_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_math: $(MATH_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(MATH_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_parallel: $(PARALLEL_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PARALLEL_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_backprop: $(BACKPROP_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(BACKPROP_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_gradient_check: $(GRADIENT_CHECK_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(GRADIENT_CHECK_TEST_SOURCES) -o $@ $(LDLIBS)

test: build/tests/test_core build/tests/test_model build/tests/test_persistence build/tests/test_math build/tests/test_parallel build/tests/test_backprop build/tests/test_gradient_check
	./build/tests/test_core
	./build/tests/test_model
	./build/tests/test_persistence
	./build/tests/test_math
	./build/tests/test_parallel
	./build/tests/test_backprop
	./build/tests/test_gradient_check

test-defaults:
	mkdir -p build/tests
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) \
		-DNEURAL_DEFAULT_TEXT_INITIAL_CAPACITY=64U \
		-DNEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH=8192U \
		-DNEURAL_DEFAULT_TOKEN_CAPACITY=2U \
		-DNEURAL_DEFAULT_LAYER_CAPACITY=2U \
		-DNEURAL_DEFAULT_SAMPLE_CAPACITY=2U \
		-DNEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY=256U \
		-DNEURAL_DEFAULT_THREAD_COUNT=3U \
		-DNEURAL_DEFAULT_GRADIENT_CHECK_EPSILON=1e-5 \
		-DNEURAL_DEFAULT_GRADIENT_CHECK_ABSOLUTE_TOLERANCE=1e-6 \
		-DNEURAL_DEFAULT_GRADIENT_CHECK_RELATIVE_TOLERANCE=1e-4 \
		$(CORE_TEST_SOURCES) -o build/tests/test_custom_defaults $(LDLIBS)
	./build/tests/test_custom_defaults

test-sanitize:
	mkdir -p build/tests
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(CORE_TEST_SOURCES) -o build/tests/test_core_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_core_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(MODEL_TEST_SOURCES) -o build/tests/test_model_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_model_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(PERSISTENCE_TEST_SOURCES) -o build/tests/test_persistence_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_persistence_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(MATH_TEST_SOURCES) -o build/tests/test_math_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_math_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(PARALLEL_TEST_SOURCES) -o build/tests/test_parallel_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_parallel_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(BACKPROP_TEST_SOURCES) -o build/tests/test_backprop_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_backprop_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(GRADIENT_CHECK_TEST_SOURCES) -o build/tests/test_gradient_check_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_gradient_check_sanitize

test-thread-sanitize:
	mkdir -p build/tests
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=thread -fno-omit-frame-pointer \
		$(PARALLEL_TEST_SOURCES) -o build/tests/test_parallel_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 $(TSAN_RUNNER) ./build/tests/test_parallel_tsan

check: test test-defaults build-native
	bash -n neural-c.sh
	bash -n tests/test_cli.sh
	./tests/test_cli.sh ./build/x86_64/neural-c
	./neural-c.sh inspect projects/xor

verify-binaries: build
	file build/x86_64/neural-c | grep -q 'x86-64'
	file build/ppc64le/neural-c | grep -Eq 'PowerPC|ppc64le'

clean:
	rm -rf build
