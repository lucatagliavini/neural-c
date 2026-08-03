NATIVE_CC ?= cc
PPC64LE_CC ?= powerpc64le-linux-gnu-gcc

CPPFLAGS += -Iinclude
CFLAGS += -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
LDLIBS += -lm

LIBRARY_SOURCES := src/activation.c src/cli_options.c src/error.c src/init.c src/model.c src/parse.c src/path.c src/project.c src/random.c src/training.c src/version.c
PROGRAM_SOURCES := src/main.c $(LIBRARY_SOURCES)
PUBLIC_HEADERS := $(wildcard include/neural/*.h)
CORE_TEST_SOURCES := tests/test_core.c $(LIBRARY_SOURCES)
MODEL_TEST_SOURCES := tests/test_model.c $(LIBRARY_SOURCES)

.PHONY: all build build-native build-ppc64le test test-defaults test-sanitize check verify-binaries clean

all: build-native

build: build-native build-ppc64le

build-native: build/x86_64/neural-c

build-ppc64le: build/ppc64le/neural-c

build/x86_64/neural-c: $(PROGRAM_SOURCES) $(PUBLIC_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PROGRAM_SOURCES) -o $@ $(LDLIBS)

build/ppc64le/neural-c: $(PROGRAM_SOURCES) $(PUBLIC_HEADERS)
	mkdir -p $(@D)
	$(PPC64LE_CC) $(CPPFLAGS) $(CFLAGS) $(PROGRAM_SOURCES) -o $@ $(LDLIBS)

build/tests/test_core: $(CORE_TEST_SOURCES) $(PUBLIC_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(CORE_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_model: $(MODEL_TEST_SOURCES) $(PUBLIC_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(MODEL_TEST_SOURCES) -o $@ $(LDLIBS)

test: build/tests/test_core build/tests/test_model
	./build/tests/test_core
	./build/tests/test_model

test-defaults:
	mkdir -p build/tests
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) \
		-DNEURAL_DEFAULT_TEXT_INITIAL_CAPACITY=64U \
		-DNEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH=8192U \
		-DNEURAL_DEFAULT_TOKEN_CAPACITY=2U \
		-DNEURAL_DEFAULT_LAYER_CAPACITY=2U \
		-DNEURAL_DEFAULT_SAMPLE_CAPACITY=2U \
		-DNEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY=256U \
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
