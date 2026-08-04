NATIVE_CC ?= cc
PPC64LE_CC ?= powerpc64le-linux-gnu-gcc
PPC64LE_QEMU ?= qemu-ppc64le
PPC64LE_SYSROOT ?= /usr/powerpc64le-linux-gnu
PPC64LE_RUNNER ?= $(PPC64LE_QEMU) -L $(PPC64LE_SYSROOT)
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

LIBRARY_SOURCES := src/activation.c src/atomic_file.c src/backprop.c src/batch.c src/cli_options.c src/compensated_sum.c src/dense.c src/digest.c src/error.c src/evaluation.c src/executor.c src/gradient.c src/gradient_check.c src/init.c src/loss.c src/model.c src/parallel.c src/parse.c src/path.c src/persistence.c src/predict_project.c src/project.c src/project_checkpoint.c src/project_lock.c src/random.c src/sha256.c src/tensor_ops.c src/train_project.c src/training.c src/version.c
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
BATCH_TEST_SOURCES := tests/test_batch.c $(LIBRARY_SOURCES)
TRAINING_ENGINE_TEST_SOURCES := tests/test_training_engine.c $(LIBRARY_SOURCES)
CHECKPOINT_OBSERVER_TEST_SOURCES := tests/test_checkpoint_observer.c $(LIBRARY_SOURCES)
PROJECT_LOCK_TEST_SOURCES := tests/test_project_lock.c $(LIBRARY_SOURCES)
TRAINING_RESUME_TEST_SOURCES := tests/test_training_resume.c $(LIBRARY_SOURCES)
PREDICTION_TEST_SOURCES := tests/test_prediction.c $(LIBRARY_SOURCES)
EVALUATION_TEST_SOURCES := tests/test_evaluation.c $(LIBRARY_SOURCES)
TEST_NAMES := core model persistence math parallel backprop gradient_check batch
TEST_NAMES += training_engine checkpoint_observer project_lock training_resume
TEST_NAMES += prediction evaluation
PPC64LE_TEST_BINARIES := $(addprefix build/ppc64le/tests/test_,$(TEST_NAMES))

.PHONY: all build build-native build-ppc64le test test-defaults
.PHONY: test-sanitize test-thread-sanitize check verify-binaries clean
.PHONY: test-ppc64le test-ppc64le-cli test-cross-runtime
.PHONY: check-cross-runtime

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

build/tests/test_batch: $(BATCH_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(BATCH_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_training_engine: $(TRAINING_ENGINE_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(TRAINING_ENGINE_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_checkpoint_observer: $(CHECKPOINT_OBSERVER_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(CHECKPOINT_OBSERVER_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_project_lock: $(PROJECT_LOCK_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PROJECT_LOCK_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_training_resume: $(TRAINING_RESUME_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(TRAINING_RESUME_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_prediction: $(PREDICTION_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(PREDICTION_TEST_SOURCES) -o $@ $(LDLIBS)

build/tests/test_evaluation: $(EVALUATION_TEST_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) $(EVALUATION_TEST_SOURCES) -o $@ $(LDLIBS)

build/ppc64le/tests/test_%: tests/test_%.c $(LIBRARY_SOURCES) $(PUBLIC_HEADERS) $(INTERNAL_HEADERS)
	mkdir -p $(@D)
	$(PPC64LE_CC) $(CPPFLAGS) $(CFLAGS) $< $(LIBRARY_SOURCES) -o $@ $(LDLIBS)

test: build/tests/test_core build/tests/test_model build/tests/test_persistence build/tests/test_math build/tests/test_parallel build/tests/test_backprop build/tests/test_gradient_check build/tests/test_batch build/tests/test_training_engine build/tests/test_checkpoint_observer build/tests/test_project_lock build/tests/test_training_resume build/tests/test_prediction build/tests/test_evaluation
	./build/tests/test_core
	./build/tests/test_model
	./build/tests/test_persistence
	./build/tests/test_math
	./build/tests/test_parallel
	./build/tests/test_backprop
	./build/tests/test_gradient_check
	./build/tests/test_batch
	./build/tests/test_training_engine
	./build/tests/test_checkpoint_observer
	./build/tests/test_project_lock
	./build/tests/test_training_resume
	./build/tests/test_prediction
	./build/tests/test_evaluation

test-ppc64le: $(PPC64LE_TEST_BINARIES)
	@for test_binary in $(PPC64LE_TEST_BINARIES); do \
		printf 'Running %s\n' "$$test_binary"; \
		$(PPC64LE_RUNNER) "$$test_binary" || exit $$?; \
	done

test-ppc64le-cli: build/ppc64le/neural-c
	bash -n tests/run_ppc64le.sh
	bash -n tests/test_cli.sh
	PPC64LE_QEMU=$(PPC64LE_QEMU) PPC64LE_SYSROOT=$(PPC64LE_SYSROOT) \
		./tests/test_cli.sh ./tests/run_ppc64le.sh

test-cross-runtime: build-native build-ppc64le
	bash -n tests/test_cross_runtime.sh
	PPC64LE_QEMU=$(PPC64LE_QEMU) PPC64LE_SYSROOT=$(PPC64LE_SYSROOT) \
		./tests/test_cross_runtime.sh \
		./build/x86_64/neural-c ./build/ppc64le/neural-c

check-cross-runtime: verify-binaries
	+$(MAKE) test-ppc64le
	+$(MAKE) test-ppc64le-cli
	+$(MAKE) test-cross-runtime

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
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(BATCH_TEST_SOURCES) -o build/tests/test_batch_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_batch_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(TRAINING_ENGINE_TEST_SOURCES) -o build/tests/test_training_engine_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_training_engine_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(CHECKPOINT_OBSERVER_TEST_SOURCES) -o build/tests/test_checkpoint_observer_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_checkpoint_observer_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(PROJECT_LOCK_TEST_SOURCES) -o build/tests/test_project_lock_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_project_lock_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(TRAINING_RESUME_TEST_SOURCES) -o build/tests/test_training_resume_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_training_resume_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(PREDICTION_TEST_SOURCES) -o build/tests/test_prediction_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_prediction_sanitize
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=address,undefined -fno-omit-frame-pointer \
		$(EVALUATION_TEST_SOURCES) -o build/tests/test_evaluation_sanitize $(LDLIBS)
	ASAN_OPTIONS=detect_leaks=0 ./build/tests/test_evaluation_sanitize

test-thread-sanitize:
	mkdir -p build/tests
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=thread -fno-omit-frame-pointer \
		$(PARALLEL_TEST_SOURCES) -o build/tests/test_parallel_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 $(TSAN_RUNNER) ./build/tests/test_parallel_tsan
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=thread -fno-omit-frame-pointer \
		$(TRAINING_ENGINE_TEST_SOURCES) -o build/tests/test_training_engine_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 $(TSAN_RUNNER) ./build/tests/test_training_engine_tsan
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=thread -fno-omit-frame-pointer \
		$(TRAINING_RESUME_TEST_SOURCES) -o build/tests/test_training_resume_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 $(TSAN_RUNNER) ./build/tests/test_training_resume_tsan
	$(NATIVE_CC) $(CPPFLAGS) $(CFLAGS) -O1 -g \
		-fsanitize=thread -fno-omit-frame-pointer \
		$(PREDICTION_TEST_SOURCES) -o build/tests/test_prediction_tsan $(LDLIBS)
	TSAN_OPTIONS=halt_on_error=1 $(TSAN_RUNNER) ./build/tests/test_prediction_tsan

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
