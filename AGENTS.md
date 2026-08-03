# Repository Guidelines

## Project Structure & Module Organization

Keep the root limited to project-wide configuration, documentation, and the CLI launcher. The established layout is:

- `src/` for production code and the CLI entry point.
- `include/neural/` for public C headers.
- `tests/` for automated tests named after the unit under test.
- `projects/<name>/` for model, dataset, configuration, weights, and checkpoint.
- `docs/` for persistent architecture decisions and the milestone roadmap.

Avoid committing generated binaries, object files, coverage output, or editor-specific files.

## Build, Test, and Development Commands

Use the root `Makefile`:

- `make` or `make build-native` — compile the x86-64 executable.
- `make build-ppc64le` — cross-compile for little-endian PowerPC 64.
- `make build` — compile both target architectures.
- `make test` — run the complete automated test suite.
- `make test-defaults` — validate supported compile-time overrides.
- `make test-sanitize` — run native tests with AddressSanitizer and UBSan.
- `make check` — run tests and validate the Bash launcher.
- `make verify-binaries` — verify both generated ELF architectures.
- `make clean` — remove generated build artifacts only.

Commands should work from the repository root and produce artifacts in an ignored directory such as `build/`.

## Coding Style & Naming Conventions

Use C11, four-space indentation, no tabs except in Make recipes, and keep lines reasonably short. Use `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for macros and constants, and descriptive lowercase filenames such as `tensor_ops.c`. Use `neural_real` (currently `double`) for numerical values. Pair public headers and implementations where practical. Compile with warnings enabled and treat new warnings as defects.

Place overridable limits and conventions in `include/neural/defaults.h` with the `NEURAL_DEFAULT_` prefix. Keep runtime training values in project files; follow `docs/specification.md` for ownership rules.

## Testing Guidelines

Add tests with every behavior change and bug fix. Name test files after the unit under test, for example `tests/test_tensor_ops.c`, and keep tests deterministic and independent of execution order. The dependency-free harness uses inputs under `tests/fixtures/`; run `make test` and `make test-sanitize`.

## Commit & Pull Request Guidelines

No usable commit history exists. Use focused commits with imperative subjects, such as `Add tensor allocation checks`. Pull requests should explain motivation, verification, and linked issues; attach logs or screenshots only when useful.

Before changing model execution, training, or persistence, read `docs/model-runtime.md`, `docs/training-resume.md`, and `docs/roadmap.md`. Preserve documented parameter layout and file ownership.
