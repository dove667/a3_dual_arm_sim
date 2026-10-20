# Repository Guidelines

## Project Structure & Module Organization

Python 3.12+ package for A3 dual-arm MuJoCo simulation, recording, and SmolVLA/ACT workflows.

- `src/a3_dual_arm_sim/`: implementation. Keep physics in `sim/`, success conditions in `tasks/`, experts in `controllers/`, inference in `policies/`, recording in `data/`, training in `learning/`, and orchestration in `workflows/`.
- `examples/`: thin script entry points; put reusable logic inside the package.
- `tests/`: pytest regression tests.
- `configs/`, `assets/`, and `models/*.xml`: simulation configuration and bundled resources. `setup.py` copies assets and configs into wheels.
- `docs/`: architecture, contracts, and workflow guides; start with `docs/workflow_guide.md`.

## Build, Test, and Development Commands

Run from the repository root in an isolated Python environment:

```bash
python -m pip install -e '.[dev]'  # Editable install with pytest and Ruff
a3-sim --help                    # Inspect CLI commands
a3-sim inspect                   # Inspect the simulation model
MUJOCO_GL=egl a3-sim smoke --steps 20  # Linux headless smoke check
python -m pytest                # Run regression tests
ruff check .                    # Lint Python code
ruff format --check .           # Check formatting
python -m pip wheel . --no-deps -w dist  # Build a resource-bundled wheel
```

Install `.[train]` for training features; retain the pinned `lerobot==0.5.1`. EGL rendering requires compatible drivers; see `docs/limitations.md` for platform constraints.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, `snake_case` functions/modules, `PascalCase` classes, and uppercase constants. Ruff targets Python 3.12 with a 100-character line limit. Separate independent operations and semantic stages; keep comments focused on intent. Load Torch and LeRobot lazily for optional features. Preserve action/observation contracts in `contracts.py` and `docs/contracts.md`.

## Testing Guidelines

Name files `test_<feature>.py` and functions `test_<behavior>`. Add regression tests for changed behavior, especially action dimensions, seeded resets, recording cleanup, and resource packaging. Use temporary directories for generated files and close environments reliably. Run focused tests first, then the relevant suite. No numeric coverage threshold is configured. Report rendering or dependency skips explicitly.

## Commit & Pull Request Guidelines

History favors imperative subjects such as `Simplify SmolVLA training and checkpoint inference`; scoped messages such as `fix(dataset): ...` also appear. Keep commits focused. PRs should explain behavior changes, linked issues, validation, and configuration/contract impacts. Include screenshots for visual simulation changes.

## Runtime & Artifact Management

Keep datasets, downloaded weights, and generated results in `datasets/`, `models/<name>/`, `outputs/`, and `artifacts/`; leave them untracked. Preserve resource provenance in `assets/a3/PROVENANCE.md`. Prefer the `AI` conda environment on Dove's Mac; use the L40 server for this project's training and model inference.
