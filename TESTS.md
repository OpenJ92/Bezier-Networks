# Testing Notes

This project uses a Docker-backed test harness because local Python environments may not have
NumPy or PyTorch installed. Prefer the wrapper script over ad hoc host commands when validating
behavior that imports `bezier_network`.

## Commands

Run the clean CI-style path:

```sh
./test.sh
```

Build the image without running tests:

```sh
./test.sh build
```

Run tests inside the persistent development container:

```sh
./test.sh dev-test
```

Open a shell inside the persistent development container:

```sh
./test.sh shell
```

Run an arbitrary command inside the persistent development container:

```sh
./test.sh exec python -m unittest discover -s tests -v
./test.sh exec python -m unittest tests.test_bezier -v
```

Remove the persistent development container:

```sh
./test.sh clean
```

## Harness Construction

The Docker image is named `bezier-network-test`. The reusable development container is named
`bezier-network-dev`.

The Dockerfile installs expensive dependencies before copying `src` and `tests`. This keeps the
NumPy/PyTorch layers cacheable when source files change. The default Torch install uses the CPU
wheel index:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch
```

The persistent container mounts the repository at `/workspace/BezierNetwork` and installs the
package editable with `--no-deps`. This gives fast test cycles while preserving the dependency
environment:

```sh
python -m pip install --no-deps --editable /workspace/BezierNetwork
```

## Current Coverage

Current tests live in `tests/` and use `unittest`.

- `tests/test_imports.py` confirms the main package modules import.
- `tests/test_bezier.py` covers the newer Bernstein-basis `Bezier` implementation, identity maps,
  error handling for parameter arity, and the legacy `bezierCurve` wrapper's shape/rounding
  behavior.

## Known Local Limitation

The host Python on this machine may not have NumPy installed. `python3 -m compileall -q src tests
examples` is still useful as a syntax check, but host `unittest` can fail at import time with
`ModuleNotFoundError: No module named 'numpy'`. Use `./test.sh dev-test` or `./test.sh` for the
real test signal.

## Future Test Expansion

Useful next test layers:

- Bezier endpoint invariants for generated control points.
- Equivalence checks between de Casteljau evaluation and Bernstein-basis evaluation.
- Shape quantization and truncation policy tests.
- Conv/Dense layer-plan tests that compare expected tensor shapes with actual PyTorch forwards.
- Property tests for random control point arrays once Hypothesis is added.
