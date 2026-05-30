# Agent Notes

This repository is being modernized from an old Bezier neural-network prototype into a standard
Python package. Read `TESTS.md` before running tests or changing the Docker harness.

## Project Shape

- Package source lives in `src/bezier_network`.
- Tests live in `tests`.
- Examples live in `examples`.
- The current active integration branch is `develop`.
- Feature branches should use the `codex/` prefix unless the user asks otherwise.

## Testing

Use `./test.sh dev-test` for normal development validation. It reuses the named Docker container
`bezier-network-dev`, which avoids paying the full PyTorch setup cost on every test run.

Use `./test.sh` when you want the clean one-shot image run that mirrors CI more closely.

See `TESTS.md` for details and command variants.

## Dependency Expectations

Do not assume host Python has project dependencies. The host has been observed to compile the source
successfully while failing `unittest` imports because NumPy is missing. Docker is the authoritative
test environment for now.

The Dockerfile intentionally installs dependency layers before copying `src` and `tests`; preserve
that ordering unless there is a concrete reason to change it.

## Current Bezier Direction

The old `bezierCurve` API remains for compatibility with the Dense/Conv modules. The newer
`Bezier` class in `src/bezier_network/bezier/bezier.py` is the preferred core for new work. It uses
Bernstein-basis evaluation over one or more collapse axes and has an identity constructor.

When extending Bezier behavior:

- Keep continuous curve evaluation separate from integer tensor-shape quantization.
- Put truncation/rounding policy in explicit objects or functions rather than hiding it in core
  curve evaluation.
- Preserve tests around the legacy wrapper until the Dense/Conv code has migrated.

## Code Style

Prefer lowercase snake_case module names and package paths. Existing class names such as
`bezierCurve` and `controlPointsUniformRandomEnclosingPrism` are legacy compatibility names; avoid
adding new names in that style.

Do not reintroduce tracked `__pycache__`, `.pyc`, or editor swap files. `.gitignore` and
`.dockerignore` already exclude them.

## Branch And PR Hygiene

Start new work from `develop` unless the user says otherwise. Commit focused changes on feature
branches and PR back into `develop`.

Local `main` exists as the renamed successor to `master`, but the remote default may still be
`origin/master`. Do not delete or rewrite remote branches without an explicit user request.

## Useful Commands

```sh
python3 -m compileall -q src tests examples
./test.sh dev-test
./test.sh exec python -m unittest tests.test_bezier -v
./test.sh shell
```
