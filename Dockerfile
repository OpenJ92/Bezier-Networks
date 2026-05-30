FROM python:3.12-slim

WORKDIR /workspace/BezierNetwork

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --no-cache-dir \
    filelock \
    fsspec \
    jinja2 \
    matplotlib \
    networkx \
    numpy \
    "setuptools<82" \
    sympy \
    typing-extensions

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps \
    torch

RUN python -m pip install --no-cache-dir --no-deps .

WORKDIR /workspace/BezierNetwork

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
