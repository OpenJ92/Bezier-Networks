#!/usr/bin/env sh
set -eu

image_name="bezier-network-test"

docker build --tag "$image_name" .
docker run --rm "$image_name"
