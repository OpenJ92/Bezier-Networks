#!/usr/bin/env sh
set -eu

image_name="bezier-network-test"
container_name="bezier-network-dev"

build_image() {
    docker build --tag "$image_name" .
}

ensure_dev_container() {
    build_image

    if ! docker container inspect "$container_name" >/dev/null 2>&1; then
        docker create \
            --name "$container_name" \
            --workdir /workspace/BezierNetwork \
            --volume "$PWD:/workspace/BezierNetwork" \
            --entrypoint sleep \
            "$image_name" \
            infinity >/dev/null
    fi

    if [ "$(docker inspect --format '{{.State.Running}}' "$container_name")" != "true" ]; then
        docker start "$container_name" >/dev/null
    fi

    docker exec "$container_name" \
        python -m pip install --no-deps --editable /workspace/BezierNetwork >/dev/null
}

case "${1:-test}" in
    test)
        build_image
        docker run --rm "$image_name"
        ;;
    build)
        build_image
        ;;
    dev-test)
        ensure_dev_container
        docker exec "$container_name" python -m unittest discover -s tests -v
        ;;
    shell)
        ensure_dev_container
        docker exec -it "$container_name" sh
        ;;
    exec)
        shift
        ensure_dev_container
        if [ "$#" -eq 0 ]; then
            docker exec "$container_name" python -m unittest discover -s tests -v
        else
            docker exec "$container_name" "$@"
        fi
        ;;
    clean)
        docker rm --force "$container_name" >/dev/null 2>&1 || true
        ;;
    *)
        echo "usage: ./test.sh [test|build|dev-test|shell|exec|clean]" >&2
        exit 2
        ;;
esac
