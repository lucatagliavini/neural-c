#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
machine=$(uname -m)

case "$machine" in
    x86_64)
        target="x86_64"
        build_target="build-native"
        ;;
    ppc64le)
        target="ppc64le"
        build_target="build-ppc64le"
        ;;
    *)
        printf "neural-c: unsupported architecture '%s'\n" "$machine" >&2
        exit 1
        ;;
esac

executable="$script_dir/build/$target/neural-c"
if [[ ! -x "$executable" ]]; then
    printf "neural-c: executable not found for %s: %s\n" \
        "$target" "$executable" >&2
    printf "Build it with: make %s\n" "$build_target" >&2
    exit 1
fi

exec "$executable" "$@"
