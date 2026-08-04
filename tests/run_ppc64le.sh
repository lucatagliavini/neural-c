#!/usr/bin/env bash

set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
qemu=${PPC64LE_QEMU:-qemu-ppc64le}
sysroot=${PPC64LE_SYSROOT:-/usr/powerpc64le-linux-gnu}
executable=${PPC64LE_EXECUTABLE:-$repository_root/build/ppc64le/neural-c}

if [[ ! -x $executable ]]; then
    printf 'neural-c: ppc64le executable not found: %s\n' "$executable" >&2
    exit 1
fi
if [[ ! -d $sysroot ]]; then
    printf 'neural-c: ppc64le sysroot not found: %s\n' "$sysroot" >&2
    exit 1
fi

exec "$qemu" -L "$sysroot" "$executable" "$@"
