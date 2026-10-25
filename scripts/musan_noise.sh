#!/usr/bin/env bash
# Optional: MUSAN noise subset (OpenSLR 17, CC BY 4.0) for MUSAN-style noise. The tarball is
# about 11 GB; only musan/noise is extracted. Without it, synthetic colored noise is used.
set -euo pipefail
source "$(dirname "$0")/env.sh"
dest="$ROOT/musan"
[ -d "$dest/musan/noise" ] && { echo "MUSAN noise present: $dest/musan/noise"; exit 0; }
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 15 ] || { echo "need ~15 GB free on $ROOT, have ${free_gb} GB"; exit 1; }
mkdir -p "$dest"
curl -L --retry 5 -C - -o "$dest/musan.tar.gz" https://openslr.trmal.net/resources/17/musan.tar.gz
tar -xzf "$dest/musan.tar.gz" -C "$dest" musan/noise
rm -f "$dest/musan.tar.gz"
echo "NOISE_DIR=$dest/musan/noise"
