#!/usr/bin/env bash
# Measured room impulse responses for GPU augmentation: the real RIRs of OpenSLR SLR28
# ("Room Impulse Response and Noise Database", RIRS_NOISES; licence on https://www.openslr.org/28/:
# Apache 2.0). Downloads rirs_noises.zip (1,311,166,223 bytes) once into $ROOT/rirs, extracts only
# the RIR wavs that RIRS_NOISES/real_rirs_isotropic_noises/rir_list names (325 in all: 107 AIR,
# 36 REVERB 2014, 182 RWCP; the isotropic noise recordings beside them are not extracted), deletes
# the zip and prints RIR_DIR for `ablate train --rir-dir`.
#
# RIR_SETS picks the source databases (prefixes in rir_list). The default "air,RVB2014" leaves out
# RWCP, whose own OpenSLR page (https://www.openslr.org/13/) says "Research and development use
# only"; SLR28 repackages all three under Apache 2.0, but the underlying AIR and REVERB 2014 terms
# were not readable from here (AIR's OpenSLR mirror, https://www.openslr.org/20/, says "Not stated
# in the download"). RIR_SETS=air,RVB2014,RWCP takes all 325. Safe to rerun.
#   bash scripts/rirs.sh            # -> RIR_DIR=$ROOT/rirs
set -euo pipefail
source "$(dirname "$0")/env.sh"
dest="$ROOT/rirs"
sets="${RIR_SETS:-air,RVB2014}"
marker="$dest/rirs.json"
size=1311166223
list="RIRS_NOISES/real_rirs_isotropic_noises/rir_list"
if [ -e "$marker" ] && grep -q "\"sets\": \"$sets\"" "$marker"; then
  echo "real RIRs present ($(grep -o '"files": [0-9]*' "$marker")): RIR_DIR=$dest"
  exit 0
fi
command -v unzip >/dev/null || { echo "unzip is missing (apt-get install -y unzip)"; exit 1; }
mkdir -p "$dest"
zip="$dest/rirs_noises.zip"
have=$( [ -e "$zip" ] && wc -c < "$zip" | tr -dc 0-9 || echo 0 )
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 3 ] || [ "$have" = "$size" ] || { echo "need ~3 GB free on $ROOT, have ${free_gb} GB"; exit 1; }
if [ "$have" != "$size" ]; then
  ok=""
  for mirror in https://www.openslr.org/resources/28 https://openslr.trmal.net/resources/28 \
                https://openslr.elda.org/resources/28; do
    if curl -L --fail --retry 5 -C - -o "$zip" "$mirror/rirs_noises.zip"; then ok=1; break; fi
  done
  [ -n "$ok" ] || { echo "rirs_noises.zip: every mirror failed"; exit 1; }
fi
[ "$(wc -c < "$zip" | tr -dc 0-9)" = "$size" ] || { echo "rirs_noises.zip has the wrong size; delete $zip and rerun"; exit 1; }
unzip -o -q "$zip" "$list" -d "$dest"
pattern="$(echo "$sets" | tr ',' '|')"
# rir_list lines: --rir-id N --room-id ROOM RIRS_NOISES/real_rirs_isotropic_noises/<set>_type..._<name>.wav
awk -v p="^(${pattern})_" '{ n = $NF; sub(".*/", "", n); if (n ~ p) print }' "$dest/$list" > "$dest/$list.selected"
mv "$dest/$list.selected" "$dest/$list"
members=()
while IFS= read -r member; do members+=("$member"); done < <(awk '{ print $NF }' "$dest/$list")
[ "${#members[@]}" -gt 0 ] || { echo "no RIRs match RIR_SETS=$sets"; exit 1; }
unzip -o -q "$zip" "${members[@]}" -d "$dest"
rm -f "$zip"
cat > "$marker" <<JSON
{"source": "OpenSLR SLR28 rirs_noises.zip (real_rirs_isotropic_noises, rir_list entries)",
 "url": "https://www.openslr.org/28/", "zip_bytes": $size, "sets": "$sets", "files": ${#members[@]},
 "licence": "Apache 2.0 (SLR28 page); RWCP's own page says research and development use only; AIR and REVERB 2014 terms not verified"}
JSON
echo "extracted ${#members[@]} real RIRs (sets $sets)"
echo "RIR_DIR=$dest"
