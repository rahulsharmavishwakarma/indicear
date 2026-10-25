# Shared transport for the workstation-side scripts (bash). Source it, then call transport_init SESSION.
#
# Colab allows ONE SSH connection per runtime (more return HTTP 429), so every ssh/rsync goes
# through one multiplexed control socket: ProxyCommand `colab ssh --proxy-mode -s SESSION`,
# ControlMaster auto, ControlPersist 10m. Launching, log tailing and syncing share it.
# SESSION=local treats remote paths as local paths (tests, or runs on the same machine).
# When SSH fails, pulls fall back to `colab download -s SESSION` and pushes to `colab upload`,
# which need no SSH slot.

INDICEAR_SSH_DIR="${INDICEAR_SSH_DIR:-$HOME/.config/indicear}"
COLAB_USER="${COLAB_USER:-root}"
PYTHON="${PYTHON:-python3}"

transport_init() {
  SESSION="$1"
  if [[ "$SESSION" == "local" ]]; then
    return 0
  fi
  mkdir -p "$INDICEAR_SSH_DIR"
  chmod 700 "$INDICEAR_SSH_DIR"
  SSH_CONF="$INDICEAR_SSH_DIR/ssh-$SESSION.conf"
  SSH_HOST="indicear-$SESSION"
  # %C hashes the connection; keeps the socket path short (macOS limit ~104 bytes).
  cat > "$SSH_CONF" <<CONF
Host $SSH_HOST
  User $COLAB_USER
  ProxyCommand colab ssh --proxy-mode -s $SESSION
  ControlMaster auto
  ControlPath $INDICEAR_SSH_DIR/cm-%C
  ControlPersist 10m
  ServerAliveInterval 30
  ServerAliveCountMax 4
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $INDICEAR_SSH_DIR/known_hosts-$SESSION
CONF
  SSH_CMD=(ssh -F "$SSH_CONF")
}

# rssh CMD...  - run a remote shell command over the shared socket.
rssh() {
  if [[ "$SESSION" == "local" ]]; then
    bash -c "$*"
  else
    "${SSH_CMD[@]}" "$SSH_HOST" "$@"
  fi
}

_rsync_remote() {  # rsync ARGS... over the shared socket
  rsync -e "ssh -F $SSH_CONF" "$@"
}

# pull REMOTE_FILE LOCAL_FILE  - copy one file down; colab download if SSH fails.
pull() {
  local src="$1" dst="$2"
  if [[ "$SESSION" == "local" ]]; then
    rsync -a --partial "$src" "$dst" 2>/dev/null
    return
  fi
  _rsync_remote -a --partial "$SSH_HOST:$src" "$dst" 2>/dev/null && return 0
  echo "ssh pull failed for $src; trying colab download" >&2
  colab download -s "$SESSION" "$src" "$dst"
}

# push LOCAL_FILE REMOTE_FILE  - copy one file up; colab upload if SSH fails.
push() {
  local src="$1" dst="$2"
  if [[ "$SESSION" == "local" ]]; then
    mkdir -p "$(dirname "$dst")"
    rsync -a --partial "$src" "$dst"
    return
  fi
  if rssh "mkdir -p '$(dirname "$dst")'" 2>/dev/null && _rsync_remote -a --partial "$src" "$SSH_HOST:$dst"; then
    return 0
  fi
  echo "ssh push failed for $src; trying colab upload" >&2
  colab upload -s "$SESSION" "$src" "$dst"
}

# pull_matching REMOTE_DIR LOCAL_DIR PATTERN...  - small text files (logs, CSVs), best effort.
pull_matching() {
  local src="$1" dst="$2"
  shift 2
  local filters=() pattern
  for pattern in "$@"; do
    filters+=(--include "$pattern")
  done
  filters+=(--exclude "*")
  if [[ "$SESSION" == "local" ]]; then
    rsync -a "${filters[@]}" "$src/" "$dst/" 2>/dev/null || true
  else
    _rsync_remote -a "${filters[@]}" "$SSH_HOST:$src/" "$dst/" 2>/dev/null || true
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# manifest_entries LATEST.json  - one "file<TAB>sha256<TAB>kind" line per complete file.
manifest_entries() {
  "$PYTHON" -c '
import json, sys
for entry in json.load(open(sys.argv[1]))["files"]:
    print(entry["file"], entry["sha256"], entry["kind"], sep="\t")
' "$1"
}

# manifest_field LATEST.json KEY  - a top-level field (run, run_dir, status); keep: one per line.
manifest_field() {
  "$PYTHON" -c '
import json, sys
value = json.load(open(sys.argv[1])).get(sys.argv[2]) or ""
print("\n".join(value) if isinstance(value, list) else value)
' "$1" "$2"
}
