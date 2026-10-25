#!/usr/bin/env bash
# Run a command on the Colab VM over the shared SSH control socket (one connection per runtime).
#   bash scripts/colab_ssh.sh SESSION 'bash /content/indicear/scripts/launch.sh prep /content/indicear/scripts/prep.sh'
#   bash scripts/colab_ssh.sh SESSION 'tail -n 50 /content/indicear/logs/ablation.log'
set -euo pipefail
source "$(dirname "$0")/lib/transport.sh"
transport_init "$1"
shift
rssh "$@"
