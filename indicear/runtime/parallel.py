"""Run several jobs concurrently on one GPU (run_langs.py's --parallel pattern).

    python -m indicear.runtime.parallel --parallel 3 --logs LOGDIR \
        --job v2 "python -m indicear.ablate train --variants v2 --vram-share 0.33 --cache-placement mmap" \
        --job v3 "python -m indicear.ablate train --variants v3 --vram-share 0.33 --cache-placement mmap"

Each job writes LOGDIR/<name>.log; GPU utilization and memory go to LOGDIR/gpu.csv every 5 s.
Jobs are themselves resumable, so rerunning the same launcher continues every unfinished job.
SIGTERM/SIGINT are forwarded to the jobs so each writes its final checkpoint.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from indicear.runtime.gpu import GpuSampler, runtime_banner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--job", nargs=2, action="append", metavar=("NAME", "COMMAND"), required=True)
    args = parser.parse_args(argv)
    args.logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHON_GIL": "0", "TOKENIZERS_PARALLELISM": "false"}
    pending = list(args.job)
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    failed: list[str] = []
    stopping = False

    def forward(signum, _frame):
        nonlocal stopping
        stopping = True
        for process, _ in running.values():
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    print(f"{runtime_banner()}; {len(pending)} jobs, {args.parallel} at a time", flush=True)
    started = time.time()
    with GpuSampler(args.logs / "gpu.csv") as gpu:
        while pending or running:
            while pending and len(running) < args.parallel and not stopping:
                name, command = pending.pop(0)
                log = (args.logs / f"{name}.log").open("a")
                process = subprocess.Popen(shlex.split(command), stdout=log, stderr=subprocess.STDOUT, env=env)
                running[name] = (process, log)
                print(f"started {name}: {command}", flush=True)
            for name, (process, log) in list(running.items()):
                if process.poll() is not None:
                    log.close()
                    del running[name]
                    if process.returncode:
                        failed.append(name)
                    print(f"{name} exited {process.returncode} after {(time.time() - started) / 60:.1f} min", flush=True)
            if stopping and not running:
                break
            time.sleep(5)
        print(gpu.summary(), flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min; failed: {failed or 'none'}", flush=True)
    return 1 if failed or stopping else 0


if __name__ == "__main__":
    sys.exit(main())
