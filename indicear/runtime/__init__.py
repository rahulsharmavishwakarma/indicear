"""Model-agnostic run infrastructure (candidate framework; no imports from ASR modules).

- batching: length-bucketed frame-budget batches, VRAM sizing, OOM backoff, LR scaling
- checkpoint: atomic resumable checkpoints, LATEST.json manifest, signal handling
- shards: sharded row cache with completion markers, df check, VRAM/pinned/mmap placement;
  generic `<stem>.done.json` markers for any sharded job (the data ingest uses them)
- gpu: nvidia-smi sampler, runtime banner
- io: atomic JSON/JSONL writes
- parallel: run several jobs concurrently on one GPU
"""
