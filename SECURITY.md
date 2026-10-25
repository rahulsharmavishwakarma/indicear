# Security

Please report a vulnerability privately through GitHub's
[security advisory form](https://github.com/rahulsharmavishwakarma/indicear/security/advisories/new),
not in a public issue. Include the affected version and steps to reproduce.

An exported model package is loaded from safetensors, JSON and SentencePiece files only. By default the
transcriber checks every file of a package against the sha256 listed in its `model.json` before
loading it (`--no-verify` turns that off). Training checkpoints are different: the trainer, warm start
and export read them with `torch.load(weights_only=False)`, which can run code, so only load
checkpoints you produced yourself.
