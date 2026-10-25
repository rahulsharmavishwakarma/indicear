"""IndicEAR ASR training code for design A on the frozen EmbeddingGemma 2 audio encoder.

Research code (see README.md). Encoder weights are never modified: layers 1-6 are cached once,
layers 7-12 run every step with trainable residual adapters attached through forward hooks.
"""
