"""Placeholder for openai-clip: DiP uses a DistilBERT text encoder, so CLIP is never called.

The CLoSD model module imports `clip` at top level; this shim satisfies the import and
fails loudly if anything actually tries to use it.
"""


def load(*args, **kwargs):
    raise RuntimeError("openai-clip is not installed; this DiP checkpoint uses the BERT text encoder")


def tokenize(*args, **kwargs):
    raise RuntimeError("openai-clip is not installed")
