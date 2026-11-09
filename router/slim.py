"""GLiNER2.5-Decide without the two heads the router never runs: the span head and the count head with its GRU.

The router only classifies (the gate, then the route call). In gliner2 2.1.0 a classification task reads the
encoder's vectors at its [L] positions and scores each with `classifier`; it never reaches `span_rep`,
`count_pred` or `count_embed` (inference/runtime.py: classification tasks are skipped before the count head, and
span scoring runs only when a span task has queries). Those three hold 50,421,780 of the checkpoint's 486,444,053
parameters (202 MB in fp32), so this model never builds them.

It loads a checkpoint with or without them: their keys are dropped before a strict load, and every other key
must match exactly. A call that did reach a dropped head would fail loudly (no such attribute), never quietly use
random weights.

`slim_checkpoint` writes such a file from a full one.
"""

from __future__ import annotations

from pathlib import Path

HEADS = ("span_rep.", "count_pred.", "count_embed.")
GLINER2 = "2.1.0"           # __init__ below mirrors this version's SpanExtractorModel.__init__


def _extractor_class():
    import gliner2
    from gliner2.inference.engine import SpanExtractor
    from gliner2.layers import create_mlp
    from gliner2.models.span.model import SpanExtractorModel
    from gliner2.processor import SchemaTransformer

    if gliner2.__version__ != GLINER2:
        raise RuntimeError(f"the slim model mirrors gliner2 {GLINER2}'s span model, not {gliner2.__version__}")

    class SlimExtractor(SpanExtractor):
        """gliner2's span extractor with only the encoder and the classifier."""

        def __init__(self, config, encoder_config=None, tokenizer=None, use_flashdeberta=None, word_splitter=None):
            # SpanExtractorModel.__init__ (gliner2 2.1.0), minus span_rep, count_pred and count_embed
            super(SpanExtractorModel, self).__init__(config)
            self.config = config
            self.max_width = config.max_width
            self.processor = SchemaTransformer(
                tokenizer=tokenizer, token_pooling=config.token_pooling, word_splitter=word_splitter,
            ) if tokenizer is not None else SchemaTransformer(
                config.model_name, token_pooling=config.token_pooling, word_splitter=word_splitter,
            )
            self.encoder = self._load_encoder(config.model_name, encoder_config,
                                              getattr(config, "attn_implementation", "sdpa"),
                                              use_flashdeberta=use_flashdeberta)
            self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
            self.hidden_size = self.encoder.config.hidden_size
            self.classifier = create_mlp(input_dim=self.hidden_size, intermediate_dims=[self.hidden_size * 2],
                                         output_dim=1, dropout=0., activation="relu", add_layer_norm=False)
            self._lora_layers = {}
            self._adapter_config = None

        def task_module_names(self) -> tuple:
            return ("classifier",)

        def load_state_dict(self, state_dict, strict=True, assign=False):
            kept = {k: v for k, v in state_dict.items() if not k.startswith(HEADS)}
            return super().load_state_dict(kept, strict=strict, assign=assign)

    return SlimExtractor


def load_slim(model_dir: str | Path):
    """The model in `model_dir` as gliner2's AutoExtractor would load it, without the span and count heads."""
    return _extractor_class().from_pretrained(str(model_dir))


def slim_checkpoint(src: str | Path, dst: str | Path) -> dict:
    """Write `src` (a model.safetensors) to `dst` without the heads' tensors, keeping its metadata. Returns counts."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    kept, dropped = {}, {}
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            t = f.get_tensor(k)
            (dropped if k.startswith(HEADS) else kept)[k] = t
    save_file(kept, str(dst), metadata=meta)
    return {"kept": len(kept), "dropped": len(dropped),
            "kept_params": sum(t.numel() for t in kept.values()),
            "dropped_params": sum(t.numel() for t in dropped.values())}
