"""router/slim.py: the model and its checkpoint without the span and count heads.

The file test needs torch and safetensors (the package's environment, `uv run`); the model test also needs
$ROUTER_TEST_MODEL_DIR, a checkpoint folder. CI has neither, so both skip there.
"""

import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None

from router.slim import HEADS, slim_checkpoint

MODEL_DIR = os.environ.get("ROUTER_TEST_MODEL_DIR")
STATE = "@james: can someone look at the build?"


@unittest.skipUnless(torch, "needs torch and safetensors")
class SlimFileTests(unittest.TestCase):
    def test_only_the_heads_tensors_go_and_the_rest_and_the_metadata_stay(self):
        t = {"encoder.w": torch.arange(6.).reshape(2, 3), "classifier.0.weight": torch.ones(4, 2),
             "span_rep.x": torch.ones(5), "count_pred.0.weight": torch.ones(2, 2), "count_embed.gru.w": torch.ones(3)}
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "full.safetensors", Path(d) / "slim.safetensors"
            save_file(t, str(src), metadata={"format": "pt"})
            counts = slim_checkpoint(src, dst)
            got = load_file(str(dst))
            with safe_open(str(dst), framework="pt") as f:
                meta = f.metadata()
        self.assertEqual(sorted(got), ["classifier.0.weight", "encoder.w"])
        self.assertTrue(torch.equal(got["encoder.w"], t["encoder.w"]))
        self.assertEqual(meta, {"format": "pt"})
        self.assertEqual(counts, {"kept": 2, "dropped": 3, "kept_params": 14, "dropped_params": 12})


@unittest.skipUnless(MODEL_DIR and torch, "needs $ROUTER_TEST_MODEL_DIR and the package's environment")
class SlimModelTests(unittest.TestCase):
    def test_it_builds_no_heads_and_gates_as_the_full_model(self):
        from router.model import GATE
        from router.slim import load_slim

        with safe_open(str(Path(MODEL_DIR) / "model.safetensors"), framework="pt") as f:
            has_heads = any(k.startswith(HEADS) for k in f.keys())
        slim = load_slim(MODEL_DIR).eval()
        self.assertFalse([h for h in ("span_rep", "count_pred", "count_embed") if hasattr(slim, h)])
        got = slim.classify_text(STATE, GATE, include_confidence=True)
        if has_heads:                     # a full checkpoint: gliner2's own loader answers exactly the same
            from gliner2 import AutoExtractor
            want = AutoExtractor.from_pretrained(MODEL_DIR).eval().classify_text(STATE, GATE, include_confidence=True)
            self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
