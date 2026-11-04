"""The extension's process as a node runs it, through the SDK's stand-in node.

Needs the node's Python SDK, where $POINTMAN_EXTENSIONS_SDK names its folder; skipped elsewhere. The real
model's test also needs the package's environment (uv sync) and $ROUTER_TEST_MODEL_DIR, a folder with the
weights: it never runs in CI.
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = os.environ.get("POINTMAN_EXTENSIONS_SDK")
if SDK:
    sys.path.insert(0, SDK)

try:
    from pointman_extension_testing import ExtensionFailed, StandInNode
except ImportError:
    StandInNode = None

MODEL_DIR = os.environ.get("ROUTER_TEST_MODEL_DIR")
PARAMS = {"post": "post_1", "state": "@james: can someone look at the build?",
          "candidates": {"ci": "Runs the builds and their checks", "docs": "Writes the docs site"},
          "need_router": True, "cutoff": 0.05}


@unittest.skipUnless(StandInNode, "needs the node's Python SDK ($POINTMAN_EXTENSIONS_SDK)")
class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def node(self, **settings):
        return StandInNode(ROOT, settings={"model_dir": str(Path(self.tmp.name) / "none"), **settings})

    def test_it_says_hello_at_once_without_weights(self):
        t = time.monotonic()
        with self.node() as node:
            self.assertEqual((node.info.id, node.info.version), ("router", "0.1.2"))
        self.assertLess(time.monotonic() - t, 10)

    def test_no_weights_is_a_failure_a_caller_can_branch_on(self):
        with self.node() as node:
            with self.assertRaises(ExtensionFailed) as e:
                node.run("router.decide", PARAMS)
            self.assertEqual(e.exception.code, "no_weights")
            self.assertTrue(str(e.exception).startswith("no_weights: "), str(e.exception))
            self.assertNotIn("build", node.log)

    def test_other_weights_are_refused(self):
        d = Path(self.tmp.name) / "jev-v1"
        (d / "encoder_config").mkdir(parents=True)
        for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            (d / f).write_bytes(b"{}")
        with self.node(model_dir=str(d), weights_sha256="0" * 64) as node:
            with self.assertRaises(ExtensionFailed) as e:
                node.run("router.decide", PARAMS)
            self.assertEqual(e.exception.code, "bad_weights")

    def test_bad_params_say_so(self):
        with self.node() as node:
            with self.assertRaises(ExtensionFailed) as e:
                node.run("router.decide", {**PARAMS, "need_router": "yes"})
            self.assertEqual(e.exception.code, "bad_params")

    @unittest.skipUnless(MODEL_DIR, "needs $ROUTER_TEST_MODEL_DIR and the package's environment")
    def test_the_real_model_decides(self):
        from router.model import sha256_of

        sha = sha256_of(Path(MODEL_DIR) / "model.safetensors")
        with StandInNode(ROOT, settings={"model_dir": MODEL_DIR, "weights_sha256": sha, "model_name": "jev-v1",
                                         "device": os.environ.get("ROUTER_TEST_DEVICE", "auto")}) as node:
            a = node.run("router.decide", PARAMS, timeout=120).result
            self.assertTrue(0 <= a["p_reply"] <= 1)
            self.assertEqual(a["weights_sha256"], sha[:12])
            if a["p_reply"] >= 0.05:
                self.assertEqual(set(a["scores"]), {"ci", "docs"})
            b = node.run("router.decide", {**PARAMS, "need_router": False}).result
            self.assertEqual(b["scores"], {})
            self.assertEqual(b["load_ms"], 0)
            self.assertAlmostEqual(a["p_reply"], b["p_reply"], places=4)
            self.assertNotIn("build", node.log)


if __name__ == "__main__":
    unittest.main()
