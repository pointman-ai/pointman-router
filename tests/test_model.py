"""The decision, the weights check, the device pick and idle unloading, on a fake model: no torch needed."""

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from router import model
from router.model import GATE, Failure, Router, ordered, params_of, pick_device, scores_of

STATE = "@james: the build's red again, can someone look?"


class FakeModel:
    def __init__(self, reply=0.6, no_reply=0.2, route=None):
        self.reply, self.no_reply = reply, no_reply
        self.route = route or {}
        self.calls = []                    # real posts' calls
        self.warm = []                     # the warm-up's, after each load
        self.counted = []                  # count_tokens' calls

    def count_tokens(self, text, candidates):
        """Words in the state and the descriptions: a stand-in for the processor's count."""
        self.counted.append((text, dict(candidates)))
        return len(text.split()) + sum(len(d.split()) for d in candidates.values())

    def classify_text(self, text, tasks, include_confidence=False):
        (self.warm if text in (model.WARM_TEXT, model.WARM_LONG) else self.calls).append((text, tasks, include_confidence))
        out = {}
        if "respond" in tasks:
            out["respond"] = [{"label": "reply needed", "confidence": self.reply},
                              {"label": "no reply needed", "confidence": self.no_reply}]
        if "route" in tasks:
            out["route"] = [{"label": h, "confidence": self.route.get(h, 0.1)} for h in tasks["route"]["labels"]]
        return out


class Weights:
    """A model folder with every file, and model.safetensors' sha256."""

    def __init__(self, body=b"weights"):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "jev-v1"
        (self.dir / "encoder_config").mkdir(parents=True)
        for f in model.FILES:
            (self.dir / f).write_bytes(body if f == "model.safetensors" else b"{}")
        self.sha = hashlib.sha256(body).hexdigest()

    def settings(self, **more):
        return {"model_dir": str(self.dir), "model_name": "jev-v1", "weights_sha256": self.sha, "device": "cpu",
                "threads": 0, "idle_unload_minutes": 0, **more}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def router(settings, fake=None, clock=None):
    fake = fake or FakeModel()
    loads = []

    def loader(d, s):
        loads.append(d)
        return fake, "cpu", 4

    r = Router(settings, loader=loader, clock=clock or Clock())
    return r, fake, loads


def params(**more):
    return {"post": "post_1", "state": STATE, "candidates": {"ci": "Runs the builds", "docs": "Writes the docs"},
            "need_router": True, "cutoff": 0.05, **more}


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def test_the_gate_and_the_router_ask_what_the_model_was_scored_with(self):
        r, fake, _ = router(self.w.settings(), FakeModel(route={"ci": 0.9, "docs": 0.2}))
        a = r.decide(params())
        self.assertEqual(fake.calls[0], (STATE, GATE, True))
        self.assertEqual(GATE, {"respond": {"labels": ["reply needed", "no reply needed"], "multi_label": True,
                                            "cls_threshold": 0.0}})
        self.assertEqual(fake.calls[1], (STATE, {"route": {"labels": {"ci": "Runs the builds", "docs": "Writes the docs"},
                                                           "multi_label": True, "cls_threshold": 0.0}}, True))
        self.assertAlmostEqual(a["p_reply"], 0.75)       # 0.6 of 0.6 + 0.2
        self.assertEqual(a["scores"], {"ci": 0.9, "docs": 0.2})

    def test_the_answer_has_every_field_of_the_interface(self):
        r, _, _ = router(self.w.settings())
        a = r.decide(params())
        self.assertEqual(set(a), {"p_reply", "scores", "model", "weights_sha256", "device", "threads", "ms", "load_ms",
                                  "desc_source", "order_source", "one_call", "earlier"})
        self.assertIs(a["one_call"], False)
        self.assertEqual(a["model"], "jev-v1")
        self.assertEqual(a["weights_sha256"], self.w.sha[:12])
        self.assertEqual((a["device"], a["threads"]), ("cpu", 4))
        self.assertGreaterEqual(a["ms"], 0)

    def test_the_route_call_reads_the_candidates_in_the_order_sent(self):
        r, fake, _ = router(self.w.settings())
        a = r.decide(params(order=["docs", "ci"]))
        self.assertEqual(list(fake.calls[1][1]["route"]["labels"]), ["docs", "ci"])
        self.assertEqual(a["order_source"], "core")

    def test_without_an_order_the_candidates_are_sorted_so_a_post_always_gets_one_answer(self):
        r, fake, _ = router(self.w.settings())
        a = r.decide(params(candidates={"docs": "Writes the docs", "ci": "Runs the builds"}))
        self.assertEqual(list(fake.calls[1][1]["route"]["labels"]), ["ci", "docs"])
        self.assertEqual(a["order_source"], "sorted")

    def test_no_router_unless_asked(self):
        r, fake, _ = router(self.w.settings())
        a = r.decide(params(need_router=False))
        self.assertEqual(a["scores"], {})
        self.assertEqual(len(fake.calls), 1)

    def test_no_router_below_the_cutoff(self):
        r, fake, _ = router(self.w.settings(), FakeModel(reply=0.01, no_reply=0.99))
        a = r.decide(params(cutoff=0.05))
        self.assertAlmostEqual(a["p_reply"], 0.01)
        self.assertEqual(a["scores"], {})
        self.assertEqual(len(fake.calls), 1)

    def test_at_the_cutoff_the_router_runs(self):
        r, fake, _ = router(self.w.settings(), FakeModel(reply=0.05, no_reply=0.95))
        r.decide(params(cutoff=0.05))
        self.assertEqual(len(fake.calls), 2)

    def test_the_cutoff_defaults_to_the_models(self):
        r, fake, _ = router(self.w.settings(), FakeModel(reply=0.04, no_reply=0.96))
        p = params()
        del p["cutoff"]
        self.assertEqual(r.decide(p)["scores"], {})

    def test_no_candidates_means_no_router(self):
        r, fake, _ = router(self.w.settings())
        self.assertEqual(r.decide(params(candidates={}))["scores"], {})
        self.assertEqual(len(fake.calls), 1)

    def test_it_loads_once(self):
        r, _, loads = router(self.w.settings())
        first = r.decide(params())
        r.decide(params())
        self.assertEqual(len(loads), 1)
        self.assertEqual(r.decide(params())["load_ms"], 0)
        self.assertGreaterEqual(first["load_ms"], 0)

    def test_each_load_warms_up_with_a_made_up_decision_counted_in_load_ms(self):
        clock = Clock()
        r, fake, loads = router(self.w.settings(idle_unload_minutes=1), clock=clock)
        r.decide(params())
        self.assertEqual(fake.warm, [(model.WARM_LONG, GATE, True),
                                     (model.WARM_LONG, model.route_task(model.WARM_MANY), True),
                                     (model.WARM_TEXT, GATE, True),
                                     (model.WARM_TEXT, model.route_task(model.WARM_CANDIDATES), True)])
        self.assertEqual([c[0] for c in fake.calls], [STATE, STATE])
        r.decide(params())
        self.assertEqual(len(fake.warm), 4)          # warm already: no warm-up
        clock.t += 120
        r.unload_if_idle()
        r.decide(params())
        self.assertEqual(len(fake.warm), 8)          # loaded again: warmed again

    def test_the_warm_up_is_in_load_ms_and_not_in_the_next_calls_ms(self):
        import time

        class SlowFirstCall(FakeModel):
            def classify_text(self, text, tasks, include_confidence=False):
                if text == model.WARM_TEXT and "respond" in tasks:
                    time.sleep(0.2)                       # compiling kernels
                return super().classify_text(text, tasks, include_confidence)

        r, _, _ = router(self.w.settings(), SlowFirstCall())
        a = r.decide(params())
        self.assertGreaterEqual(a["load_ms"], 200)
        self.assertGreaterEqual(r.warm_ms, 200)
        self.assertLess(a["ms"], 100)

    def test_a_model_that_fails_its_warm_up_isnt_kept(self):
        class Broken(FakeModel):
            def classify_text(self, text, tasks, include_confidence=False):
                raise Failure("out of memory", "model_error")

        r, _, loads = router(self.w.settings(), Broken())
        for _ in range(2):
            with self.assertRaises(Failure):
                r.decide(params())
            self.assertIsNone(r.model)
        self.assertEqual(len(loads), 2)

    def test_a_single_label_answer_is_read_too(self):
        self.assertEqual(scores_of({"route": {"label": "ci", "confidence": 0.7}}, "route"), {"ci": 0.7})



class OneCallTests(unittest.TestCase):
    """one_call: both tasks in one call whenever there's a candidate, the gate first, as the one-call model was
    scored. Off, the calls are the two they always were."""

    SCORES = {"ci": 0.9, "docs": 0.2}
    BOTH = {"respond": {"labels": ["reply needed", "no reply needed"], "multi_label": True, "cls_threshold": 0.0},
            "route": {"labels": {"ci": "Runs the builds", "docs": "Writes the docs"}, "multi_label": True,
                      "cls_threshold": 0.0}}

    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def decide(self, fake=None, on=True, **more):
        fake = fake or FakeModel(route=self.SCORES)
        r, _, _ = router(self.w.settings(one_call=on), fake)
        return r.decide(params(**more)), fake

    def test_one_call_asks_both_tasks_gate_first(self):
        a, fake = self.decide()
        self.assertEqual(fake.calls, [(STATE, self.BOTH, True)])
        self.assertEqual(list(fake.calls[0][1]), ["respond", "route"])
        self.assertAlmostEqual(a["p_reply"], 0.75)
        self.assertEqual(a["scores"], self.SCORES)
        self.assertIs(a["one_call"], True)

    def test_the_one_call_is_made_when_the_router_isnt_asked_for_but_its_scores_arent_kept(self):
        a, fake = self.decide(need_router=False)
        self.assertEqual(fake.calls, [(STATE, self.BOTH, True)])
        self.assertAlmostEqual(a["p_reply"], 0.75)
        self.assertEqual(a["scores"], {})

    def test_under_the_cutoff_means_nobody(self):
        a, fake = self.decide(FakeModel(reply=0.04, no_reply=0.96, route=self.SCORES), cutoff=0.05)
        self.assertEqual(len(fake.calls), 1)
        self.assertAlmostEqual(a["p_reply"], 0.04)
        self.assertEqual(a["scores"], {})

    def test_at_the_cutoff_its_scores_are_kept(self):
        a, _ = self.decide(FakeModel(reply=0.05, no_reply=0.95, route=self.SCORES), cutoff=0.05)
        self.assertEqual(a["scores"], self.SCORES)

    def test_one_candidate_is_one_call_too(self):
        a, fake = self.decide(candidates={"ci": "Runs the builds"})
        self.assertEqual(list(fake.calls[0][1]), ["respond", "route"])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(a["scores"], {"ci": 0.9})

    def test_no_candidate_is_the_gate_alone(self):
        a, fake = self.decide(candidates={})
        self.assertEqual(fake.calls, [(STATE, GATE, True)])
        self.assertEqual(a["scores"], {})

    def test_it_reads_the_candidates_in_the_order_sent_with_the_models_own_text(self):
        (self.w.dir / model.DESCRIPTIONS).write_text(json.dumps({"docs": "The docs, as trained"}))
        a, fake = self.decide(order=["docs", "ci"])
        labels = fake.calls[0][1]["route"]["labels"]
        self.assertEqual(list(labels), ["docs", "ci"])
        self.assertEqual(labels["docs"], "The docs, as trained")
        self.assertEqual((a["order_source"], a["desc_source"]), ("core", {"model": 1, "core": 1}))

    def test_the_trim_counts_the_route_input_as_without_it(self):
        state = "\n".join(["Earlier post, by @a: " + "word " * 30, "Earlier post, by @b: " + "word " * 30,
                           "@james: the build's red again, can someone look?"])
        for on in (False, True):
            with self.subTest(one_call=on):
                fake = FakeModel(route=self.SCORES)
                r, _, _ = router(self.w.settings(one_call=on, fit_tokens=50), fake)
                a = r.decide(params(state=state))
                self.assertEqual(a["earlier"], 1)
                self.assertTrue(all(set(c) == {"ci", "docs"} for _, c in fake.counted))

    def test_calibration_reads_the_one_calls_numbers(self):
        (self.w.dir / model.CALIBRATION).write_text(json.dumps({"temperature": 1.1, "gate_temperature": 1.96}))
        a, _ = self.decide()
        self.assertEqual(a["confidence"], model.confidence(self.SCORES, 1.1))
        self.assertAlmostEqual(a["p_reply_calibrated"], model.calibrated_gate(0.75, 1.96))
        b, _ = self.decide(need_router=False)
        self.assertNotIn("confidence", b)
        self.assertIn("p_reply_calibrated", b)

    def test_the_warm_up_makes_the_one_call(self):
        _, fake = self.decide()
        self.assertEqual(fake.warm, [(model.WARM_LONG, model.both_tasks(model.WARM_MANY), True),
                                     (model.WARM_TEXT, model.both_tasks(model.WARM_CANDIDATES), True)])

    def test_off_is_the_two_calls_as_before(self):
        for settings in ({"one_call": False}, {}):
            with self.subTest(settings=settings):
                fake = FakeModel(route=self.SCORES)
                r, _, _ = router(self.w.settings(**settings), fake)
                a = r.decide(params())
                self.assertEqual(fake.calls, [(STATE, GATE, True),
                                              (STATE, model.route_task({"ci": "Runs the builds",
                                                                        "docs": "Writes the docs"}), True)])
                self.assertEqual(len(fake.warm), 4)
                self.assertIs(a["one_call"], False)

class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def test_no_folder_is_no_weights(self):
        r, _, loads = router(self.w.settings(model_dir=str(self.w.dir.parent / "missing")))
        with self.assertRaises(Failure) as e:
            r.decide(params())
        self.assertEqual(e.exception.code, "no_weights")
        self.assertEqual(loads, [])

    def test_a_missing_file_is_no_weights(self):
        (self.w.dir / "tokenizer.json").unlink()
        r, _, _ = router(self.w.settings())
        with self.assertRaises(Failure) as e:
            r.ensure_loaded()
        self.assertEqual(e.exception.code, "no_weights")
        self.assertIn("tokenizer.json", str(e.exception))

    def test_each_message_starts_with_its_code_once(self):
        self.assertEqual(str(Failure("missing", "no_weights")), "no_weights: missing")
        self.assertEqual(str(Failure("no_device: no GPU", "no_device")), "no_device: no GPU")   # from the worker

    def test_other_weights_are_refused_before_they_load(self):
        r, _, loads = router(self.w.settings(weights_sha256="0" * 64))
        with self.assertRaises(Failure) as e:
            r.decide(params())
        self.assertEqual(e.exception.code, "bad_weights")
        self.assertEqual(loads, [])

    def test_the_hash_is_read_once_while_the_file_is_unchanged(self):
        clock = Clock()
        r, _, loads = router(self.w.settings(idle_unload_minutes=1), clock=clock)
        with mock.patch.object(model, "sha256_of", wraps=model.sha256_of) as sha:
            r.decide(params())
            clock.t += 120
            self.assertTrue(r.unload_if_idle())
            r.decide(params())
        self.assertEqual(len(loads), 2)
        self.assertEqual(sha.call_count, 1)

    def test_a_changed_file_is_hashed_again(self):
        r, _, _ = router(self.w.settings())
        r.check_weights()
        (self.w.dir / "model.safetensors").write_bytes(b"other weights")
        with self.assertRaises(Failure) as e:
            r.check_weights()
        self.assertEqual(e.exception.code, "bad_weights")

    def test_a_home_relative_folder_is_found(self):
        with mock.patch.dict("os.environ", {"HOME": str(self.w.dir.parent)}):
            r, _, _ = router(self.w.settings(model_dir="~/jev-v1"))
            self.assertEqual(r.check_weights(), self.w.sha)


class DescriptionsTests(unittest.TestCase):
    """The model folder's descriptions.json: the text the model was trained with, in place of the job's."""

    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def write(self, text):
        (self.w.dir / model.DESCRIPTIONS).write_text(text if isinstance(text, str) else json.dumps(text))

    def routed_labels(self, fake):
        return [c[1]["route"]["labels"] for c in fake.calls if "route" in c[1]]

    def test_the_models_text_is_used_for_each_candidate_it_lists(self):
        self.write({"ci": "Runs CI, as it was in October", "other": "Not a candidate here"})
        r, fake, _ = router(self.w.settings())
        a = r.decide(params())
        self.assertEqual(self.routed_labels(fake), [{"ci": "Runs CI, as it was in October", "docs": "Writes the docs"}])
        self.assertEqual(a["desc_source"], {"model": 1, "core": 1})
        self.assertEqual(list(a["scores"]), ["ci", "docs"])          # the candidates' order is kept

    def test_without_the_file_the_jobs_text_is_used(self):
        r, fake, _ = router(self.w.settings())
        a = r.decide(params())
        self.assertEqual(self.routed_labels(fake), [{"ci": "Runs the builds", "docs": "Writes the docs"}])
        self.assertEqual(a["desc_source"], {"model": 0, "core": 2})

    def test_the_counts_are_the_candidates_whether_or_not_the_router_runs(self):
        self.write({"ci": "Runs CI", "docs": "Docs"})
        r, fake, _ = router(self.w.settings())
        self.assertEqual(r.decide(params(need_router=False))["desc_source"], {"model": 2, "core": 0})
        self.assertEqual(r.decide(params(candidates={}))["desc_source"], {"model": 0, "core": 0})
        self.assertEqual(self.routed_labels(fake), [])

    def test_a_bad_file_is_refused_before_the_model_loads(self):
        for bad in ("{not json", ["ci"], {"ci": 3}, "\udcff"):
            with self.subTest(bad=bad):
                if bad == "\udcff":
                    (self.w.dir / model.DESCRIPTIONS).write_bytes(b"\xff\xfe{")
                else:
                    self.write(bad)
                r, _, loads = router(self.w.settings())
                with self.assertRaises(Failure) as e:
                    r.decide(params())
                self.assertEqual(e.exception.code, "bad_descriptions")
                self.assertEqual(loads, [])
                self.assertIsNone(r.model)

    def test_the_file_is_read_with_each_load(self):
        clock = Clock()
        self.write({"ci": "first"})
        r, fake, _ = router(self.w.settings(idle_unload_minutes=1), clock=clock)
        r.decide(params())
        self.write({"ci": "second"})
        r.decide(params())                                   # still loaded: the text it loaded with
        clock.t += 120
        self.assertTrue(r.unload_if_idle())
        r.decide(params())
        self.assertEqual([l["ci"] for l in self.routed_labels(fake)], ["first", "first", "second"])


class ConfidenceTests(unittest.TestCase):
    """The model folder's calibration.json: calibrated numbers beside the raw scores and the raw gate."""

    SCORES = {"ci": 0.9, "docs": 0.2}

    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def write(self, text):
        (self.w.dir / model.CALIBRATION).write_text(text if isinstance(text, str) else json.dumps(text))

    def decide(self, **kw):
        r, _, _ = router(self.w.settings(), FakeModel(route=self.SCORES))
        return r.decide(params(**kw))

    def test_it_is_the_softmax_of_the_log_odds_over_the_temperature(self):
        self.write({"temperature": 1.1})
        c = self.decide()["confidence"]
        odds = math.exp((math.log(0.9 / 0.1) - math.log(0.2 / 0.8)) / 1.1)
        self.assertAlmostEqual(c["ci"], odds / (odds + 1))
        self.assertAlmostEqual(c["docs"], 1 / (odds + 1))
        self.assertAlmostEqual(sum(c.values()), 1.0)

    def test_the_raw_scores_and_the_gate_are_the_same_with_or_without_it(self):
        without = self.decide()
        self.write({"temperature": 1.1, "gate_temperature": 1.96})
        with_it = self.decide()
        self.assertEqual(with_it["scores"], self.SCORES)
        self.assertEqual(with_it["scores"], without["scores"])
        self.assertEqual(with_it["p_reply"], without["p_reply"])

    def test_without_the_file_there_is_no_field(self):
        self.assertNotIn("confidence", self.decide())

    def test_no_route_call_means_no_field(self):
        self.write({"temperature": 1.1})
        self.assertNotIn("confidence", self.decide(need_router=False))

    def test_a_lone_candidate_the_route_call_scored_gets_one(self):
        self.write({"temperature": 1.1})
        r, _, _ = router(self.w.settings(), FakeModel(route={"ci": 0.3}))
        self.assertEqual(r.decide(params(candidates={"ci": "Runs the builds"}))["confidence"], {"ci": 1.0})

    def test_scores_at_the_ends_are_clamped(self):
        c = model.confidence({"a": 1.0, "b": 0.0}, 1.1)
        self.assertGreater(c["a"], c["b"])
        self.assertAlmostEqual(sum(c.values()), 1.0)

    def test_the_gate_is_its_log_odds_over_the_gate_temperature(self):
        self.write({"gate_temperature": 1.96})
        a = self.decide()
        self.assertAlmostEqual(a["p_reply"], 0.75)
        self.assertAlmostEqual(a["p_reply_calibrated"], 1 / (1 + math.exp(-math.log(3) / 1.96)))

    def test_the_gate_field_is_on_every_answer(self):
        self.write({"gate_temperature": 1.96})
        self.assertIn("p_reply_calibrated", self.decide(need_router=False))
        self.assertIn("p_reply_calibrated", self.decide(cutoff=0.9))
        self.assertIn("p_reply_calibrated", self.decide(candidates={}))

    def test_the_cutoff_reads_the_raw_gate(self):
        self.write({"gate_temperature": 1.96})
        fake = FakeModel(reply=0.04, no_reply=0.96, route=self.SCORES)
        r, _, _ = router(self.w.settings(), fake)
        a = r.decide(params())
        self.assertAlmostEqual(a["p_reply"], 0.04)
        self.assertGreater(a["p_reply_calibrated"], 0.05)
        self.assertEqual(a["scores"], {})
        self.assertEqual([tasks for _, tasks, _ in fake.calls if "route" in tasks], [])

    def test_each_temperature_gives_only_its_own_field(self):
        cases = (({"temperature": 1.1}, True, False), ({"gate_temperature": 1.96}, False, True), ({}, False, False),
                 ({"temperature": 1.1, "gate_temperature": 1.96}, True, True))
        for body, route, gate in cases:
            with self.subTest(body=body):
                self.write(body)
                a = self.decide()
                self.assertEqual("confidence" in a, route)
                self.assertEqual("p_reply_calibrated" in a, gate)

    def test_the_gate_at_the_ends_is_clamped(self):
        self.assertAlmostEqual(model.calibrated_gate(1.0, 1.96), model.calibrated_gate(1 - 1e-6, 1.96))
        self.assertAlmostEqual(model.calibrated_gate(0.0, 1.96), model.calibrated_gate(1e-6, 1.96))
        self.assertAlmostEqual(model.calibrated_gate(0.5, 1.96), 0.5)

    def test_a_very_small_gate_temperature_does_not_overflow(self):
        self.assertAlmostEqual(model.calibrated_gate(0.9, 1e-6), 1.0)
        self.assertAlmostEqual(model.calibrated_gate(0.1, 1e-6), 0.0)

    def test_a_bad_file_is_refused_before_the_model_loads(self):
        for bad in ("{not json", ["x"], {"temperature": 0}, {"temperature": -1}, {"temperature": True},
                    {"temperature": "1.1"}, {"temperature": None}, '{"temperature": NaN}', {"gate_temperature": 0},
                    {"gate_temperature": "1.96"}, {"temperature": 1.1, "gate_temperature": -1}):
            with self.subTest(bad=bad):
                self.write(bad)
                r, _, loads = router(self.w.settings())
                with self.assertRaises(Failure) as e:
                    r.decide(params())
                self.assertEqual(e.exception.code, "bad_calibration")
                self.assertEqual(loads, [])


class IdleTests(unittest.TestCase):
    def setUp(self):
        self.w = Weights()
        self.clock = Clock()

    def tearDown(self):
        self.w.tmp.cleanup()

    def test_it_unloads_after_the_idle_minutes_and_loads_again(self):
        r, _, loads = router(self.w.settings(idle_unload_minutes=15), clock=self.clock)
        r.decide(params())
        self.clock.t += 14 * 60
        self.assertFalse(r.unload_if_idle())
        r.decide(params())                     # a post resets the wait
        self.clock.t += 14 * 60
        self.assertFalse(r.unload_if_idle())
        self.clock.t += 2 * 60
        self.assertTrue(r.unload_if_idle())
        self.assertIsNone(r.model)
        r.decide(params())
        self.assertEqual(len(loads), 2)

    def test_zero_keeps_it_loaded(self):
        r, _, _ = router(self.w.settings(idle_unload_minutes=0), clock=self.clock)
        r.decide(params())
        self.clock.t += 7 * 24 * 3600
        self.assertFalse(r.unload_if_idle())
        self.assertIsNotNone(r.model)

    def test_nothing_loaded_is_nothing_to_unload(self):
        r, _, _ = router(self.w.settings(idle_unload_minutes=15), clock=self.clock)
        self.clock.t += 3600
        self.assertFalse(r.unload_if_idle())


class ParamsTests(unittest.TestCase):
    def test_bad_params_name_the_field_never_the_text(self):
        bad = [params(state=""), params(state=None), params(candidates=["ci"]), params(candidates={"ci": 1}),
               params(need_router="yes"), params(cutoff=2), params(cutoff=True), params(post=7)]
        no_router = params()
        del no_router["need_router"]
        for p in bad + [no_router]:
            with self.subTest(p=p), self.assertRaises(Failure) as e:
                params_of(p)
            self.assertEqual(e.exception.code, "bad_params")
            self.assertNotIn("build", str(e.exception))

    def test_an_order_must_be_exactly_the_candidates(self):
        cands = {"ci": "Runs the builds", "docs": "Writes the docs"}
        for order in ["ci", ["ci"], ["ci", "docs", "ops"], ["ci", "ci"], ["ci", 7], {"ci": 1, "docs": 2}]:
            with self.subTest(order=order), self.assertRaises(Failure) as e:
                ordered(cands, order)
            self.assertEqual(e.exception.code, "bad_params")
        self.assertEqual(ordered(cands, ["docs", "ci"]), ({"docs": "Writes the docs", "ci": "Runs the builds"}, "core"))
        self.assertEqual(ordered({}, None), ({}, "sorted"))

    def test_good_params(self):
        self.assertEqual(params_of(params()), ("post_1", STATE, {"ci": "Runs the builds", "docs": "Writes the docs"},
                                               True, 0.05))


FAKE_WORKER = r"""
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "load" in req:
        print(json.dumps({"ok": {"device": "cpu", "threads": 2}}), flush=True)
    elif req["text"] == "die":
        sys.exit(3)
    elif req["text"] == "fail":
        print(json.dumps({"error": "out of memory", "code": "model_error"}), flush=True)
    else:
        print(json.dumps({"ok": {"respond": [{"label": "reply needed", "confidence": 0.9},
                                             {"label": "no reply needed", "confidence": 0.1}]}}), flush=True)
"""


class WorkerTests(unittest.TestCase):
    def worker(self):
        w = model.Worker([sys.executable, "-c", FAKE_WORKER])
        self.addCleanup(w.close)
        return w

    def test_it_answers_each_call(self):
        w = self.worker()
        self.assertEqual(w.ask({"load": {}}), {"device": "cpu", "threads": 2})
        self.assertEqual(scores_of(w.classify_text("hi", GATE), "respond")["reply needed"], 0.9)
        self.assertTrue(w.alive)

    def test_an_error_keeps_its_code(self):
        with self.assertRaises(Failure) as e:
            self.worker().classify_text("fail", GATE)
        self.assertEqual(e.exception.code, "model_error")

    def test_a_stopped_process_is_started_again_on_the_next_post(self):
        w = Weights()
        self.addCleanup(w.tmp.cleanup)
        workers = []

        def loader(d, s):
            workers.append(self.worker())
            return workers[-1], "cpu", 2

        r = Router(w.settings(), loader=loader)
        with self.assertRaises(Failure) as e:
            r.decide(params(state="die"))
        self.assertEqual(e.exception.code, "model_stopped")
        self.assertFalse(workers[0].alive)
        self.assertAlmostEqual(r.decide(params(need_router=False))["p_reply"], 0.9)
        self.assertEqual(len(workers), 2)

    def test_unloading_stops_the_process(self):
        w = Weights()
        self.addCleanup(w.tmp.cleanup)
        clock = Clock()
        workers = []

        def loader(d, s):
            workers.append(self.worker())
            return workers[-1], "cpu", 2

        r = Router(w.settings(idle_unload_minutes=1), loader=loader, clock=clock)
        r.decide(params(need_router=False))
        clock.t += 61
        self.assertTrue(r.unload_if_idle())
        self.assertFalse(workers[0].alive)


class PriorityTests(unittest.TestCase):
    def test_a_background_worker_runs_at_nice_10_whichever_thread_starts_it(self):
        import os
        import threading

        got = {}

        def start():                       # as the lane's thread does, on a first post or after a crash
            w = model.Worker()
            try:
                try:
                    w.ask({"load": {"model_dir": "/nonexistent", "settings": {"background": True, "device": "cpu"}}})
                except Failure:
                    pass                   # no torch here, or no checkpoint: the priority is set before either
                got["nice"] = os.getpriority(os.PRIO_PROCESS, w.proc.pid)
            finally:
                w.close()

        mine = os.getpriority(os.PRIO_PROCESS, 0)
        t = threading.Thread(target=start)
        t.start()
        t.join(60)
        self.assertEqual(got["nice"], max(mine, 10))
        self.assertEqual(os.getpriority(os.PRIO_PROCESS, 0), mine)

    def test_no_background_keeps_the_priority(self):
        from router import worker

        with mock.patch.object(worker.os, "setpriority") as set_it, \
                mock.patch.object(worker.os, "getpriority", return_value=0):
            worker.prioritise({"background": False})
            set_it.assert_not_called()
            worker.prioritise({"background": True})
            set_it.assert_called_once_with(worker.os.PRIO_PROCESS, 0, 10)


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


class DeviceTests(unittest.TestCase):
    def test_auto_picks_cuda_where_theres_an_nvidia_driver(self):
        self.assertEqual(pick_device("auto", fake_torch(cuda=True), nvidia=True, apple=False), "cuda")

    def test_a_driver_torch_cant_reach_is_an_error_not_the_cpu(self):
        with self.assertRaises(Failure) as e:
            pick_device("auto", fake_torch(cuda=False), nvidia=True, apple=False)
        self.assertEqual(e.exception.code, "no_device")

    def test_auto_picks_the_apple_gpu_on_a_mac(self):
        self.assertEqual(pick_device("auto", fake_torch(mps=True), nvidia=False, apple=True), "mps")
        with self.assertRaises(Failure):
            pick_device("auto", fake_torch(mps=False), nvidia=False, apple=True)

    def test_auto_elsewhere_is_the_cpu(self):
        self.assertEqual(pick_device("auto", fake_torch(), nvidia=False, apple=False), "cpu")

    def test_a_setting_wins(self):
        self.assertEqual(pick_device("cpu", fake_torch(cuda=True), nvidia=True, apple=False), "cpu")
        with self.assertRaises(Failure):
            pick_device("cuda", fake_torch(), nvidia=False, apple=False)



FIT_STATE = "\n".join([
    "Thread: Loft lighting",
    "Earlier post, by @scene: one two three four five",
    "Earlier post, by @editor: six seven eight nine ten",
    "Previous post, by @james: eleven twelve",
    "New post, by @james: too blue?",
])


class FitTests(unittest.TestCase):
    """The earlier posts that fit (ctxfit.py's rule): the oldest go first, at least one stays, the gate and the
    router read the same trimmed state, and it's off unless fit_tokens is set."""

    def setUp(self):
        self.w = Weights()

    def tearDown(self):
        self.w.tmp.cleanup()

    def test_it_is_off_unless_fit_tokens_is_set(self):
        r, fake, _ = router(self.w.settings())
        a = r.decide(params(state=FIT_STATE))
        self.assertEqual(fake.counted, [])
        self.assertEqual(fake.calls[0][0], FIT_STATE)
        self.assertEqual(a["earlier"], 3)

    def test_the_oldest_earlier_posts_go_until_it_fits(self):
        words = lambda s: len(s.split())
        full = words(FIT_STATE) + words("Runs the builds") + words("Writes the docs")
        r, fake, _ = router(self.w.settings(fit_tokens=full - 1))       # one word over: the oldest goes
        a = r.decide(params(state=FIT_STATE))
        read = fake.calls[0][0]
        self.assertNotIn("@scene", read)
        self.assertIn("@editor", read)
        self.assertEqual(a["earlier"], 2)
        self.assertEqual([c[0] for c in fake.calls], [read, read], "the gate and the router read the same state")
        self.assertTrue(read.startswith("Thread: Loft lighting") and read.endswith("New post, by @james: too blue?"))

    def test_at_least_one_earlier_post_stays(self):
        r, fake, _ = router(self.w.settings(fit_tokens=1))
        a = r.decide(params(state=FIT_STATE))
        self.assertEqual(a["earlier"], 1)
        self.assertIn("Previous post, by @james", fake.calls[0][0])

    def test_it_counts_with_the_final_candidates_and_the_models_own_text(self):
        (self.w.dir / model.DESCRIPTIONS).write_text(json.dumps({"ci": "Builds and tests every push on the Air"}))
        r, fake, _ = router(self.w.settings(fit_tokens=1000))
        r.decide(params(state=FIT_STATE))
        self.assertEqual(fake.counted[-1][1], {"ci": "Builds and tests every push on the Air", "docs": "Writes the docs"})

    def test_a_state_with_no_earlier_post_is_left_as_it_is(self):
        r, fake, _ = router(self.w.settings(fit_tokens=1))
        a = r.decide(params())
        self.assertEqual(fake.calls[0][0], STATE)
        self.assertEqual(a["earlier"], 0)

    def test_the_route_schema_is_the_one_training_counted_with(self):
        self.assertEqual(model.route_schema({"ci": "Runs the builds"}),
                         {"classifications": [{"task": "route", "labels": ["ci"], "multi_label": True,
                                               "label_descriptions": {"ci": "Runs the builds"}}]})
        self.assertEqual(model.route_schema({}),
                         {"classifications": [{"task": "route", "labels": ["nobody"], "multi_label": True,
                                               "label_descriptions": {"nobody": ""}}]})


if __name__ == "__main__":
    unittest.main()
