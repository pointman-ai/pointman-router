"""The router extension's process: `python -m router`, started and spoken to by the node.

`router.decide` answers whether a post needs a reply (p_reply) and, when asked, how well each candidate agent
fits it (scores). The node it runs on is the one the post was written on; the caller applies the cut-off's
decision, the rule and any combiner. The model loads once and stays loaded; with idle_unload_minutes set, it
frees its memory after that long without a post. Logs carry post ids and numbers, never a post's text.
"""

from .libpaths import ensure

ensure()  # before anything loads a native library

import threading  # noqa: E402

from pointman_extension import Extension  # noqa: E402

from .model import Failure, Router  # noqa: E402

ext = Extension()
_router: Router | None = None
_made = threading.Lock()


def router() -> Router:
    global _router
    with _made:
        if _router is None:
            _router = Router(ext.settings)
        return _router


@ext.on_start
def start() -> None:
    if int(ext.settings.get("idle_unload_minutes") or 0):
        return  # it loads on its first post, so a restart doesn't take the memory back
    r = router()
    try:
        ms = r.ensure_loaded()
    except Failure as e:
        ext.log(f"router: not loaded ({e.code}): {e}")
    except Exception as e:  # noqa: BLE001 - the first post reports it as its failure
        ext.log(f"router: not loaded: {type(e).__name__}: {e}")
    else:
        ext.log(f"router: {ext.settings.get('model_name')} loaded on {r.device}, {r.threads} threads, in {ms:.0f} ms "
                f"(warm-up {r.warm_ms:.0f} ms), with {len(r.descriptions)} of its own descriptions")


@ext.command("router.decide")
def decide(job):
    answer = router().decide(job.params)
    src = answer["desc_source"]
    ext.log(f"router.decide {job.params.get('post', '')}: p_reply {answer['p_reply']:.3f}, "
            f"{len(answer['scores'])} scored (descriptions: {src['model']} model, {src['core']} core), "
            f"{answer['ms']:.0f} ms on {answer['device']}"
            + (f", after loading for {answer['load_ms']:.0f} ms (warm-up {router().warm_ms:.0f} ms)"
               if answer["load_ms"] else ""))
    return answer


def idle() -> None:
    if _router is not None and _router.unload_if_idle():
        ext.log("router: unloaded after idle_unload_minutes without a post")


ext.every(30, idle, lane="main")
ext.run()
