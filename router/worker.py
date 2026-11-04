"""The model's own process: `python -m router.worker`, started by the extension, one JSON line in, one out.

Holding the model in a child process means unloading it gives all of its memory back (torch keeps much of a
process's heap after the model is gone), and a crash in torch costs one decision, not the extension. The
first line asks it to load ({"load": {"model_dir", "settings"}}); each line after is one classify_text call
({"text", "tasks"}). Every answer is {"ok": …} or {"error", "code"}. Its own stdout goes to stderr (the
node's log), so nothing a library prints can get into the answers.
"""

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    answers = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)                          # print() and C libraries now write to the log
    sys.stdout = sys.stderr

    from .model import Failure, load_gliner

    def say(obj):
        answers.write(json.dumps(obj) + "\n")
        answers.flush()

    model = None
    for line in sys.stdin:
        req = json.loads(line)
        try:
            if "load" in req:
                model, device, threads = load_gliner(Path(req["load"]["model_dir"]), req["load"]["settings"])
                say({"ok": {"device": device, "threads": threads}})
            else:
                say({"ok": model.classify_text(req["text"], req["tasks"], include_confidence=True)})
        except Failure as e:
            say({"error": str(e), "code": e.code})
        except Exception as e:  # noqa: BLE001 - the extension reports it as the decision's failure
            traceback.print_exc()
            say({"error": f"{type(e).__name__}: {e}"[:500], "code": "model_error"})


if __name__ == "__main__":
    main()
