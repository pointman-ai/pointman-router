"""The extension's process: the node starts it with `python -m hello` and talks to it over stdin and
stdout (pointman_extension handles that). Log with ext.log(...): stdout is the node's."""

from pointman_extension import Extension

ext = Extension()


@ext.command("hello.greet")
def greet(job):
    text = f"{ext.settings['greeting']}, {job.params['name']}!"
    job.progress(0.5, "writing")
    out = job.work / "greeting.txt"
    out.write_text(text + "\n", encoding="utf-8")
    job.output(out, kind="text")           # posted where the request came from
    ext.event("hello.last", {"name": job.params["name"]})
    return {"text": text}


ext.run()
