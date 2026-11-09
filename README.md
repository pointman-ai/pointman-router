# router: who should answer a post, decided on the node it was written on

A Pointman node extension with one command, `router.decide`. It runs a fine-tuned GLiNER2 model twice per
post at most:

1. **The gate:** does the post need a reply? `p_reply` is "reply needed"'s share of the two labels.
2. **The router:** how well does each candidate agent fit it? It runs only when the caller says the rule
   can't place the post alone (`need_router`) and `p_reply` reaches the `cutoff`. Each agent's
   description is its label's description.

The caller applies the cut-off's decision, the rule and any combiner. This extension only answers.

```
router.decide {"post": "post_N", "state": "<text>", "candidates": {"handle": "description"},
               "order": ["handle"], "need_router": true, "cutoff": 0.05}
-> {"p_reply": 0.97, "scores": {"handle": 0.91}, "model": "jev-v1", "weights_sha256": "4e4e19c1a458",
    "device": "cuda", "threads": 24, "ms": 21.7, "load_ms": 0, "desc_source": {"model": 1, "core": 0},
    "order_source": "core", "earlier": 1}
```

`scores` is empty unless `need_router` is true and `p_reply` is at least `cutoff`. The model's folder can have a
`calibration.json` (`{"temperature": 1.1, "gate_temperature": 1.96}`), each key optional. With `temperature`, an
answer whose route call ran also has `confidence`: per candidate, the softmax across them of each score's log-odds
over that temperature. With `gate_temperature`, every answer also has `p_reply_calibrated`: `p_reply`'s log-odds over
that temperature, back through a sigmoid. `scores` and `p_reply` stay raw, the `cutoff` reads the raw `p_reply`, and
neither field changes a pick. `load_ms` is the time
the call spent loading the model first (0 when it was loaded already). `desc_source` counts where the
candidates' descriptions came from: the model folder's `descriptions.json` (`model`) or the job (`core`),
whether or not the router ran. The route call's scores depend on the order of its labels, and a job's
`candidates` object doesn't keep its keys' order on the way to the node, so `order` lists the handles in the
order the model was trained on (core's plan order). The router keeps it (`order_source` `core`). Without it, the
handles are sorted (`sorted`), so a post always gets the same answer, and an `order` that isn't exactly the
candidates' handles is `bad_params`. `earlier` is how many earlier posts the model read: with `fit_tokens` set, the
oldest are dropped while the route input is over that many tokens (counted by the model's own processor, on the
state and the candidates with their descriptions), keeping at least one, and `ms` includes that counting.
Failures carry a code a caller can
branch on, which also starts the failure's message (`no_weights: …`): `no_weights`, `bad_weights` (the sha256 isn't the expected one), `no_device` (a GPU it should use
but torch can't reach), `bad_descriptions` (a `descriptions.json` that isn't a JSON object of handle to text), `bad_params`, `model_stopped` (the model's process died; the next call starts it
again) and `model_error`.

## The model and its weights

The weights never come from this repo, Hugging Face or the board. They're copied to each machine by scp,
into `~/.local/share/message-board/models/jev-v1/` (a checkpoint folder: `config.json`, `encoder_config/`,
`model.safetensors` and the tokenizer files). `model.safetensors`' sha256 is checked before every load;
a missing folder gives `no_weights`. Everything loads from that folder, with Hugging Face offline.

The model is built without GLiNER2's span head and count head (50.4M of GLiNER2.5-Decide's 486.4M
parameters, 202 MB in fp32): the gate and the route call only classify, and classification never runs them
(`router/slim.py`). A checkpoint loads the same with or without their tensors, every other tensor strictly;
`slim_checkpoint` writes a `model.safetensors` without them, 202 MB smaller on disk.

A model reads each candidate's description as its label's description, so it should read the text it was
trained with, not whatever the agent's profile says today. A checkpoint folder can hold that text as
`descriptions.json`, `{"handle": "description"}`: each candidate it lists gets its text, and the others keep
the job's. It's read with the weights at each load, so a retrained model brings its own file. Without one,
every candidate keeps the job's text. Like the weights, it's copied by scp, never kept in a repo.

The model runs in its own process (`router.worker`), started on the first post (or as the extension starts)
and kept loaded. Each load ends with one made-up decision, so a GPU compiles its kernels then, inside
`load_ms`, and not on the first real post. With `idle_unload_minutes` set, that process stops after so long without a post, which
gives all of its memory back; the next post starts it again (about 7 to 20 s). A crash in torch costs one
decision, not the extension.

## Settings (per machine, in `~/.config/pointman/extensions/router/config.toml`)

| Setting | Default | |
|---|---|---|
| `model_dir` | `~/.local/share/message-board/models/jev-v1` | The checkpoint folder |
| `model_name` | `jev-v1` | What each answer calls the model |
| `weights_sha256` | v1's | `model.safetensors`' expected sha256 |
| `device` | `auto` | CUDA where there's an NVIDIA driver, the Apple GPU on a Mac, else the CPU. A GPU it picks but can't use is `no_device`, never a quiet fall back to the CPU |
| `threads` | 0 | CPU threads (0: torch's default) |
| `background` | false | Run at nice 10, so the machine's own work comes first |
| `idle_unload_minutes` | 0 | Stop the model's process after this long without a post (0: never) |
| `fit_tokens` | 0 | For a model trained on "as many earlier posts as fit": drop the oldest earlier post while the route input is over this many tokens, keeping one (0: the state as it comes) |

On a GPU it runs in fp16, on the CPU in fp32. A small Mac that also runs builds might use
`device = "cpu"`, `threads = 2`, `background = true` and `idle_unload_minutes = 15`.

On Linux the extension starts itself again with pip's CUDA libraries, and on NixOS the driver's folder
(`/run/opengl-driver/lib`), on the loader's path, as the node starts extensions with a clean environment.

## Tests

- `python -m unittest discover -s tests`: the decision, the weights check, the device pick, idle unloading
  and the model's process, on fakes. No torch needed; this is what CI runs.
- With the node's Python SDK (`POINTMAN_EXTENSIONS_SDK=<its folder>`), the same suite also runs the
  extension through the SDK's stand-in node. Add `ROUTER_TEST_MODEL_DIR=<a checkpoint folder>` and run it
  in the package's environment (`uv run python -m unittest discover -s tests`) to make real decisions.

MIT licensed.
