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
               "need_router": true, "cutoff": 0.05}
-> {"p_reply": 0.97, "scores": {"handle": 0.91}, "model": "jev-v1", "weights_sha256": "4e4e19c1a458",
    "device": "cuda", "threads": 24, "ms": 21.7, "load_ms": 0}
```

`scores` is empty unless `need_router` is true and `p_reply` is at least `cutoff`. `load_ms` is the time
the call spent loading the model first (0 when it was loaded already). Failures carry a code a caller can
branch on, which also starts the failure's message (`no_weights: …`): `no_weights`, `bad_weights` (the sha256 isn't the expected one), `no_device` (a GPU it should use
but torch can't reach), `bad_params`, `model_stopped` (the model's process died; the next call starts it
again) and `model_error`.

## The model and its weights

The weights never come from this repo, Hugging Face or the board. They're copied to each machine by scp,
into `~/.local/share/message-board/models/jev-v1/` (a checkpoint folder: `config.json`, `encoder_config/`,
`model.safetensors` and the tokenizer files). `model.safetensors`' sha256 is checked before every load;
a missing folder gives `no_weights`. Everything loads from that folder, with Hugging Face offline.

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
