# LoRA Adapter Support

oMLX can load a LoRA adapter alongside a base model. Adapters live in their own root at `~/.omlx/adapters/<name>/`, separate from base models at `~/.omlx/models/<name>/`. Each adapter declares which bases it supports via an `omlx.json` sidecar. To swap adapters, change the `adapter_id` setting on the base model (or update a symlink in `~/.omlx/adapters/`) and trigger a reload.

## Directory Layout

```
~/.omlx/
├── models/
│   └── gpt-oss-20b-MXFP4-Q8/            # base model
│       ├── config.json
│       └── model-00001-of-00003.safetensors
└── adapters/
    └── rnd-001/                          # LoRA adapter
        ├── adapter_config.json           # PEFT/mlx-lm format (rank, keys, etc.)
        ├── adapters.safetensors          # weights
        └── omlx.json                     # oMLX sidecar — required
```

Adapters never live under `~/.omlx/models/`. Any adapter directory placed there is silently ignored. The split root is intentional — discovery treats the two roots as completely separate namespaces.

## omlx.json sidecar (required)

Minimum contents:

```json
{
  "supported_models": ["gpt-oss-20b-MXFP4-Q8"]
}
```

`supported_models` is a list — an adapter can declare compatibility with multiple bases. Each entry is matched against discovered bases by direct `model_id` first, then by basename (strips everything before the last `/`, so `"mlx-community/gpt-oss-20b-MXFP4-Q8"` matches a locally-discovered dir named `gpt-oss-20b-MXFP4-Q8`).

If `omlx.json` is missing or malformed, oMLX logs a warning at startup and skips the adapter. You must write the sidecar manually after training (or have your training pipeline produce it) — there is no fallback.

Example — one adapter compatible with two quant variants of the same base:

```json
{
  "supported_models": [
    "gpt-oss-20b-MXFP4-Q4",
    "gpt-oss-20b-MXFP4-Q8"
  ]
}
```

oMLX trusts this list. Compatibility errors (architectural mismatch, rank/layer shape disagreement) surface at engine load time via `mlx_lm.load` / `mlx_vlm.utils.load`, not at discovery.

## `adapter_config.json` (unchanged format)

oMLX reads `adapter_config.json` for mlx-lm's own consumption — rank, scale, dropout, num_layers, target keys, fine_tune_type. This is the file your training tool writes. Don't hand-edit it.

oMLX does **not** use `base_model_name_or_path` (PEFT) or `model` (mlx-lm) for discovery matching — that is what `omlx.json` is for. Discovery is layout-agnostic with respect to what the upstream tool wrote into the config file's base-ref field.

Minimum required fields:

```json
{
  "fine_tune_type": "lora",
  "num_layers": 16,
  "lora_parameters": {
    "rank": 16,
    "scale": 2.0,
    "dropout": 0.0,
    "keys": [
      "self_attn.q_proj",
      "self_attn.k_proj",
      "self_attn.v_proj",
      "self_attn.o_proj"
    ]
  }
}
```

Full fine-tunes (`fine_tune_type: "full"`) are rejected with a warning — full fine-tunes should be registered as base models under `~/.omlx/models/`, not as adapters.

## Loading an adapter

Drop the adapter directory into `~/.omlx/adapters/`, write the `omlx.json` sidecar, restart oMLX. Discovery picks it up at startup. Then:

```bash
curl -X PUT http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/settings \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "rnd-001"}'
```

The model's next engine load merges the adapter. If the model was already loaded when you changed `adapter_id`, oMLX auto-unloads the engine so the next inference request reloads with the new weights.

## Inference

Inference proceeds against the base's model_id, never the adapter's:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-MXFP4-Q8",
    "messages": [{"role": "user", "content": "..."}]
  }'
```

Note: the request's `model` field is the **base** model_id, not the adapter_id. Adapter selection is a property of the loaded engine, not the request.

## Fine-tune iteration pattern

For a training loop that produces rnd-001, rnd-002, ... adapters in sequence:

```bash
# One-time setup — stable symlink so adapter_id stays constant across iterations
ln -sfn /abs/path/to/rnd-001/adapter ~/.omlx/adapters/finetune-current
cat > ~/.omlx/adapters/finetune-current/omlx.json <<EOF
{"supported_models": ["gpt-oss-20b-MXFP4-Q8"]}
EOF

curl -X PUT http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/settings \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "finetune-current"}'

# Per iteration — retarget symlink, unload
ln -sfn /abs/path/to/rnd-N/adapter ~/.omlx/adapters/finetune-current
curl -X POST http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/unload
```

The symlink target changes each iteration; `adapter_id` and the sidecar both stay put. Next inference triggers a lazy load with the new weights.

If the adapter changes rank or target_modules across iterations (rare), call `/admin/api/adapters/rescan` first so the pool's `available_adapters` entry picks up the new metadata before reload.

Caveat: if your symlink target writes its own `adapter_config.json` on each iteration, and that file's rank or keys change across iterations, `mlx_lm.load` may fail at load time. For iteration loops, keep rank and target_modules constant across rnd-N to avoid that.

## Admin dashboard

The settings modal's Advanced section has a "LoRA Adapter" dropdown showing all adapters whose `supported_models` matched this base. Selecting one saves to the model's settings and auto-unloads the engine.

The active models status widget shows `(+ adapter-id)` next to loaded models that have an adapter attached.

## Known limitations

- **BatchedEngine and VLMBatchedEngine only.** Embedding, reranker, STT, TTS, STS engines ignore `adapter_id` at load.
- **One adapter at a time per base.** Sequential attach, not stacking.
- **No runtime hot-swap.** Changing `adapter_id` unloads and lazy-reloads. The simplicity is the feature.
- **Discovery runs at startup.** To pick up an adapter directory added after the server started, POST `/admin/api/adapters/rescan` — no restart needed. Removal is detected the same way.
- **Full fine-tunes as adapters are rejected.** Register those as base models.

## CLI introspection

Inspect adapters from the shell without hitting the server:

```bash
omlx adapter list           # table of all adapters with their attached bases
omlx adapter show rnd-001   # details for a single adapter
```

Both commands run offline — they walk `~/.omlx/models/` + `~/.omlx/adapters/` directly via `discover_models()` + `discover_adapters()`. Useful for training pipelines verifying a freshly-written adapter is discoverable, or for "why isn't this showing up?" debugging before the server is even running.

## Troubleshooting

### "My adapter doesn't show up in the model settings dropdown."

1. Does the adapter directory exist at `~/.omlx/adapters/<name>/`? Adapters under `~/.omlx/models/` are silently ignored — hard cut, no backward compat.
2. Does the directory contain all three files? `adapter_config.json`, `adapters.safetensors`, **and** `omlx.json`. Missing `omlx.json` is the most common gotcha.
3. Does `omlx.json` have a `supported_models` list with at least one string that matches a discovered base's `model_id` (or the HF-style basename thereof)?
4. Check server startup logs for `Skipping adapter ...` or `Adapter ... references unknown base ...` warnings — they name the specific reason.
5. Was the adapter added after server start? Either POST to `/admin/api/adapters/rescan` or restart the server.
6. Quick offline check: `omlx adapter list` — confirms whether discovery sees it at all.

### "Adapter attached successfully but inference looks identical to base."

1. Check the Active Models status widget in the dashboard for the `(+ <adapter-id>)` suffix next to the model name. If it's absent, the engine is loaded without the adapter.
2. Or hit `/admin/api/models` and compare `settings.adapter_id` against `loaded_adapter_id` for the model:
   - If they disagree, the engine was loaded before your settings change. POST to `/admin/api/models/<id>/unload`; next inference reloads with the new adapter.
   - If they agree (both point at the right adapter) but output still matches base, the adapter's training may not produce a behavioral delta on your test prompt. Try a prompt closer to the training distribution.

### "Model settings page shows 'Selected adapter is no longer available'."

Common after a rename, deletion, or layout migration — the `adapter_id` setting persists across filesystem changes.

1. Click the **Clear** button in the LoRA section to null the `adapter_id`, then Save. Or pick the current correct adapter from the dropdown and Save.
2. Alternatively, edit `~/.omlx/model_settings.json` directly — delete the `adapter_id` key for the affected model or update it to the current valid value. Restart the server or POST to `/admin/api/adapters/rescan` after manual edits.
3. The startup log also prints a prominent WARNING listing every stale setting and the valid options, so watching `/tmp/omlx-server.log` (or your server's log destination) at startup will tell you what needs fixing.

## Writing the sidecar from a training script

If your training pipeline outputs to `/some/path/rnd-N/adapter/`, add a step that writes the sidecar alongside:

```python
import json
from pathlib import Path

output_dir = Path("/some/path/rnd-N/adapter")
sidecar = {"supported_models": ["gpt-oss-20b-MXFP4-Q8"]}
(output_dir / "omlx.json").write_text(json.dumps(sidecar, indent=2))
```

Then symlink or copy the directory into `~/.omlx/adapters/<name>/`. Restart the server (or the oMLX daemon) to pick up the adapter.

## Verified against

`mlx-community/gpt-oss-20b-MXFP4-Q8` + rank-16 LoRA adapter — `tests/integration/test_adapter_from_settings.py` exercises the full load path.
