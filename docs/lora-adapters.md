# LoRA Adapter Support

oMLX can load a LoRA adapter alongside a base model. Adapters are **not** peer models — each base can have one adapter attached via its per-model settings. To swap adapters, change the setting (or the adapter directory itself) and reload the model.

## Directory Layout

Drop adapter directories alongside base models in `~/.omlx/models/`:

```
~/.omlx/models/
├── gpt-oss-20b-MXFP4-Q8/          # base
│   ├── config.json
│   └── model-00001-of-00003.safetensors
├── rnd-001-adapter/                # LoRA adapter for gpt-oss-20b-MXFP4-Q8
│   ├── adapter_config.json
│   └── adapters.safetensors
└── rnd-002-adapter/                # another LoRA adapter for the same base
    ├── adapter_config.json
    └── adapters.safetensors
```

An adapter directory is identified by the presence of `adapter_config.json`. The adapter's base reference lives in its config (either `base_model_name_or_path`, PEFT convention, or `model`, mlx-lm convention). oMLX matches the ref against both model_ids and their basenames — so `"mlx-community/gpt-oss-20b-MXFP4-Q8"` attaches to a local dir named `gpt-oss-20b-MXFP4-Q8`.

## `adapter_config.json` reference

Minimum required fields:

```json
{
  "base_model_name_or_path": "mlx-community/gpt-oss-20b-MXFP4-Q8",
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

Full fine-tunes (`fine_tune_type: "full"`) are rejected — those should be registered as base models. Orphan adapters (base not locally discovered) are skipped with a warning at startup.

## Selecting an adapter

### Via the admin dashboard

1. Open the model's settings modal.
2. Scroll to the Advanced section.
3. In the "LoRA Adapter" dropdown, pick an adapter (or "None (base only)").
4. Save.
5. **Reload the model** for the change to take effect — either via the Unload + Load buttons in the dashboard, or by restarting the server.

The dropdown lists only adapters whose base matches this model (auto-detected at startup).

### Via the API

PATCH the model settings:

```bash
curl -X PUT http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/settings \
  -H "Content-Type: application/json" \
  -d '{"adapter_id": "rnd-001-adapter"}'
```

Then reload:

```bash
curl -X POST http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/unload
curl -X POST http://localhost:8080/admin/api/models/gpt-oss-20b-MXFP4-Q8/load
```

### Via symlink (no UI required)

If you're driving from a fine-tune iteration loop, the simplest programmatic approach is:

1. Point a stable symlink at the current adapter version:
   ```bash
   ln -sfn /path/to/rnd-002/adapter ~/.omlx/models/my-adapter
   ```
2. Configure the model's setting once: `adapter_id: "my-adapter"`.
3. For each iteration, update the symlink to the new adapter dir + restart the server (or reload the model).

The adapter_id stays stable; only the symlink changes. Simpler than pushing API calls on every training step.

## Inference

After the model is loaded with an adapter, hit it at the normal OpenAI-compatible endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-MXFP4-Q8",
    "messages": [{"role": "user", "content": "Wake me in 3 seconds."}]
  }'
```

Note: the request's `model` field is the **base** model_id, not the adapter_id. Adapter selection is a property of the loaded engine, not the request.

## Changing adapters

Reload the model. No hot-swap in v1 — the simplicity is the feature. The flow:

1. Change the `adapter_id` setting (via UI or PATCH) OR update the symlink.
2. Unload + Load the model, or restart the server.
3. Subsequent requests use the new adapter.

## Limitations

- **BatchedEngine only.** VLM, embedding, reranker, STT, TTS engines don't support adapters yet. If an adapter's base is one of those model types, oMLX will still discover + attach metadata, but engine load will ignore the adapter_path.
- **One adapter at a time per base.** To A/B test two adapters, configure them as separate ModelSettings entries or evict + reload.
- **No hot-swap.** Changing `adapter_id` doesn't apply to an already-loaded engine until it's reloaded. This is intentional.

## Verified against

`mlx-community/gpt-oss-20b-MXFP4-Q8` + rank-16 LoRA adapter — `tests/integration/test_adapter_from_settings.py` exercises the full load path.
