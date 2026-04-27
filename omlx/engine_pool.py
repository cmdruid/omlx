# SPDX-License-Identifier: Apache-2.0
"""
Engine pool for oMLX multi-model serving.

This module manages multiple model engines with LRU-based eviction
when memory limits are exceeded. It supports:

- Pre-load memory checking to ensure models fit before loading
- LRU eviction of least recently used models
- Model pinning to keep specific models always loaded
- BatchedEngine for all LLM models (continuous batching)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .model_settings import ModelSettingsManager

import mlx.core as mx

from .engine import BaseEngine, BatchedEngine
from .engine.embedding import EmbeddingEngine
from .engine.reranker import RerankerEngine
from .engine.stt import STTEngine
from .engine.sts import STSEngine
from .engine.tts import TTSEngine
from .engine.vlm import VLMBatchedEngine
from .exceptions import (
    AdapterSwapError,
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from .adapter_utils import AdapterInfo
from .model_discovery import DiscoveredModel, discover_models, format_size, estimate_model_size
from .engine_core import get_mlx_executor
from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)




@dataclass
class EngineEntry:
    """Per-model state in the engine pool."""

    model_id: str  # Directory name (e.g., "llama-3b")
    model_path: str  # Full path to model directory
    model_type: Literal["llm", "vlm", "embedding", "reranker", "audio_stt", "audio_tts", "audio_sts"]  # Model type
    engine_type: Literal["batched", "simple", "embedding", "reranker", "vlm", "audio_stt", "audio_tts", "audio_sts"]  # Engine type to use
    estimated_size: int  # Pre-calculated from safetensors (bytes)
    config_model_type: str = ""  # Raw model_type from config.json (e.g., "deepseekocr_2")
    thinking_default: bool | None = None  # True if model thinks by default, False if not, None if unknown
    preserve_thinking_default: bool | None = None  # True when template supports preserve_thinking (Qwen 3.6+)
    engine: BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine | None = None  # Loaded engine instance
    last_access: float = 0.0  # Timestamp for LRU (0 if never loaded)
    is_loading: bool = False  # Prevent concurrent loads
    is_pinned: bool = False  # Never evict if True
    abort_loading: bool = False  # Set by memory enforcer to abort in-progress load
    available_adapters: list[AdapterInfo] = field(default_factory=list)  # LoRA adapters for this base model
    _adapter_size_bump: int = 0  # Bytes added to estimated_size for the currently-tracked adapter
    swap_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Per-engine lock for adapter swaps


def _loaded_adapter_id(entry: "EngineEntry") -> Optional[str]:
    """Return the adapter_id currently attached to a loaded engine, or None.

    Derives the id from ``engine._adapter_path`` (the final path segment).
    Returns None when the engine is not loaded, or when it was loaded without
    an adapter.
    """
    engine = getattr(entry, "engine", None)
    if engine is None:
        return None
    adapter_path = getattr(engine, "_adapter_path", None)
    if not adapter_path:
        return None
    return Path(adapter_path).name


def _health_fields_for(adapter_path: str) -> dict:
    """Read AdapterHealth from the cache and return its serializable fields.

    Returns {} if no sidecar / cache miss — caller's existing keys are unaffected.
    """
    from .adapter_health import get_adapter_health_cache
    h = get_adapter_health_cache().get(Path(adapter_path))
    if h is None:
        return {}
    return {
        "installed_at": h.installed_at,
        "last_loaded_at": h.last_loaded_at,
        "last_request_at": h.last_request_at,
        "load_count": h.load_count,
        "compatible": h.compatible,
        "last_validated_at": h.last_validated_at,
        "last_load_error": h.last_load_error,
    }


def _build_engine_kwargs_with_adapter(
    entry: EngineEntry,
    settings,
    *,
    log_kind: str = "adapter",
) -> dict:
    """Shared engine-kwargs builder with adapter resolution.

    Returns a kwargs dict containing ``model_name`` (always) and
    ``adapter_path`` (when ``settings.adapter_id`` matches one of
    ``entry.available_adapters``). Mutates ``entry.estimated_size`` (tracked
    via a private ``_adapter_size_bump`` field) so the LRU's memory
    accounting reflects the full loaded footprint.

    Idempotent: calling this twice with the same arguments leaves the entry
    in the same state — the previous bump is reverted before the new one is
    applied. Switching adapters or clearing the selection both behave
    correctly.

    Unknown ``adapter_id`` logs a warning and the engine loads without an
    adapter — advisory, never a hard error.

    ``log_kind`` appears in the "attaching {log_kind}" info log; callers pass
    ``"adapter"`` for LLMs and ``"VLM adapter"`` for VLMs to disambiguate.
    """
    kwargs: dict = {"model_name": entry.model_path}

    # Revert any previous adapter size bump so repeated calls are idempotent.
    prev_bump = getattr(entry, "_adapter_size_bump", 0)
    entry.estimated_size -= prev_bump
    entry._adapter_size_bump = 0

    if settings is not None and getattr(settings, "adapter_id", None):
        adapter_id = settings.adapter_id
        matching = next(
            (a for a in entry.available_adapters if a.adapter_id == adapter_id),
            None,
        )
        if matching is not None:
            kwargs["adapter_path"] = matching.path
            try:
                adapter_size = estimate_model_size(Path(matching.path))
                entry._adapter_size_bump = adapter_size
                entry.estimated_size += adapter_size
            except (ValueError, FileNotFoundError) as e:
                logger.debug(
                    f"Could not estimate adapter size for {matching.adapter_id}: {e}"
                )
            logger.info(
                f"Model {entry.model_id}: attaching {log_kind} "
                f"{matching.adapter_id} (rank={matching.rank}, path={matching.path})"
            )
        else:
            logger.warning(
                f"Model {entry.model_id}: requested adapter_id "
                f"{adapter_id!r} not found in available_adapters "
                f"{[a.adapter_id for a in entry.available_adapters]!r} — "
                f"loading without adapter"
            )
    return kwargs


def _build_batched_engine_kwargs(entry: EngineEntry, settings=None) -> dict:
    """Kwargs for BatchedEngine. Thin wrapper over the shared helper."""
    return _build_engine_kwargs_with_adapter(entry, settings, log_kind="adapter")


def _build_vlm_engine_kwargs(entry: EngineEntry, settings=None) -> dict:
    """Kwargs for VLMBatchedEngine. Thin wrapper over the shared helper."""
    return _build_engine_kwargs_with_adapter(entry, settings, log_kind="VLM adapter")


class EnginePool:
    """
    Manages multiple model engines with LRU-based memory management.

    Features:
    - Pre-load memory checking (evict before load, not after)
    - LRU eviction when memory limit is exceeded
    - Model pinning to prevent eviction
    - Automatic engine type selection based on model type
    """

    def __init__(
        self,
        max_model_memory: int | None,
        scheduler_config: SchedulerConfig | None = None,
    ):
        """
        Initialize the engine pool.

        Args:
            max_model_memory: Maximum memory for loaded models in bytes,
                or None for no limit (disabled)
            scheduler_config: Configuration for BatchedEngine schedulers
        """
        self._entries: dict[str, EngineEntry] = {}
        self._lock = asyncio.Lock()
        self._max_model_memory = max_model_memory
        self._current_model_memory = 0
        self._scheduler_config = scheduler_config or SchedulerConfig()
        self._process_memory_enforcer: object | None = None  # Set by server
        self._settings_manager: object | None = None  # Set by server
        self._suppress_ttl: bool = False  # Suppress TTL during benchmarks

    @property
    def max_model_memory(self) -> int | None:
        """Maximum memory for loaded models in bytes, or None if disabled."""
        return self._max_model_memory

    @property
    def current_model_memory(self) -> int:
        """Current memory used by loaded models in bytes."""
        return self._current_model_memory

    @property
    def model_count(self) -> int:
        """Total number of discovered models."""
        return len(self._entries)

    @property
    def loaded_model_count(self) -> int:
        """Number of currently loaded models."""
        return sum(1 for e in self._entries.values() if e.engine is not None)

    def discover_models(
        self, model_dirs: str | list[str], pinned_models: list[str] | None = None
    ) -> None:
        """
        Discover models in the specified directory or directories.

        Args:
            model_dirs: Path or list of paths to directories containing model subdirectories
            pinned_models: List of model IDs to pin (never evict)
        """
        from pathlib import Path

        from .model_discovery import discover_adapters, discover_models_from_dirs

        if isinstance(model_dirs, str):
            dirs = [Path(model_dirs)]
        else:
            dirs = [Path(d) for d in model_dirs]

        if len(dirs) == 1:
            discovered = discover_models(dirs[0])
            # Attach adapters from the sibling adapters/ root.
            for model_dir in dirs:
                adapter_dir = model_dir.parent / "adapters"
                discover_adapters(adapter_dir, discovered)
        else:
            # discover_models_from_dirs handles adapter attachment internally.
            discovered = discover_models_from_dirs(dirs)

        pinned_set = set(pinned_models or [])

        for model_id, info in discovered.items():
            existing = self._entries.get(model_id)
            if existing is not None and existing.engine is not None:
                # Loaded model: preserve runtime state, refresh metadata
                existing.is_pinned = model_id in pinned_set
                existing.available_adapters = list(getattr(info, "available_adapters", []))
            else:
                # New or unloaded model: create fresh entry
                self._entries[model_id] = EngineEntry(
                    model_id=model_id,
                    model_path=info.model_path,
                    model_type=info.model_type,
                    engine_type=info.engine_type,
                    estimated_size=info.estimated_size,
                    config_model_type=getattr(info, "config_model_type", ""),
                    thinking_default=getattr(info, "thinking_default", None),
                    preserve_thinking_default=getattr(info, "preserve_thinking_default", None),
                    is_pinned=model_id in pinned_set,
                    available_adapters=list(getattr(info, "available_adapters", [])),
                )

            if model_id in pinned_set:
                logger.info(f"Pinned model: {model_id}")

        # Remove entries no longer discovered and not loaded
        discovered_ids = set(discovered.keys())
        stale = [
            mid
            for mid in self._entries
            if mid not in discovered_ids and self._entries[mid].engine is None
        ]
        for mid in stale:
            del self._entries[mid]

        # Warn about pinned models not found
        found_models = set(self._entries.keys())
        for model_id in pinned_set:
            if model_id not in found_models:
                logger.warning(f"Pinned model not found: {model_id}")

        mem_display = "disabled" if self._max_model_memory is None else format_size(self._max_model_memory)
        logger.info(
            f"Discovered {len(self._entries)} models, "
            f"max memory: {mem_display}"
        )

        self._warn_stale_adapter_ids()

    def _warn_stale_adapter_ids(self) -> None:
        """Log a WARNING for each model whose settings.adapter_id doesn't match
        any of its available_adapters. Non-destructive — just logs."""
        if self._settings_manager is None:
            return
        for model_id, entry in self._entries.items():
            try:
                settings = self._settings_manager.get_settings(model_id)
            except Exception:
                continue  # settings read is best-effort; don't crash discovery
            configured = getattr(settings, "adapter_id", None)
            if not configured:
                continue
            available_ids = {a.adapter_id for a in entry.available_adapters}
            if configured not in available_ids:
                logger.warning(
                    f"Model {model_id}: settings.adapter_id={configured!r} does "
                    f"not match any available adapter ({sorted(available_ids) or 'none'}). "
                    f"Setting will be ignored at load; clear or update via the admin UI."
                )

    _MODEL_TYPE_TO_ENGINE: dict[str, str] = {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
    }

    def apply_settings_overrides(
        self, settings_manager: "ModelSettingsManager"
    ) -> None:
        """Apply model_type_override from persisted settings to discovered entries."""
        for model_id, entry in self._entries.items():
            settings = settings_manager.get_settings(model_id)
            if settings.model_type_override:
                entry.model_type = settings.model_type_override
                entry.engine_type = self._MODEL_TYPE_TO_ENGINE.get(
                    settings.model_type_override, "batched"
                )
                logger.info(
                    f"Applied model_type override for {model_id}: "
                    f"type={entry.model_type}, engine={entry.engine_type}"
                )

    def get_model_ids(self) -> list[str]:
        """Get list of all discovered model IDs."""
        return list(self._entries.keys())

    def get_loaded_model_ids(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return [mid for mid, e in self._entries.items() if e.engine is not None]

    def get_entry(self, model_id: str) -> EngineEntry | None:
        """Get entry for a specific model, or None if not found."""
        return self._entries.get(model_id)

    def set_pinned(self, model_id: str, pinned: bool) -> bool:
        """
        Set the pinned status for a model.

        Args:
            model_id: The model ID to update
            pinned: Whether to pin (True) or unpin (False) the model

        Returns:
            True if successful, False if model not found.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False
        entry.is_pinned = pinned
        return True

    def _case_insensitive_entry_match(self, name: str) -> str | None:
        """Find a model entry matching *name* case-insensitively.

        Returns the actual model_id if found, None otherwise.
        """
        lower = name.lower()
        for mid in self._entries:
            if mid.lower() == lower:
                return mid
        return None

    def resolve_model_id(self, model_id_or_alias: str, settings_manager) -> str:
        """Resolve a model alias to its actual model_id (directory name).

        Tries exact match in _entries first, then case-insensitive match,
        then scans model settings for alias match. If those fail and input
        contains a provider prefix (e.g. "omlx/my-model"), strips the prefix
        and retries. Returns the original string if no match found.
        """
        if model_id_or_alias in self._entries:
            return model_id_or_alias

        # Case-insensitive fallback
        ci_match = self._case_insensitive_entry_match(model_id_or_alias)
        if ci_match is not None:
            return ci_match

        all_settings = None
        if settings_manager is not None:
            all_settings = settings_manager.get_all_settings()
            for mid, ms in all_settings.items():
                if ms.model_alias and ms.model_alias == model_id_or_alias:
                    return mid

        # Strip provider prefix (e.g. "omlx/qwen3.5-35b" -> "qwen3.5-35b")
        if "/" in model_id_or_alias:
            stripped = model_id_or_alias.split("/", 1)[1]
            if stripped in self._entries:
                return stripped
            ci_match = self._case_insensitive_entry_match(stripped)
            if ci_match is not None:
                return ci_match
            if all_settings is not None:
                for mid, ms in all_settings.items():
                    if ms.model_alias and ms.model_alias == stripped:
                        return mid

        return model_id_or_alias

    async def get_engine(
        self, model_id: str, force_lm: bool = False,
    ) -> BaseEngine | EmbeddingEngine | RerankerEngine | STTEngine | STSEngine | TTSEngine:
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded → return immediately
        2. Check if model is too large for memory limit → raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit
            InsufficientMemoryError: If can't free enough memory (all pinned)
            ModelLoadingError: If model is already being loaded
        """
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))

            # Already loaded - just update access time
            if entry.engine is not None:
                # If force_lm requested but current engine is VLM, unload and reload
                if force_lm and isinstance(entry.engine, VLMBatchedEngine):
                    logger.info(
                        f"Unloading VLM engine for {model_id} "
                        f"(force_lm=True, reloading as LM)"
                    )
                    await self._unload_engine(model_id)
                else:
                    entry.last_access = time.time()
                    return entry.engine

            # Check if model is too large for memory limit
            if (
                self._max_model_memory is not None
                and entry.estimated_size > self._max_model_memory
            ):
                raise ModelTooLargeError(
                    model_id, entry.estimated_size, self._max_model_memory
                )

            # Pre-load eviction: reserve 25% extra for KV cache headroom
            # so other models get evicted earlier, leaving room for context.
            # Always try to evict with headroom first. If all evictable models
            # are gone and the model still fits without headroom, allow it.
            # Skip entirely when model memory limit is disabled (None).
            # Audio engines (STT/TTS) don't use KV cache, so headroom is 0.
            if self._max_model_memory is not None:
                if entry.engine_type in ("audio_stt", "audio_tts", "audio_sts"):
                    kv_headroom = 0
                else:
                    kv_headroom = int(entry.estimated_size * 0.25)
                required_with_headroom = entry.estimated_size + kv_headroom
                try:
                    await self._ensure_memory_available(required_with_headroom)
                except InsufficientMemoryError:
                    # Can't fit with headroom even after evicting everything possible.
                    # Fall back to weights-only if that fits.
                    if self._current_model_memory + entry.estimated_size <= self._max_model_memory:
                        logger.info(
                            f"Loading {model_id} without KV headroom "
                            f"(need {format_size(required_with_headroom)}, "
                            f"available {format_size(self._max_model_memory - self._current_model_memory)})"
                        )
                    else:
                        await self._ensure_memory_available(entry.estimated_size)

            # Check process memory limit before loading.
            # Try evicting LRU models first to free actual Metal memory.
            # max_bytes <= 0 means enforcement is disabled (no limit).
            if self._process_memory_enforcer is not None:
                enforcer = self._process_memory_enforcer
                if enforcer.max_bytes > 0:
                    while True:
                        current_active = mx.get_active_memory()
                        projected = current_active + entry.estimated_size
                        if projected <= enforcer.max_bytes:
                            break
                        # Try to evict an LRU model to free memory
                        victim = self._find_lru_victim()
                        if victim is not None:
                            logger.info(
                                f"Evicting '{victim}' to fit '{model_id}' "
                                f"within process memory limit "
                                f"({format_size(projected)} > "
                                f"{format_size(enforcer.max_bytes)})"
                            )
                            await self._unload_engine(victim)
                            continue
                        # No more victims — cannot fit
                        raise InsufficientMemoryError(
                            required=entry.estimated_size,
                            current=current_active,
                            message=(
                                f"Cannot load {model_id}: projected memory "
                                f"{format_size(projected)} would exceed process "
                                f"limit {format_size(enforcer.max_bytes)} "
                                f"(current: {format_size(current_active)}, "
                                f"model: {format_size(entry.estimated_size)})"
                            ),
                        )

            # Now load the model
            await self._load_engine(model_id, force_lm=force_lm)

            return self._entries[model_id].engine

    async def _ensure_memory_available(self, required: int) -> None:
        """
        Evict LRU models BEFORE loading to ensure we don't exceed memory limit.

        Args:
            required: Required memory in bytes

        Raises:
            InsufficientMemoryError: If can't free enough memory
        """
        if self._max_model_memory is None:
            return  # No model memory limit
        while self._current_model_memory + required > self._max_model_memory:
            victim = self._find_lru_victim()
            if not victim:
                raise InsufficientMemoryError(
                    required=required,
                    current=self._current_model_memory,
                    message=(
                        f"Cannot free enough memory. "
                        f"Need {format_size(required)}, "
                        f"current usage {format_size(self._current_model_memory)}, "
                        f"all loaded models are pinned."
                    ),
                )
            await self._unload_engine(victim)

    def _find_lru_victim(self) -> str | None:
        """
        Find the least recently used non-pinned loaded model.

        Skips models with active inference requests to avoid interrupting
        in-flight generation.

        Returns:
            Model ID of the LRU victim, or None if no evictable model found
        """
        candidates = []
        for mid, e in self._entries.items():
            if e.engine is None or e.is_pinned:
                continue
            try:
                if e.engine.has_active_requests():
                    logger.debug(
                        f"Skipping victim '{mid}': has active requests"
                    )
                    continue
            except AttributeError:
                pass
            candidates.append((e.last_access, mid))
        if not candidates:
            return None
        candidates.sort()  # Sort by last_access (oldest first)
        return candidates[0][1]

    async def _unload_engine(self, model_id: str) -> None:
        """
        Immediately stop and unload an engine with memory settle barrier.

        After stopping the engine, polls mx.get_active_memory() to verify
        Metal buffers are actually reclaimed before updating the memory
        tracking counter.

        Args:
            model_id: The model ID to unload
        """
        entry = self._entries.get(model_id)
        if not entry or entry.engine is None:
            return

        logger.info(f"Unloading model: {model_id} (immediate abort)")
        pre_unload_active = mx.get_active_memory()

        try:
            await entry.engine.stop()
        except Exception as e:
            logger.warning(f"Error stopping engine for {model_id}: {e}")

        # Clear engine reference before settle barrier
        entry.engine = None
        entry.last_access = 0.0

        # Force garbage collection to release memory.
        # Run mx.clear_cache on the global MLX executor to avoid concurrent
        # Metal operations with running engines. See issue #85.
        # Synchronize before clearing to prevent releasing Metal buffers
        # still referenced by in-flight command buffers. See issue #300.
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )

        # Memory settle barrier: poll actual freed memory instead of
        # trusting the cumulative _current_model_memory estimate.
        # Scale tolerance with model size: estimated_size includes a 5%
        # overhead factor (model_discovery.py) that may not be reflected in
        # actual freed memory. Use 2 GB floor for small models. See #768.
        settle_tolerance = max(2 * 1024**3, int(entry.estimated_size * 0.05))
        min_expected_freed = max(0, entry.estimated_size - settle_tolerance)
        settled = False
        for _settle_round in range(10):
            active_now = mx.get_active_memory()
            actual_freed = pre_unload_active - active_now
            if actual_freed >= min_expected_freed:
                settled = True
                logger.debug(
                    f"Settle round {_settle_round + 1} for '{model_id}': "
                    f"freed={format_size(actual_freed)} "
                    f"(need>={format_size(min_expected_freed)}) - settled"
                )
                break
            logger.debug(
                f"Settle round {_settle_round + 1} for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)}) - retry"
            )
            await asyncio.sleep(0.5)
            gc.collect()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )

        # Release memory tracking AFTER barrier
        self._current_model_memory -= entry.estimated_size

        if settled:
            logger.info(
                f"Unloaded model: {model_id}, "
                f"freed={format_size(actual_freed)} "
                f"(expected>={format_size(min_expected_freed)}), "
                f"active_memory: {format_size(active_now)} (settled)"
            )
        else:
            # Barrier timed out - try emergency reclaim
            logger.warning(
                f"Settle barrier timed out for '{model_id}': "
                f"freed={format_size(actual_freed)} "
                f"(need>={format_size(min_expected_freed)})"
            )
            for _ in range(3):
                gc.collect()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                await asyncio.sleep(1.0)
            active_after = mx.get_active_memory()
            if active_after > self._current_model_memory + 5 * 1024**3:
                logger.error(
                    f"Emergency reclaim failed for '{model_id}': "
                    f"active_memory={format_size(active_after)} "
                    f"exceeds safe threshold "
                    f"({format_size(self._current_model_memory + 5 * 1024**3)})"
                )
            else:
                logger.info(
                    f"Emergency reclaim succeeded: "
                    f"active_memory={format_size(active_after)}"
                )

    async def _load_engine(self, model_id: str, force_lm: bool = False) -> None:
        """
        Load an engine for the specified model.

        Args:
            model_id: The model ID to load
            force_lm: Force loading as BatchedEngine even for VLM models.

        Raises:
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries[model_id]
        if entry.is_loading:
            raise ModelLoadingError(model_id)

        entry.is_loading = True
        entry.abort_loading = False
        try:
            effective_type = entry.engine_type
            if force_lm and effective_type == "vlm":
                effective_type = "batched"
                logger.info(f"Loading model as LM (force_lm=True): {model_id}")
            else:
                logger.info(f"Loading model: {model_id}")

            # Retrieve per-model settings for post-load transforms
            model_settings = None
            if self._settings_manager is not None:
                model_settings = self._settings_manager.get_settings(model_id)

            # Check if DFlash is enabled — takes priority over engine type
            # since DFlash has its own model loading pipeline
            engine = None
            if model_settings is not None:
                dflash_enabled = getattr(model_settings, "dflash_enabled", False)
                dflash_draft = getattr(model_settings, "dflash_draft_model", None)
                if dflash_enabled and dflash_draft:
                    try:
                        from .engine.dflash import DFlashEngine
                        engine = DFlashEngine(
                            model_name=entry.model_path,
                            draft_model_path=dflash_draft,
                            draft_quant_bits=getattr(model_settings, "dflash_draft_quant_bits", None),
                            model_settings=model_settings,
                            fallback_engine_type=effective_type,
                            scheduler_config=self._scheduler_config,
                        )
                        logger.info(f"DFlash enabled for {model_id}, draft={dflash_draft}")
                    except ImportError:
                        logger.warning(
                            f"DFlash enabled for {model_id} but dflash-mlx is not installed. "
                            f"Falling back to default engine."
                        )
                    except Exception as e:
                        logger.warning(
                            f"DFlash init failed for {model_id}: {e}. "
                            f"Falling back to default engine."
                        )

            # Create engine based on engine type (if DFlash not active)
            if engine is None:
                if effective_type == "embedding":
                    engine = EmbeddingEngine(model_name=entry.model_path)
                elif effective_type == "reranker":
                    engine = RerankerEngine(model_name=entry.model_path)
                elif effective_type == "vlm":
                    _vlm_kwargs = _build_vlm_engine_kwargs(entry, model_settings)
                    engine = VLMBatchedEngine(
                        **_vlm_kwargs,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                elif entry.engine_type == "audio_stt":
                    engine = STTEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_tts":
                    engine = TTSEngine(model_name=entry.model_path)
                elif entry.engine_type == "audio_sts":
                    engine = STSEngine(
                        model_name=entry.model_path,
                        config_model_type=entry.config_model_type,
                    )
                else:
                    _batched_kwargs = _build_batched_engine_kwargs(entry, model_settings)
                    engine = BatchedEngine(
                        **_batched_kwargs,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )

            _is_dflash_engine = engine is not None and type(engine).__name__ == "DFlashEngine"

            try:
                await engine.start()
            except Exception as start_error:
                if _is_dflash_engine:
                    # DFlash engine failed to start — fall back to the
                    # model's natural engine type (VLM or Batched)
                    logger.warning(
                        f"DFlash start failed for {model_id}: {start_error}. "
                        f"Falling back to {effective_type} engine."
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    if effective_type == "vlm":
                        _vlm_kwargs = _build_vlm_engine_kwargs(entry, model_settings)
                        engine = VLMBatchedEngine(
                            **_vlm_kwargs,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                        )
                    else:
                        _batched_kwargs = _build_batched_engine_kwargs(entry, model_settings)
                        engine = BatchedEngine(
                            **_batched_kwargs,
                            scheduler_config=self._scheduler_config,
                            model_settings=model_settings,
                        )
                    await engine.start()
                    logger.info(
                        f"Successfully loaded {model_id} as {effective_type} "
                        f"(fallback from DFlash)"
                    )

                elif force_lm and entry.engine_type == "vlm":
                    # force_lm created a BatchedEngine but mlx-lm can't
                    # load this VLM model — fall back to VLMBatchedEngine.
                    logger.warning(
                        f"LM loading failed for VLM model {model_id} "
                        f"(force_lm=True), falling back to VLM engine: "
                        f"{start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    _vlm_kwargs = _build_vlm_engine_kwargs(entry, model_settings)
                    engine = VLMBatchedEngine(
                        **_vlm_kwargs,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                    await engine.start()

                    logger.info(
                        f"Successfully loaded {model_id} as VLM "
                        f"(fallback from force_lm)"
                    )
                elif entry.engine_type == "vlm":
                    # VLM loading failed — fall back to LLM (BatchedEngine)
                    logger.warning(
                        f"VLM loading failed for {model_id}, "
                        f"falling back to LLM: {start_error}"
                    )
                    try:
                        await engine.stop()
                    except Exception:
                        pass
                    gc.collect()
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        get_mlx_executor(),
                        lambda: (mx.synchronize(), mx.clear_cache()),
                    )

                    _batched_kwargs = _build_batched_engine_kwargs(entry, model_settings)
                    engine = BatchedEngine(
                        **_batched_kwargs,
                        scheduler_config=self._scheduler_config,
                        model_settings=model_settings,
                    )
                    await engine.start()

                    entry.model_type = "llm"
                    entry.engine_type = "batched"
                    logger.info(
                        f"Successfully loaded {model_id} as LLM "
                        f"(fallback from VLM)"
                    )
                else:
                    raise

            # Check if memory enforcer requested abort during loading
            if entry.abort_loading:
                logger.warning(
                    f"Model load aborted by memory enforcer: {model_id}"
                )
                try:
                    await engine.stop()
                except Exception as e:
                    logger.warning(
                        f"Error stopping aborted engine for {model_id}: {e}"
                    )
                gc.collect()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
                raise ModelLoadingError(
                    f"Model {model_id} load aborted: "
                    f"process memory limit exceeded"
                )

            entry.engine = engine
            entry.last_access = time.time()
            self._current_model_memory += entry.estimated_size

            # Record adapter-load success in the health sidecar (if attached).
            _adapter_path = getattr(engine, "_adapter_path", None)
            if _adapter_path:
                from .adapter_health import get_adapter_health_cache
                try:
                    _health_cache = get_adapter_health_cache()
                    _adapter_id_from_path = Path(_adapter_path).name
                    # Defensive: ensure the entry exists in the cache.
                    _health_cache.get_or_create(Path(_adapter_path), adapter_id=_adapter_id_from_path)
                    _health_cache.record_load_success(Path(_adapter_path))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "Failed to record adapter load_success for %s: %s",
                        _adapter_path, exc,
                    )

            # Propagate memory limit to new engine's scheduler
            if self._process_memory_enforcer is not None:
                self._process_memory_enforcer._propagate_memory_limit()

            # Release intermediate Metal buffers from model loading.
            # mlx_lm.load() creates large temporaries (weight transforms,
            # quantization intermediates) that stay in the Metal buffer pool
            # because mx.set_cache_limit(total_mem) prevents automatic release.
            # Without this, memory stays at ~2x model size until the first
            # inference request triggers a clear. (#429)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(),
                lambda: (mx.synchronize(), mx.clear_cache()),
            )

            logger.info(
                f"Loaded model: {model_id} "
                f"(estimated: {format_size(entry.estimated_size)}, "
                f"total: {format_size(self._current_model_memory)})"
            )
        finally:
            entry.is_loading = False
            entry.abort_loading = False

    async def swap_adapter(
        self, model_id: str, adapter_id: Optional[str]
    ) -> None:
        """Atomically swap the engine's loaded adapter.

        - ``adapter_id=None`` detaches the current adapter (engine reloads
          without one).
        - Otherwise, ``adapter_id`` must match an entry in ``available_adapters``.

        Holds the per-engine ``swap_lock`` for the unload+load cycle so
        concurrent chat-completion requests on the same engine queue
        cleanly. Raises ``AdapterSwapError`` on validation failure or load
        failure; the caller is responsible for reverting any settings
        changes.

        Args:
            model_id: Target model id.
            adapter_id: Adapter id to load, or None to detach.

        Raises:
            AdapterSwapError(error_type='model_not_found'): no such model.
            AdapterSwapError(error_type='adapter_not_found'): adapter_id not in
                available_adapters.
            AdapterSwapError(error_type='adapter_load_failed'): load attempt failed.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            raise AdapterSwapError(
                "model_not_found", f"Model not found: {model_id}"
            )

        if adapter_id is not None:
            ids = {a.adapter_id for a in entry.available_adapters}
            if adapter_id not in ids:
                raise AdapterSwapError(
                    "adapter_not_found",
                    f"Adapter {adapter_id!r} not in available_adapters {sorted(ids)!r}",
                )

        async with entry.swap_lock:
            # Apply settings.adapter_id so _load_engine picks up the right adapter.
            if self._settings_manager is not None:
                ms = self._settings_manager.get_settings(model_id)
                ms.adapter_id = adapter_id
                self._settings_manager.set_settings(model_id, ms)

            # Unload current engine if loaded, then load with the new adapter.
            if entry.engine is not None:
                await self._unload_engine(model_id)
            try:
                await self._load_engine(model_id)
            except Exception as exc:  # noqa: BLE001
                raise AdapterSwapError("adapter_load_failed", str(exc)) from exc

    async def swap_adapter_for_request(
        self, model_id: str, adapter_id: Optional[str]
    ) -> None:
        """Swap the engine to ``adapter_id`` without mutating settings.adapter_id.

        Used by the per-request override path. The caller MUST already hold
        ``entry.swap_lock`` (we don't re-acquire it here; that would deadlock).

        Temporarily overrides the in-memory adapter_id inside the settings
        manager (bypassing ``set_settings`` so nothing is persisted to disk),
        reloads the engine, then restores the original adapter_id in a
        ``finally`` block so the restore is guaranteed even on failure.

        On failure, raises ``AdapterSwapError(error_type='adapter_load_failed')``.
        Caller wraps as needed. Raises ``RuntimeError`` if no settings_manager
        is configured.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            raise AdapterSwapError(
                "model_not_found", f"Model not found: {model_id}"
            )

        if self._settings_manager is None:
            raise RuntimeError(
                "swap_adapter_for_request requires a settings_manager"
            )

        # Access the live settings object directly (not a copy) so we can
        # restore it without going through set_settings (which persists).
        # If no entry exists yet in _settings, we create a temporary one so
        # _load_engine sees the override, then remove it on restore.
        sm = self._settings_manager
        _created_entry = False
        with sm._lock:
            if model_id not in sm._settings:
                from .model_settings import ModelSettings
                sm._settings[model_id] = ModelSettings()
                _created_entry = True
            live_settings = sm._settings[model_id]
            original_adapter_id = live_settings.adapter_id
            live_settings.adapter_id = adapter_id

        try:
            if entry.engine is not None:
                await self._unload_engine(model_id)
            await self._load_engine(model_id)
        except Exception as exc:  # noqa: BLE001
            raise AdapterSwapError("adapter_load_failed", str(exc)) from exc
        finally:
            # Restore settings.adapter_id in memory — never persisted.
            with sm._lock:
                if _created_entry and model_id in sm._settings:
                    del sm._settings[model_id]
                elif model_id in sm._settings:
                    sm._settings[model_id].adapter_id = original_adapter_id

    async def preload_pinned_models(self) -> None:
        """
        Preload all pinned models at startup.

        This ensures pinned models are always available.
        """
        pinned_models = [
            model_id for model_id, e in self._entries.items() if e.is_pinned
        ]

        for model_id in pinned_models:
            try:
                logger.info(f"Preloading pinned model: {model_id}")
                await self.get_engine(model_id)
            except Exception as e:
                logger.error(f"Failed to preload pinned model {model_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown all engines gracefully."""
        async with self._lock:
            for model_id in list(self._entries.keys()):
                entry = self._entries.get(model_id)
                if entry and entry.engine is not None:
                    try:
                        await self._unload_engine(model_id)
                    except Exception as e:
                        logger.error(f"Error unloading {model_id} during shutdown: {e}")

        logger.info("Engine pool shutdown complete")

    def get_status(self) -> dict:
        """
        Get pool status for monitoring endpoints.

        Returns:
            Dictionary with pool status information
        """
        return {
            "max_model_memory": self._max_model_memory,
            "current_model_memory": self._current_model_memory,
            "model_count": len(self._entries),
            "loaded_count": sum(1 for e in self._entries.values() if e.engine is not None),
            "models": [
                {
                    "id": mid,
                    "model_path": e.model_path,
                    "loaded": e.engine is not None,
                    "is_loading": e.is_loading,
                    "estimated_size": e.estimated_size,
                    "pinned": e.is_pinned,
                    "engine_type": e.engine_type,
                    "model_type": e.model_type,
                    "config_model_type": e.config_model_type,
                    "thinking_default": e.thinking_default,
                    "preserve_thinking_default": e.preserve_thinking_default,
                    "last_access": e.last_access if e.last_access > 0 else None,
                    "available_adapters": [
                        {
                            "adapter_id": a.adapter_id,
                            "path": a.path,
                            "rank": a.rank,
                            "num_layers": a.num_layers,
                            "fine_tune_type": a.fine_tune_type,
                            **_health_fields_for(a.path),
                        }
                        for a in e.available_adapters
                    ],
                    "loaded_adapter_id": _loaded_adapter_id(e),
                }
                for mid, e in sorted(self._entries.items())
            ],
        }

    async def _validate_adapter(
        self, model_id: str, adapter_info: "AdapterInfo"  # type: ignore[name-defined]
    ) -> "tuple[bool, Optional[str]]":
        """Verify the adapter loads cleanly against its base model.

        Returns ``(compatible, last_load_error)``: ``(True, None)`` on success,
        ``(False, <error message>)`` on failure.

        Calls ``swap_adapter_for_request`` to attempt the load. After validation
        (success or failure), restores the adapter that was loaded before the
        validation attempt. Settings.adapter_id is never mutated by this method
        (validation is a read-only inspection from the user's perspective).

        The caller MUST NOT hold ``entry.swap_lock`` — this method acquires it.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return False, f"model not found: {model_id}"

        async with entry.swap_lock:
            # Snapshot what was loaded before validation so we can restore it.
            # Taken inside the lock so a concurrent override can't change the
            # engine state between snapshot and the validation swap.
            current_adapter_id = (
                _loaded_adapter_id(entry) if entry.engine is not None else None
            )

            try:
                await self.swap_adapter_for_request(model_id, adapter_info.adapter_id)
            except AdapterSwapError as exc:
                # Restore best-effort to whatever was loaded before validation.
                try:
                    await self.swap_adapter_for_request(model_id, current_adapter_id)
                except Exception:  # noqa: BLE001
                    pass
                return False, exc.message
            except Exception as exc:  # noqa: BLE001
                # Defensive — unexpected failures are treated as incompatible.
                try:
                    await self.swap_adapter_for_request(model_id, current_adapter_id)
                except Exception:  # noqa: BLE001
                    pass
                return False, str(exc)

            # Validation succeeded — restore to what was loaded before.
            try:
                await self.swap_adapter_for_request(model_id, current_adapter_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Validation succeeded for %s but restore-swap failed: %s",
                    adapter_info.adapter_id,
                    exc,
                )

        return True, None

    async def rescan_adapters(self, *, validate: bool = False) -> dict:
        """Re-walk ~/.omlx/adapters/ and refresh each base's available_adapters.

        Picks up adapters added/removed since server startup. Holds the pool lock
        for the duration to serialize against loads/unloads — rescan is O(adapter-count)
        filesystem reads, so the pause is brief.

        Pass ``validate=True`` to additionally exercise each adapter with a load
        cycle via ``_validate_adapter``. Results are persisted to each adapter's
        ``.health.json`` sidecar via ``AdapterHealthCache.record_validation``.

        v1 constraint: always scans Path.home() / ".omlx" / "adapters". Multi-root
        support (deriving adapter dirs from configured model_dirs) can be added later
        if needed; the settings manager does not store model_dirs at runtime, so there
        is no clean way to derive them here without a larger refactor.

        Returns:
            {"attached": <int>, "removed": <int>, "total": <int>}
        """
        from .model_discovery import discover_adapters

        async with self._lock:
            # Snapshot pre-state per model so we can compute the diff.
            before: dict[str, set[str]] = {
                mid: {a.adapter_id for a in e.available_adapters}
                for mid, e in self._entries.items()
            }
            # Zero out each entry's adapter list — discover_adapters only appends.
            for e in self._entries.values():
                e.available_adapters.clear()

            # Scan the default adapter root under the user's home dir.
            adapter_root = Path.home() / ".omlx" / "adapters"
            if adapter_root.exists():
                # _try_attach_adapter accesses .available_adapters on values,
                # which EngineEntry provides — structurally compatible with
                # the DiscoveredModel type annotation in discover_adapters.
                discover_adapters(adapter_root, self._entries)  # type: ignore[arg-type]

            # Ensure each discovered adapter has a .health.json sidecar.
            from .adapter_health import ensure_health_for_adapter
            for entry in self._entries.values():
                for ai in entry.available_adapters:
                    try:
                        ensure_health_for_adapter(Path(ai.path), adapter_id=ai.adapter_id)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            "Failed to ensure health sidecar for %s: %s", ai.path, exc,
                        )

            # Compute diff counts.
            after: dict[str, set[str]] = {
                mid: {a.adapter_id for a in e.available_adapters}
                for mid, e in self._entries.items()
            }
            attached = sum(len(after[m] - before.get(m, set())) for m in after)
            removed = sum(len(before.get(m, set()) - after[m]) for m in after)
            total = sum(len(s) for s in after.values())

            logger.info(
                "Adapter rescan complete: %d attached, %d removed, %d total across %d models",
                attached, removed, total, len(self._entries),
            )

        # Validation is done outside the pool lock — each adapter load acquires
        # entry.swap_lock independently, which is sufficient for atomicity.
        if validate:
            from .adapter_health import get_adapter_health_cache
            cache = get_adapter_health_cache()
            for entry_id, entry in list(self._entries.items()):
                for ai in entry.available_adapters:
                    ok, err = await self._validate_adapter(entry_id, ai)
                    cache.record_validation(Path(ai.path), compatible=ok, error=err)
            # Surface results to disk immediately so callers reading the
            # sidecar right after rescan see the validation outcome (the
            # 300s periodic flush is too slow for deploy-gate consumers).
            cache.flush()

        return {"attached": attached, "removed": removed, "total": total}

    async def check_ttl_expirations(
        self,
        settings_manager: ModelSettingsManager,
        global_idle_timeout_seconds: int | None = None,
    ) -> list[str]:
        """Check and unload models that have exceeded their TTL.

        Pinned models are skipped (TTL is ignored for pinned models).
        Models with active requests are skipped and their last_access is refreshed.
        Suppressed during benchmark runs via _suppress_ttl flag.

        Args:
            settings_manager: The settings manager to read TTL values from.
            global_idle_timeout_seconds: Global idle timeout fallback (None = no global TTL).

        Returns:
            List of model IDs that were unloaded.
        """
        if self._suppress_ttl:
            return []

        now = time.time()
        expired: list[str] = []

        async with self._lock:
            for model_id, entry in self._entries.items():
                if entry.engine is None or entry.is_loading or entry.is_pinned:
                    continue

                settings = settings_manager.get_settings(model_id)
                effective_ttl = settings.ttl_seconds
                if effective_ttl is None:
                    effective_ttl = global_idle_timeout_seconds
                if effective_ttl is None:
                    continue

                idle_time = now - entry.last_access
                if idle_time < effective_ttl:
                    continue

                # Check if model has active requests
                has_active = entry.engine.has_active_requests()

                if has_active:
                    entry.last_access = now
                    continue

                logger.info(
                    f"TTL expired for model '{model_id}' "
                    f"(idle {idle_time:.0f}s > ttl {effective_ttl}s)"
                )
                await self._unload_engine(model_id)
                expired.append(model_id)

        return expired
