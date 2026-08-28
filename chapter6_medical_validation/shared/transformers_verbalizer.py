"""Experiment-side local Transformers backend for strict SLM verbalization.

The FuzzyXAI verbalization core remains backend-agnostic.  This adapter only
implements its public ``VerbalizationBackend`` protocol and is deliberately
kept outside the framework package.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fuzzyxai.verbalization.contracts import (
    BackendTimeoutError,
    InvalidBackendResponseError,
    ModelNotFoundError,
)


class LocalTransformersBackend:
    """Pinned local causal-LM backend with deterministic generation settings."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str,
        revision: str,
        max_new_tokens: int = 256,
    ) -> None:
        if not model_path.is_dir():
            raise ModelNotFoundError(f"local model snapshot does not exist: {model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ModelNotFoundError("transformers backend dependencies are unavailable") from exc
        self.model = f"{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype="auto",
            ).eval()
        except OSError as exc:  # pragma: no cover - depends on a downloaded model
            raise ModelNotFoundError(f"cannot load local model snapshot {model_path}: {exc}") from exc

    def generate(self, prompt: str, *, response_schema: Mapping[str, Any] | None = None) -> str:
        del response_schema  # Strict SLM validation remains in the framework guard.
        try:
            messages = [{"role": "system", "content": "Return only the requested JSON object."}, {"role": "user", "content": prompt}]
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            with self._torch.inference_mode():
                generated = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            return self._tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True).strip()
        except TimeoutError as exc:  # pragma: no cover - backend/device-specific
            raise BackendTimeoutError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - backend/device-specific
            raise InvalidBackendResponseError(str(exc)) from exc
