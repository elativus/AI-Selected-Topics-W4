from __future__ import annotations

import contextlib
import inspect
import io
from typing import Any, Dict, Sequence


@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def maybe_make_lora_request(name: str, adapter_path: str, adapter_id: int = 1) -> Any:
    """Best-effort helper for vLLM LoRA requests across versions."""
    try:
        from vllm.lora.request import LoRARequest  # type: ignore
    except Exception:
        try:
            from vllm import LoRARequest  # type: ignore
        except Exception:
            return None
    return LoRARequest(str(name), int(adapter_id), str(adapter_path))


def generate_with_vllm(
    llm: Any,
    prompts: Sequence[str],
    sampling_params: Any,
    *,
    lora_request: Any = None,
    use_tqdm: bool = False,
    suppress_output: bool = True,
):
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(llm.generate)
        if "use_tqdm" in sig.parameters:
            kwargs["use_tqdm"] = bool(use_tqdm)
    except Exception:
        pass

    if lora_request is not None:
        kwargs["lora_request"] = lora_request

    if suppress_output:
        with suppress_stdout_stderr(True):
            return llm.generate(list(prompts), sampling_params, **kwargs)
    return llm.generate(list(prompts), sampling_params, **kwargs)
