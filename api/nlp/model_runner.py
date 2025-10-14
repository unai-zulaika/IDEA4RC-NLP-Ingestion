# prompts_runtime.py
import os
import json
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pynvml
except ImportError:
    pynvml = None

from llama_cpp import Llama


# ---------- Global state (kept hot in production) ----------
_LLM: Optional[Llama] = None
_PROMPTS: Dict[str, Dict] = {}


# ---------- 1) Prompt loading & building ----------
def load_prompts_from_json(json_path: str | Path) -> None:
    """Load (or hot-reload) all prompt templates into memory."""
    global _PROMPTS
    with open(json_path, "r", encoding="utf-8") as f:
        _PROMPTS = json.load(f)


def get_prompt(task_key: str,
               # list of (note_text, annotation)
               fewshots: List[Tuple[str, str]],
               note_text: str) -> str:
    """
    Build the final prompt string from the JSON template by key.
    - fewshots: list of (note_text, annotation) pairs
    - note_text: the incoming note to process
    """
    if task_key not in _PROMPTS:
        raise KeyError(f"No prompt found for task '{task_key}'. "
                       f"Known: {list(_PROMPTS.keys())}")
    template = _PROMPTS[task_key]["template"]

    fewshots_text = "\n".join(
        f"Example:\n- Medical Note: {n}\n- Annotation: {a}\n"
        for (n, a) in fewshots
    )

    return template.format(fewshots=fewshots_text, note=note_text)


# ---------- 2) Model init (GPU → CPU fallback) & inference ----------
def _default_threads() -> int:
    env = os.getenv("LLAMA_N_THREADS")
    if env and env.isdigit():
        return int(env)
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or psutil.cpu_count(
            logical=True) or 4
    except Exception:
        n = os.cpu_count() or 4
    return max(1, n - 1)


def _detect_vram_gb() -> Optional[float]:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return float(info.total / (1024**3))
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _pick_n_gpu_layers(model_layers: int = 32) -> int:
    env = os.getenv("LLAMA_N_GPU_LAYERS")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    if platform.system() == "Darwin":
        return -1  # Metal build → request "all", runtime will cap

    vram = _detect_vram_gb()
    if vram is None:
        return -1  # optimistically try all; fallback catches failures
    if vram >= 6.0:
        return -1
    if vram >= 4.0:
        return int(model_layers * 0.75)
    if vram >= 2.5:
        return int(model_layers * 0.33)
    return 0


def init_model(model_path: str,
               n_ctx: int = 4096,
               model_layers: int = 32) -> None:
    """
    Initialize a global Llama() instance once (keep it hot).
    Tries GPU offload first, falls back to CPU.
    """
    global _LLM
    if _LLM is not None:
        return  # already initialized

    n_threads = _default_threads()
    try_gpu = _pick_n_gpu_layers(model_layers=model_layers)

    for n_gpu_layers in ([try_gpu] if try_gpu == 0 else [try_gpu, 0]):
        try:
            print(
                f"[llama.cpp] init n_gpu_layers={n_gpu_layers}, n_threads={n_threads}")
            _LLM = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=512,
                n_gpu_layers=n_gpu_layers,   # -1: all, 0: CPU
                verbose=False,
            )
            print("[llama.cpp] model ready")
            return
        except Exception as e:
            print(
                f"[llama.cpp] init failed (n_gpu_layers={n_gpu_layers}): {e}")
            _LLM = None
    raise RuntimeError("Could not initialize llama.cpp (GPU+CPU both failed).")


def run_model_with_prompt(prompt: str,
                          max_new_tokens: int = 128,
                          temperature: float = 0.7) -> Dict[str, str]:
    """
    Run the (already-initialized) LLM on a ready-to-go prompt string.
    Returns {"raw": full_text, "normalized": first_line_after_response}.
    """
    if _LLM is None:
        raise RuntimeError(
            "Model not initialized. Call init_model(...) first.")

    out = _LLM.create_chat_completion(
        messages=[
            {"role": "system",
                "content": "You are a concise medical text annotation assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    raw = out["choices"][0]["message"]["content"]

    # Normalize: extract the first line (your prompts end with "Annotation: ")
    # If you prefer to split by "### Response:", do it here.
    first_line = raw.strip().splitlines()[0].strip()

    return {"raw": raw, "normalized": first_line}
