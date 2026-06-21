from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from functools import lru_cache

MODEL_ID = "rofl1st/llama-3.1-8b-essay-grader"

def get_dtype():
    # Prefer float16 to save memory; fallback to float32 if needed
    try:
        return torch.float16
    except Exception:
        return torch.float32

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

@lru_cache(maxsize=1)
def get_model():
    dtype = get_dtype()
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
