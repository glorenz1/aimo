"""Prompt Endpoint Probe (PEP): frozen shared-encoder predictor. No research or training imports.

Only original inputs enter a BF16 prefill. No answer generation is performed.
The scalar semantic contribution is reused across target-model calls.
"""
import functools
import hashlib
import json
import math
from pathlib import Path
import time

MODELS = ('Qwen/Qwen3.5-4B', 'Skywork/Skywork-OR1-Math-7B',
          'allenai/Olmo-3-7B-Think', 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')
ENCODER_ID = 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'
_MODEL = None
_TOKENIZER = None
_ARTIFACT = None
MEASUREMENTS = {'loads': 0, 'forwards': 0, 'load_seconds': 0., 'prefill_seconds': 0., 'prompt_tokens': 0}


def artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        value = json.loads(Path(__file__).with_name('classifier.json').read_text())
        if value['schema'] != 'frozen-shared-representation-v1' or value['width'] != 4096:
            raise ValueError('incompatible frozen representation head')
        _ARTIFACT = value
    return _ARTIFACT


def metadata(model_id, effort, text):
    n = max(1, len(text))
    return [math.log1p(len(text)), math.log1p(len(text.split())), sum(c.isdigit() for c in text)/n,
            sum(c in '+-*/=^{}\\' for c in text)/n, float(text.count('?')),
            *[float(model_id == m) for m in MODELS],
            *[float(effort == e) for e in ('default','low','medium','high')],
            float(model_id == 'openai/gpt-oss-120b')]


def score_from_vector(model_id, effort, text, vector):
    """Portable exact fitted preprocessing, also used for export parity tests."""
    a = artifact()
    if len(vector) != a['width']:
        raise ValueError('wrong hidden-state width')
    hidden = math.fsum(w*(float(x)-m)/s for w,x,m,s in zip(a['hidden_coef'],vector,a['hidden_mean'],a['hidden_scale']))
    return _score(model_id,effort,text,hidden)


def _score(model_id, effort, text, hidden_margin):
    a = artifact()
    margin = a['intercept'] + hidden_margin + math.fsum(w*(x-m)/s for w,x,m,s in
        zip(a['metadata_coef'],metadata(model_id,effort,text),a['metadata_mean'],a['metadata_scale']))
    if not math.isfinite(margin): raise ValueError('nonfinite frozen classifier score')
    return margin


def build_prompt(tokenizer, problem):
    system = artifact()['encoder']['system_prompt']
    if system and hasattr(tokenizer,'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':problem}],
                                                 tokenize=False,add_generation_prompt=True)
        except Exception:
            pass  # Exact historical probe formatting fallback.
    return system+'\n\n'+problem if system else problem


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is not None: return _TOKENIZER, _MODEL
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('PEP requires a BF16-capable CUDA GPU; no precision fallback')
    if torch.cuda.device_count() != 1: raise RuntimeError('one visible GPU is required')
    free, _ = torch.cuda.mem_get_info()
    if free < 24*1024**3: raise RuntimeError('insufficient free VRAM for frozen BF16 prefill')
    spec = artifact()['encoder']; tick = time.perf_counter()
    snapshot = Path(snapshot_download(ENCODER_ID,revision=spec['revision'],local_files_only=True,allow_patterns=list(spec['files'])+['*.safetensors']))
    for name, expected in spec['files'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('offline checkpoint/tokenizer metadata mismatch: '+name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot),dtype=torch.bfloat16,device_map={'':'cuda:0'},
        local_files_only=True,trust_remote_code=False,attn_implementation='sdpa')
    model.eval()
    if any(p.device.type!='cuda' or p.dtype!=torch.bfloat16 for p in model.parameters()):
        raise RuntimeError('frozen model must be fully GPU resident in BF16')
    torch.cuda.synchronize();MEASUREMENTS['load_seconds'] += time.perf_counter()-tick
    MEASUREMENTS['loads'] += 1
    _TOKENIZER, _MODEL = tokenizer, model
    return tokenizer, model


def encode_vector(problem):
    """Same full forward and hidden-state indexing as the frozen cached probe."""
    import torch
    tokenizer, model = _load(); tick = time.perf_counter()
    inputs = tokenizer(build_prompt(tokenizer,problem),return_tensors='pt',truncation=True)
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        # Only the unused LM-head logits are sliced. Decoder hidden states and
        # their indexing remain unchanged; qualification checks cached vectors.
        output = model(**inputs,output_hidden_states=True,return_dict=True,use_cache=False,logits_to_keep=1)
    vector = output.hidden_states[artifact()['encoder']['layer']][0,-1,:].detach().float().cpu().tolist()
    torch.cuda.synchronize();MEASUREMENTS['prefill_seconds'] += time.perf_counter()-tick
    MEASUREMENTS['forwards'] += 1;MEASUREMENTS['prompt_tokens'] += inputs['input_ids'].shape[1]
    return vector


@functools.lru_cache(maxsize=20000)
def hidden_margin(problem):
    a = artifact(); vector = encode_vector(problem)
    return math.fsum(w*(x-m)/s for w,x,m,s in zip(a['hidden_coef'],vector,a['hidden_mean'],a['hidden_scale']))


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    if model_id not in MODELS: raise ValueError('unsupported Small Models checkpoint')
    if not problems: return []
    return [bool(_score(model_id,reasoning_effort,p,hidden_margin(p)) >= 0.) for p in problems]
