"""NRC numerical features, body mask, alignment and no-regional fallback head."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

_ROOT=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_nrc_pep',_ROOT/'pep.py')
pep=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pep)
MODELS=pep.MODELS
_CORRECTION=None
NUMBER=re.compile(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?')
MEASUREMENTS=dict(forwards=0,forward_seconds=0.,fallback_predictions=0,predictions=0)
OBSERVER=None  # Optional benchmark receipt writer; never changes predictions.


def correction():
    global _CORRECTION
    if _CORRECTION is None:
        _CORRECTION=json.loads((_ROOT/'correction.json').read_text())
        if _CORRECTION['schema']!='f01-experimental-correction-v1':
            raise ValueError('invalid experimental correction schema')
    return _CORRECTION


def numerical(problem):
    tokens=NUMBER.findall(problem); magnitudes=[]
    for token in tokens:
        v=abs(float(token))
        magnitudes.append(math.log1p(v) if math.isfinite(v) else len(token.lstrip('+-'))*math.log(10))
    return [max(magnitudes,default=0.)-min(magnitudes,default=0.),
        sum(t.startswith('-') for t in tokens)/max(len(tokens),1),
        sum('.' in t for t in tokens)/max(len(tokens),1)]


def problem_mask(prompt,problem,ids,offsets,special_ids):
    if not problem or prompt.count(problem)!=1:
        return []
    start=prompt.index(problem); end=start+len(problem)
    if len(ids)!=len(offsets): raise ValueError('offset/input length mismatch')
    return [i for i,(token,(a,b)) in enumerate(zip(ids,offsets))
        if b>a and a>=start and b<=end and token not in special_ids]


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def anchored_alignment(lexical,contextual,endpoint):
    import numpy as np
    u=np.asarray(contextual,float)-np.asarray(lexical,float)
    v=np.asarray(endpoint,float)-np.asarray(lexical,float)
    nu,nv=float(np.linalg.norm(u)),float(np.linalg.norm(v))
    if nu==0 or nv==0 or not np.isfinite(u).all() or not np.isfinite(v).all(): return None
    return float(u@v/max(nu*nv,1e-30))


