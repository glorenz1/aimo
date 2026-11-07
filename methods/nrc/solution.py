"""Numerical and Regional Correction (NRC) over the frozen PEP score."""
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=_module('_nrc_reference_'+ROOT.name,ROOT/'reference.py')
E=_module('_nrc_features_'+ROOT.name,ROOT/'regional_features.py')
PIPELINE=json.loads((ROOT/'pipeline.json').read_text())
MODELS=B.MODELS
OBSERVER=None
MEASUREMENTS=dict(forwards=0,seconds=0.,fallbacks=0)


def role_features(row,states):
    import numpy as np
    masks=E.role_masks(row);x=np.asarray(states,float);end=x[-1]
    means={r:x[masks[r]].mean(0) if masks[r] else np.zeros(4096) for r in ['body','numeric','relation','identifier']}
    out={}
    for second in ['relation','identifier']:
        out['bf31_numeric_'+second+'_endpoint_gap']=E.cos(means['numeric'],end)-E.cos(means[second],end)
        out['bf31_numeric_'+second+'_translation']=E.cos(means['numeric']-means['body'],means[second]-means['body'])
    return out


@functools.lru_cache(maxsize=20000)
def measure(problem):
    import numpy as np
    import torch
    tok,model=B.pep._load();t=time.perf_counter();prompt=B.pep.build_prompt(tok,problem)
    encoded=tok(prompt,return_tensors='pt',truncation=True,return_offsets_mapping=True)
    ids=encoded['input_ids'][0].tolist();offsets=encoded.pop('offset_mapping')[0].tolist();mask=B.problem_mask(prompt,problem,ids,offsets,set(tok.all_special_ids))
    inp={k:v.to(model.device) for k,v in encoded.items()};observed={};raw=PIPELINE['requires_raw_roles']
    def emb(_m,_a,out):
        if mask:observed['lexical']=out[0,mask,:].detach().float().cpu().numpy().mean(0,dtype=np.float32)
    def layer(_m,_a,out):
        if raw:
            observed['states']=out[0].detach().float().cpu().numpy().copy();observed['endpoint']=observed['states'][-1].copy()
            if mask:observed['contextual']=observed['states'][mask].mean(0,dtype=np.float32)
        else:
            observed['endpoint']=out[0,-1,:].detach().float().cpu().numpy().copy()
            if mask:observed['contextual']=out[0,mask,:].detach().float().cpu().numpy().mean(0,dtype=np.float32)
    handles=[]
    try:
        handles.append(model.get_input_embeddings().register_forward_hook(emb));handles.append(model.model.layers[30].register_forward_hook(layer))
        with torch.no_grad():out=model(**inp,output_hidden_states=False,return_dict=True,use_cache=False,logits_to_keep=1)
        del out
    finally:
        for h in handles:h.remove()
    end=observed['endpoint'];artifact=B.pep.artifact()
    hidden=math.fsum(w*(float(x)-m)/s for w,x,m,s in zip(artifact['hidden_coef'],end,artifact['hidden_mean'],artifact['hidden_scale']))
    d={};nums=B.numerical(problem);valid=bool(mask)
    align=B.anchored_alignment(observed['lexical'],observed['contextual'],end) if valid else None
    valid=bool(valid and align is not None)
    d.update(zip(E.CORE_FEATURES,nums+[align]));hashes={'endpoint':hashlib.sha256(end.tobytes()).hexdigest()}
    if mask:
        for k in ['lexical','contextual']:hashes[k]=hashlib.sha256(observed[k].tobytes()).hexdigest()
    if raw:hashes['full_states']=hashlib.sha256(observed['states'].tobytes()).hexdigest()
    if raw and valid:
        start=prompt.index(problem);row=dict(problem=problem,original_span=[start,start+len(problem)],mask=mask,offsets=offsets)
        d.update(role_features(row,observed['states']))
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    record=dict(features={k:d.get(k) for k in set(PIPELINE['keys']+E.CORE_FEATURES)},
        hidden=hidden,valid=valid,hashes=hashes,input_ids_hash=B.digest(ids),mask_hash=B.digest(mask),prompt_hash=B.digest(prompt),tokens=len(ids),seconds=elapsed)
    MEASUREMENTS['forwards']+=1;MEASUREMENTS['seconds']+=elapsed
    if OBSERVER is not None:OBSERVER(problem,record)
    return record


def score_probability(margin,values,head=None):
    if head is None:head=PIPELINE['head']
    p=1/(1+math.exp(-margin)) if margin>=0 else math.exp(margin)/(1+math.exp(margin));p=min(1-1e-8,max(1e-8,p));z=math.log(p/(1-p))
    raw=[z]+list(values)
    if len(raw)!=len(head['mean']):raise ValueError('head feature width mismatch')
    s=z+head['coef'][0]+math.fsum(w*(x-m)/v for w,x,m,v in zip(head['coef'][1:],raw,head['mean'],head['scale']))
    return 1/(1+math.exp(-s)) if s>=0 else math.exp(s)/(1+math.exp(s))


def probability(model,effort,problem):
    observed=measure(problem);margin=B.pep._score(model,effort,problem,observed['hidden'])
    if effort!='default' or not observed['valid']:
        MEASUREMENTS['fallbacks']+=1;p=1/(1+math.exp(-margin)) if margin>=0 else math.exp(margin)/(1+math.exp(margin))
        return math.nextafter(.5,0) if margin<0 and p>=.5 else p
    v=[observed['features'].get(k) for k in PIPELINE['keys']]
    if not all(x is not None and math.isfinite(x) for x in v):
        MEASUREMENTS['fallbacks']+=1
        return score_probability(margin,[observed['features'][k] for k in E.CORE_FEATURES],B.correction()['head'])
    return score_probability(margin,v)


def are_robust(model_id: str,reasoning_effort: str,problems: list[str])->list[bool]:
    if model_id not in MODELS:raise ValueError('unsupported Small Models checkpoint')
    return [bool(probability(model_id,reasoning_effort,p)>=.5) for p in problems]
