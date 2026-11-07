"""Portable raw-input -> exact features -> training bank, without repository imports.

Requires permitted public TRAIN exports and either certified raw-state caches or
an explicitly requested fresh GPU prefill. No data/model download is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
NRC_SHA = '4b482b26cf8ec0bd51f5cb2bcb1ad6c964479091e0f828f27414114aef8312d2'
REVISION = '6e8885a6ff5c1dc5201574c8fd700323f23c25fa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_package(bundle, directory):
    if sha(bundle) != NRC_SHA:
        raise ValueError('Not the frozen NRC package')
    with zipfile.ZipFile(bundle) as z:
        # A known immutable archive; still reject path traversal.
        for name in z.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member')
        z.extractall(directory)
    spec = importlib.util.spec_from_file_location('exact_nrc_rebuild', Path(directory)/'solution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_inputs(manifest, releases):
    """Match only the frozen allowlist; do not expand training to extra rows."""
    selected = {}
    expected = {(r['source_revision'],r['problem_sha256'],r['model_id'],r['reasoning_effort']):r
                for r in manifest['records']}
    if len(expected) != len(manifest['records']):
        raise ValueError('Manifest identity collision')
    for revision, path in releases.items():
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row = row.get('row', row)
                text = row.get('problem', row.get('original_problem'))
                if not isinstance(text,str):
                    continue
                key = (revision,hashlib.sha256(text.encode()).hexdigest(),row.get('model_id'),row.get('reasoning_effort') or 'default')
                if key not in expected:
                    continue
                label = row.get('is_robust',row.get('model_is_robust'))
                if label is not None and type(label) is not bool:
                    raise ValueError('Labels must be Boolean or null, never inferred from correctness')
                value = dict(expected[key],problem=text,label=label)
                if key in selected and selected[key] != value:
                    raise ValueError('Conflicting duplicate selected input')
                selected[key] = value
    if set(selected) != set(expected):
        raise ValueError(f'Missing {len(set(expected)-set(selected))} frozen authorized inputs')
    return [selected[(r['source_revision'],r['problem_sha256'],r['model_id'],r['reasoning_effort'])]
            for r in manifest['records']]


def certify_cache(directory, index, pid, text):
    item = index['records'][pid]
    if index['encoder_revision'] != REVISION or index['layer'] != 31:
        raise ValueError('Checkpoint/layer mismatch')
    if item['problem_sha256'] != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError('Raw cache belongs to another original')
    path = (directory/item['states']).resolve()
    if not path.is_relative_to(directory.resolve()) or sha(path) != item['states_sha256']:
        raise ValueError('Invalid raw-state cache path/hash')
    x = np.load(path, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != 4096 or not np.isfinite(x).all():
        raise ValueError('Raw token-state cache shape/value mismatch')
    row = dict(item['row'],problem=text)
    if len(row['offsets']) != len(x):
        raise ValueError('Tokenizer offsets do not align with states')
    lexical = np.asarray(item['lexical'],dtype=np.float32)
    if lexical.shape != (4096,) or not np.isfinite(lexical).all():
        raise ValueError('Invalid cached lexical embedding mean')
    return row, x, lexical


def extract_features(mod, row, states, lexical):
    masks = mod.E.role_masks(row)
    x = np.asarray(states,float)
    body = masks['body']
    if not body:
        raise ValueError('Original reproduction cohort requires populated bodies')
    E = x[-1]
    B = x[body].mean(0)
    regions = [masks[k] for k in ['numeric','relation','identifier']]
    nums = mod.B.numerical(row['problem'])
    a = mod.B.anchored_alignment(lexical,states[body].mean(0,dtype=np.float32),states[-1])
    if a is None:
        raise ValueError('Original reproduction cohort requires valid alignment')
    anchor = nums+[a]
    def geometry(rr):
        N,R,I = [x[ix].mean(0) if len(ix) else np.zeros(4096) for ix in rr]
        co = mod.E.cos
        return [co(N,E)-co(R,E),co(N,E)-co(I,E),co(N-B,R-B),co(N-B,I-B)]
    q = geometry(regions)
    exact = mod.role_features(row,states)
    keys = ['bf31_numeric_relation_endpoint_gap','bf31_numeric_identifier_endpoint_gap',
            'bf31_numeric_relation_translation','bf31_numeric_identifier_translation']
    if not np.allclose(q,[exact[k] for k in keys],rtol=0,atol=1e-12):
        raise ValueError('Packaged regional arithmetic disagrees')
    counts = [len(ix) for ix in regions]
    uniform = [[body[i] for i in np.linspace(0,len(body)-1,n).astype(int)] if n else [] for n in counts]
    generic = sorted(set(body)-set().union(*map(set,regions)))
    matched = []
    for ix in regions:
        if len(generic) < len(ix):
            matched = None
            break
        if not ix:
            matched.append([])
            continue
        _, b = linear_sum_assignment(abs(np.asarray(ix)[:,None]-np.asarray(generic)[None,:]))
        matched.append([generic[j] for j in b])
    means = [x[ix].mean(0) if ix else np.zeros(4096) for ix in regions]+[B]
    extras = {'no_q':[],'q12':q[:2],'q34':q[2:],'full':q,
              'presence3':[int(n>0) for n in counts],'counts3':counts,
              'fractions3':[n/len(x) for n in counts],
              'counts_length4':counts+[len(x)],'full_plus_counts_length4':q+counts+[len(x)],
              'uniform4':geometry(uniform),
              'nonrole_position4':geometry(matched) if matched is not None else [None]*4,
              'radial4':[math.log1p(float(np.linalg.norm(v))/max(float(np.linalg.norm(E)),1e-30)) for v in means]}
    for seed in [20261105,20261106,20261107]:
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{row['problem_id']}".encode()).hexdigest()[:16],16))
        lookup = dict(zip(body,rng.permutation(body)))
        extras[f'shuffle4_seed{seed}'] = geometry([[int(lookup[i]) for i in ix] for ix in regions])
    return {k:anchor+v for k,v in extras.items()}|{'numeric3':nums}


def collect_fresh(mod, selected, destination):
    """One actual packaged prefill per original; no generation. GPU path untested here."""
    destination.mkdir(parents=True,exist_ok=False)
    tok, model = mod.B.shadow._load()
    records = {}
    for pid, record in {r['problem_id']:r for r in selected}.items():
        text = record['problem']
        prompt = mod.B.shadow.build_prompt(tok,text)
        encoded = tok(prompt,return_tensors='pt',truncation=True,return_offsets_mapping=True)
        ids = encoded['input_ids'][0].tolist()
        offsets = encoded['offset_mapping'][0].tolist()
        mask = mod.B.problem_mask(prompt,text,ids,offsets,set(tok.all_special_ids))
        start = prompt.index(text)
        row = dict(problem_id=pid,original_span=[start,start+len(text)],mask=mask,offsets=offsets)
        captured = {}
        def emb(_module,_args,out):
            captured['lexical'] = out[0,mask,:].detach().float().cpu().numpy().mean(0,dtype=np.float32)
        def layer(_module,_args,out):
            captured['states'] = out[0].detach().float().cpu().numpy().copy()
        handles = [model.get_input_embeddings().register_forward_hook(emb),model.model.layers[30].register_forward_hook(layer)]
        try:
            mod.measure.cache_clear()
            observed = mod.measure(text)
        finally:
            for h in handles:h.remove()
        states = captured['states']
        path = destination/(pid.replace(':','_')+'.npy')
        np.save(path,states,allow_pickle=False)
        records[pid] = dict(states=path.name,states_sha256=sha(path),row=row,
                            lexical=captured['lexical'].tolist(),problem_sha256=record['problem_sha256'])
        calculated = extract_features(mod,dict(row,problem=text),states,captured['lexical'])['full']
        packaged = [observed['features'][k] for k in mod.PIPELINE['keys']]
        if not np.allclose(calculated,packaged,rtol=0,atol=1e-12):
            raise ValueError('Fresh features differ from actual frozen measurement')
    save(destination/'cache-index.json',dict(encoder_revision=REVISION,layer=31,
        cache_kind='fresh frozen NRC prefill plus passive measurement hooks',records=records))


def rebuild(manifest, selected, mod, cache_dir, output):
    output.mkdir(parents=True,exist_ok=False)
    cache = json.loads((cache_dir/'cache-index.json').read_text())
    inputs = {r['problem_id']:r for r in selected}
    vectors, features = {}, {}
    small_ids = {r['problem_id'] for r in selected if r['model_id'] != 'openai/gpt-oss-120b' and type(r['label']) is bool}
    for pid,r in inputs.items():
        row,states,lexical = certify_cache(cache_dir,cache,pid,r['problem'])
        vectors[pid] = np.asarray(states[-1],float)
        if pid in small_ids:
            features[pid] = extract_features(mod,row,states,lexical)
    fitted = [dict(id=r['id'],problem_id=r['problem_id'],group_id=r['group_id'],model_id=r['model_id'],
                   reasoning_effort=r['reasoning_effort'],label=int(r['label']),auxiliary=r['model_id']=='openai/gpt-oss-120b',
                   metadata=mod.B.shadow.metadata(r['model_id'],r['reasoning_effort'],r['problem']),
                   partition='train',evidence='official_public_training_label')
              for r in selected if type(r['label']) is bool]
    bank = {k:manifest[k] for k in ['all_groups','folds','variants','influential_families']}
    bank.update(rows=[r for r in fitted if not r['auxiliary']],aux=[r for r in fitted if r['auxiliary']],features=features)
    save(output/'training-bank.json',bank)
    np.savez_compressed(output/'endpoints.npz',**vectors)
    save(output/'rebuild-receipt.json',dict(selected_input_rows=len(selected),small_binary_rows=len(bank['rows']),
        auxiliary_binary_rows=len(bank['aux']),feature_originals=len(features),endpoint_originals=len(vectors),
        raw_cache_kind=cache['cache_kind'],new_labels=False,original_statements_redistributed=False))
    return bank,vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=HERE/'input-selection.json')
    parser.add_argument('--release',action='append',required=True,help='REVISION=authorized-export.jsonl; repeat for both releases')
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--fresh-prefill',action='store_true',help='Explicitly run frozen GPU encoder; creates --cache, never downloads')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = select_inputs(manifest,dict(x.split('=',1) for x in args.release))
    with tempfile.TemporaryDirectory() as tmp:
        mod = load_package(args.bundle,tmp)
        if args.fresh_prefill:collect_fresh(mod,selected,args.cache)
        rebuild(manifest,selected,mod,args.cache,args.out)


if __name__ == '__main__':
    main()
