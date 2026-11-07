#!/usr/bin/env python3
"""Standalone CPU refits from derived public-TRAIN caches; no model inference.
Reproduces the original PEP/NRC objective and nested family exclusions.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def splits(rows,seed):
    groups=[r['group_id'] for r in rows]
    return list(GroupKFold(n_splits=min(3,len(set(groups))),shuffle=True,random_state=seed).split(np.zeros(len(rows)),groups=groups))
def fit_head(z,x,y,groups,C):
    if len(set(y))<2:return None
    scale=StandardScaler().fit(np.c_[z,x]);D=np.c_[np.ones(len(y)),scale.transform(np.c_[z,x])]
    wgt=np.array([1/groups.count(g) for g in groups],float);wgt*=len(y)/wgt.sum();y=np.asarray(y,float)
    def objective(w):
        s=z+D@w;p=expit(s)
        return float(np.sum(wgt*(np.logaddexp(0,s)-y*s))+w@w/(2*C)),D.T@(wgt*(p-y))+w/C
    def hess(w):
        p=expit(z+D@w)
        return D.T@((wgt*p*(1-p))[:,None]*D)+np.eye(D.shape[1])/C
    r=minimize(objective,np.zeros(D.shape[1]),jac=True,hess=hess,method='trust-exact',options={'maxiter':100,'gtol':1e-8});w=r.x.copy()
    for _ in range(8):
        loss,g=objective(w)
        if max(abs(g))<=1e-7:break
        c=w-np.linalg.solve(hess(w),g);new,ng=objective(c)
        if not np.isfinite(c).all() or new>loss+16*np.finfo(float).eps*max(1.,abs(loss)) or np.linalg.norm(ng)>=np.linalg.norm(g):break
        w=c
    residual=float(max(abs(objective(w)[1])));assert residual<=1e-7
    return {'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'coef':w.tolist(),'C':C,'gradient_max_abs':residual}
def apply(head,z,x):
    if head is None:return expit(z)
    return expit(z+np.c_[np.ones(len(z)),(np.c_[z,x]-head['mean'])/head['scale']]@np.asarray(head['coef']))
def loss(rows,p):
    return float(np.mean([np.mean([(p[i]-r['label'])**2 for i,r in enumerate(rows) if r['group_id']==g]) for g in sorted({r['group_id'] for r in rows})]))

class Engine:
    def __init__(self,bank,endpoints,exclude=(),metadata=False):
        self.bank=bank;self.excluded=set(exclude);self.rows=[r for r in bank['rows'] if r['group_id'] not in self.excluded];self.aux=[r for r in bank['aux'] if r['group_id'] not in self.excluded]
        self.groups=set(bank['all_groups'])-self.excluded;self.vectors=endpoints;self.metadata=metadata
        self.basecache={};self.contexts={};self.fitcache={};self.receipts=[];self.base_receipts=[]
    def fit_base(self,tr):
        y=np.asarray([r['label'] for r in tr],float);weights=np.asarray([.25 if r['auxiliary'] else 1 for r in tr]);m=np.asarray([r['metadata'] for r in tr]);sm=StandardScaler().fit(m)
        matrices=[sparse.csr_matrix(sm.transform(m))];head={'metadata_mean':sm.mean_.tolist(),'metadata_scale':sm.scale_.tolist()}
        if not self.metadata:
            h=np.asarray([self.vectors[r['problem_id']] for r in tr]);sh=StandardScaler().fit(h);matrices.append(sparse.csr_matrix(sh.transform(h)));head.update(hidden_mean=sh.mean_.tolist(),hidden_scale=sh.scale_.tolist())
        if len(set(y))<2:return {'constant':float(np.average(y,weights=weights))}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');clf=LogisticRegression(C=.01,class_weight='balanced',solver='liblinear',dual=True,max_iter=2000,random_state=2026).fit(sparse.hstack(matrices,format='csr'),y,sample_weight=weights)
        assert not any('converg' in str(w.message).lower() for w in ws)
        head.update(metadata_coef=clf.coef_[0,:14].tolist(),intercept=float(clf.intercept_[0]))
        if not self.metadata:head['hidden_coef']=clf.coef_[0,14:].tolist()
        return head
    def predict_base(self,head,te):
        if not te:return np.empty(0)
        if 'constant' in head:return np.full(len(te),head['constant'])
        m=np.asarray([r['metadata'] for r in te]);s=head['intercept']+((m-head['metadata_mean'])/head['metadata_scale'])@np.asarray(head['metadata_coef'])
        if not self.metadata:
            h=np.asarray([self.vectors[r['problem_id']] for r in te]);s+=((h-head['hidden_mean'])/head['hidden_scale'])@np.asarray(head['hidden_coef'])
        return expit(s)
    def base(self,allowed,te):
        assert not set(r['group_id'] for r in te)&set(allowed)
        key=tuple(sorted(allowed))
        if key not in self.basecache:
            tr=[r for r in self.rows+self.aux if r['group_id'] in allowed];head=self.fit_base(tr)
            self.basecache[key]=head
            self.base_receipts.append({'allowed_groups':list(key),'training_ids':[r['id'] for r in tr],'training_groups':sorted({r['group_id'] for r in tr})})
        return self.predict_base(self.basecache[key],te)
    def context(self,tr,te,allowed,seed):
        key=digest([[r['id'] for r in tr],[r['id'] for r in te],sorted(allowed),seed])
        if key not in self.contexts:
            pt=np.full(len(tr),np.nan)
            for _,b in splits(tr,seed):
                held=[tr[i] for i in b];pt[b]=self.base(allowed-{r['group_id'] for r in held},held)
            assert np.isfinite(pt).all()
            self.contexts[key]={'id':key,'train':tr,'test':te,'allowed':allowed,'seed':seed,'pt':pt,'pv':self.base(allowed,te)}
        return self.contexts[key]
    def children(self,ctx):
        tr=ctx['train'];out=[]
        for j,(a,b) in enumerate(splits(tr,ctx['seed'])):
            fit=[tr[i] for i in a];val=[tr[i] for i in b]
            out.append(self.context(fit,val,ctx['allowed']-{r['group_id'] for r in val},ctx['seed']+100+j))
        return out
    def fit(self,ctx,variant,C):
        key=ctx['id'],variant,C
        if key not in self.fitcache:
            tr,te=ctx['train'],ctx['test'];width=len(self.bank['variants'][variant])
            get=lambda rows:np.asarray([self.bank['features'][r['problem_id']][variant] for r in rows],float).reshape(len(rows),width)
            x,v=get(tr),get(te);valid=np.isfinite(x).all(1);seen=np.isfinite(v).all(1)
            z=logit(np.clip(ctx['pt'],1e-8,1-1e-8));zv=logit(np.clip(ctx['pv'],1e-8,1-1e-8))
            fitted=[r for i,r in enumerate(tr) if valid[i]]
            head=fit_head(z[valid],x[valid],[r['label'] for r in fitted],[r['group_id'] for r in fitted],C)
            p=ctx['pv'].copy()
            if not seen.all() and variant!='no_q':p=self.model(ctx,'no_q')[0].copy()
            if seen.any():p[seen]=apply(head,zv[seen],v[seen])
            receipt={'context':ctx['id'],'variant':variant,'C':C,'head':head,'training_ids':[r['id'] for r in fitted],'test_ids':[r['id'] for r in te],'training_groups':sorted({r['group_id'] for r in fitted}),'test_groups':sorted({r['group_id'] for r in te}),'unsupported_test':int((~seen).sum())}
            assert not set(receipt['training_groups'])&set(receipt['test_groups'])
            self.receipts.append(receipt);self.fitcache[key]=(p,receipt)
        return self.fitcache[key]
    def model(self,ctx,variant):
        trials={}
        for C in [.1,1.]:
            rr=[];pp=[]
            for child in self.children(ctx):
                p,_=self.fit(child,variant,C);rr+=child['test'];pp+=p.tolist()
            trials[C]=loss(rr,pp)
        C=min(trials,key=lambda c:(trials[c],c));p,receipt=self.fit(ctx,variant,C)
        return p,dict(receipt,selected_C=C,inner_brier=trials)
    def evaluate(self,variants):
        pred=[];selected=[]
        for fold in self.bank['folds']:
            held=set(fold['evaluation_groups']);tr=[r for r in self.rows if r['group_id'] not in held];te=[r for r in self.rows if r['group_id'] in held]
            if not te:continue
            ctx=self.context(tr,te,self.groups-held,fold['seed']+fold['fold']);outputs={}
            for v in variants:
                outputs[v],receipt=self.model(ctx,v);selected.append(dict(receipt,seed=fold['seed'],fold=fold['fold']))
            for i,r in enumerate(te):pred.append({'id':r['id'],'problem_id':r['problem_id'],'family':r['group_id'],'model':r['model_id'],'label':r['label'],'seed':fold['seed'],'fold':fold['fold'],'probabilities':{'pep':float(ctx['pv'][i]),**{v:float(p[i]) for v,p in outputs.items()}}})
        return {'predictions':pred,'selected':selected,'base_fits':len(self.basecache),'correction_fits':len(self.receipts),'base_fit_audit':self.base_receipts,'excluded_families':sorted(self.excluded)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--out',type=Path,default=Path('runs/refits'));parser.add_argument('--influence',action='store_true');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);bank=read(args.bank);v=np.load(args.bank.with_name('endpoints.npz'));vectors={k:v[k] for k in v.files};tick=time.monotonic()
    for name,meta,variants,exclude in [('controls',False,list(bank['variants']),[]),('no-whitebox',True,['numeric3'],[])]+([('exclude-circumradius',False,['no_q','full'],[bank['influential_families'][0]]),('exclude-walking',False,['no_q','full'],[bank['influential_families'][1]]),('exclude-both',False,['no_q','full'],bank['influential_families'])] if args.influence else []):
        path=args.out/(name+'.json')
        if path.exists():continue
        print('Starting',name,flush=True);e=Engine(bank,vectors,exclude,meta);result=e.evaluate(variants)
        if name=='controls':
            final=e.context(e.rows,[],e.groups,2026);_,head=e.model(final,'full');result['final_nrc']=head
            result['final_pep']=e.fit_base(e.rows+e.aux)
        result['seconds']=time.monotonic()-tick;save(path,result);print('Completed',name,result['base_fits'],result['correction_fits'],flush=True)
    import scipy,sklearn
    save(args.out/'environment.json',{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'elapsed_seconds':time.monotonic()-tick,'new_LM_forwards':0})
if __name__=='__main__':main()
