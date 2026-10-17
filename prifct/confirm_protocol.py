"""Immutable role protocol for the reviewer-directed next experimental round.

Role assignment is a public-benchmark curator operation, NOT a private release
operation. Adjacent protected cohorts must keep these role identities fixed.
Office31 labels are never consumed by development or parameter selection.
"""
import hashlib,itertools,json
from functools import lru_cache
from pathlib import Path
import numpy as np
from prifct.data import load
from prifct.calibrated_data import FrozenMap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/models/eswa_fedkapt/original_v5_confirm'
SEEDS=list(range(62101,62121))
DEV_SEEDS=[48101,48102]
DIRECTIONS=[('art','clipart'),('product','real_world')]
CONDITIONS=['full','quarter','half','LT10']

@lru_cache(maxsize=3)
def dataset(name):
    if name=='HAR':
        views,y,subjects,_=load('HAR')
        return {'accel':views['accel']},{'accel':y},{'accel':subjects},{'accel':np.arange(len(y)).astype(str)},6
    domains=['art','clipart','product','real_world'] if name=='OfficeHome' else ['amazon','dslr','webcam']
    root=ROOT/'data'/('office_home' if name=='OfficeHome' else 'office31')/'.cache'
    k=65 if name=='OfficeHome' else 31
    xx,yy,ii={},{},{}
    for domain in domains:
        r=np.load(root/f'{domain}_resnet50_all{k}.npz',allow_pickle=True)
        xx[domain]=r['features'];yy[domain]=r['labels']
        ii[domain]=np.array([domain+'/'+Path(p).parent.name+'/'+Path(p).name for p in r['paths']])
    return xx,yy,None,ii,k

def _allocate(ids,weights):
    expected=len(ids)*np.asarray(weights);sizes=np.floor(expected).astype(int)
    for j in np.argsort(-(expected-sizes),kind='stable')[:len(ids)-sizes.sum()]:sizes[j]+=1
    split=np.split(np.asarray(ids),np.cumsum(sizes)[:-1])
    return split

def freeze_roles():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'identity_roles.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        if previous['version']=='review-round1-immutable-roles-2':return previous
        # Protocol implementation repair before any external scores. Preserve
        # both prior role hash and preregistration rather than erase history.
        archive=OUT/'protocol_repair_archive';archive.mkdir(exist_ok=True)
        for name in ['identity_roles.json','identity_roles.sha256','preregistration.json','preregistration.sha256']:
            src=OUT/name
            if src.exists():src.replace(archive/name)
    role_manifest={'version':'review-round1-immutable-roles-2','seed':502701,'datasets':{},
        'adjacency_note':'Fixed public benchmark identities/role metadata under full feature-label replacement. Do not rerun class-dependent splitting on protected records.',
        'reuse_note':'No global identity-role overlap. A fixed source/public/adaptation/test role may be reused across directions and repeat noise seeds; these are not independent domain populations.'}
    for name in ['OfficeHome','Office31','HAR']:
        x,y,subjects,identities,k=dataset(name);domains={}
        for dn,features in x.items():
            rng=np.random.default_rng(502701+sum(map(ord,name+dn)))
            if name=='HAR':
                group=rng.permutation(np.unique(subjects[dn]))
                role_groups=dict(zip(['public','development','source','adaptation','test'],np.split(group,[4,10,18,24])))
                roles={r:np.flatnonzero(np.isin(subjects[dn],g)).tolist() for r,g in role_groups.items()}
            else:
                order=rng.permutation(len(features));npublic=min(400,int(.2*len(features)))
                roles={'public':order[:npublic].tolist()};remaining=order[npublic:]
                names=['development','source','adaptation','test'] if name=='OfficeHome' else ['source','adaptation','test']
                weights=[.375,.25,.225,.15] if name=='OfficeHome' else [.5,.3,.2]
                roles.update({r:[] for r in names})
                for label in range(k):
                    index=remaining[y[dn][remaining]==label]
                    for r,part in zip(names,_allocate(index,weights)):roles[r].extend(part.tolist())
            all_ids=[i for rows in roles.values() for i in rows]
            assert len(set(all_ids))==len(features)==len(all_ids)
            if subjects is not None:
                for a,b in itertools.combinations(roles,2):
                    assert not(set(subjects[dn][roles[a]])&set(subjects[dn][roles[b]]))
            domains[dn]={'roles':roles,'ids':identities[dn].tolist(),'rows':len(features),'dimension':features.shape[1]}
            if name=='HAR':
                domains[dn]['subject_groups']={r:g.tolist() for r,g in role_groups.items()}
                groups=role_groups['development']
                domains[dn]['development_parts']={r:np.flatnonzero(np.isin(subjects[dn],g)).tolist() for r,g in zip(['source','adaptation','validation'],np.split(groups,[2,4]))}
                dev_source=np.asarray(domains[dn]['development_parts']['source'])
            else:dev_source=np.asarray(roles['development'],int) if 'development' in roles else None
            domains[dn]['conditions']={'main':{c:subset(roles['source'],y[dn],c,64103).tolist() for c in CONDITIONS}}
            if dev_source is not None:domains[dn]['conditions']['development']={c:subset(dev_source,y[dn],c,63107).tolist() for c in CONDITIONS}
        role_manifest['datasets'][name]={'classes':k,'domains':domains,'status':'unseen external benchmark' if name=='Office31' else 'previously observed benchmark; constrained exploratory replication'}
    serial=json.dumps(role_manifest,indent=2,ensure_ascii=False)
    path.write_bytes(serial.encode('utf8'))
    (OUT/'identity_roles.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    return role_manifest

def subset(index,y,condition,seed):
    index=np.asarray(index,int);rng=np.random.default_rng(seed+31870)
    if condition=='full':return index
    if condition in ['quarter','half']:
        return rng.permutation(index)[:max(2,int(len(index)*(.25 if condition=='quarter' else .5)))]
    if condition=='LT10':
        # Curator predefines this labelled public-benchmark cohort ONCE.
        # During protected adjacency, the resulting identities do not change.
        selected=[];k=int(y.max()+1)
        for c in range(k):
            rows=index[y[index]==c]
            keep=max(1,int(len(rows)*10**(-c/max(1,k-1))))
            selected.extend(rng.permutation(rows)[:keep])
        return np.asarray(selected,int)
    raise ValueError(condition)

def case(name,a,b,seed,condition='full',development=False):
    manifest=freeze_roles();views,labels,subjects,ids,k=dataset(name)
    if development and name=='Office31':raise ValueError('No Office31 development permitted')
    source_roles=manifest['datasets'][name]['domains'][a]['roles']
    target_roles=manifest['datasets'][name]['domains'][b]['roles']
    pub=np.asarray(source_roles['public'],int)
    if development:
        source_indices=np.asarray(source_roles['development'],int)
        target_pool=np.asarray(target_roles['development'],int)
        if name=='HAR':
            part=manifest['datasets'][name]['domains'][b]['development_parts']
            source_indices=np.asarray(part['source']);fit=np.asarray(part['adaptation']);test=np.asarray(part['validation'])
        else:
            rng=np.random.default_rng(seed);fit=[];test=[]
            for c in range(k):
                rows=rng.permutation(target_pool[labels[b][target_pool]==c]);cut=max(1,int(.6*len(rows)))
                fit.extend(rows[:cut]);test.extend(rows[cut:])
            fit,test=np.asarray(fit),np.asarray(test)
    else:
        source_indices=np.asarray(source_roles['source'],int)
        fit=np.asarray(target_roles['adaptation'],int);test=np.asarray(target_roles['test'],int)
    # The source subset seed is independent of replicate noise/clustering seed.
    source_indices=np.asarray(manifest['datasets'][name]['domains'][a]['conditions']['development' if development else 'main'][condition],int)
    if a==b:
        assert not(set(source_indices)&set(fit) or set(source_indices)&set(test) or set(fit)&set(test))
    rank=16 if name=='HAR' else 128
    mode='log_standard' if name=='HAR' else 'raw_center'
    sm=FrozenMap(views[a][pub],rank,pca=name!='HAR',mode=mode)
    # Small Office31 public pools can make actual PCA dimension <128. Follow
    # the predeclared SVD rule, expose actual rank, and copy the same projection.
    tm=FrozenMap(views[b][fit],rank,mode=mode);tm.projection=sm.projection.copy()
    return {'source':sm.transform(views[a][source_indices]),'source_y':labels[a][source_indices],
       'train':tm.transform(views[b][fit]),'train_y':labels[b][fit],
       'test':tm.transform(views[b][test]),'test_y':labels[b][test],
       'public':sm.transform(views[a][pub]),'source_map':sm,'target_map':tm,
       'source_indices':source_indices,'train_indices':fit,'test_indices':test,'public_indices':pub,
       'subjects_test':None if subjects is None else subjects[b][test],'k':k,'comparable':True}

if __name__=='__main__':
    manifest=freeze_roles()
    print(json.dumps({n:{d:{r:len(i) for r,i in v['roles'].items()} for d,v in ds['domains'].items()} for n,ds in manifest['datasets'].items()},indent=2))
