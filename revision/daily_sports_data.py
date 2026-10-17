"""Official Daily and Sports Activities; fixed independent 5-second segments."""
import hashlib, io, json, re
from zipfile import ZipFile
import numpy as np
from revision.run_study import ROOT

DEST=ROOT/'data'/'daily_sports'
VIEWS=['torso','right_arm','left_arm','right_leg','left_leg']

def features(signal):
    n=signal.shape[1];center=signal-signal.mean(1,keepdims=True);sd=signal.std(1)
    standard=center/np.maximum(sd[:,None,:],1e-12)
    basic=[signal.mean(1),sd,np.sqrt(np.mean(signal**2,1)),np.median(signal,axis=1),
        signal.min(1),signal.max(1),np.mean(standard**3,1),np.mean(standard**4,1)-3,
        np.mean(np.abs(np.diff(signal,axis=1)),1)]
    power=np.abs(np.fft.rfft(signal,axis=1))**2/n**2
    bands=[np.log1p(power[:,ix,:].sum(1)) for ix in np.array_split(np.arange(1,power.shape[1]),8)]
    corr=[np.mean(standard[:,:,i]*standard[:,:,j],axis=1)[:,None] for i,j in [(0,1),(0,2),(1,2)]]
    return np.hstack(basic+bands+corr)

def load_daily():
    archive=DEST/'official.zip'; cache=DEST/'physical_views162.npz'; manifest=DEST/'manifest.json'
    if cache.exists() and manifest.exists():
        meta=json.loads(manifest.read_text());assert hashlib.sha256(cache.read_bytes()).hexdigest()==meta['cache_sha256']
        d=np.load(cache);return {v:d[v] for v in VIEWS},d['y'],d['subjects'],meta
    z=ZipFile(archive);assert z.testzip() is None
    pattern=re.compile(r'(?:^|/)a(\d{2})/p(\d)/s(\d{2})\.txt$')
    names=sorted((n,pattern.search(n)) for n in z.namelist() if pattern.search(n))
    assert len(names)==9120, len(names)
    signal=[];labels=[];subjects=[];ids=[]
    for i,(name,m) in enumerate(names):
        x=np.loadtxt(io.BytesIO(z.read(name)),delimiter=',');assert x.shape==(125,45) and np.isfinite(x).all()
        signal.append(x);labels.append(int(m[1])-1);subjects.append(int(m[2]));ids.append(name)
        if (i+1)%1000==0:print('Daily segments',i+1,flush=True)
    signal=np.asarray(signal);views={}
    for j,v in enumerate(VIEWS):
        block=signal[:,:,j*9:(j+1)*9]
        views[v]=np.hstack([features(block[:,:,s:s+3]) for s in [0,3,6]])
        assert views[v].shape==(9120,162) and np.isfinite(views[v]).all()
    y=np.asarray(labels);subjects=np.asarray(subjects)
    np.savez_compressed(cache,**views,y=y,subjects=subjects)
    meta={'url':'https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip',
       'doi':'10.24432/C5C59F','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
       'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'rows':9120,'classes':19,'subjects':8,
       'sampling_Hz':25,'segment_seconds':5,'overlap':0,'views':VIEWS,'view_dimension':162,
       'feature_recipe':'54 fixed features separately for acceleration, angular rate, magnetic field; 162 per physical device',
       'record_ids':ids,'privacy_unit':'one provided labelled five-second segment replacement; not subject or participation DP'}
    manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return views,y,subjects,meta

def daily_partition(subjects,seed):
    order=np.random.default_rng(seed+91000).permutation(np.unique(subjects))
    groups=dict(public_fit=order[:1],public_calibration=order[1:2],source=order[2:4],target_train=order[4:6],target_test=order[6:])
    parts={k:np.flatnonzero(np.isin(subjects,v)) for k,v in groups.items()}
    return parts,groups

if __name__=='__main__':
    views,y,s,meta=load_daily();print(meta['rows'],meta['classes'],{k:v.shape for k,v in views.items()})
