"""UCI HAR mirror validation and subject-disjoint sensor views.

The provided 561-feature benchmark upstream normalisation is not re-proved
private. DP here is conditional on supplied benchmark features, protecting
one labelled WINDOW replacement, not an entire subject or recording.
"""
import hashlib,json
import io
from zipfile import ZipFile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'data'/'uci_har'

def subject_partition(subjects,seed):
    universe=np.unique(subjects)
    if len(universe)!=30:raise ValueError('Expected all 30 UCI HAR subjects')
    ids=np.random.default_rng(seed+61000).permutation(universe)
    groups=dict(public_fit=ids[:4],public_calibration=ids[4:6],source=ids[6:16],target_train=ids[16:23],target_test=ids[23:])
    indices={name:np.flatnonzero(np.isin(subjects,group)) for name,group in groups.items()}
    assert len(set(np.concatenate(list(groups.values()))))==30
    assert sum(map(len,indices.values()))==len(subjects)
    return indices,groups

def load_har():
    files=['features.txt','activity_labels.txt','train/X_train.txt','train/y_train.txt',
        'train/subject_train.txt','test/X_test.txt','test/y_test.txt','test/subject_test.txt']
    provenance={}
    for name in files:
        path=ROOT/name;info=path.with_suffix(path.suffix+'.retrieval.json')
        if not(path.exists() and info.exists()):raise FileNotFoundError(f'Complete verified retrieval required: {path}')
        meta=json.loads(info.read_text());actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=meta['sha256']:raise ValueError('File changed after retrieval: '+name)
        provenance[name]=meta
    names=[line.split(maxsplit=1)[1].strip() for line in (ROOT/'features.txt').read_text().splitlines()]
    # Exclude every angle feature: several combine accelerometer and gyro.
    # These are separate sensor-derived columns, not mean/worst homologues.
    accel=np.array([i for i,name in enumerate(names) if 'Acc' in name and 'Gyro' not in name and not name.startswith('angle')])
    gyro=np.array([i for i,name in enumerate(names) if 'Gyro' in name and 'Acc' not in name and not name.startswith('angle')])
    excluded=sorted(set(range(561))-set(accel)-set(gyro))
    assert len(names)==561 and len(set(accel)&set(gyro))==0
    parts=[];labels=[];subjects=[]
    for split,n in [('train',7352),('test',2947)]:
        x=np.loadtxt(ROOT/split/('X_'+split+'.txt'));y=np.loadtxt(ROOT/split/('y_'+split+'.txt'),dtype=int)-1
        s=np.loadtxt(ROOT/split/('subject_'+split+'.txt'),dtype=int)
        assert x.shape==(n,561) and y.shape==s.shape==(n,)
        assert np.isfinite(x).all() and set(y)==set(range(6))
        parts.append(x);labels.append(y);subjects.append(s)
    x=np.vstack(parts);y=np.concatenate(labels);s=np.concatenate(subjects)
    assert len(np.unique(s))==30 and set(np.unique(s))==set(range(1,31))
    assert not(set(subjects[0])&set(subjects[1]))
    manifest={'n':len(y),'classes':6,'subjects':30,'source_view_columns':len(accel),'target_view_columns':len(gyro),
        'accel_indices':accel.tolist(),'gyro_indices':gyro.tolist(),'excluded':[{ 'index':i,'name':names[i]} for i in excluded],
        'feature_names':names,'retrieval':provenance,
        'privacy_unit':'one labelled window replacement; conditional on benchmark upstream preprocessing; not subject DP',
        'split_note':'official train/test reconstructed then all 30 subjects repartitioned; not official leaderboard split'}
    (ROOT/'manifest_sensor_views.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {'accel':x[:,accel],'gyro':x[:,gyro]},y,s,manifest


def window_features(signal):
    """Fixed per-window 54D features; no cohort mean/scale or learned filter.

    Input shape [window,128,3], supplied filtered measurements. Eight equal
    positive-frequency power bands use log1p. All operations are row-local.
    Final private-vector bound is imposed after a public-only frozen bridge.
    """
    signal=np.asarray(signal,float)
    if signal.ndim!=3 or signal.shape[1:]!=(128,3):raise ValueError('128-sample triaxial windows required')
    centered=signal-signal.mean(1,keepdims=True);sd=signal.std(1)
    standard=centered/np.maximum(sd[:,None,:],1e-12)
    basic=[signal.mean(1),sd,np.sqrt(np.mean(signal**2,1)),np.median(signal,axis=1),
        signal.min(1),signal.max(1),np.mean(standard**3,1),np.mean(standard**4,1)-3,
        np.mean(np.abs(np.diff(signal,axis=1)),1)]
    power=np.abs(np.fft.rfft(signal,axis=1))**2/128**2
    bands=[np.log1p(power[:,ix,:].sum(1)) for ix in np.array_split(np.arange(1,65),8)]
    correlation=[np.mean(standard[:,:,i]*standard[:,:,j],axis=1)[:,None] for i,j in [(0,1),(0,2),(1,2)]]
    return np.hstack(basic+bands+correlation)


def load_har_raw():
    _,y,subjects,_=load_har()
    verification=json.loads((ROOT/'official_verification.json').read_text())
    archive=ROOT/'official_complete.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==verification['sha256']
    cache=ROOT/'raw_sensor54.npz';meta_path=ROOT/'manifest_raw_sensor_views.json'
    if cache.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if hashlib.sha256(cache.read_bytes()).hexdigest()==meta['cache_sha256']:
            data=np.load(cache);return {v:data[v] for v in ['accel','gyro']},y,subjects,meta
    outer=ZipFile(archive);inner=ZipFile(io.BytesIO(outer.read('UCI HAR Dataset.zip')))
    views={};members={}
    for view,prefix in [('accel','total_acc'),('gyro','body_gyro')]:
        blocks=[]
        for split,n in [('train',7352),('test',2947)]:
            channels=[]
            for axis in ['x','y','z']:
                name=f'UCI HAR Dataset/{split}/Inertial Signals/{prefix}_{axis}_{split}.txt'
                raw=inner.read(name);members[name]=hashlib.sha256(raw).hexdigest()
                channel=np.loadtxt(io.BytesIO(raw));assert channel.shape==(n,128)
                channels.append(channel)
            blocks.append(window_features(np.stack(channels,axis=2)))
        views[view]=np.vstack(blocks)
        assert views[view].shape==(10299,54) and np.isfinite(views[view]).all()
    np.savez_compressed(cache,**views,y=y,subjects=subjects)
    manifest={'n':len(y),'classes':6,'subjects':30,'view_dimensions':{'accel':54,'gyro':54},
        'physical_channels':{'accel':'total_acc_x/y/z','gyro':'body_gyro_x/y/z'},
        'feature_recipe':'9 row-local statistics and 8 positive-frequency log1p-power bands per axis, plus 3 axis correlations',
        'excluded_preprocessing':'561-column globally normalised feature matrix is NOT used for predictions',
        'source_members_sha256':members,'official_archive_sha256':verification['sha256'],
        'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
        'privacy_unit':'one labelled provided filtered window replacement; no raw stream, subject, participation or add/remove DP claim',
        'upstream_boundary':'supplied filtered and overlapping windows are benchmark inputs; original stream filtering is outside mechanism',
        'split_note':'all 30 subjects repartitioned; public/source/target train/test subjects disjoint; not official leaderboard split'}
    meta_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return views,y,subjects,manifest
