"""Fixed MHEALTH development windows, separate from any external evaluation.

Time windows are formed before labels. The benchmark curator excludes
majority-null windows once and saves fixed identities; this selection is not
claimed to be private. Features, sampling and unit conversion are predeclared.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/mhealth/raw/MHEALTHDATASET'
OUT=ROOT/'results/models/eswa_fedkapt/original_v6_research/mhealth_features'
SENSORS={'ankle_accel':[5,6,7],'ankle_gyro':[8,9,10],
         'wrist_accel':[14,15,16],'wrist_gyro':[17,18,19]}
ROLES={'public':[1,6],'source':[2,7,3],'adaptation':[9,8],'validation':[4,5,10]}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def window_features(blocks,hz=50.,device='auto'):
    """Twenty-four fixed features; accepts signals only, no labels/subjects."""
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    outputs=[];n=blocks.shape[1]
    frequency=torch.fft.rfftfreq(n,1/hz,device=device)
    factor=torch.full((len(frequency),),2.,dtype=torch.float64,device=device)
    factor[0]=1.
    if n%2==0:factor[-1]=1.
    low=(frequency>=.5)&(frequency<2.);high=(frequency>=2.)&(frequency<=5.)
    for start in range(0,len(blocks),2048):
        x=torch.as_tensor(blocks[start:start+2048],device=device,dtype=torch.float64)
        x=torch.cat([x,torch.linalg.vector_norm(x,dim=2,keepdim=True)],dim=2)
        mean=x.mean(1);centered=x-mean[:,None,:]
        power=torch.abs(torch.fft.rfft(centered,dim=1))**2/n**2*factor[None,:,None]
        feature=torch.stack([mean,torch.sqrt((centered**2).mean(1)),
            torch.sqrt((x**2).mean(1)),x.max(1).values-x.min(1).values,
            power[:,low,:].sum(1),power[:,high,:].sum(1)],dim=2).reshape(len(x),24)
        outputs.append(feature.cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0,24))


def build_cache():
    OUT.mkdir(parents=True,exist_ok=True)
    specification={'window_seconds':2,'samples':100,'sampling_hz':50,
        'roles':ROLES,'sensors_zero_based':SENSORS,'gyro_conversion':'degrees/s times pi/180',
        'features':['mean','population_std','RMS','peak_to_peak','PSD .5<=f<2','PSD 2<=f<=5'],
        'channels':['x','y','z','magnitude'],'dimension':24,
        'labels':'time windows first; majority original label; lowest-number tie; curator excludes majority-null0 once',
        'quality':'all23sensorvalues finite throughout complete100-row window; incomplete final window dropped',
        'class_universe':list(range(1,13)),
        'privacy':'fixed selected window identities conditional full labelled-window replacement; selection and subjects not private'}
    path=OUT/'feature_protocol.json'
    if path.exists():assert json.loads(path.read_text())==specification
    else:path.write_text(json.dumps(specification,indent=2),encoding='utf-8')
    audits={}
    for subject in range(1,11):
        rawpath=RAW/f'mHealth_subject{subject}.log';cache=OUT/f'subject{subject}.npz'
        rawhash=sha(rawpath)
        if cache.exists():
            saved=np.load(cache)
            assert str(saved['raw_sha256'])==rawhash
            audits[str(subject)]=json.loads(str(saved['audit_json']))
            continue
        raw=np.loadtxt(rawpath,dtype=np.float64)
        assert raw.ndim==2 and raw.shape[1]==24
        count=len(raw)//100;blocks=raw[:count*100].reshape(count,100,24)
        quality=np.isfinite(blocks[:,:,:23]).all((1,2))
        annotations=blocks[:,:,23]
        valid_labels=np.isfinite(annotations).all(1)&np.isin(annotations,np.arange(13)).all(1)
        # Annotation validity is an offline benchmark scoring requirement,
        # not a sensor pairing or classifier confidence filter.
        votes=np.stack([(annotations==c).sum(1) for c in range(13)],axis=1)
        majority=votes.argmax(1);mixed=votes.max(1)<100
        ids=np.flatnonzero(quality&valid_labels&(majority!=0))
        saved={}
        for name,columns in SENSORS.items():
            signal=blocks[ids][:,:,columns]
            if name.endswith('gyro'):signal=signal*(np.pi/180)
            saved[name]=window_features(signal)
        report={'raw_rows':len(raw),'full_time_windows':count,'incomplete_tail_rows':len(raw)-count*100,
            'sensor_quality_rejections':int((~quality).sum()),'invalid_annotation_windows':int((~valid_labels).sum()),
            'curator_null_windows':int((quality&valid_labels&(majority==0)).sum()),
            'retained_windows':len(ids),'mixed_retained_windows':int(mixed[ids].sum()),
            'class_support':np.bincount(majority[ids],minlength=13)[1:].tolist(),
            'sampling_or_pairing_uses_label':False,'curator_idle_exclusion_uses_label':True}
        np.savez_compressed(cache,**saved,label=majority[ids]-1,subject=np.full(len(ids),subject),
            window_id=ids,start_sample=ids*100,raw_sha256=np.array(rawhash),
            audit_json=np.array(json.dumps(report)))
        audits[str(subject)]=report
        print('MHEALTH subject',subject,len(ids),'windows cached',flush=True)
    (OUT/'feature_cache_audit.json').write_text(json.dumps({'protocol_sha256':sha(path),'subjects':audits},indent=2),encoding='utf-8')
    return audits


def cohort(role,view):
    """Return immutable development/public records; no external data here."""
    if role not in ROLES or view not in SENSORS:raise ValueError('Known fixed role/view required')
    features=[];labels=[];subjects=[];identities=[]
    for subject in ROLES[role]:
        with np.load(OUT/f'subject{subject}.npz') as saved:
            features.append(saved[view]);labels.append(saved['label']);subjects.append(saved['subject'])
            identities.extend([f'MHEALTH/{subject}/sample{start}' for start in saved['start_sample']])
    return np.concatenate(features),np.concatenate(labels),np.concatenate(subjects),np.array(identities)


if __name__=='__main__':build_cache()
