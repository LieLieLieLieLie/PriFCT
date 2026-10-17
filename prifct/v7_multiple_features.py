"""Six real disjoint descriptor families; fixed label-blind record roles.

This benchmark was previously inspected elsewhere in the project. It supplies
additional DEVELOPMENT evidence, not a newly untouched confirmation dataset.
All30 directed families are retained, including the six-dimensional family.
"""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/multiple_features'
OUT=ROOT/'results/models/eswa_fedkapt/original_v7_research/multiple_features_development'
FAMILIES={'fou':76,'fac':216,'kar':64,'pix':240,'zer':47,'mor':6}


def load():
    views={};hashes={}
    for name,dimension in FAMILIES.items():
        path=DATA/('mfeat-'+name);array=np.loadtxt(path)
        if array.shape!=(2000,dimension) or not np.isfinite(array).all():raise ValueError('Invalid original descriptor family '+name)
        views[name]=array;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    # UCI documents ten contiguous digit classes, 200 examples per class.
    labels=np.repeat(np.arange(10),200)
    return views,labels,hashes


def roles():
    # Global record permutation: not selected or balanced by private labels.
    order=np.random.default_rng(77001).permutation(2000)
    return dict(public=order[:400],source=order[400:1200],adaptation=order[1200:1400],
        validation=order[1400:1700],reserved=order[1700:])


def register():
    OUT.mkdir(exist_ok=True)
    views,_,hashes=load();partition=roles()
    sets=[set(ids) for ids in partition.values()]
    assert sum(map(len,sets))==len(set.union(*sets))==2000
    protocol=dict(scope='Known benchmark additional DEVELOPMENT, not pristine external confirmation',
        dataset='UCI Multiple Features',doi='10.24432/C5HC70',raw_source_sha256=hashes,
        families=FAMILIES,directions=[(s,t) for s in FAMILIES for t in FAMILIES if s!=t],
        role_indices={name:ids.tolist() for name,ids in partition.items()},
        role_rule='Fixed global77001 permutation independent of labels, not per-class private selection',
        feature_disjoint='Actual independently defined descriptor families and different dimensions, no column shuffling or artificial label alignment',
        public_resource='Same-record pairs ONLY among independent400 public samples; public labels never fit maps',
        privacy='Fixed800 private labelled-record cohort, entire feature/label replacement; counts not declared public; no record pairs from private cohort are disclosed',
        labels='UCI documented digit row order used only at source query and development scoring; no map/target training labels',
        reserved='Remaining300 records excluded from this development; no new independent confirmation claim because benchmark has earlier project exposure',
        public_fit='336 fitting and64 held-out public pairs by independent fixed77002 permutation, no labels',
        source_point='Public logscale/PCA capped32, 95% public norm scale, source row unitclip',
        target_function='Public target128 landmarks, bandwidth factors.25/1 and ridge.01/.1/1 selected on held-out projected function MSE',
        source_function='Existing linear+two64RFF blocks, public uncentred orthogonal projection32 (source point dimension may be6)',
        output='One private Gaussian joint feature/count query and exact semi-relaxed entropy transport; pilot runner must separately freeze grid and code before scores')
    path=OUT/'data_protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==json.loads(json.dumps(protocol))
    else:path.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    print(json.dumps(dict(records=2000,roles={name:len(ids) for name,ids in partition.items()},directions=30,
        shapes={name:list(array.shape) for name,array in views.items()},status='No new classifier utility opened')),flush=True)
    return views,partition,protocol


if __name__=='__main__':register()
