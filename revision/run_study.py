"""Run all replacement experiments. Results are append-only JSONL per stage.

Usage: python -m revision.run_study --stage real|synthetic|audit|all
Run from FedKAPT/code. No test labels are passed into protocol.align/screen.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, hashlib, json, time, zipfile, itertools, sys
from pathlib import Path
import numpy as np
import scipy, sklearn, ot
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from core.preprocessing import fixed_record_map
from revision.protocol import release, Message, clip, costs, transport, align, assignment, screen, classify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "paper" / "KBS复投" / "experiments"
SEEDS = [42,123,456,789,2026,3141,2718,8080,9001,10007]
EPS = [.1,.5,1.,2.,5.,10.,float("inf")]
FAMILIES = ["fou","fac","kar","pix","zer","mor"]
PRIMARY_PAIRS = [("fou","fac"),("fac","kar"),("kar","zer"),
                 ("fac","fou"),("kar","fac"),("zer","kar")]


def safe_json(o):
    if isinstance(o, dict): return {str(k):safe_json(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [safe_json(x) for x in o]
    if isinstance(o, np.ndarray): return safe_json(o.tolist())
    if isinstance(o, np.generic): return safe_json(o.item())
    if isinstance(o, float) and not np.isfinite(o): return "inf" if o>0 else None
    return o


def write_row(stage, row):
    with (OUT / (stage+".jsonl")).open("a",encoding="utf-8") as f:
        f.write(json.dumps(safe_json(row),ensure_ascii=False,allow_nan=False)+"\n")


def done(stage):
    path=OUT/(stage+".jsonl")
    return {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def read_mf():
    folder=ROOT/"data"/"multiple_features"
    with zipfile.ZipFile(folder/"multiple_features.zip") as z:
        z.extractall(folder)
    data={name:np.loadtxt(folder/("mfeat-"+name)) for name in FAMILIES}
    for name,x in data.items():
        assert x.shape[0]==2000, (name,x.shape)
    return data, np.repeat(np.arange(10),200)


def split_views(xs,xt,y,seed):
    src,tgt=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=seed)
    tr,te=train_test_split(tgt,test_size=.4 if len(y)==569 else .3,
                          stratify=y[tgt],random_state=seed+11)
    assert not (set(src)&set(tr) or set(src)&set(te) or set(tr)&set(te))
    log=len(y)==569
    return fixed_record_map(xs[src],log),y[src],fixed_record_map(xt[tr],log),y[tr],\
           fixed_record_map(xt[te],log),y[te],{"source_indices":src,"train_indices":tr,"test_indices":te}


def cwru_views(src,tgt,seed,noise=0.):
    def read(load):
        z=np.load(ROOT/"data"/"cwru"/".cache"/f"load{load}_stat64_v2.npz")
        x,y=z["features"][:,32:64],z["labels"]
        tr,te=[],[]
        for c in range(4):
            ix=np.flatnonzero(y==c)
            # Chronological 30% holdout; one intervening window removes 50% overlap.
            cut=int(.7*len(ix)); tr.extend(ix[:cut-1]);te.extend(ix[cut:])
        return fixed_record_map(x,True),y,np.array(tr),np.array(te)
    sx,sy,si,_=read(src);tx,ty,ti,te=read(tgt)
    if noise:
        tx=clip(tx+np.random.default_rng(seed+55).normal(size=tx.shape)*noise)
    return sx[si],sy[si],tx[ti],ty[ti],tx[te],ty[te],\
           {"source_indices":si,"train_indices":ti,"test_indices":te,"temporal_gap_windows":1,
            "source_load":src,"target_load":tgt,"added_target_noise":noise}


def truth_mapping(clusters,y,k,ct=None):
    ct=ct or k
    cm=np.array([np.bincount(y[clusters==c],minlength=k) for c in range(ct)])
    if ct==k:
        rr,cc=linear_sum_assignment(-cm)
        return cc[np.argsort(rr)]
    return cm.argmax(1)


def evaluate_case(name,data,seed,eps,rho,baselines=False,sensitivity=False):
    sx,sy,xtr,ytr,xte,yte,split=data
    k=len(np.unique(sy))
    km=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=seed)
    clusters=km.fit_predict(xtr)
    centers=km.cluster_centers_
    test_clusters=km.predict(xte)
    # Same private draw across all comparable methods and paired eps values.
    msg=release(sx,sy,k,eps,seed+80000)
    s,mapping,gap,warnings=align(centers,msg,rho,declared_coordinates=rho>0)
    ref=truth_mapping(clusters,ytr,k)
    correct=bool(np.array_equal(mapping,ref))
    scr=screen(centers,msg,rho,s,mapping,gap,seed+90000,
               multipliers=(.5,1.,2.) if sensitivity else (1.,),
               counts=(25,50,100) if sensitivity else (100,),declared_coordinates=rho>0)
    for row in scr: row["correct_assignment"]=correct
    pseudo=mapping[clusters]
    condition=s@msg.prototypes
    main=classify(np.c_[xtr,condition[clusters]],np.c_[xte,condition[test_clusters]],pseudo,yte)
    result={"dataset":name,"seed":seed,"epsilon":eps,"rho":rho,"k":k,
            "source_dim":sx.shape[1],"target_dim":xtr.shape[1],"counts":msg.counts,
            "source_n":len(sx),"target_train_n":len(xtr),"split":split,"sigma":msg.sigma,
            "mapping":mapping,"reference_mapping":ref,"correct_assignment":correct,
            "fraction_correct_clusters":np.mean(mapping==ref),"pseudo_label_accuracy":np.mean(pseudo==ytr),
            "ARI":adjusted_rand_score(ytr,clusters),"NMI":normalized_mutual_info_score(ytr,clusters),
            "oracle_cluster_test_accuracy":np.mean(ref[test_clusters]==yte),
            "screen":scr,"results":{"FedKAPT-core":main},"solver_warnings":warnings,
            "payload_bytes":int(msg.prototypes.astype("<f8").nbytes+msg.counts.astype("<i8").nbytes)}
    # Clean-plan comparison is evaluator-only, never supplied to screen or training.
    clean=release(sx,sy,k,float("inf"),seed+80000)
    sc,mc,_,_=align(centers,clean,rho,declared_coordinates=rho>0)
    result["private_clean_plan_error"]=float(np.max(abs(s-sc)))
    result["private_clean_same_assignment"]=bool(np.array_equal(mapping,mc))
    if baselines:
        result["results"]["Rel-Assignment"]=classify(xtr,xte,pseudo,yte)
        result["results"]["OneHot-condition"]=classify(np.c_[xtr,np.eye(k)[pseudo]],
               np.c_[xte,np.eye(k)[mapping[test_clusters]]],pseudo,yte)
        rnd=np.random.default_rng(seed+5050).permutation(k)
        result["results"]["RandomNaming-target-only"]=classify(xtr,xte,rnd[clusters],yte)
        for mode,label in [("full","FullOT-matched"),("uot","UOT-matched")]:
            sb,mb,_,nw=align(centers,msg,rho,mode=mode,declared_coordinates=rho>0)
            result["results"][label]=classify(xtr,xte,mb[clusters],yte)
            result["results"][label]["mapping"]=mb
            result["results"][label]["solver_warnings"]=nw
        a,b=np.ones(k)/k,np.ones(k)/k
        gw=ot.gromov.gromov_wasserstein(cdist(centers,centers)/2.,
              cdist(msg.prototypes,msg.prototypes)/2.,a,b,"square_loss",max_iter=1000)
        mg,_=assignment(gw/gw.sum(1,keepdims=True))
        result["results"]["GW-message"]=classify(xtr,xte,mg[clusters],yte)
        result["results"]["GW-message"]["mapping"]=mg
        if rho>0:
            md,_=assignment(-cdist(centers,msg.prototypes)**2)
            result["results"]["Direct-message"]=classify(xtr,xte,md[clusters],yte)
            # CORAL analogue uses ONLY covariance of the released prototypes.
            def power(c,p):
                w,v=np.linalg.eigh(c+1e-3*np.eye(len(c))); return (v*(np.maximum(w,1e-8)**p))@v.T
            wt=power(np.cov(xtr,rowvar=False),-.5)@power(np.cov(msg.prototypes,rowvar=False),.5)
            ctr=xtr.mean(0);cs=msg.prototypes.mean(0)
            result["results"]["CORAL-prototype"]=classify((xtr-ctr)@wt+cs,(xte-ctr)@wt+cs,pseudo,yte)
        # True one-step entropy self-training, initialised by the same private mapping.
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,multi_class="multinomial"))
        model.fit(xtr,pseudo);p=model.predict_proba(xtr);conf=p.max(1)
        keep=conf>=np.quantile(conf,.6)
        labels=model.classes_[p.argmax(1)]
        result["results"]["Entropy-ST-message"]=classify(xtr[keep],xte,labels[keep],yte)
    # Separate saved message proves all target baselines consumed the same object.
    ident=f"{name}_s{seed}_e{eps}_r{rho}".replace("/","-")
    np.savez_compressed(OUT/"messages"/(ident+".npz"),prototypes=msg.prototypes,
                         counts=msg.counts,sigma=msg.sigma,centers=centers,plan=s,clusters=clusters)
    result["message_file"]="messages/"+ident+".npz"
    return result


def real_study(controls_only=False):
    mf,my=({},None) if controls_only else read_mf()
    w=np.loadtxt(ROOT/"data"/"wdbc"/"wdbc.data",delimiter=",",dtype=str)
    wx=w[:,2:].astype(float);wy=(w[:,1]=="M").astype(int)
    completed=done("real");t0=time.time()
    cases=[("WDBC",None)]+([] if controls_only else [("MF-"+a+"-"+b,(a,b)) for a,b in itertools.permutations(FAMILIES,2)])
    cases += [(f"CWRU-{a}-{b}",(a,b)) for a,b in itertools.permutations(range(4),2)]
    for name,pair in cases:
        primary=name=="WDBC" or name=="CWRU-0-2" or (name.startswith("MF") and pair in PRIMARY_PAIRS)
        for seed in SEEDS:
            if name=="WDBC": data=split_views(wx[:,:10],wx[:,20:30],wy,seed)
            elif name.startswith("MF"): data=split_views(mf[pair[0]],mf[pair[1]],my,seed)
            else: data=cwru_views(*pair,seed)
            for eps in EPS if primary else [2.]:
                rho=.5 if not name.startswith("MF") else 0.
                ident=f"{name}|{seed}|{eps}|{rho}"
                if ident in completed: continue
                row=evaluate_case(name,data,seed,eps,rho,baselines=eps==2.,sensitivity=primary and eps in [.5,2.,10.])
                row["id"]=ident;write_row("real",row)
            print(f"real {name} seed {seed}; elapsed {time.time()-t0:.1f}s",flush=True)


def ablation_study():
    mf,my=read_mf();completed=done("ablations")
    for seed in SEEDS:
        for name,data,rhos in [("CWRU-0-2",cwru_views(0,2,seed),[0.,.25,.5,.75,1.]),
                               ("MF-fac-kar",split_views(mf["fac"],mf["kar"],my,seed),[0.])]:
            sx,sy,xtr,ytr,xte,yte,split=data;k=len(np.unique(sy))
            msg=release(sx,sy,k,2.,seed+80000)
            for ct in [k-2,k,k+2] if k==10 else [k]:
                km=KMeans(n_clusters=ct,n_init=20,random_state=seed).fit(xtr)
                clusters=km.labels_;centers=km.cluster_centers_;testc=km.predict(xte)
                for rho,mass,length in itertools.product(rhos,[.5,.7,.9,1.],[8,32,64]):
                    ident=f"{name}|{seed}|{ct}|{rho}|{mass}|{length}"
                    if ident in completed:continue
                    s,m,g,nw=align(centers,msg,rho,mass=mass,length=length,declared_coordinates=rho>0)
                    scores=classify(xtr,xte,m[clusters],yte)
                    write_row("ablations",{"id":ident,"dataset":name,"seed":seed,"clusters":ct,
                       "rho":rho,"mass":mass,"length":length,"gap":g,"status":"not-screened" if ct!=k else "not-evaluated",
                       "mapping":m,"metrics":scores,"mean_row_confidence":float(s.max(1).mean()),"solver_warnings":nw})
        # Explicit noise stress; default control has no injected noise.
        for noise in [.05,.15,.3]:
            ident=f"CWRU-noise|{seed}|{noise}"
            if ident in completed:continue
            row=evaluate_case("CWRU-noise",cwru_views(0,2,seed,noise),seed,2.,.5)
            row["id"]=ident;row["noise"]=noise;write_row("ablations",row)
        print(f"ablation seed {seed}",flush=True)


def synthetic_study():
    completed=done("synthetic")
    for geometry in ["asymmetric","near-symmetric","symmetric","semantic-swap"]:
        for replicate in range(30):
            rng=np.random.default_rng(210000+replicate)
            k=10
            angle=2*np.pi*np.arange(k)/k
            ring=np.c_[np.cos(angle),np.sin(angle)]*.65
            if geometry=="asymmetric":
                source=clip(rng.normal(size=(k,5))) *.7
            elif geometry=="near-symmetric":source=np.c_[ring,np.arange(k)*.003]
            else:source=np.c_[ring,np.zeros(k)]
            p=rng.permutation(k);target=source[p].copy()
            if geometry=="semantic-swap":
                source=clip(rng.normal(size=(k,5)))*.7;target=source[p].copy()
                # Same observable geometry, different semantic truth: no screen can detect this.
                truth=p.copy();truth[[0,1]]=truth[[1,0]]
            else:truth=p.copy()
            counts=np.full(k,100);sigma_base=2./counts
            for eps in EPS:
                ident=f"{geometry}|{replicate}|{eps}"
                if ident in completed:continue
                from core.privacy import compute_sigma
                sigma=np.asarray(compute_sigma(sigma_base,eps,1e-5))[:,None]
                msg=Message(clip(source+np.random.default_rng(310000+replicate).normal(size=source.shape)*sigma),counts,sigma,eps)
                s,m,g,nw=align(target,msg)
                scr=screen(target,msg,0.,s,m,g,seed=410000+replicate,
                      multipliers=(.5,1.,2.),counts=(25,50,100))
                write_row("synthetic",{"id":ident,"geometry":geometry,"replicate":replicate,
                         "epsilon":eps,"gap":g,"correct_assignment":bool(np.array_equal(m,truth)),
                         "fraction_correct":float(np.mean(m==truth)),"mapping":m,"truth":truth,"screen":scr})
        print("synthetic "+geometry,flush=True)


def stress_study():
    completed=done('stress')
    w=np.loadtxt(ROOT/'data'/'wdbc'/'wdbc.data',delimiter=',',dtype=str)
    wx=w[:,2:].astype(float);wy=(w[:,1]=='M').astype(int)
    for name in ['CWRU-rel-only','WDBC-rel-only']:
        for seed in SEEDS:
            data=cwru_views(0,2,seed) if name.startswith('CWRU') else split_views(wx[:,:10],wx[:,20:30],wy,seed)
            for eps in EPS:
                ident=f'{name}|{seed}|{eps}'
                if ident in completed:continue
                row=evaluate_case(name,data,seed,eps,0.)
                row['id']=ident;write_row('stress',row)
        print('stress '+name,flush=True)


def audit():
    # Legacy suspicious coincidences are examined, not silently called defects.
    records=[]
    for p in (ROOT/"results"/"models").glob("**/*runs*.json"):
        if "checkpoint" in str(p):continue
        try:
            data=json.loads(p.read_text(encoding="utf-8"))
            records.append({"file":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                            "root_type":type(data).__name__,"size":len(data)})
        except Exception as e:records.append({"file":str(p),"error":str(e)})
    manifest={"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
             "sklearn":sklearn.__version__,"pot":ot.__version__,"seeds":SEEDS,
             "epsilons":safe_json(EPS),"C":1,"delta":1e-5,"mass":.9,"tau":.01,
             "legacy_result_files":records,"files":{}}
    for p in list((ROOT/"revision").glob("*.py"))+list((ROOT/"data"/"multiple_features").glob("mfeat-*")):
        manifest["files"][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["real","controls","ablations","synthetic","stress","audit","all"],default="all")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);(OUT/"messages").mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage in ["real","all"]:real_study()
        if args.stage=="controls":real_study(controls_only=True)
        if args.stage in ["ablations","all"]:ablation_study()
        if args.stage in ["synthetic","all"]:synthetic_study()
        if args.stage in ["stress","all"]:stress_study()
        if args.stage in ["audit","all"]:audit()
