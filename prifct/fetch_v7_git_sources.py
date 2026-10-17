"""Acquire pinned small author sources without API limits or model downloads.

Partial clone stores tree metadata; only selected source blobs are requested.
Each exported byte stream is checked against its Git blob id and SHA256.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent/'author_baselines_v7'
REPOS=['ParkLabML/DP-KIP','ParkLabML/DP-HP','XJTU-XGU/KPG-RL','pinardemetci/AGW-AISTATS24']


def git(arguments,cwd=None):
    env=dict(os.environ)
    for name in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(name,None)
    process=subprocess.run(['git','-c','http.proxy=','-c','https.proxy=']+arguments,
        cwd=cwd,env=env,capture_output=True,timeout=90)
    if process.returncode:raise RuntimeError(process.stderr.decode(errors='replace'))
    return process.stdout


def fetch(repo):
    name=repo.split('/')[-1]
    cache=ROOT/'git_sources'/name
    cache.parent.mkdir(exist_ok=True,parents=True)
    if not cache.exists():git(['clone','--filter=blob:none','--no-checkout','--depth','1','https://github.com/'+repo+'.git',str(cache)])
    commit=git(['rev-parse','HEAD'],cache).decode().strip()
    # Asking ls-tree for sizes would lazily download unrelated data/model blobs.
    entries=git(['ls-tree','-r','HEAD'],cache).decode().splitlines()
    selected=[]
    for line in entries:
        metadata,relative=line.split('\t',1)
        mode,kind,blob=metadata.split()
        if kind!='blob':continue
        if relative.split('/')[0].lower() in {'data','datasets','results','models','checkpoints'}:continue
        if not (relative.endswith(('.py','.md','.txt','.toml','.cfg','.yml','.yaml')) or
                Path(relative).name.lower().startswith(('license','copying'))):continue
        selected.append((relative,blob))
    exported=ROOT/name/commit;exported.mkdir(exist_ok=True,parents=True)
    files=[]
    for relative,expected in selected:
        path=exported/relative
        previous=ROOT.parent/'author_baselines_v6'/name/relative
        if path.exists():content=path.read_bytes()
        elif expected=='e69de29bb2d1d6434b8b29ae775ad8c2e48c5391':content=b''
        elif previous.exists():
            candidate=previous.read_bytes()
            candidate_sha=hashlib.sha1(b'blob '+str(len(candidate)).encode()+b'\0'+candidate).hexdigest()
            content=candidate if candidate_sha==expected else git(['cat-file','blob',expected],cache)
        else:content=git(['cat-file','blob',expected],cache)
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if actual!=expected:raise RuntimeError('Author content mismatch '+relative)
        if len(content)+sum(x['bytes'] for x in files)>8*2**20:raise RuntimeError('8MiB source bound')
        if not path.exists():path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(content)
        files.append(dict(path=relative,git_blob_sha1=actual,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content)))
    result=dict(repository='https://github.com/'+repo,commit=commit,files=files,
        status='Source-only snapshot; fidelity audit and new utility pending')
    (exported/'author_source_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(repo=repo,commit=commit,files=len(files),bytes=sum(x['bytes'] for x in files))),flush=True)
    return result


if __name__=='__main__':
    def safe_fetch(repo):
        try:return fetch(repo)
        except Exception as error:
            result=dict(repository=repo,status='Incomplete; retry or inspect required',error=repr(error))
            print(json.dumps(result),flush=True)
            return result
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(safe_fetch,REPOS))
    (ROOT/'git_acquisition_summary.json').write_text(json.dumps(results,indent=2),encoding='utf8')
