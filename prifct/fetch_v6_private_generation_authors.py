"""Fetch and Git-blob-verify author implementations, without running utility."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent/'author_baselines_v6'
REPOS=['ParkLabML/DP-MERF','ParkLabML/DP-HP']


def fetch(repo):
    session=requests.Session();session.trust_env=False
    response=session.get('https://api.github.com/repos/'+repo+'/git/trees/master?recursive=1',timeout=20)
    response.raise_for_status();tree=response.json()
    if tree.get('truncated'): raise RuntimeError('Incomplete author tree')
    folder=ROOT/repo.split('/')[-1]
    folder.mkdir(parents=True,exist_ok=True)
    items=[v for v in tree['tree'] if v['type']=='blob' and
        (v['path'].endswith(('.py','.md','.txt','.yml','.yaml')) or
         Path(v['path']).name.lower().startswith(('license','copying')))]
    def one(item):
        r=session.get('https://raw.githubusercontent.com/'+repo+'/master/'+item['path'],timeout=20)
        r.raise_for_status();content=r.content
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=item['sha']: raise RuntimeError('Author tree changed: '+item['path'])
        path=folder/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=content: raise RuntimeError('Refuse author overwrite')
        path.write_bytes(content)
        return {'path':item['path'],'git_blob_sha1':blob,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    with ThreadPoolExecutor(4) as pool: files=list(pool.map(one,items))
    manifest={'repo':'https://github.com/'+repo,'git_tree_sha1':tree['sha'],
        'files':files,'scope':'Author source snapshot only; no private arrays, utility, or packages modified'}
    path=folder/'source_manifest.json'
    path.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return {'repo':repo,'files':len(files),'bytes':sum(v['bytes'] for v in files),'tree':tree['sha']}


if __name__=='__main__':
    # Independent repositories; nested downloads are bounded to 8 workers.
    with ThreadPoolExecutor(2) as pool:
        print(json.dumps(list(pool.map(fetch,REPOS)),indent=2))
