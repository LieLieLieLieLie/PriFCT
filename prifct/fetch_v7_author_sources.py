"""Small pinned author sources only: no datasets, checkpoints or packages."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import requests

OUT=Path(__file__).resolve().parent/'author_baselines_v7'
REPOSITORIES=['Justinyangjy/DP-NTK','XJTU-XGU/KPG-RL','pinardemetci/AGW-AISTATS24','ParkLabML/DP-HP']


def fetch(repo):
    session=requests.Session();session.trust_env=False
    info=session.get('https://api.github.com/repos/'+repo,timeout=15)
    info.raise_for_status();branch=info.json()['default_branch']
    commit=session.get('https://api.github.com/repos/'+repo+'/commits/'+branch,timeout=15)
    commit.raise_for_status();commit=commit.json()['sha']
    tree=session.get('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1',timeout=15)
    tree.raise_for_status();tree=tree.json()
    if tree.get('truncated'):raise RuntimeError('Author tree truncated')
    folder=OUT/repo.split('/')[-1]/commit
    folder.mkdir(parents=True,exist_ok=True)
    items=[v for v in tree['tree'] if v['type']=='blob' and
           (v['path'].endswith(('.py','.md','.txt','.toml','.cfg','.yml','.yaml','.ipynb')) or
            Path(v['path']).name.lower().startswith(('license','copying')))]
    if sum(v.get('size',0) for v in items)>8*2**20:raise RuntimeError('Source8MiB guard')
    def one(item):
        path=folder/item['path']
        if path.exists():content=path.read_bytes()
        else:
            response=session.get('https://api.github.com/repos/'+repo+'/git/blobs/'+item['sha'],timeout=15)
            response.raise_for_status();blob_data=response.json()
            if blob_data.get('encoding')!='base64':raise RuntimeError('Unknown author blob encoding')
            content=base64.b64decode(blob_data['content'])
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if blob!=item['sha']:raise RuntimeError('Author blob mismatch '+item['path'])
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(content)
        return dict(path=item['path'],git_blob_sha1=blob,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
    with ThreadPoolExecutor(3) as pool:files=list(pool.map(one,items))
    manifest=dict(repository='https://github.com/'+repo,commit=commit,tree_sha1=tree['sha'],files=files,
                  status='Author source acquisition only; algorithm equivalence and utility pending')
    (folder/'author_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return dict(repo=repo,commit=commit,files=len(files),bytes=sum(v['bytes'] for v in files))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo');args=parser.parse_args()
    repos=[args.repo] if args.repo else REPOSITORIES
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(2) as pool: result=list(pool.map(fetch,repos))
    (OUT/'acquisition_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)
