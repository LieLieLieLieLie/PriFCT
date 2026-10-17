"""Bounded HTTP source-only retry with exact pinned Git blob verification."""
import hashlib,json,subprocess
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent/'author_baselines_v7'
COMMIT='e9aec76845f8a127ba0a7a25614641878199cae7'


def main():
    cache=ROOT/'git_sources/DP-HP';out=ROOT/'DP-HP'/COMMIT
    entries=subprocess.run(['git','ls-tree','-r','HEAD'],cwd=cache,capture_output=True,check=True,timeout=10).stdout.decode().splitlines()
    session=requests.Session();session.trust_env=False;files=[];failed=[]
    for line in entries:
        metadata,name=line.split('\t',1);_,kind,blob=metadata.split()
        if kind!='blob' or name.split('/')[0]=='data':continue
        if not (name.endswith(('.py','.md')) or name=='LICENSE'):continue
        path=out/name;old=ROOT.parent/'author_baselines_v6/DP-HP'/name
        def gitsha(value):return hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
        try:
            if path.exists():data=path.read_bytes()
            elif blob=='e69de29bb2d1d6434b8b29ae775ad8c2e48c5391':data=b''
            elif old.exists() and gitsha(old.read_bytes())==blob:data=old.read_bytes()
            else:
                response=session.get('https://raw.githubusercontent.com/ParkLabML/DP-HP/'+COMMIT+'/'+name,timeout=(8,15))
                response.raise_for_status();data=response.content
            if gitsha(data)!=blob:raise RuntimeError('Pinned blob mismatch')
            if len(data)>1000000:raise RuntimeError('Unexpected large source')
            if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            files.append(dict(path=name,git_blob_sha1=blob,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            print(name,len(data),flush=True)
        except Exception as error:failed.append(dict(path=name,error=repr(error)));print(name,repr(error),flush=True)
    result=dict(repository='https://github.com/ParkLabML/DP-HP',commit=COMMIT,files=files,failed=failed,
        status='Complete selected author sources' if not failed else 'Verified partial sources; missing files explicitly recorded')
    (out/'author_source_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':main()
