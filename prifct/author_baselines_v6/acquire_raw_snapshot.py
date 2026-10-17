"""Per-file author snapshot if GitHub API/codeload is unreachable.

The branch is mutable: provenance explicitly reports an unpinned snapshot,
and SHA256 verifies the exact locally audited bytes, not a repository commit.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE=Path(__file__).resolve().parent
FILES=['README.md','requirements.txt','algorithms/algorithms.py',
 'models/models.py','models/loss.py','models/augmentations.py',
 'configs/hparams.py','configs/data_model_configs.py','dataloader/dataloader.py',
 'main.py','main_sweep.py','trainers/trainer.py','trainers/trainer_uniDA.py',
 'utils.py','trainer.py','algorithms/uniOT_utils.py']

def fetch(name):
    session=requests.Session();session.trust_env=False
    url='https://raw.githubusercontent.com/RomainMsrd/UniDABench/main/'+name
    reply=session.get(url,timeout=25)
    if reply.status_code==404:return name,{'status':404}
    reply.raise_for_status()
    value=reply.content
    target=HERE/'raw_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=value:raise RuntimeError('Refuse to replace a changed audited snapshot')
    target.write_bytes(value)
    return name,{'url':url,'status':reply.status_code,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}

def main():
    results={}
    with ThreadPoolExecutor(max_workers=6) as workers:
        for name,result in workers.map(fetch,FILES):
            results[name]=result;print(name,result.get('status'),result.get('bytes'),flush=True)
    provenance={'acquired_utc':datetime.now(timezone.utc).isoformat(),
      'repository':'https://github.com/RomainMsrd/UniDABench','requested_ref':'main',
      'commit':None,'warning':'Per-file mutable branch snapshot, not a pinned commit; do not claim complete repo checkout',
      'files':results,'executed_remote_code':False}
    (HERE/'raw_snapshot_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')

if __name__=='__main__':main()
