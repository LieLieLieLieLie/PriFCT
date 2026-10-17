"""Acquire a pinned author repository without executing its code."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
REPO = 'RomainMsrd/UniDABench'


def main():
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'FedKAPT-research-provenance/1'
    reply = session.get(f'https://api.github.com/repos/{REPO}/commits/main', timeout=60)
    reply.raise_for_status()
    commit = reply.json()['sha']
    url = f'https://codeload.github.com/{REPO}/zip/{commit}'
    reply = session.get(url, timeout=120)
    reply.raise_for_status()
    archive = reply.content
    if len(archive) > 40_000_000:
        raise RuntimeError('Repository archive unexpectedly large')
    destination = HERE/'upstream'/commit
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for item in zipped.infolist():
            if item.is_dir():
                continue
            relative = Path(*Path(item.filename).parts[1:])
            if not relative.parts or relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe repository member')
            target = (destination/relative).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError('Repository extraction escaped destination')
            if item.file_size > 30_000_000:
                raise RuntimeError('Repository member unexpectedly large')
            value = zipped.read(item)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != value:
                raise RuntimeError('Refusing to replace differing pinned source')
            target.write_bytes(value)
            files[relative.as_posix()] = {'bytes':len(value), 'sha256':hashlib.sha256(value).hexdigest()}
    provenance = {'acquired_utc':datetime.now(timezone.utc).isoformat(),
        'repository':f'https://github.com/{REPO}', 'commit':commit,
        'commit_response':reply.status_code, 'download_url':url,
        'archive_sha256':hashlib.sha256(archive).hexdigest(), 'archive_bytes':len(archive),
        'source_root':str(destination), 'files':files,
        'execution':'No source files imported or executed during acquisition'}
    (HERE/'upstream_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    print(json.dumps({'commit':commit,'archive_bytes':len(archive),'files':len(files),'source_root':str(destination)}))


if __name__ == '__main__':
    main()
