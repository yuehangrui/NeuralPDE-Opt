"""Create and verify self-contained workspaces; never borrow an old project path."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload():
    return Path(__file__).resolve().parent / '_payload'


def verify(root):
    root = Path(root).resolve()
    marker = root / 'WORKSPACE.json'
    if marker.exists():
        identity = json.loads(marker.read_text(encoding='utf-8'))
        if identity['payload_sha256'] != sha(root / 'PAYLOAD_MANIFEST.json'):
            raise ValueError('Workspace manifest identity mismatch')
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        unresolved = root / name
        target = unresolved.resolve()
        if not target.is_relative_to(root) or not target.is_file() or unresolved.is_symlink():
            raise ValueError('Unsafe or missing source: ' + name)
        if sha(target) != digest:
            raise ValueError('Source identity mismatch: ' + name)
    return manifest


def initialize(path):
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a new workspace; existing directories are never overwritten')
    verify(payload())
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(payload(), destination)
    manifest = verify(destination)
    (destination / 'WORKSPACE.json').write_text(json.dumps(dict(
        version='1.0.0', payload_sha256=sha(destination/'PAYLOAD_MANIFEST.json'),
        source_files=len(manifest['files']), external_project_paths=[],
        training_started=False), indent=2)+'\n',encoding='utf-8')
    return destination


def run(path, arguments):
    root = Path(path).expanduser().resolve()
    verify(root)
    if not (root / 'WORKSPACE.json').is_file():
        raise ValueError('Not an initialized independent workspace')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root),str(root/'src')]),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', MPLBACKEND='Agg',
               PYTHONUTF8='1')
    lock = root / '.book-running.lock'
    exclusive = arguments[0] in ['train','finish','verify','pipeline']
    if exclusive:
        with lock.open('x', encoding='utf-8') as handle:
            json.dump(dict(pid=os.getpid(), arguments=arguments),handle)
    try:
        return subprocess.call([sys.executable, '-B', '-u', str(root/'book_driver.py'), *arguments],
                               cwd=root, env=env)
    finally:
        if exclusive:
            lock.unlink()
