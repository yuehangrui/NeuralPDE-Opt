"""Verify this extracted download without scientific dependencies or retraining."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
bad=[]
for name,info in manifest['files'].items():
    p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        bad.append((name,'missing/unsafe'));continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    if p.stat().st_size!=info['bytes'] or h.hexdigest()!=info['sha256']:bad.append((name,'changed'))
print(json.dumps({'passed':not bad,'checked_files':len(manifest['files']),'failures':bad},indent=2))
sys.exit(bool(bad))
