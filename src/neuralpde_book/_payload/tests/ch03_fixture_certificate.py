"""Synthetic integrity metadata for unit fixtures; not a PDE quality audit."""
import hashlib
import json
from pathlib import Path

def certify_fixture(data, counts):
    data = Path(data)
    def write(name, value):
        (data/name).write_text(json.dumps(value))
    def digest(name):
        return hashlib.sha256((data/name).read_bytes()).hexdigest()
    workflow = 'ch03-resumable-fields-v1'
    config = {'purpose':'synthetic_unit_fixture', 'splits':counts}
    config_hash = hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False,
        allow_nan=False,separators=(',', ':')).encode()).hexdigest()
    write('config.json',config)
    write('identity.json',{'workflow':workflow,'config_sha256':config_hash})
    write('quality.json',{'passed':True,'scope':'synthetic fixture, no PDE accuracy claim'})
    write('reload_audit.json',{'passed':True,'failures':[],
          'independent_weak_residual_samples':sum(counts.values()),
          'split_statistics':{s:{'count':n} for s,n in counts.items()}})
    names = [p.name for p in data.glob('*.npz')]+['config.json','identity.json','quality.json']
    write('manifest.json',{name:digest(name) for name in names})
    write('COMPLETE',{'workflow':workflow,'config_sha256':config_hash,
          'manifest_sha256':digest('manifest.json'),'reload_audit_sha256':digest('reload_audit.json')})
