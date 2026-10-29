"""Audit complete Chapter 2 budgets, final-weight reloads and classical convergence."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from neuralpde_ch02.models import CoordinateMLP
from neuralpde_ch02.problems import Poisson
from neuralpde_ch02.metrics import evaluate
from neuralpde_ch02.training import environment

ROOT=Path(__file__).resolve().parent
out=ROOT/'results/ch02/full'
config=json.loads((out/'suite_config.json').read_text())
if json.loads((out/'environment.json').read_text())['source_sha256']!=environment()['source_sha256']:
    raise ValueError('Source identity changed')
expected={f'poisson{d}d_{m}_seed{s}' for d in [1,2] for m in config['methods'] for s in config['seeds']}
folders={p.parent.name:p.parent for p in out.glob('poisson*d_*_seed*/metrics.json')}
if set(folders)!=expected or len(expected)!=18: raise ValueError('Missing formal neural runs')
checks=[]
torch.set_num_threads(2)
for name,folder in sorted(folders.items()):
    m=json.loads((folder/'metrics.json').read_text())
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    if ck['progress']['adam_done']!=config['adam_steps'] or ck['progress']['lbfgs_done']!=config['lbfgs_steps']:
        raise ValueError('Training budget incomplete: '+name)
    pc=config['problems'][str(m['dim'])]
    model=CoordinateMLP(Poisson(m['dim']),pc['width'],pc['depth']).double()
    weights=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(weights)
    for key,value in weights.items():
        if not torch.equal(value,ck['model'][key]): raise ValueError('Final checkpoint mismatch')
    metric=evaluate(model,Poisson(m['dim']),pc['validation_n'],config['validation_order'],strong=False)
    for key in ['l2_rel','h1_semi_rel','boundary_max','energy_identity_defect']:
        np.testing.assert_allclose(metric[key],m[key],rtol=1e-7,atol=1e-12)
    if m['boundary_max']>1e-12 or m['energy_identity_defect']>1e-7:
        raise ValueError('Physical identity failed')
    if max(m['quadrature_l2_relative_change'],m['quadrature_h1_relative_change'])>1e-3:
        raise ValueError('Unresolved quadrature')
    checks.append(dict(name=name,reload_passed=True,final_iterate=True))
fem=[]
for dim in [1,2]:
    rows=sorted([json.loads(p.read_text()) for p in (out/'fem').glob(f'poisson{dim}d_n*.json')],key=lambda r:r['n'])
    if [r['n'] for r in rows]!=config['fem_levels'][str(dim)]:raise ValueError('FEM grids missing')
    a,b=rows[-2:];ratio=b['n']/a['n']
    orders={k:math.log(a[k]/b[k])/math.log(ratio) for k in ['l2_abs','h1_semi_abs']}
    if orders['l2_abs']<1.8 or orders['h1_semi_abs']<.9:raise ValueError('FEM convergence failure')
    fem.append(dict(dim=dim,grids=len(rows),orders=orders))
record=dict(passed=True,neural_runs=18,fem_runs=8,checks=checks,fem=fem,
            files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(out.rglob('*')) if p.is_file() and p.name!='DELIVERY_AUDIT.json'})
(out/'DELIVERY_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
print('CH02_AUDIT_PASSED',len(checks),flush=True)
