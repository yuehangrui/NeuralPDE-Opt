"""Whole-book lifecycle; explicit training, unchanged budgets, independent outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CHAPTERS=['ch02','ch03','ch04','ch04-depth','ch05']
RESULTS={'ch02':'results/ch02/full','ch03':'results/ch03/final_test_v1',
         'ch04':'results/ch04/unified_delivery_v1','ch04-depth':'results/ch04/nonlinear_depth_v1',
         'ch05':'results/ch05/unified_delivery_v1'}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def execute(arguments,name):
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    target=logs/(name+'.log')
    if target.exists(): target=logs/(name+'-'+str(time.time_ns())+'.log')
    write(ROOT/'CURRENT_TASK.json',dict(task=name,started=time.time(),log=str(target.relative_to(ROOT))))
    print('START',name,flush=True)
    with target.open('w') as out:
        code=subprocess.call([sys.executable,'-B','-u',*map(str,arguments)],cwd=ROOT,
                             stdout=out,stderr=subprocess.STDOUT)
    if code:
        write(ROOT/'CURRENT_TASK.json',dict(task=name,state='failed',exit_code=code,log=str(target.relative_to(ROOT))))
        raise RuntimeError(f'{name} failed; retained outputs. See {target}')
    write(ROOT/'CURRENT_TASK.json',dict(task=name,state='passed',completed=time.time(),log=str(target.relative_to(ROOT))))
    print('DONE',name,flush=True)


def require_gates():
    for f in ['CHECKS.json','PREPARATION.json','SMOKE.json']:
        report=read(ROOT/'verification'/f)
        if not report['passed']: raise ValueError('Required gate failed: '+f)
        if f in ['CHECKS.json','SMOKE.json'] and report.get('payload_sha256')!=hashlib.sha256((ROOT/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest():
            raise ValueError('Gate was run against different source: '+f)
    free=shutil.disk_usage(ROOT).free
    if free<4*1024**3: raise RuntimeError('Less than 4 GiB available; training not launched')


def inventory():
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(ROOT/'chapters/ch03'),str(ROOT),str(ROOT/'src')])}
    code='import json; from registry import runs; print(json.dumps(runs()))'
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=ROOT,env=env,text=True))


def train_chapter(chapter,resume=False):
    require_gates()
    marker=ROOT/'verification'/f'{chapter}_TRAINING.json'
    if marker.exists():
        if not read(marker)['passed']: raise ValueError('Incomplete prior stage')
        print('RETAIN_COMPLETED',chapter,flush=True);return
    if chapter=='ch02':
        execute(['-m','neuralpde_ch02.cli','--config','chapters/ch02/configs/full.json',
                 '--output',RESULTS[chapter],'--device','cuda'],'ch02_train')
    elif chapter=='ch03':
        rows=inventory()
        # Train source maps before initializing the residual models from this run.
        rows.sort(key=lambda r:({'varmion':0,'pod':1,'varmion_v2':2,'physics':3,'baseline':4,'hints':5}[r['family']],r['id']))
        for row in rows:
            parent=(ROOT/row['path']).parent
            arguments=['book_ch03.py','case','--case',row['id']]
            if parent.exists():
                if not resume: raise FileExistsError(f'Existing case {parent}; use explicit --resume')
                arguments+=['--resume']
            execute(arguments,'ch03_'+row['id'])
        execute(['book_ch03.py','audit'],'ch03_training_audit')
    else:
        module={'ch04':'chapters.ch04_delivery.cli','ch04-depth':'chapters.ch04_depth.cli','ch05':'chapters.ch05_delivery.cli'}[chapter]
        execute(['-m',module,'train','--root',RESULTS[chapter]],chapter+'_train')
    write(marker,dict(passed=True,completed=time.time(),formal_budget=True,reused_old_weights=False))


def finish_chapter(chapter):
    if not read(ROOT/'verification'/f'{chapter}_TRAINING.json')['passed']: raise ValueError('Training not complete')
    marker=ROOT/'verification'/f'{chapter}_DELIVERY.json'
    if marker.exists(): return
    if chapter=='ch02':
        execute(['book_ch02_audit.py'],'ch02_audit')
    elif chapter=='ch03':
        execute(['book_ch03.py','finish'],'ch03_final_test')
    else:
        module={'ch04':'chapters.ch04_delivery.cli','ch04-depth':'chapters.ch04_depth.cli','ch05':'chapters.ch05_delivery.cli'}[chapter]
        arguments=['-m',module,'finish','--root',RESULTS[chapter]]
        if chapter in ['ch04','ch05']: arguments+=['--resume-assessment']
        execute(arguments,chapter+'_finish')
    write(marker,dict(passed=True,completed=time.time(),result=RESULTS[chapter]))


def smoke():
    # A separate generated workspace prevents pilots from entering formal statistics.
    if not read(ROOT/'verification/PREPARATION.json')['passed']: raise ValueError('Prepare data first')
    dest=ROOT/'development/smoke'
    if dest.exists(): raise FileExistsError('Smoke workspace exists; inspect it before rerunning')
    manifest=read(ROOT/'PAYLOAD_MANIFEST.json')
    dest.mkdir(parents=True)
    for name in list(manifest['files'])+['PAYLOAD_MANIFEST.json','WORKSPACE.json']:
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    shutil.copytree(ROOT/'data',dest/'data')
    shutil.copytree(ROOT/'results/ch03',dest/'results/ch03')
    env={**os.environ,'PYTHONPATH':os.pathsep.join([str(dest),str(dest/'src')])}
    rows=[r for r in inventory() if r['seed']==0]
    rows.sort(key=lambda r:({'varmion':0,'pod':1,'varmion_v2':2,'physics':3,'baseline':4,'hints':5}[r['family']],r['id']))
    reports=[]
    for row in rows:
        path=dest/'logs'/f"{row['id']}.log";path.parent.mkdir(exist_ok=True)
        with path.open('w') as log:
            code=subprocess.call([sys.executable,'-B','-u',str(dest/'book_ch03.py'),'case',
                                  '--case',row['id'],'--smoke-epochs','2'],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT)
        reports.append(dict(id=row['id'],exit_code=code))
        print('SMOKE',reports[-1],flush=True)
        if code: break
    passed=len(reports)==len(rows) and all(r['exit_code']==0 for r in reports)
    write(ROOT/'verification/SMOKE.json',dict(passed=passed,chapter3_cases=reports,epochs=2,
          payload_sha256=hashlib.sha256((ROOT/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest(),
          other_chapters='Short training/reload/derivative tests included in retained regression suites',
          used_for_selection=False,formal_initialization='fresh; no pilot checkpoint reused'))
    if not passed: raise RuntimeError('Smoke failed; formal training blocked')


def status():
    result={'training':{},'delivery':{},'free_gib':round(shutil.disk_usage(ROOT).free/1024**3,2)}
    for ch in CHAPTERS:
        for key,suffix in [('training','TRAINING'),('delivery','DELIVERY')]:
            p=ROOT/'verification'/f'{ch}_{suffix}.json'
            result[key][ch]=read(p) if p.exists() else {'passed':False}
    if (ROOT/'CURRENT_TASK.json').exists(): result['current']=read(ROOT/'CURRENT_TASK.json')
    result['ch03_complete_cases']=sum((ROOT/r['path']/'COMPLETE').exists() for r in inventory())
    print(json.dumps(result,ensure_ascii=False,indent=2))


def verify_chapter(chapter):
    if chapter=='ch02': execute(['book_ch02_audit.py'],'ch02_verify')
    elif chapter=='ch03': execute(['book_ch03.py','verify'],'ch03_verify')
    else:
        module={'ch04':'chapters.ch04_delivery.cli','ch04-depth':'chapters.ch04_depth.cli','ch05':'chapters.ch05_delivery.cli'}[chapter]
        execute(['-m',module,'verify','--root',RESULTS[chapter]],chapter+'_verify')
    write(ROOT/'verification'/f'{chapter}_VERIFIED.json',dict(passed=True,completed=time.time()))


def pipeline(resume=False):
    require_gates()
    path=ROOT/'verification/PIPELINE.json'
    state=dict(state='running',started=time.time(),planned_training_configurations=156,
               payload_sha256=hashlib.sha256((ROOT/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest())
    write(path,state)
    try:
        for chapter in CHAPTERS: train_chapter(chapter,resume)
        for chapter in CHAPTERS: finish_chapter(chapter)
        for chapter in CHAPTERS: verify_chapter(chapter)
        execute(['book_report.py'],'final_book_report')
    except BaseException:
        state.update(state='failed',stopped=time.time());write(path,state)
        raise
    state.update(state='passed',completed=time.time());write(path,state)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['check','prepare','smoke','train','finish','verify','status','pipeline'])
    p.add_argument('--chapter',choices=CHAPTERS+['all'],default='all')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.action=='check': return execute(['book_checks.py'],'regression')
    if a.action=='prepare': return execute(['book_ch03.py','prepare'],'prepare_train_val')
    if a.action=='smoke': return smoke()
    if a.action=='status': return status()
    if a.action=='pipeline': return pipeline(a.resume)
    chapters=CHAPTERS if a.chapter=='all' else [a.chapter]
    if a.action=='train':
        for chapter in chapters: train_chapter(chapter,a.resume)
    elif a.action=='finish':
        for chapter in chapters: finish_chapter(chapter)
    else:
        for chapter in chapters: verify_chapter(chapter)


if __name__=='__main__': main()
