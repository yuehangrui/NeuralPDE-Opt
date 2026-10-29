"""Run the retained mathematical regression suites without historical project paths."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'verification'
    output.mkdir(exist_ok=True)
    suites=[('ch02',['discover','-s','chapters/ch02/tests','-v'],ROOT),
            ('ch03_core',['discover','-s','tests','-p','test*.py','-v'],ROOT/'tests')]
    suites += [(name,['discover','-s','experiments/'+name,'-p','test*.py','-v'],ROOT/'experiments'/name)
               for name in ['ch03_pod_v1','ch03_physics_v1','ch03_varmion_v1','ch03_varmion_v2','ch03_hints_v1']]
    suites += [('ch03_entry',['discover','-s','chapters/ch03','-p','test*.py','-v'],ROOT/'chapters/ch03'),
               ('ch04',[f'chapters.{name}.{test}' for name,test in [
                   ('ch04','test_ch04'),('ch04_adaptation','test_adaptation'),
                   ('ch04_operator_learning','test_operators'),('ch04_operator_adaptation','test_adaptation'),
                   ('ch04_completion','test_completion'),('ch04_delivery','test_delivery'),('ch04_depth','test_depth')]]+['-v'],ROOT),
               ('ch05',['chapters.ch05.test_ch05','chapters.ch05_delivery.test_delivery','-v'],ROOT)]
    reports=[]
    for name,arguments,import_dir in suites:
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[import_dir,ROOT,ROOT/'src',
                 ROOT/'experiments/ch03_varmion_v1',ROOT/'experiments/ch03_physics_v1'])))
        with (output/(name+'.log')).open('w') as log:
            code=subprocess.call([sys.executable,'-B','-m','unittest',*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        text=(output/(name+'.log')).read_text()
        n=re.search(r'Ran (\d+) tests? in',text)
        reports.append(dict(suite=name,exit_code=code,tests=int(n[1]) if n else 0))
        print(json.dumps(reports[-1]),flush=True)
    with (output/'optimized_ch03.log').open('w') as log:
        code=subprocess.call([sys.executable,'-O','-B',str(ROOT/'chapters/ch03/test_release.py')],
                             cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(passed=not code and all(r['exit_code']==0 and r['tests'] for r in reports),
                payload_sha256=hashlib.sha256((ROOT/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest(),
                suites=reports,total_tests=sum(r['tests'] for r in reports),optimized_mode_passed=not code)
    (output/'CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__': main()
