"""Read-only import probe from outside the legacy project."""
import hashlib,json,os,platform,sys
from pathlib import Path
ROOT=Path('/home/yhr/Desktop/NeuralPDE-Opt-independent-v1')
OLD=Path('/home/yhr/Desktop/NeuralPDE-Opt')
os.chdir('/tmp')
sys.path=[str(ROOT/'src')]+[p for p in sys.path if not Path(p or '.').resolve().is_relative_to(OLD)]
import neuralpde_book,numpy,scipy,torch,matplotlib
from neuralpde_opt.chapter3.upstream_operators import DeepXDEOperator,GalerkinOperator
from neuralpde_ch02.models import CoordinateMLP
modules={name:str(Path(m.__file__).resolve()) for name,m in list(sys.modules.items())
         if getattr(m,'__file__',None) and not str(m.__file__).startswith('<')}
bad={n:p for n,p in modules.items() if Path(p).is_relative_to(OLD)}
if bad:raise RuntimeError('Import depends on old project: '+str(bad))
value=dict(passed=True,working_directory='/tmp',old_project_modules=bad,
           python=platform.python_version(),
           versions={name:m.__version__ for name,m in [('numpy',numpy),('scipy',scipy),('torch',torch),('matplotlib',matplotlib)]},
           package_path=neuralpde_book.__file__,
           numerical_import_paths={n:p for n,p in modules.items() if n.startswith(('neuralpde_opt','neuralpde_ch02'))},
           scientific_packages_shared=True,
           payload_sha256=hashlib.sha256((ROOT/'PAYLOAD_MANIFEST.json').read_bytes()).hexdigest())
(ROOT/'verification/DEPENDENCIES.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items() if k!='numerical_import_paths'},indent=2))
