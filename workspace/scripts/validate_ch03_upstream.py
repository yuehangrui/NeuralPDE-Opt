"""Bounded, non-resuming train/val-only validation for pinned upstream ports.

Never selects or loads test arrays. Every output directory is create-once.
Preserves exact original P1 objective/metrics and train-only normalization.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, random, argparse, time, hashlib
from pathlib import Path
import numpy as np
import torch
from neuralpde_opt.chapter3 import operator_training as t
from neuralpde_opt.chapter3.upstream_operators import make_model
ROOT=Path(__file__).resolve().parents[1]


def source_identity():
    paths=[ROOT/'src/neuralpde_opt/chapter3/operator_training.py',
           ROOT/'src/neuralpde_opt/chapter3/upstream_operators.py',Path(__file__).resolve()]
    paths+=sorted((ROOT/'src/neuralpde_opt/chapter3/upstream_vendor').glob('*.py'))
    paths+=sorted(p for p in (ROOT/'third_party/ch03_upstream').rglob('*') if p.is_file())
    return {str(p.relative_to(ROOT)):t.sha256(p) for p in paths}


def guard_npz(data):
    """Runtime deny-by-default NPZ guard, including resolved symlinks."""
    original=np.load; opened=[]
    allowed={(data/name).resolve() for name in ('train.npz','val.npz','observation_geometry.npz')}
    def safe(path,*args,**kwargs):
        resolved=Path(path).resolve()
        if resolved not in allowed: raise RuntimeError('NPZ access denied: '+str(resolved))
        opened.append(str(resolved))
        return original(path,*args,**kwargs)
    np.load=safe
    return original,opened


def load_model(output,device='cpu'):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    for name,sha in manifest.items():
        if t.sha256(output/name)!=sha:raise ValueError('Artifact checksum mismatch: '+name)
    cfg=json.loads((output/'config.json').read_text())
    norm=json.loads((output/'normalization.json').read_text())
    identity=json.loads((output/'identity.json').read_text())
    if identity['sources']!=source_identity(): raise ValueError('Upstream source identity mismatch')
    state=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
    if state['config']!=cfg or state['normalization']!=norm or state['identity']!=identity:
        raise ValueError('Checkpoint identity mismatch')
    model=t.PhysicalOperator(make_model(cfg['model']),norm).to(device)
    model.load_state_dict(state['model_state'])
    return model.eval()


def run(cfg,data,output,device):
    if cfg['schema']!='ch03-upstream-trainval-v1' or cfg['seed']!=0:
        raise ValueError('Only predeclared seed-0 adaptation plans are supported')
    if cfg['epochs'] not in (100,200) or cfg['train_count'] not in (8,128):raise ValueError('Budget outside bounded plan')
    if cfg['batch_size'] not in (8,16) or cfg['microbatch']!=4:raise ValueError('Invalid batch plan')
    if cfg['selection_split']!=('train' if cfg['train_count']==8 else 'val'):raise ValueError('Invalid checkpoint selection')
    if cfg['objective']!='relative_l2_squared':raise ValueError('Objective changed')
    if not 0<cfg['lr']<=.001:raise ValueError('Invalid learning rate')
    data,output=Path(data).resolve(),Path(output).resolve()
    if data!=ROOT/'data/ch03/experiment_v1':raise ValueError('Only certified experiment_v1 is allowed')
    output.mkdir(parents=True,exist_ok=False)
    original,opened=guard_npz(data)
    try:
        certificate=t.verify_data_manifest(data)
        train_all=t.read_split(data,'train');val=t.read_split(data,'val')
        if len(train_all['y'])!=128 or len(val['y'])!=32:raise ValueError('Unexpected split counts')
        if set(train_all['sample_id']) & set(val['sample_id']):raise ValueError('Overlapping split IDs')
        geometry=t.P1Geometry(data/'observation_geometry.npz')
        norm=t.fit_normalization(train_all,geometry)
        train={k:v[:cfg['train_count']] for k,v in train_all.items()}
        identity={'sources':source_identity(),'data':t.data_fingerprints(data),'certificate':certificate}
        for name,value in [('config',cfg),('normalization',norm),('identity',identity)]:t.dump_json(output/(name+'.json'),value)
        torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rng=np.random.default_rng(cfg['seed'])
        model=t.PhysicalOperator(make_model(cfg['model']),norm).to(device)
        t.dump_json(output/'representation.json',model.representation_metadata())
        arrays={k:torch.as_tensor(train[k],device=device,dtype=torch.float32) for k in ('a','f','y')}
        loss_norm=geometry.torch_loss(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs']) if cfg['schedule']=='cosine' else None
        if cfg['schedule'] not in ('cosine','constant'):raise ValueError('Invalid scheduler')
        history=[];actual_steps=[0]
        hook=opt.register_step_post_hook(lambda *args:actual_steps.__setitem__(0,actual_steps[0]+1))
        best_value=float('inf');best_epoch=None;best_metrics=None;start=time.monotonic()
        t.dump_json(output/'environment.json',{'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
             'device':str(device),'gpu':torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
             'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
             'real_parameters':sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())})
        for epoch in range(1,cfg['epochs']+1):
            model.train();order=rng.permutation(len(train['y']));epoch_loss=0.
            lr=opt.param_groups[0]['lr']
            for offset in range(0,len(order),cfg['batch_size']):
                idx=order[offset:offset+cfg['batch_size']]
                part=t.batch_update(model,arrays,idx,cfg['microbatch'],opt,loss_norm,norm['y_rms'],cfg['objective'],grad_clip=1.)
                epoch_loss+=float(part)*len(idx)/len(order)
            if sched is not None:sched.step()
            metrics={}
            for name,split in [('train',train),('val',val)]:
                prediction=t.predict(model,split,device,4)
                metrics[name]=t.evaluate_arrays(prediction,split['y'],geometry,split['sample_id'])
            score=metrics[cfg['selection_split']]['mean']['relative_l2']
            record={'epoch':epoch,'optimizer_steps':actual_steps[0],'sample_visits':epoch*len(order),
                    'lr':lr,'objective':epoch_loss,'train':metrics['train']['mean'],'val':metrics['val']['mean']}
            history.append(record)
            if score < best_value:
                best_value,best_epoch,best_metrics=score,epoch,metrics
                t.atomic_write(output/'best.pt',{'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'epoch':epoch,'optimizer_steps':actual_steps[0],'config':cfg,'normalization':norm,'identity':identity})
            if epoch%10==0 or epoch==1:
                t.dump_json(output/'history.json',history)
                print(json.dumps(record),flush=True)
        hook.remove()
        expected_steps=cfg['epochs']*((len(train['y'])+cfg['batch_size']-1)//cfg['batch_size'])
        assert actual_steps[0]==expected_steps
        optimizer_counts=sorted({int(v['step']) for v in opt.state.values()})
        assert optimizer_counts==[actual_steps[0]],optimizer_counts
        t.atomic_write(output/'last.pt',{'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer_state':t._cpu_copy(opt.state_dict()),'scheduler_state':sched.state_dict() if sched else None,
            'epoch':cfg['epochs'],'optimizer_steps':actual_steps[0],'config':cfg,'normalization':norm,'identity':identity,
            'rng':t.capture_rng(rng),'resume_supported':False})
        last_metrics=metrics
        payload=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(payload['model_state']);model.eval()
        reload_differences={}
        for name,split in [('train',train),('val',val)]:
            pred=t.predict(model,split,device,4)
            actual=t.evaluate_arrays(pred,split['y'],geometry,split['sample_id'])
            diff=max(abs(actual['mean'][k]-best_metrics[name]['mean'][k]) for k in actual['mean'])
            assert diff<1e-10
            reload_differences[name]=diff
            np.savez_compressed(output/(name+'_predictions.npz'),prediction=pred,sample_id=split['sample_id'])
        if source_identity()!=identity['sources'] or t.data_fingerprints(data)!=identity['data']:
            raise RuntimeError('Source/data changed during training')
        result={'status':'complete','selected_epoch':best_epoch,'selection_split':cfg['selection_split'],
            'selection_metric':'mean per-sample relative P1 L2; earliest evaluated minimum',
            'best':best_metrics,'last':last_metrics,'optimizer_steps':actual_steps[0],
            'optimizer_state_steps':optimizer_counts,'sample_visits':cfg['epochs']*len(train['y']),
            'elapsed_seconds':time.monotonic()-start,'test_evaluations':0,'test_arrays_opened':False,
            'opened_npz_files':sorted(set(opened)),'checkpoint_reload_metric_difference':reload_differences}
        t.dump_json(output/'history.json',history);t.dump_json(output/'metrics.json',result)
        files=sorted(p for p in output.iterdir() if p.is_file() and p.name!='manifest.json')
        t.dump_json(output/'manifest.json',{p.name:t.sha256(p) for p in files})
        print('COMPLETE',output.name,best_epoch,best_metrics['train']['mean']['relative_l2'],best_metrics['val']['mean']['relative_l2'],flush=True)
        return result
    finally:np.load=original

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding='utf-8-sig')),ROOT/'data/ch03/experiment_v1',args.output,args.device)
