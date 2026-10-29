"""Prepare train/val assets and run declared Chapter 3 cases in this workspace."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'chapters/ch03'))
import bootstrap
from integrity import read,dump,sha,check_manifest
from registry import runs
from predictors import array_file


def prepare():
    from neuralpde_opt.chapter3.reaction_data import generate,verify_dataset
    from neuralpde_opt.chapter3.operator_training import P1Geometry
    from neuralpde_opt.chapter3.reaction_operators import fit_reaction_normalization
    from pod import fit_pod
    from data import generate as generate_cf,load as load_cf
    from physics import P1System
    from prepare_v2 import prepare as prepare_v2,reload_prepared
    target=ROOT/'data/ch03/reaction_v1'
    if not target.exists(): generate(read(ROOT/'configs/ch03_reaction_v1.json'),target)
    data=verify_dataset(target)
    geometry=P1Geometry(target/'observation_geometry.npz')
    pod=ROOT/'results/ch03/pod_v1'
    if not pod.exists():
        pod.mkdir(parents=True)
        basis=fit_pod(data['train'],geometry,32)
        np.savez_compressed(pod/'basis.npz',**basis)
        dump(pod/'normalization.json',fit_reaction_normalization(data['train'],geometry))
        dump(pod/'prepared_manifest.json',{n:sha(pod/n) for n in ['basis.npz','normalization.json']})
    else: check_manifest(pod,read(pod/'prepared_manifest.json'))
    cfg=read(ROOT/'experiments/ch03_varmion_v1/config.json')
    target=ROOT/'data/ch03/varmion_v1'
    if not target.exists(): generate_cf(cfg,target)
    data=load_cf(target)
    geometry=P1Geometry(target/'observation_geometry.npz')
    prepared=ROOT/'results/ch03/varmion_v1'
    if not prepared.exists():
        prepared.mkdir(parents=True)
        engine=bootstrap.module('prepare_varmion','experiments/ch03_varmion_v1/run.py')
        engine.prepare(prepared,data,geometry,cfg)
        dump(prepared/'config.json',cfg)
        names=['basis.npz','normalization.json','train_coordinates.npz','val_coordinates.npz','representation.json','config.json']
        dump(prepared/'prepared_manifest.json',{n:sha(prepared/n) for n in names})
    else: check_manifest(prepared,read(prepared/'prepared_manifest.json'))
    v2=ROOT/'results/ch03/varmion_v2'
    if not v2.exists():
        v2.mkdir(parents=True)
        prepare_v2(v2,read(ROOT/'experiments/ch03_varmion_v2/config.json'))
    else: reload_prepared(v2)
    hints=ROOT/'data/ch03/hints_v1'
    residual=bootstrap.module('book_residual_data','experiments/ch03_hints_v1/residual_data.py')
    if not hints.exists():
        residual.generate(hints,data,array_file(prepared/'basis.npz')['basis'],P1System(64,.01),
                          read(ROOT/'experiments/ch03_hints_v1/config.json'))
    else: residual.load(hints)
    (ROOT/'verification').mkdir(exist_ok=True)
    dump(ROOT/'verification/PREPARATION.json',dict(passed=True,created_from_generators=True,
         reused_external_checkpoints=False,test_generated=False,
         manifests={str(p.relative_to(ROOT)):sha(p) for p in [
             ROOT/'data/ch03/reaction_v1/manifest.json',target/'manifest.json',
             hints/'manifest.json',pod/'prepared_manifest.json',prepared/'prepared_manifest.json',v2/'prepared_manifest.json']}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','case','audit','finish','verify'])
    p.add_argument('--case')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-epochs',type=int)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.action=='prepare': return prepare()
    if a.action=='case':
        from training import run
        row=next(r for r in runs() if r['id']==a.case)
        return run(a.case,(ROOT/row['path']).parent,a.resume,a.smoke_epochs)
    if a.action=='audit':
        from audit import run
        target=ROOT/'results/ch03/final_v1_support'
        target.mkdir(parents=True,exist_ok=True)
        return run(target/'training_audit.json')
    final=ROOT/'results/ch03/final_test_v1'
    if a.action=='finish':
        from heldout import register,generate
        from evaluation import run
        if not final.exists(): register(final)
        generate(final)
        return run(final)
    from heldout import protocol_at
    from integrity import verify
    protocol_at(final)
    verify(final)


if __name__=='__main__': main()
