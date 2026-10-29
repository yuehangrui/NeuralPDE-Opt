#!/usr/bin/env python3
"""Fixed-budget multi-seed experiment; freeze all jobs before opening test arrays."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
# Initialize NumPy's BLAS before PyTorch's OpenMP runtime in the server env.
import numpy as np
import torch
from neuralpde_opt.chapter3.operator_training import (
    artifact_hashes, atomic_write, data_fingerprints, dump_json, exclusive_run_lock,
    finalize, run, sha256, source_fingerprints, verify_completed,
)
from evaluate_ch03_operator_transfer import evaluate_transfer
from report_ch03_experiment import build_report


def preserve_incomplete(directory):
    if directory.exists() and not (directory/'COMPLETE').is_file():
        saved = directory.with_name(directory.name+f'.incomplete-{time.time_ns()}')
        directory.rename(saved)
        print(f'Preserved incomplete diagnostic at {saved}', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--finalize-test', action='store_true',
                        help='Explicitly enable reused-test evaluation for adapted epoch experiments')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.train_only and args.finalize_test:
        parser.error('--train-only and --finalize-test are mutually exclusive')
    seeds, base = config['seeds'], config['training']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int for s in seeds):
        raise ValueError('Need unique integer seeds')
    if not base.get('defer_test'):
        raise ValueError('All experiment training must defer test evaluation')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(base.get('cpu_threads', 2))
    scripts = ('run_ch03_experiment.py','report_ch03_experiment.py',
               'evaluate_ch03_operator_transfer.py','audit_ch03_operator_outputs.py')
    identity = {'config': config, 'data_sha256': data_fingerprints(args.data),
                'source_sha256': source_fingerprints(base['models']),
                'scripts_sha256': {name: sha256(Path(__file__).parent/name) for name in scripts},
                'device': args.device}
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with exclusive_run_lock(args.output):
        protocol = args.output/'experiment_identity.json'
        if args.resume:
            if json.loads(protocol.read_text()) != identity:
                raise ValueError('Experiment identity mismatch')
            if (args.output/'COMPLETE').is_file():
                verify_completed(args.output)
                print('Verified completed experiment; no retraining or retesting.', flush=True)
                return
        else:
            dump_json(protocol, identity)
            dump_json(args.output/'experiment_config.json', config)
        configs = args.output/'seed_configs'
        configs.mkdir(exist_ok=True)
        index = {}
        for seed in seeds:
            path = configs/f'seed_{seed}.json'
            seed_config = {**base, 'seed': seed}
            if path.exists() and json.loads(path.read_text()) != seed_config:
                raise ValueError('Seed config mismatch')
            if not path.exists():
                dump_json(path, seed_config)
            folder = args.output/f'seed_{seed}'
            print(f'TRAIN seed={seed}; test deferred', flush=True)
            run(args.data, path, folder, device=args.device, resume=folder.exists())
            index[str(seed)] = {'directory': folder.name,
                                'state': 'COMPLETE' if (folder/'COMPLETE').exists() else 'FROZEN'}
            dump_json(args.output/'run_index.json', index)
        frozen = {}
        for seed in seeds:
            directory = args.output/f'seed_{seed}'
            if not ((directory/'FROZEN').is_file() or (directory/'COMPLETE').is_file()):
                raise ValueError('Every seed must finish training before any test evaluation')
            for spec in base['models']:
                name = spec['name']; model = directory/name
                metrics = json.loads((model/'metrics.json').read_text())
                frozen[f'{seed}/{name}'] = {
                    'best_sha256': sha256(model/'best.pt'),
                    'config_sha256': sha256(model/'config.json'),
                    'normalization_sha256': sha256(model/'normalization.json'),
                    'selected_step': metrics['selected_step'],
                    'training_relative_l2': metrics['train']['mean']['relative_l2'],
                    'validation_relative_l2': metrics['val']['mean']['relative_l2']}
        ledger = args.output/'frozen_selection.json'
        if ledger.exists() and json.loads(ledger.read_text()) != frozen:
            raise ValueError('Frozen model ledger mismatch')
        if not ledger.exists():
            dump_json(ledger, frozen)
        atomic_write(args.output/'ALL_MODELS_FROZEN', 'All seed/model choices committed before test arrays are opened.\n')
        if args.train_only or (base.get('epoch_training') is not None and not args.finalize_test):
            print('All models frozen; no test arrays opened.', flush=True)
            return
        for seed in seeds:
            directory = args.output/f'seed_{seed}'
            print(f'FINALIZE seed={seed}; frozen checkpoint test evaluation', flush=True)
            finalize(args.data, directory, device=args.device)
            index[str(seed)]['state'] = 'COMPLETE'
            dump_json(args.output/'run_index.json', index)
        for seed in seeds:
            resolution = args.output/'resolution'/f'seed_{seed}'
            preserve_incomplete(resolution)
            if resolution.exists():
                verify_completed(resolution)
            else:
                evaluate_transfer(args.data, args.output/f'seed_{seed}', resolution, args.device)
            predictions = args.output/'inference'/f'seed_{seed}'
            preserve_incomplete(predictions)
            if predictions.exists():
                verify_completed(predictions)
            else:
                with (args.output/f'inference_seed_{seed}.log').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).parent/'audit_ch03_operator_outputs.py'),
                                    '--data', str(args.data), '--runs', str(args.output/f'seed_{seed}'),
                                    '--transfer', str(resolution), '--output', str(predictions),
                                    '--device', args.device], check=True, stdout=log, stderr=subprocess.STDOUT)
        build_report(args.data, args.output, device=args.device)
        dump_json(args.output/'manifest.json', artifact_hashes(args.output))
        atomic_write(args.output/'COMPLETE', 'Fixed-budget Chapter 3 first comparative experiment and diagnostics completed.\n')
        print((args.output/'summary.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
