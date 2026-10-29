#!/usr/bin/env python3
"""Verify all selected operators through the raw-field prediction entry point.

No parameter updates. Checks immutable manifests, reconstructs unlabelled test
inputs and compares reloaded predictions with the committed selected outputs.
Optional finer-grid output checks use an already completed transfer diagnostic.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from neuralpde_opt.chapter3.operator_inference import run_inference
from neuralpde_opt.chapter3.operator_training import dump_json, sha256, verify_completed


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transfer', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_completed(args.runs)
    data_manifest = json.loads((args.data/'manifest.json').read_text())
    if sha256(args.data/'test.npz') != data_manifest['test.npz']:
        raise ValueError('Test data checksum mismatch')
    if args.transfer is not None:
        verify_completed(args.transfer)
        if sha256(args.data/'reference_geometry.npz') != data_manifest['reference_geometry.npz']:
            raise ValueError('Finer geometry checksum mismatch')
    args.output.mkdir(parents=True)
    inputs = args.output/'unlabelled_inputs.npz'
    with np.load(args.data/'test.npz', allow_pickle=False) as z:
        ids = z['sample_id'].copy()
        np.savez_compressed(inputs, a=z['a'], f=z['f'], sample_id=ids)
    if args.transfer is not None:
        with np.load(args.data/'reference_geometry.npz', allow_pickle=False) as z:
            np.savez(args.output/'fine_queries.npz', query_points=z['coordinates'])
    config = json.loads((args.runs/'config.json').read_text())
    report = {'purpose': 'Saved-model reload/inference audit only; no training or checkpoint selection.',
              'input_sha256': sha256(inputs), 'sample_ids': list(map(str, ids)),
              'script_sha256': sha256(__file__), 'models': {}}
    for spec in config['models']:
        name = spec['name']
        prediction = run_inference(args.runs/name, inputs, args.output/name,
                                   device=args.device, batch_size=config['batch_size'])
        with np.load(args.runs/name/'test_predictions.npz', allow_pickle=False) as z:
            if not np.array_equal(ids, z['sample_id']):
                raise ValueError('Test ID mismatch: '+name)
            stored = z['y'].copy()
        if not np.allclose(prediction, stored, rtol=5e-5, atol=1e-8):
            raise ValueError('Reloaded prediction mismatch: '+name)
        record = {'coarse_max_absolute_difference': float(np.max(np.abs(prediction-stored))),
                  'finite': bool(np.isfinite(prediction).all())}
        if args.transfer is not None and name in ('deeponet', 'transformer'):
            fine = run_inference(args.runs/name, inputs, args.output/(name+'_fine'),
                                 device=args.device, batch_size=1,
                                 queries=args.output/'fine_queries.npz', query_chunk_size=257)
            with np.load(args.transfer/(name+'_fine_predictions.npz'), allow_pickle=False) as z:
                target = z['y'].reshape(fine.shape)
            if not np.allclose(fine, target, rtol=5e-5, atol=1e-8):
                raise ValueError('Chunked fine prediction mismatch: '+name)
            record['fine_chunk_max_absolute_difference'] = float(np.max(np.abs(fine-target)))
        report['models'][name] = record
    dump_json(args.output/'audit.json', report)
    dump_json(args.output/'manifest.json', {str(p.relative_to(args.output)): sha256(p)
              for p in sorted(args.output.rglob('*')) if p.is_file()})
    (args.output/'COMPLETE').write_text('All selected model inference checks passed.\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
