#!/usr/bin/env python3
"""Evaluate selected operators on saved finer-grid labels, without tuning.

Grid operators re-evaluate the saved raw a/f on the finer grid. Query operators
keep their trained input sensors and change output coordinates. Each prediction
is compared with P1 interpolation of that checkpoint's coarse-grid prediction.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from neuralpde_opt.chapter3.operator_training import (
    P1Geometry, dump_json, evaluate_arrays, load_trained_model, sha256, source_fingerprints,
)
from neuralpde_opt.chapter3 import reference
from neuralpde_opt.chapter3.reference import evaluate_p1


def evaluate_transfer(data, runs, output, device='cpu', models=None):
    data, runs, output = Path(data), Path(runs), Path(output)
    if not (runs/'COMPLETE').is_file():
        raise ValueError('Completed restricted operator runs required')
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((runs/'config.json').read_text())
    names = [spec['name'] for spec in config['models']]
    if models is not None:
        if not models or len(models) != len(set(models)) or not set(models) <= set(names):
            raise ValueError('Choose unique model names from the completed suite')
        names = list(models)
    manifest = json.loads((data/'manifest.json').read_text())
    required = ('test.npz', 'observation_geometry.npz', 'reference_geometry.npz')
    for name in required:
        if name not in manifest or sha256(data/name) != manifest[name]:
            raise ValueError('Dataset checksum mismatch: '+name)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    torch.set_num_threads(2)
    coarse_geometry = P1Geometry(data/'observation_geometry.npz')
    fine_geometry = P1Geometry(data/'reference_geometry.npz')
    with np.load(data/'test.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ('a', 'f', 'y', 'a_reference', 'f_reference', 'y_reference', 'sample_id')}
    n = coarse_geometry.grid_shape[0]-1
    fine_points = fine_geometry.coordinates.reshape(-1, 2)
    ids = arrays['sample_id']
    label_interpolation = np.stack([
        evaluate_p1(values.ravel(), n, fine_points)[0].reshape(fine_geometry.grid_shape)
        for values in arrays['y']])
    report = {
        'scope': 'Held-out pilot PDEs; resolution diagnostic only, no tuning or method ranking.',
        'fine_target': 'Saved finer reference P1 state, not an exact continuum solution.',
        'baseline': 'P1 interpolation of the same selected model coarse-grid prediction.',
        'data_sha256': {name: sha256(data/name) for name in required},
        'script_sha256': sha256(__file__),
        'math_backend': {'deterministic_algorithms': True, 'cudnn_benchmark': False,
                         'cudnn_deterministic': True, 'matmul_allow_tf32': False,
                         'cudnn_allow_tf32': False},
        'source_sha256': {**source_fingerprints([s for s in config['models'] if s['name'] in names]),
                          'reference.py': sha256(reference.__file__)},
        'coarse_label_interpolation': evaluate_arrays(
            label_interpolation, arrays['y_reference'], fine_geometry, ids),
        'models': {},
    }
    output.mkdir(parents=True)
    for name in names:
        run_dir = runs/name
        manifest = json.loads((run_dir/'manifest.json').read_text())
        for filename in ('best.pt', 'test_predictions.npz', 'provenance.json'):
            if filename not in manifest or sha256(run_dir/filename) != manifest[filename]:
                raise ValueError('Run checksum mismatch: '+name+'/'+filename)
        provenance = json.loads((run_dir/'provenance.json').read_text())
        for filename in ('test.npz', 'observation_geometry.npz', 'manifest.json'):
            if provenance['data_sha256'].get(filename) != sha256(data/filename):
                raise ValueError('Training-data provenance mismatch: '+filename)
        model = load_trained_model(run_dir, device)
        dtype = next(model.parameters()).dtype
        queries = bool(getattr(model.model, 'supports_queries', False))
        with np.load(run_dir/'test_predictions.npz', allow_pickle=False) as stored:
            if not np.array_equal(ids, stored['sample_id']):
                raise ValueError('Held-out sample ID mismatch')
            coarse = stored['y'].copy()
        if coarse.shape != arrays['y'].shape:
            raise ValueError('Coarse prediction shape mismatch')
        interpolation = np.stack([
            evaluate_p1(values.ravel(), n, fine_points)[0].reshape(fine_geometry.grid_shape)
            for values in coarse])
        predicted = []
        with torch.inference_mode():
            for index in range(len(ids)):
                keys = ('a', 'f') if queries else ('a_reference', 'f_reference')
                a, f = [torch.as_tensor(arrays[key][index:index+1], dtype=dtype, device=device)
                        for key in keys]
                options = {'query_points': torch.as_tensor(fine_points, dtype=dtype, device=device),
                           'query_chunk_size': 1024} if queries else {}
                result = model(a, f, **options)
                predicted.append(result.cpu().numpy().reshape(fine_geometry.grid_shape))
        predicted = np.stack(predicted)
        report['models'][name] = {
            'checkpoint_sha256': manifest['best.pt'],
            'representation': ('Keep trained input sensors; query the finer grid.' if queries
                               else 'Re-evaluate saved a/f on the finer grid, then apply the grid operator.'),
            'model_representation': model.model.representation_metadata(),
            'direct_fine': evaluate_arrays(predicted, arrays['y_reference'], fine_geometry, ids),
            'interpolated_coarse': evaluate_arrays(interpolation, arrays['y_reference'], fine_geometry, ids),
        }
        np.savez_compressed(output/f'{name}_fine_predictions.npz', sample_id=ids,
                            y=predicted, interpolated_coarse=interpolation)
    dump_json(output/'metrics.json', report)
    lines = ['# Chapter 3 pilot resolution diagnostic', '',
             'Same selected checkpoints; held-out pilot PDEs; no tuning or method ranking.', '',
             '| Model | Direct fine relative L2 | Interpolated coarse relative L2 | Direct fine relative H1 |',
             '|---|---:|---:|---:|']
    for name, row in report['models'].items():
        direct, baseline = row['direct_fine']['mean'], row['interpolated_coarse']['mean']
        lines.append(f'| {name} | {direct["relative_l2"]:.6e} | {baseline["relative_l2"]:.6e} | '
                     f'{direct["relative_h1_seminorm"]:.6e} |')
    lines += ['', 'Finer queries or shared parameters permit evaluation; they do not certify resolution accuracy.',
              'Fine labels retain finite-element discretization error. Public-grid P1 label interpolation '
              'error is recorded separately in metrics.json.', '']
    (output/'summary.md').write_text('\n'.join(lines))
    dump_json(output/'manifest.json', {p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    (output/'COMPLETE').write_text('Held-out diagnostic completed; no parameter or checkpoint tuning.\n')
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--models', nargs='+')
    args = parser.parse_args()
    evaluate_transfer(args.data, args.runs, args.output, args.device, args.models)
    print((args.output/'summary.md').read_text())


if __name__ == '__main__':
    main()
