#!/usr/bin/env python3
"""Read-only train/val diagnosis of frozen models. Never opens test arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from neuralpde_opt.chapter3.operator_training import (
    P1Geometry, fit_normalization, load_trained_model, read_split, sha256,
    verify_data_manifest, verify_completed, verify_frozen,
)


def trunk_projection(model, targets, geometry):
    """Best per-target coefficients in the frozen trunk span, not a predictor.

    SVD uses the exact elementwise mass square root, avoiding squared condition
    numbers from normal equations. Report two tolerances because a float32
    trunk can have nearly dependent columns. Bias is allowed a free coefficient
    per target, so this is an optimistic representation diagnostic.
    """
    net = model.model
    with torch.no_grad():
        points = net.grid_points.to(dtype=net.bias.dtype, device=net.bias.device)
        basis = net.trunk(net._coordinate_features(points)).cpu().double().numpy()
        xy = points.cpu().double().numpy()
    boundary = 16*xy[:, 0]*(1-xy[:, 0])*xy[:, 1]*(1-xy[:, 1]) if net.hard_boundary else np.ones(len(xy))
    basis = np.column_stack((basis, np.ones(len(xy))))*boundary[:, None]
    local_mass = (np.eye(3)+np.ones((3, 3)))/12
    root = np.linalg.cholesky(local_mass).T
    weighted_basis = (np.einsum('ij,ejk->eik', root, basis[geometry.elements])
                      *np.sqrt(geometry.areas)[:, None, None]).reshape(-1, basis.shape[1])
    u, s, vt = np.linalg.svd(weighted_basis, full_matrices=False)
    report = {'singular_values_relative': (s/s[0]).tolist(), 'tolerances': {}}
    for tol in (1e-6, 1e-10):
        rank = int(np.sum(s > tol*s[0]))
        rows = {}
        for split, y in targets.items():
            flat = y.reshape(len(y), -1)
            rhs = (np.einsum('ij,bej->bei', root, flat[:, geometry.elements])
                   *np.sqrt(geometry.areas)[None, :, None]).reshape(len(y), -1).T
            coefficients = (vt[:rank].T/s[:rank]) @ (u[:, :rank].T@rhs)
            residual = (flat-(basis@coefficients).T).reshape(y.shape)
            rows[split] = float(np.mean(np.sqrt(geometry.l2_squared(residual)/geometry.l2_squared(y))))
        report['tolerances'][str(tol)] = {'rank': rank, 'oracle_relative_l2': rows}
    return report


def diagnostic_models(runs):
    """Derive complete coverage from saved configuration, never an empty glob."""
    runs = Path(runs)
    if (runs/'experiment_config.json').is_file():
        config = json.loads((runs/'experiment_config.json').read_text())
        seeds, specs = config['seeds'], config['training']['models']
        expected = {f'seed_{seed}/{s["name"]}' for seed in seeds for s in specs}
        count = len(seeds)*len(specs)
        actual = {str(p.parent.relative_to(runs)) for p in runs.glob('seed_*/*/best.pt')}
    elif (runs/'config.json').is_file():
        config = json.loads((runs/'config.json').read_text())
        specs = config.get('models', [])
        expected = {s['name'] for s in specs}
        count = len(specs)
        actual = {p.parent.name for p in runs.glob('*/best.pt')}
    else:
        raise ValueError('Diagnostic requires an experiment or seed directory with saved configuration')
    if not expected or len(expected) != count or actual != expected:
        raise ValueError(f'Diagnostic model coverage mismatch: expected={sorted(expected)}, actual={sorted(actual)}')
    return [(key, runs/key) for key in sorted(expected)]


def diagnostic_inputs(data, directory, splits, geometry):
    """Verify frozen artifacts and training data identity before interpreting errors."""
    if (directory/'COMPLETE').is_file():
        verify_completed(directory)
    elif (directory/'FROZEN').is_file():
        verify_frozen(directory)
    else:
        raise ValueError('Diagnostic requires a completed or frozen model: '+str(directory))
    provenance = json.loads((directory/'provenance.json').read_text())
    hashes = provenance['data_sha256']
    # Do not read real test arrays. The manifest/certificate also binds that split.
    for name in ('train.npz', 'val.npz', 'observation_geometry.npz', 'config.json', 'manifest.json', 'COMPLETE'):
        if name not in hashes or sha256(Path(data)/name) != hashes[name]:
            raise ValueError('Diagnostic data identity mismatch: '+name)
    train = splits['train']
    ids = provenance['training_sample_ids']
    lookup = {str(sid): i for i, sid in enumerate(train['sample_id'])}
    if len(ids) != len(set(ids)) or not ids or any(sid not in lookup for sid in ids):
        raise ValueError('Diagnostic training subset IDs mismatch')
    selected = {k: v[[lookup[sid] for sid in ids]].copy() for k, v in train.items()}
    norm = json.loads((directory/'normalization.json').read_text())
    if norm != fit_normalization(selected, geometry):
        raise ValueError('Diagnostic saved normalization differs from the actual training subset')
    return {'train': selected, 'val': splits['val']}, norm


def diagnose(data, runs, output, device='cpu'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    certificate = verify_data_manifest(data)
    directories = diagnostic_models(runs)
    torch.set_num_threads(2)
    splits = {key: read_split(data, key) for key in ('train', 'val')}
    geometry = P1Geometry(Path(data)/'observation_geometry.npz')
    norm = fit_normalization(splits['train'], geometry)
    report = {'scope': 'read-only train/val; no test loading or optimizer updates',
              'normalization': norm, 'physical_mse_gradient_scale_vs_scaled_mse': norm['y_rms']**2,
              'normalization_scope': 'full training split; each model uses its saved training-subset normalization',
              'data_certificate': certificate,
              'expected_model_count': len(directories), 'checked_model_count': 0,
              'splits': {}, 'models': {}}
    for key, split in splits.items():
        energy = geometry.l2_squared(split['y'])
        report['splits'][key] = {'sample_count': len(energy), 'target_l2_min_median_max':
            np.quantile(np.sqrt(energy), [0, .5, 1]).tolist(),
            'squared_norm_max_over_min': float(energy.max()/energy.min())}
    for key, directory in directories:
        model_splits, model_norm = diagnostic_inputs(data, directory, splits, geometry)
        before = sha256(directory/'best.pt')
        model = load_trained_model(directory, device)
        row = {'best_sha256': before, 'split_metrics': {}, 'normalization': model_norm,
               'physical_mse_gradient_scale_vs_scaled_mse': model_norm['y_rms']**2,
               'data_identity_verified': True}
        for split in model_splits:
            with np.load(directory/f'{split}_predictions.npz', allow_pickle=False) as z:
                if not np.array_equal(z['sample_id'], model_splits[split]['sample_id']):
                    raise ValueError('Prediction ID mismatch')
                prediction = z['y'].copy()
            target = model_splits[split]['y']
            energy = geometry.l2_squared(target)
            error = geometry.l2_squared(prediction-target)
            row['split_metrics'][split] = {
                'mean_relative_l2': float(np.sqrt(error/energy).mean()),
                'global_relative_l2': float(np.sqrt(error.sum()/energy.sum())),
                'physical_nodal_mse': float(np.mean((prediction-target)**2)),
                'scaled_nodal_mse': float(np.mean((prediction-target)**2)/model_norm['y_rms']**2)}
        # Backward-only scale audit: computes gradients, never calls an optimizer.
        train = model_splits['train']
        dtype = next(model.parameters()).dtype
        a, f, y = [torch.as_tensor(train[k][:4], dtype=dtype, device=device) for k in ('a', 'f', 'y')]
        prediction = model(a, f)
        physical_loss = (prediction-y).square().mean()
        gradients = torch.autograd.grad(physical_loss, tuple(model.parameters()), allow_unused=True)
        values = torch.cat([g.detach().abs().flatten() for g in gradients if g is not None])
        row['physical_mse_gradient_audit_first_4_train'] = {
            'loss': float(physical_loss.detach()), 'gradient_norm': float(torch.linalg.vector_norm(values)),
            'fraction_abs_gradient_below_adam_default_eps': float((values < 1e-8).float().mean()),
            'caveat': 'A gradient magnitude audit, not evidence that Adam eps alone caused the error; no optimizer update.'}
        val = model_splits['val']
        with torch.no_grad():
            av, fv = [torch.as_tensor(val[k][:4], dtype=dtype, device=device) for k in ('a', 'f')]
            reloaded = model(av, fv).cpu().numpy()
        with np.load(directory/'val_predictions.npz', allow_pickle=False) as z:
            saved = z['y'][:4]
        np.testing.assert_allclose(reloaded, saved, rtol=1e-5, atol=1e-7)
        row['reload_val_first_4_max_absolute_difference'] = float(np.max(np.abs(reloaded-saved)))
        if directory.name == 'deeponet':
            with torch.no_grad():
                net = model.model
                i, j = net.sensor_indices.unbind(-1)
                an, fn = (a-model.a_mean)/model.a_std, (f-model.f_mean)/model.f_std
                features = torch.cat((an[:, i, j], fn[:, i, j]), dim=-1)
                activations = []
                for layer in net.branch:
                    features = layer(features)
                    if isinstance(layer, torch.nn.Tanh):
                        activations.append(float((features.abs()>.99).float().mean()))
                row['branch_tanh_fraction_abs_above_099_first_4_train'] = activations
            row['frozen_trunk_projection'] = trunk_projection(model, {k:v['y'] for k,v in model_splits.items()}, geometry)
        if sha256(directory/'best.pt') != before:
            raise RuntimeError('Checkpoint changed')
        report['models'][key] = row
        report['checked_model_count'] += 1
        del model
    report['read_only_checks_passed'] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--runs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    diagnose(args.data, args.runs, args.output, args.device)
