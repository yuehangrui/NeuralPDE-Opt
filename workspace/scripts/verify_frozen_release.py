"""Read-only SHA256 verification of a frozen source release and optional artifacts.

Uses only the Python standard library. Never loads arrays/checkpoints or launches
training. A code-only archive can be checked without copying datasets or models.
"""
import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_RELEASE = 'ch03-reaction-v1.0-20261022'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Expected a project-relative path: ' + name)
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path leaves project: ' + name)
    return target


def verify_release(root, release, artifacts=False, archive=False):
    root = Path(root).resolve()
    directory = inside(root, 'releases/' + release)
    descriptor = directory / 'release.json'
    expected = (directory / 'release.json.sha256').read_text().split()[0]
    if sha256(descriptor) != expected:
        raise ValueError('Release descriptor checksum mismatch')
    info = json.loads(descriptor.read_text())
    if info.get('schema_version') != 'neuralpde-frozen-release-v1' or info.get('release') != release:
        raise ValueError('Unsupported or inconsistent release identity')
    groups = ['code'] + (['artifacts'] if artifacts else [])
    problems, counts = [], {}
    for group in groups:
        spec = info['manifests'][group]
        path = inside(root, spec['path'])
        if sha256(path) != spec['sha256']:
            raise ValueError(group + ' manifest checksum mismatch')
        records = json.loads(path.read_text())
        counts[group] = len(records)
        for name, record in records.items():
            p = inside(root, name)
            if p.is_symlink() or not p.is_file():
                problems.append({'path': name, 'reason': 'missing, non-file, or symlink'})
            elif p.stat().st_size != record['bytes'] or sha256(p) != record['sha256']:
                problems.append({'path': name, 'reason': 'content changed'})
        if group == 'code':
            for base in info['managed_code_directories']:
                for p in inside(root, base).rglob('*'):
                    if '__pycache__' in p.parts or p.suffix in {'.pyc', '.pyo'}:
                        continue
                    if p.is_file() and p.relative_to(root).as_posix() not in records:
                        problems.append({'path': p.relative_to(root).as_posix(), 'reason': 'unfrozen extra file'})
    if archive:
        item = info['archive']
        path = inside(root, item['path'])
        expected_archive = Path(str(path) + '.sha256').read_text().split()[0]
        if not path.is_file() or sha256(path) != expected_archive:
            problems.append({'path': item['path'], 'reason': 'archive missing or changed'})
    return {'passed': not problems, 'release': release, 'verified_files': counts,
            'archive_checked': archive, 'test_arrays_loaded': False, 'training_started': False,
            'problems': problems}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--release', default=DEFAULT_RELEASE)
    parser.add_argument('--artifacts', action='store_true', help='Also hash preserved data/models/results/references')
    parser.add_argument('--archive', action='store_true', help='Also verify the source archive in dist/')
    args = parser.parse_args()
    try:
        report = verify_release(args.root, args.release, args.artifacts, args.archive)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report = {'passed': False, 'error': str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
