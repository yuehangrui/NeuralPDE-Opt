"""Explicit lifecycle commands; every run stays in a verified workspace."""
import argparse
import json
from .workspace import initialize, run, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('directory')
    for command in ['check','prepare','train','finish','verify','status','smoke','pipeline']:
        p = sub.add_parser(command)
        p.add_argument('--workspace', required=True)
        if command in ['train','finish','verify']:
            p.add_argument('--chapter', choices=['all','ch02','ch03','ch04','ch04-depth','ch05'],default='all')
        if command in ['train','pipeline']:
            p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        root = initialize(args.directory)
        print(json.dumps({'workspace':str(root),'source_files':len(verify(root)['files'])}))
        return
    arguments = [args.command]
    if hasattr(args,'chapter'): arguments += ['--chapter',args.chapter]
    if getattr(args,'resume',False): arguments += ['--resume']
    raise SystemExit(run(args.workspace,arguments))


if __name__ == '__main__':
    main()
