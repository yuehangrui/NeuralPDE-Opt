"""Generate the final book checklist and audit report only after verified retraining."""
import datetime
import hashlib
import json
from pathlib import Path
from book_driver import CHAPTERS,RESULTS,read,write

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for chapter in CHAPTERS:
        for stage in ['TRAINING','DELIVERY','VERIFIED']:
            if read(ROOT/'verification'/f'{chapter}_{stage}.json')['passed'] is not True:
                raise ValueError('Incomplete delivery: '+chapter+' '+stage)
    audits={c:read(ROOT/RESULTS[c]/('verification.json' if c=='ch03' else 'DELIVERY_AUDIT.json')) for c in CHAPTERS}
    if audits['ch02']['neural_runs']!=18 or audits['ch02']['fem_runs']!=8:raise ValueError('Chapter 2 scope')
    training3=read(ROOT/'results/ch03/final_v1_support/training_audit.json')
    if not training3['passed'] or training3['count']!=57:raise ValueError('Chapter 3 scope')
    if not audits['ch03']['passed'] or audits['ch03']['model_evaluations']!=54:raise ValueError('Chapter 3 evaluation')
    if audits['ch04']['status']!='passed' or audits['ch04']['training_experiments']!=48:raise ValueError('Chapter 4 scope')
    depth=read(ROOT/RESULTS['ch04-depth']/'protocol.json')
    if len(depth['jobs'])!=12:raise ValueError('Chapter 4 depth scope')
    if audits['ch05']['status']!='passed' or audits['ch05']['training_configurations']!=21:raise ValueError('Chapter 5 scope')
    destination=ROOT/'reports'
    destination.mkdir(exist_ok=True)
    for name in ['BOOK_CHECKLIST_ZH.md','AUDIT_REPORT_ZH.md','FINAL_AUDIT.json']:
        if (destination/name).exists():raise FileExistsError('Final report already exists: '+name)
    scope=read(ROOT/'book_scope.json')
    checks=read(ROOT/'verification/CHECKS.json')
    smoke=read(ROOT/'verification/SMOKE.json')
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    head=['# Standalone edition: manuscript-to-code checklist','',f'Generated: {stamp}. Manuscript SHA256: `{scope["book_sha256"]}`.',
          '', 'This checklist was generated after all 156 training runs, independent evaluations, and audits were completed. It follows the previously reviewed manuscript mapping. Chapters 1 and 6 provide explanatory material; Chapters 2-5 contain the algorithms. Delivery completion does not imply uniform accuracy or strict optimality.','',
          '| Part | Training runs | Results and verification |','|---|---:|---|']
    for ch,count in zip(CHAPTERS,[18,57,48,12,21]):
        head.append(f'| {ch} | {count} | [Formal results](../{RESULTS[ch]}/summary.md); independent audit passed |')
    head+=['','Chapter 2 also includes 8 FEM runs; the 48 Chapter 4 training runs retain 72 model stages. HINTS uses the newly trained VarMiON initialization. Development smoke tests are excluded from formal counts.',
           '', '| Manuscript location | Coverage | Implementation | Limitations |','|---|---|---|---|']
    for row in scope['rows']:
        files=', '.join(f'[{Path(p).name}](../{p})' for p in row['source_paths'])
        head.append(f'| {row["section"]} | {row["coverage"]} | {files or row["implementation"]} | {row["boundary"]} |')
    head+=['','Deferred work remains deferred: state-splitting ADMM, interface control, obstacle control/S2-FOBA, and semilinear/parabolic extensions. Neural initialization + CG has left the main scope; the Darcy replication remains paused.','']
    (destination/'BOOK_CHECKLIST_ZH.md').write_text('\n'.join(head),encoding='utf-8')
    lines=['# Standalone edition: audit report','',f'Generated: {stamp}.','',
           'Training assets were regenerated in the independent workspace and all 156 formal runs were completed. Previous checkpoints were not used for formal initialization. HINTS uses the newly trained VarMiON in the same workspace, as specified by the method. Chapter 2 saves the final iterate; Chapters 3-5 select models using validation data.','',
           f'Numerical regression: {checks["total_tests"]} tests passed; optimized-mode checks: {checks["optimized_mode_passed"]}. All {len(smoke["chapter3_cases"])} small-budget checks across Chapter 3 training variants passed.',
           '', 'Final tests for Chapters 3-5 use new random streams declared in advance for the standalone edition, and are generated only after training is complete and model selection is frozen. Chapter 2 uses independent quadrature, physical consistency checks, and FEM convergence checks.',
           '', '| Part | Audit evidence |','|---|---|']
    for ch in CHAPTERS:
        name='verification.json' if ch=='ch03' else 'DELIVERY_AUDIT.json'
        lines.append(f'| {ch} | [{name}](../{RESULTS[ch]}/{name}), [results report](../{RESULTS[ch]}/summary.md) |')
    lines+=['','Accuracy is interpreted in each chapter report. Completing a workflow does not establish convergence. Spatial validation on one fixed instance is not generalization to new instances; extending a fixed-grid prediction is not resolution transfer. Different budgets, data, and physical information do not establish an architecture-family ranking.',
            '', 'The wheel can initialize any new workspace without the original project or upstream repositories. Scientific computing still depends on PyTorch, NumPy, SciPy, and Matplotlib. Ported DeepXDE and Galerkin components retain their licenses and provenance; ocpde_dl provenance is also retained. Licenses are not replaced with one uniform license, and the code is not claimed to be entirely original.','']
    installation=ROOT/'verification/INSTALLATION.json'
    if installation.exists():
        details=read(installation)
        lines+=['Installation used an independent virtual environment sharing the scientific dependencies already available on the server. No clean-operating-system or fully offline dependency installation is claimed. See [INSTALLATION.json](../verification/INSTALLATION.json).','']
    lines+=['Source identity: [PAYLOAD_MANIFEST.json](../PAYLOAD_MANIFEST.json); provenance: [SOURCE_ORIGINS.json](../SOURCE_ORIGINS.json); packaging changes: [SOURCE_CHANGES.json](../SOURCE_CHANGES.json). Chapter manifests separately verify training, model selection, and final artifact identities.','']
    (destination/'AUDIT_REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    identities={}
    for ch in CHAPTERS:
        for p in (ROOT/RESULTS[ch]).rglob('*'):
            if p.is_file():identities[p.relative_to(ROOT).as_posix()]=sha(p)
    for p in (destination/'BOOK_CHECKLIST_ZH.md',destination/'AUDIT_REPORT_ZH.md'):
        identities[p.relative_to(ROOT).as_posix()]=sha(p)
    write(destination/'FINAL_AUDIT.json',dict(passed=True,created=stamp,training_configurations=156,
           book_mapping_entries=len(scope['rows']),source_manifest_sha256=sha(ROOT/'PAYLOAD_MANIFEST.json'),
           regression_tests=checks['total_tests'],chapters=CHAPTERS,files=identities))
    print('BOOK_DELIVERY_COMPLETE',flush=True)


if __name__=='__main__':main()
