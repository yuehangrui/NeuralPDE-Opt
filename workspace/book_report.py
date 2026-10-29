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
    head=['# 全书独立版：书稿—代码清单','',f'生成时间：{stamp}。书稿SHA256：`{scope["book_sha256"]}`。',
          '', '本清单在本轮156组训练、独立评测及复核完成后生成。覆盖范围沿用已经核对的书稿映射；第1、6章是说明性配套，第2至5章是算法主体。交付完成不等于达到统一精度或严格最优性。','',
          '| 部分 | 本轮训练 | 结果与核查 |','|---|---:|---|']
    for ch,count in zip(CHAPTERS,[18,57,48,12,21]):
        head.append(f'| {ch} | {count} | [正式结果](../{RESULTS[ch]}/summary.md)；独立复核通过 |')
    head+=['','第二章另有8组FEM；第四章48组训练保留72个阶段模型。HINTS使用本轮VarMiON初值，正式统计不包含开发烟雾试验。',
           '', '| 书稿位置 | 覆盖层级 | 本版实现 | 边界 |','|---|---|---|---|']
    for row in scope['rows']:
        files='、'.join(f'[{Path(p).name}](../{p})' for p in row['source_paths'])
        head.append(f'| {row["section"]} | {row["coverage"]} | {files or row["implementation"]} | {row["boundary"]} |')
    head+=['','后置内容继续保留：状态分裂ADMM、界面、障碍控制/S2-FOBA及半线性/抛物型扩展。神经初值＋CG退出主线，Darcy复刻保持暂停。','']
    (destination/'BOOK_CHECKLIST_ZH.md').write_text('\n'.join(head),encoding='utf-8')
    lines=['# 全书独立版：核查报告','',f'生成时间：{stamp}。','',
           '本轮独立目录重新生成训练资产并完成156组正式训练。旧检查点未用于正式初始化；HINTS按既定方法使用同一工作区中新训练的VarMiON。第二章保存最终迭代，第三至五章按验证集选模。','',
           f'数值回归：{checks["total_tests"]}项通过；优化模式检查：{checks["optimized_mode_passed"]}。第三章各训练变体共{len(smoke["chapter3_cases"])}项小预算验证通过。',
           '', '第三至五章最终测试使用独立版预先声明的新随机流，全部训练结束且选模冻结后才生成。第二章以独立积分、物理一致性和FEM收敛核查。',
           '', '| 部分 | 核查证据 |','|---|---|']
    for ch in CHAPTERS:
        name='verification.json' if ch=='ch03' else 'DELIVERY_AUDIT.json'
        lines.append(f'| {ch} | [{name}](../{RESULTS[ch]}/{name})、[结果报告](../{RESULTS[ch]}/summary.md) |')
    lines+=['','精度按各章结果报告逐项解释。本报告不把流程完成等同于算法收敛；固定实例空间验证不是新实例泛化，固定网格延拓也不是分辨率迁移。不同预算、数据与物理信息不构成架构家族排名。',
            '', '本版可从wheel展开到任意新工作区，运行不需要原项目或作者仓库。科学计算仍依赖PyTorch、NumPy、SciPy和Matplotlib。DeepXDE及Galerkin移植组件和来源许可证随包保留；ocpde_dl来源记录继续保留，不统一改写许可证，也不声称全部原创。','']
    installation=ROOT/'verification/INSTALLATION.json'
    if installation.exists():
        details=read(installation)
        lines+=['本机安装使用独立虚拟环境，共享服务器已有科学计算依赖；未声称完成全新操作系统或离线全依赖安装。安装记录见[INSTALLATION.json](../verification/INSTALLATION.json)。','']
    lines+=['版本身份：[PAYLOAD_MANIFEST.json](../PAYLOAD_MANIFEST.json)；来源：[SOURCE_ORIGINS.json](../SOURCE_ORIGINS.json)；封装改动：[SOURCE_CHANGES.json](../SOURCE_CHANGES.json)。训练、选模和最终产物身份由各章清单分别核查。','']
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
