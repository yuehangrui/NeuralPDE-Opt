#!/usr/bin/env python3
"""Tables and scientific figures for already frozen/completed operator runs."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neuralpde_opt.chapter3.operator_training import (
    P1Geometry, data_fingerprints, dump_json, evaluate_arrays, load_trained_model,
    sha256, source_fingerprints, verify_completed,
)

DISPLAY = {'fno': 'FNO', 'deeponet': 'DeepONet', 'gno': 'GNO', 'transformer': 'Transformer'}
COLORS = {'fno': '#2767a1', 'deeponet': '#cc6b28', 'gno': '#36844b', 'transformer': '#8a58a3'}


def statistic(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()),
            'std_across_seeds': float(values.std(ddof=1)) if len(values)>1 else 0.,
            'values': values.tolist()}


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def measure_inference(directory, a, f, device, repeats=10):
    model = load_trained_model(directory, device)
    dtype = next(model.parameters()).dtype
    av, fv = [torch.as_tensor(v[:min(8,len(v))], dtype=dtype, device=device) for v in (a,f)]
    def synchronize():
        if torch.device(device).type == 'cuda':
            torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(3): model(av, fv)
        synchronize()
        times = []
        for _ in range(repeats):
            synchronize(); start=time.perf_counter()
            model(av, fv); synchronize()
            times.append(time.perf_counter()-start)
    return {'batch_size': len(av), 'warmup_batches': 3, 'timed_batches': repeats,
            'median_batch_seconds': float(np.median(times)),
            'median_seconds_per_pde': float(np.median(times)/len(av)),
            'scope': 'Forward on preloaded GPU/CPU tensors, raw-field normalization included; no transfer, checkpoint load, or reference solve.'}


def build_report(data, output, device='cuda'):
    data, output = Path(data), Path(output)
    config = json.loads((output/'experiment_config.json').read_text())
    seeds = config['seeds']; names=[s['name'] for s in config['training']['models']]
    identity = json.loads((output/'experiment_identity.json').read_text())
    if identity['data_sha256'] != data_fingerprints(data):
        raise ValueError('Report data identity mismatch')
    if identity['source_sha256'] != source_fingerprints(config['training']['models']):
        raise ValueError('Report model source identity mismatch')
    ledger = json.loads((output/'frozen_selection.json').read_text())
    for seed in seeds:
        if not (output/f'seed_{seed}'/'COMPLETE').is_file():
            raise ValueError('Complete every frozen seed before reporting test errors')
        verify_completed(output/f'seed_{seed}')
        verify_completed(output/'resolution'/f'seed_{seed}')
        verify_completed(output/'inference'/f'seed_{seed}')
        for name in names:
            if sha256(output/f'seed_{seed}'/name/'best.pt') != ledger[f'{seed}/{name}']['best_sha256']:
                raise ValueError('Report frozen checkpoint mismatch')
    geometry = P1Geometry(data/'observation_geometry.npz')
    with np.load(data/'test.npz', allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('sample_id','a','f','y')}
    with np.load(data/'train.npz', allow_pickle=False) as z:
        train_count = len(z['sample_id'])
        training_mean=z['y'].mean(axis=0)
    baseline = {name:evaluate_arrays(values, arrays['y'], geometry, arrays['sample_id']) for name,values in
                {'zero':np.zeros_like(arrays['y']),
                 'mean_training_state':np.broadcast_to(training_mean, arrays['y'].shape)}.items()}
    records={}; aggregate={}; histories={}; fine={}
    for name in names:
        records[name]=[];histories[name]=[];fine[name]=[]
        for seed in seeds:
            run=output/f'seed_{seed}'/name
            metrics=json.loads((run/'metrics.json').read_text())
            provenance=json.loads((run/'provenance.json').read_text())
            records[name].append({'seed':seed,'metrics':metrics,
                                  'parameter_count':provenance['parameter_count']})
            histories[name].append(json.loads((run/'history.json').read_text()))
            fine[name].append(json.loads((output/'resolution'/f'seed_{seed}'/'metrics.json').read_text())['models'][name])
        aggregate[name]={
            split:{metric:statistic([r['metrics'][split]['mean'][metric] for r in records[name]])
                   for metric in ('relative_l2','relative_h1_seminorm','absolute_l2','boundary_max')}
            for split in ('train','val','test')}
        aggregate[name].update({
            'parameters':records[name][0]['parameter_count'],
            'loop_wall_seconds':statistic([r['metrics']['loop_wall_seconds'] for r in records[name]]),
            'selected_steps':[r['metrics']['selected_step'] for r in records[name]],
            'inference_seed0': measure_inference(output/f'seed_{seeds[0]}'/name, arrays['a'],arrays['f'],device),
            'resolution':{kind:{metric:statistic([r[kind]['mean'][metric] for r in fine[name]])
                                for metric in ('relative_l2','relative_h1_seminorm')}
                          for kind in ('direct_fine','interpolated_coarse')}})
    report={'scope':('Legacy-referenced full supervised training on the existing dataset; reused test set, no new blind test or universal architecture ranking.'
                     if config['training'].get('legacy_full_epochs') else
                     'First fixed-budget comparison of four specified teaching architectures; no universal architecture ranking.'),
            'seeds':seeds,'test_ids':list(map(str,arrays['sample_id'])),
            'data_manifest_sha256':sha256(data/'manifest.json'),
            'frozen_selection_sha256':sha256(output/'frozen_selection.json'),
            'report_source_sha256':sha256(__file__), 'aggregate':aggregate,'runs':records,'baselines':baseline,
            'uncertainty':'Std uses ddof=1 across training seeds on the same test PDEs; it is not a confidence interval over independent datasets.',
            'timing_device':str(device), 'timing_note':'Training wall time includes validation, P1 metrics and checkpoint writes. No CPU-FEM/GPU speedup claim.'}
    dump_json(output/'aggregate.json',report)
    with (output/'metrics.csv').open('w',newline='') as file:
        fields=('model','seed','parameters','selected_step','train_relative_l2',
                'val_relative_l2','test_relative_l2','test_relative_h1_seminorm',
                'boundary_max','loop_wall_seconds')
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for name in names:
            for row in records[name]:
                metrics=row['metrics']
                writer.writerow({'model':name,'seed':row['seed'],'parameters':row['parameter_count'],
                    'selected_step':metrics['selected_step'],
                    'train_relative_l2':metrics['train']['mean']['relative_l2'],
                    'val_relative_l2':metrics['val']['mean']['relative_l2'],
                    'test_relative_l2':metrics['test']['mean']['relative_l2'],
                    'test_relative_h1_seminorm':metrics['test']['mean']['relative_h1_seminorm'],
                    'boundary_max':metrics['test']['max']['boundary_max'],
                    'loop_wall_seconds':metrics['loop_wall_seconds']})
    figures=output/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,name in zip(axes.flat,names):
        steps=np.array([r['step'] for r in histories[name][0]])
        for split,style in (('train','--'),('val','-')):
            values=np.array([[r[split]['relative_l2'] for r in hist] for hist in histories[name]])
            mean=values.mean(axis=0); low=values.min(axis=0); high=values.max(axis=0)
            ax.plot(steps,mean,style,label=split,color=COLORS[name],linewidth=1.8)
            ax.fill_between(steps,low,high,color=COLORS[name],alpha=.13)
        axis_unit = 'Epochs' if (config['training'].get('legacy_full_epochs') is not None
                                or config['training'].get('epoch_training') is not None) else 'Optimizer updates'
        ax.set(xlabel=axis_unit,ylabel='Mean relative P1 L2',title=DISPLAY[name],yscale='log')
        ax.legend(frameon=False)
    fig.suptitle('Fixed-budget training: mean across seeds; shade = seed range')
    save_figure(fig,figures,'convergence')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,metric,title in zip(axes,('relative_l2','relative_h1_seminorm'),('Relative L2','Relative H1 seminorm')):
        means=[aggregate[n]['test'][metric]['mean'] for n in names]
        std=[aggregate[n]['test'][metric]['std_across_seeds'] for n in names]
        for index,name in enumerate(names):
            ax.errorbar(index,means[index],yerr=std[index],fmt='o',capsize=4,
                        color=COLORS[name],markersize=7)
        ax.set_xticks(np.arange(len(names)),[DISPLAY[n] for n in names])
        ax.axhline(1,color='#777',linestyle=':',label='zero prediction = 1')
        ax.set(ylabel=title,title='Held-out test: '+title,yscale='log')
        ax.tick_params(axis='x',rotation=15);ax.legend(frameon=False)
    save_figure(fig,figures,'test_errors')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    x=np.arange(len(names));width=.36
    for ax,metric,title in zip(axes,('relative_l2','relative_h1_seminorm'),('Relative L2','Relative H1 seminorm')):
        for shift,kind,color,label in ((-.5,'direct_fine','#2767a1','direct fine'),(.5,'interpolated_coarse','#999999','coarse P1 interpolation')):
            rows=[aggregate[n]['resolution'][kind][metric] for n in names]
            ax.bar(x+shift*width,[r['mean'] for r in rows],width,
                   yerr=[r['std_across_seeds'] for r in rows],capsize=3,color=color,label=label)
        ax.set_xticks(x,[DISPLAY[n] for n in names],rotation=15)
        ax.set(ylabel=title,title='129x129 fixed-checkpoint evaluation');ax.legend(frameon=False)
    save_figure(fig,figures,'resolution')
    fig,axes=plt.subplots(1,3,figsize=(11,3.3),constrained_layout=True)
    for ax,key,label in zip(axes,('a','f','y'),('Diffusion a','Source f','Reference public-grid y')):
        values=arrays[key][0].T
        options={} if key=='a' else {'vmin':-float(np.max(np.abs(values))),
                                    'vmax':float(np.max(np.abs(values)))}
        im=ax.imshow(values,origin='lower',extent=(0,1,0,1),
                     cmap='viridis' if key=='a' else 'RdBu_r',**options)
        ax.set(title=label,xlabel='x',ylabel='y');ax.grid(False);fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Fixed test ID: '+str(arrays['sample_id'][0]))
    save_figure(fig,figures,'sample_inputs')
    fig,axes=plt.subplots(len(names),2,figsize=(8,3*len(names)),constrained_layout=True)
    target=arrays['y'][0]
    limit=max(float(np.max(np.abs(target))),1e-12)
    errors=[];predictions=[]
    for name in names:
        with np.load(output/f'seed_{seeds[0]}'/name/'test_predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['sample_id'],arrays['sample_id']):raise ValueError('ID mismatch')
            prediction=z['y'][0].copy()
        predictions.append(prediction);errors.append(prediction-target)
    limit=max(limit,float(np.max(np.abs(predictions))))
    error_limit=max(float(np.max(np.abs(errors))),1e-12)
    for row,(name,prediction,error) in enumerate(zip(names,predictions,errors)):
        for col,values,title,vmax in ((0,prediction,DISPLAY[name]+' prediction',limit),
                                     (1,error,DISPLAY[name]+' signed error',error_limit)):
            ax=axes[row,col]
            im=ax.imshow(values.T,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-vmax,vmax=vmax)
            ax.set(title=title,xlabel='x',ylabel='y');ax.grid(False);fig.colorbar(im,ax=ax,shrink=.85)
    fig.suptitle('Fixed seed '+str(seeds[0])+', fixed first test ID; common scales within each column')
    save_figure(fig,figures,'sample_predictions')
    full = config['training'].get('legacy_full_epochs') is not None
    adapted = config['training'].get('epoch_training') is not None
    training_description = (f'每组完整遍历训练集{config["training"]["steps"]}个epoch；逻辑batch由模型配置确定，计算batch={config["training"]["batch_size"]}。' if full else f'每组{config["training"]["steps"]}次Adam更新，batch={config["training"]["batch_size"]}。')
    if adapted:
        training_description = (f'每组{config["training"]["steps"]}个epoch；每epoch按逻辑batch分批更新，显存microbatch={config["training"]["batch_size"]}；'
                                '实际更新次数见各模型metrics.json。验证平均逐样本相对P1 L2选模。')
    lines=['# 第三章3.1–3.3 固定预算对照结果','',
           f'已完成四模型、{len(seeds)}个种子、共{len(names)*len(seeds)}组训练。{training_description}',
           f'数据train/val/test={train_count}/{records[names[0]][0]["metrics"]["val"]["count"]}/{len(arrays["sample_id"])}，公共网格{geometry.grid_shape}。测试前全部检查点已冻结，见frozen_selection.json。','',
           '数值为各训练种子的均值 ± 样本标准差；标准差只描述训练随机性，不是数据集置信区间。','',
           '| 模型 | 参数数 | 测试相对L2 | 测试相对H1半范数 | 训练墙钟秒 | GPU/CPU推理ms/PDE |',
           '|---|---:|---:|---:|---:|---:|']
    for name in names:
        row=aggregate[name]; l2=row['test']['relative_l2'];h1=row['test']['relative_h1_seminorm'];wall=row['loop_wall_seconds']
        lines.append(f'| {DISPLAY[name]} | {row["parameters"]} | {l2["mean"]:.4e} ± {l2["std_across_seeds"]:.2e} | '
                     f'{h1["mean"]:.4e} ± {h1["std_across_seeds"]:.2e} | {wall["mean"]:.2f} | '
                     f'{1000*row["inference_seed0"]["median_seconds_per_pde"]:.3f} |')
    lines+=['','各模型使用同一原始数据；损失、实际更新次数、参数数及输入采样见配置和metrics.json，结果只比较这些配置。',
            '训练时间含验证/度量/检查点I/O，推理时间在预载入张量上batch8测量（3次warmup、10次计时），不含传输、加载和参考求解。','',
            '| 模型 | 129×129直接相对L2 | 同一粗预测P1插值相对L2 |', '|---|---:|---:|']
    for name in names:
        r=aggregate[name]['resolution']
        lines.append(f'| {DISPLAY[name]} | {r["direct_fine"]["relative_l2"]["mean"]:.5e} | {r["interpolated_coarse"]["relative_l2"]["mean"]:.5e} |')
    lines+=['','加密评估保持检查点不变；坐标查询或共享参数不保证分辨率精度。较细标签仍有有限元误差。',
            f'零预测相对L2={baseline["zero"]["mean"]["relative_l2"]:.3f}；只用训练集均值的固定状态相对L2={baseline["mean_training_state"]["mean"]["relative_l2"]:.3f}。',
            '边界最大误差、逐样本指标、训练/验证指标和完整计时定义见aggregate.json及各seed目录。','',
            '图表：figures/convergence、test_errors、resolution、sample_inputs、sample_predictions，均含PNG与矢量PDF。',
            '展示固定seed0及第一个测试ID，未挑选最好样本。','',
            '本轮是平滑余弦分布上的固定配置对照。有限数据、有限种子与固定更新预算不足以支持架构家族的通用优劣结论，',
            '也不覆盖更高对比系数、非平滑输入、分布外测试、物理损失或第三章3.4以后的方法。','']
    if full:
        lines += ['', '本轮参照ocpde_dl的1600 epoch完整训练。原测试集在首轮已查看，本轮为复测，不是新的盲测。',
                  'GNO/Transformer在旧包中无对应训练脚本，采用DeepONet的优化方案；方程和架构适配见本轮方案。', '',
                  '| 模型 | 优化器 | 初始LR | 名义batch | 调度 |', '|---|---|---:|---:|---|']
        for spec in config['training']['models']:
            p=spec['legacy_training']
            lines.append(f'| {spec["name"]} | {p["optimizer"]} | {p["learning_rate"]} | {p["batch_size"]} | {p["scheduler"]} |')
    (output/'summary.md').write_text('\n'.join(lines))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    build_report(args.data,args.output,args.device)
