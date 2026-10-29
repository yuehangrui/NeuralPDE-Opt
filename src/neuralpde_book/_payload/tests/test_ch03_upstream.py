import ast, copy, importlib.util
from pathlib import Path
import numpy as np
import unittest
import itertools

def parametrize(name, values):
    def decorate(fn):
        fn.cases = getattr(fn, "cases", []) + [(name, values)]
        return fn
    return decorate
import torch
from neuralpde_opt.chapter3.upstream_operators import DeepXDEOperator, GalerkinOperator
from neuralpde_opt.chapter3.upstream_vendor import galerkin_layers as g
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

@parametrize('modes',[0,4])
def test_deepxde_adapter_matches_author_dot_and_gradients(modes):
    torch.manual_seed(3)
    m=DeepXDEOperator(width=20,depth=2,latent_dim=8,coordinate_modes=modes).double()
    a=torch.randn(2,65,65,dtype=torch.float64);f=torch.randn_like(a)
    branch,coord,bc=m.core_inputs(a,f)
    literal=(m.core.branch(branch)@torch.tanh(m.core.trunk(coord)).T+m.core.b[0])*bc
    expected=literal.reshape(2,65,65)
    actual=m(a,f)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    v=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*v).sum(),tuple(m.parameters()))
    ge=torch.autograd.grad((expected*v).sum(),tuple(m.parameters()))
    for x,y in zip(ga,ge):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11)
    # Flatten layout: all a, then all f; exact axis ordering and endpoints.
    x=m.grid[...,0].expand(2,-1,-1);y=m.grid[...,1].expand_as(x)
    br,_,_=m.core_inputs(x,y)
    torch.testing.assert_close(br[:,:289],x[:,::4,::4].flatten(1))
    torch.testing.assert_close(br[:,289:],y[:,::4,::4].flatten(1))

@parametrize('fn',[g.attention,g.linear_attention])
def test_attention_eval_is_deterministic_and_honors_probability(fn):
    torch.manual_seed(5)
    q,k,v=[torch.randn(2,2,9,4) for _ in range(3)]
    d=torch.nn.Dropout(.05).eval()
    x,_=fn(q,k,v,dropout=d);y,_=fn(q,k,v,dropout=d)
    ref,_=fn(q,k,v,dropout=None)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(x,ref,rtol=0,atol=0)
    d.train();torch.manual_seed(50);z,_=fn(q,k,v,dropout=d)
    torch.manual_seed(50)
    if fn is g.linear_attention:
        expected=q@torch.nn.functional.dropout(k.transpose(-2,-1)@v/9,p=.05)
    else:
        expected=torch.nn.functional.dropout((q@k.transpose(-2,-1)/2).softmax(-1),p=.05)@v
    torch.testing.assert_close(z,expected,rtol=0,atol=0)


def test_galerkin_attention_matches_original_when_dropout_disabled():
    # Original layers has no training side effects. Do not execute its model/train scripts.
    source=(ROOT/'third_party/ch03_upstream/galerkin-transformer/libs/layers.py').read_text()
    ns={};exec(compile(source,'upstream_layers_reference','exec'),ns)
    original_fn=ns['linear_attention']
    def no_dropout(*args,**kwargs):
        kwargs['dropout']=None
        return original_fn(*args,**kwargs)
    ns['linear_attention']=no_dropout
    original=ns['SimpleAttention'](n_head=2,d_model=12,pos_dim=2,attention_type='galerkin',norm=True).double().eval()
    port=g.SimpleAttention(n_head=2,d_model=12,pos_dim=2,attention_type='galerkin',norm=True).double().eval()
    port.load_state_dict(original.state_dict())
    x=torch.randn(2,9,12,dtype=torch.float64,requires_grad=True);pos=torch.rand(2,9,2,dtype=torch.float64)
    a=original(x,x,x,pos)[0];b=port(x,x,x,pos)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga=torch.autograd.grad(a.square().sum(),(x,*original.parameters()))
    gb=torch.autograd.grad(b.square().sum(),(x,*port.parameters()))
    for p,q in zip(ga,gb):torch.testing.assert_close(p,q,rtol=0,atol=0)

@parametrize('kind',['deepxde','galerkin'])
@parametrize('device',['cpu','cuda'])
def test_adapter_forward_backward_boundary_batch_reload(kind,device):
    if device=='cuda' and not torch.cuda.is_available():raise unittest.SkipTest('No CUDA')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    m=(DeepXDEOperator(width=20,depth=2,latent_dim=8) if kind=='deepxde' else
       GalerkinOperator(width=12,layers=1,modes=4,dropout=.05)).to(device).eval()
    a=torch.randn(2,65,65,device=device);f=torch.randn_like(a)
    p=m(a,f);again=m(a,f)
    torch.testing.assert_close(p,again,rtol=0,atol=0)
    solo=torch.cat([m(a[i:i+1],f[i:i+1]) for i in range(2)])
    torch.testing.assert_close(p,solo,rtol=2e-5,atol=2e-6)
    assert p.shape==a.shape and torch.isfinite(p).all()
    assert torch.count_nonzero(p[:,[0,-1],:])==0
    assert torch.count_nonzero(p[:,:,[0,-1]])==0
    with torch.no_grad():
        assert (m(a+.1,f)-p).abs().max()>1e-8
        assert (m(a,f+.1)-p).abs().max()>1e-8
    p.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m2=copy.deepcopy(m);m2.load_state_dict(m.state_dict())
    torch.testing.assert_close(m2(a,f),p.detach(),rtol=0,atol=0)
    if kind=='galerkin':torch.testing.assert_close(p,m.core(**m.core_inputs(a,f))['preds'].squeeze(-1))

@parametrize('bad',['shape','dtype','nan','empty'])
@parametrize('kind',['deepxde','galerkin'])
def test_reject_malformed_inputs(kind,bad):
    m=DeepXDEOperator(width=10,depth=1) if kind=='deepxde' else GalerkinOperator(width=12,layers=1,modes=4)
    a=torch.randn(1,65,65);f=a.clone()
    if bad=='shape':f=f[:,:-1]
    if bad=='dtype':f=f.double()
    if bad=='nan':a[0,1,1]=float('nan')
    if bad=='empty':a=a[:0];f=f[:0]
    with unittest.TestCase().assertRaises((ValueError,TypeError)):m(a,f)


def test_author_class_bodies_preserved():
    def classes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.ClassDef)}
    core=classes(ROOT/'src/neuralpde_opt/chapter3/upstream_vendor/deepxde_core.py')
    for name in ['nn','fnn','deeponet']:
        ref=classes(ROOT/f'third_party/ch03_upstream/deepxde/deepxde/nn/pytorch/{name}.py')
        for k in core.keys() & ref.keys():assert core[k]==ref[k]
    ref=classes(ROOT/'third_party/ch03_upstream/deepxde/deepxde/nn/deeponet_strategy.py')
    for k in core.keys() & ref.keys():assert core[k]==ref[k]
    port=classes(ROOT/'src/neuralpde_opt/chapter3/upstream_vendor/galerkin_model.py')
    ref=classes(ROOT/'third_party/ch03_upstream/galerkin-transformer/libs/model.py')
    for k in port:assert port[k]==ref[k]


def test_spectral_layer_preserves_double_precision():
    layer=g.SpectralConv2d(3,4,modes=4,dropout=0.).double().eval()
    x=torch.randn(2,17,17,3,dtype=torch.float64,requires_grad=True)
    y=layer(x)
    assert y.dtype==torch.float64
    y.square().mean().backward()
    assert x.grad.dtype==torch.float64 and torch.isfinite(x.grad).all()


def test_data_guard_rejects_test_and_unknown_paths():
    spec=importlib.util.spec_from_file_location('upstream_runner',ROOT/'scripts/validate_ch03_upstream.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    original,opened=runner.guard_npz(ROOT/'data/ch03/experiment_v1')
    try:
        for name in ['test.npz','extra.npz','../experiment_v1/test.npz']:
            with unittest.TestCase().assertRaises(RuntimeError):
                np.load(ROOT/'data/ch03/experiment_v1'/name)
        assert opened==[]
    finally:np.load=original


class UpstreamPortTests(unittest.TestCase):
    pass

for name, fn in list(globals().items()):
    if name.startswith('test_') and callable(fn):
        cases = getattr(fn, 'cases', [])
        combos = itertools.product(*(v for k,v in cases)) if cases else [()]
        for i, values in enumerate(combos):
            kwargs = dict(zip((k for k,v in cases), values))
            def method(self, fn=fn, kwargs=kwargs): fn(**kwargs)
            setattr(UpstreamPortTests, name+'_'+str(i), method)

del fn  # Avoid exposing a second alias of TestCase to unittest discovery.

if __name__ == '__main__': unittest.main()
