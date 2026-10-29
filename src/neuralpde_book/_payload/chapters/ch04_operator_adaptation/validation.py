"""Independent numerical audit plus matched-space neural online controls."""

from pathlib import Path
import argparse, hashlib, json, math, subprocess, sys, time
import numpy as np
import torch
from chapters.ch04_operator_learning.data import load as load_data, arrays
from chapters.ch04_operator_learning.training import metrics, write
from chapters.ch04_operator_learning.reference import Poisson
from chapters.ch04_operator_learning.online_supported import SupportedControl
from chapters.ch04.problem import Problem
from .training import load, predict, identity

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    if not np.isclose(a, b, atol=1e-10, rtol=1e-8):
        raise AssertionError((a, b))


def run(result, data):
    torch.set_num_threads(2)
    root = Path(result)
    training = root / "training"
    out = root / "validation"
    assert (training / "TRAINING_COMPLETE").exists()
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    tr, va, manifest = load_data(data)
    ref = Poisson()
    problem = Problem()
    points = ref.points.reshape(-1, 2)
    original = {
        k: getattr(problem, k)(points).reshape(33, 33) * ref.mask
        for k in ("f", "yd", "y", "u", "p")
    }
    np.testing.assert_allclose(
        ref.solve(original["f"] + original["u"]), original["y"], atol=1e-13, rtol=1e-12
    )
    report = dict(
        status="running",
        models={},
        online={},
        test_evaluations=0,
        environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__),
    )

    def status(stage, **kwargs):
        write(out / "status.json", dict(stage=stage, seconds=time.perf_counter() - start, **kwargs))

    for architecture in ("fno", "deeponet", "fedonet"):
        for role in ("state", "kkt"):
            name = architecture + "_" + role
            folder = training / name
            status("offline_audit", model=name)
            neural, cfg, ck = load(folder / "neural_best.pt")
            head, _, hc = load(folder / "best.pt")
            last, _, lc = load(folder / "last.pt")
            info = read(folder / "run.json")
            hist = [json.loads(x) for x in (folder / "history.jsonl").read_text().splitlines()]
            selected = min(hist, key=lambda row: row["validation_score"])
            assert (
                selected == read(folder / "neural_metrics.json")
                and ck["epoch"] == selected["epoch"]
            )
            assert (
                not bool(neural.head_active)
                and bool(head.head_active)
                and not bool(last.head_active)
            )
            assert (
                info["status"] == "completed"
                and info["full_budget_completed"]
                and cfg.epochs == 1600
            )
            batches = math.ceil(len(tr["ids"]) / cfg.batch)
            expected = 1600 * batches
            assert (
                info["updates"] == lc["updates"] == hist[-1]["updates"] == expected
                and lc["epoch"] == 1600
            )
            expected_tangent = (
                max(0, 1600 - cfg.tangent_warmup) * batches if cfg.tangent_weight else 0
            )
            assert info["tangent_updates"] == expected_tangent
            assert lc["scheduler"]["last_epoch"] == 1600 and len(hist) == 1601
            assert all(h["epoch"] == i and h["updates"] == i * batches for i, h in enumerate(hist))
            assert ck["data_hashes"] == hc["data_hashes"] == manifest["hashes"]
            assert (
                info["source_identity"]
                == identity()
                == read(root / "protocol.json")["source_identity"]
            )
            assert all(
                torch.equal(p, dict(neural.named_parameters())[n])
                for n, p in head.named_parameters()
            )
            tx, ty = arrays(tr, role)
            vx, vy = arrays(va, role)
            xt = torch.tensor(tx)
            yt = torch.tensor(ty)
            c = neural.inputs(xt)
            t = neural.encode_fields(yt)
            for attr, value in (
                ("input_mean", c.mean(0)),
                ("input_scale", c.std(0, unbiased=False)),
                ("output_mean", t.mean(0)),
                ("output_scale", t.std(0, unbiased=False)),
            ):
                torch.testing.assert_close(getattr(neural, attr), value, rtol=1e-11, atol=1e-12)
            variants = {}
            for variant, model, prefix in (
                ("neural", neural, "neural_"),
                ("head_polished", head, ""),
            ):
                saved = read(folder / (prefix + "metrics.json"))
                with np.load(folder / (prefix + "validation_prediction.npz")) as archive:
                    assert np.array_equal(archive["ids"], va["ids"])
                    pred = archive["prediction"]
                fresh = metrics(pred, vy, vx, role, ref)
                for k, v in fresh.items():
                    close(v, saved[k])
                errors = np.sqrt(
                    ((pred - vy)[:, :, 1:-1, 1:-1] ** 2).sum((2, 3))
                    / (vy[:, :, 1:-1, 1:-1] ** 2).sum((2, 3))
                ).mean(0)
                close(errors[0], fresh["state_relative_l2_mean"])
                if role == "kkt":
                    close(errors[1], fresh["control_relative_l2_mean"])
                again = predict(model, vx)
                np.testing.assert_allclose(again, pred, atol=1e-10, rtol=1e-9)
                zeros = torch.zeros_like(torch.tensor(vx[:1]))
                zero = float(model(zeros).detach().abs().max())
                a = torch.tensor(vx[:2])
                b = torch.tensor(vx[2:4])
                defect = float(
                    (model(a + b) - model(a) - model(b)).detach().norm()
                    / model(a + b).detach().norm()
                )
                variants[variant] = dict(
                    metrics=fresh,
                    train_metrics=read(folder / (prefix + "train_metrics.json")),
                    cpu_reload_max_abs=float(abs(again - pred).max()),
                    zero_response_max=zero,
                    additivity_relative_defect=defect,
                )
                ox = np.stack([original["f"], original["u" if role == "state" else "yd"]])[None]
                oy = (
                    original["y"][None, None]
                    if role == "state"
                    else np.stack([original["y"], original["u"]])[None]
                )
                variants[variant]["original_pinn_case"] = metrics(
                    predict(model, ox), oy, ox, role, ref
                )
                if variant == "head_polished":
                    assert zero < 1e-10 and defect < 1e-8
            # CPU refit independently uses train only, never validation targets.
            neural.fit_head(xt, yt)
            repred = predict(neural, vx)
            np.testing.assert_allclose(repred, predict(head, vx), atol=1e-9, rtol=1e-8)
            report["models"][name] = dict(
                variants=variants,
                epochs=1600,
                updates=expected,
                best_neural_epoch=ck["epoch"],
                parameter_count=info["parameter_count"],
                head_fit=read(folder / "head_fit.json"),
                frozen_backbone_verified=True,
                train_normalization_verified=True,
                train_only_head_refit_verified=True,
            )
            print(
                "AUDITED",
                name,
                variants["neural"]["metrics"]["validation_score"],
                variants["head_polished"]["metrics"]["validation_score"],
                flush=True,
            )
    write(out / "offline_audit.json", report["models"])
    for architecture in ("fno", "deeponet", "fedonet"):
        for variant, ckname in (("neural", "neural_best.pt"), ("head_polished", "best.pt")):
            key = architecture + "_" + variant
            folder = out / key
            folder.mkdir()
            ckpath = training / (architecture + "_state") / ckname
            model, cfg, ck = load(ckpath, "cuda")
            model.eval().requires_grad_(False)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            projector = SupportedControl(ref)
            project = lambda u: projector.project(torch.tensor(u)).detach().numpy()
            derivative = []
            rng = np.random.default_rng(944)
            for i in range(8):
                x = torch.tensor(
                    arrays(va, "state")[0][i : i + 1], device="cuda", requires_grad=True
                )
                w = project(ref.solve(rng.normal(size=ref.mask.shape)))
                w /= ref.norm(w)
                value = (
                    model(x, check=False)[0, 0] * torch.tensor(w, device="cuda")
                ).sum() / ref.n**2
                g = torch.autograd.grad(value, x)[0][0, 1].detach().cpu().numpy()
                truth = ref.solve(w) / ref.n**2
                derivative.append(float(ref.norm(project(g) - truth) / ref.norm(truth)))
            rows = []
            controls = {}
            for i in range(8):
                status("online", variant=key, task=i + 1)
                torch.manual_seed(444 + i)
                control = SupportedControl(ref).cuda()
                f = torch.tensor(va["f"][i], device="cuda")
                yd = torch.tensor(va["yd"][i], device="cuda")
                trace = []
                _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(va["f"][i]))
                _, _, _, jstar = ref.recover(va["f"][i], va["yd"][i], va["u"][i])

                def objective():
                    u = control()
                    y = model(torch.stack([f, u])[None], check=False)[0, 0]
                    return 0.5 * ((y - yd) ** 2 + ref.alpha * u * u).sum() / ref.n**2

                def record(stage):
                    with torch.no_grad():
                        ut = control()
                        u = ut.cpu().numpy()
                        yp = model(torch.stack([f, ut])[None], check=False)[0, 0].cpu().numpy()
                    y, p, g, j = ref.recover(va["f"][i], va["yd"][i], u)
                    trace.append(
                        dict(
                            stage=stage,
                            J_true=float(j),
                            J_surrogate=float(
                                0.5 * ref.inner(yp - va["yd"][i], yp - va["yd"][i])
                                + 0.5 * ref.alpha * ref.inner(u, u)
                            ),
                            control_relative_l2=float(
                                ref.norm(u - va["u"][i]) / ref.norm(va["u"][i])
                            ),
                            true_gradient_relative=float(ref.norm(g) / ref.norm(g0)),
                            objective_gap=float(j - jstar),
                            support_projection_error=float(ref.norm(u - project(u))),
                        )
                    )

                def backward(opt):
                    opt.zero_grad(set_to_none=True)
                    loss = objective()
                    loss.backward()
                    if not bool(torch.isfinite(loss)) or any(
                        p.grad is not None and not bool(torch.isfinite(p.grad).all())
                        for p in control.parameters()
                    ):
                        raise FloatingPointError("nonfinite online loss")
                    return loss

                record("initial")
                opt = torch.optim.Adam(control.parameters(), lr=0.01)
                for step in range(1, 101):
                    backward(opt)
                    opt.step()
                    if step % 20 == 0:
                        record("adam:" + str(step))
                opt = torch.optim.LBFGS(
                    control.parameters(),
                    lr=1.0,
                    max_iter=300,
                    tolerance_grad=1e-12,
                    tolerance_change=1e-15,
                    history_size=50,
                    line_search_fn="strong_wolfe",
                )
                opt.step(lambda: backward(opt))
                record("lbfgs:last")
                state = opt.state[next(control.parameters())]
                controls[str(i)] = control().detach().cpu().numpy()
                rows.append(
                    dict(
                        id=str(va["ids"][i]),
                        trajectory=trace,
                        adam_steps=100,
                        lbfgs_max_iter=300,
                        lbfgs_iterations=int(state["n_iter"]),
                        lbfgs_function_evaluations=int(state["func_evals"]),
                    )
                )
                torch.save(control.cpu().state_dict(), folder / ("control_" + str(i) + ".pt"))
            assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
            np.savez_compressed(folder / "controls.npz", **controls)
            for i, row in enumerate(rows):
                net = SupportedControl(ref)
                net.load_state_dict(
                    torch.load(
                        folder / ("control_" + str(i) + ".pt"),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                u = controls[str(i)]
                np.testing.assert_allclose(net().detach().numpy(), u, atol=1e-10, rtol=1e-9)
                np.testing.assert_allclose(project(u), u, atol=1e-10, rtol=1e-9)
                _, _, g, j = ref.recover(va["f"][i], va["yd"][i], u)
                _, _, g0, _ = ref.recover(va["f"][i], va["yd"][i], np.zeros_like(u))
                _, _, _, jstar = ref.recover(va["f"][i], va["yd"][i], va["u"][i])
                last = row["trajectory"][-1]
                close(j, last["J_true"])
                close(ref.norm(g) / ref.norm(g0), last["true_gradient_relative"])
                close(ref.norm(u - va["u"][i]) / ref.norm(va["u"][i]), last["control_relative_l2"])
                assert j >= jstar - 1e-12
            values = [r["trajectory"][-1]["control_relative_l2"] for r in rows]
            info = dict(
                variant=variant,
                checkpoint_sha256=sha(ckpath),
                fixed_surrogate_verified=True,
                role="validation",
                selection_use=False,
                supported_adjoint_action_relative=derivative,
                rows=rows,
                test_evaluations=0,
            )
            write(folder / "online.json", info)
            report["online"][key] = dict(
                mean_control_relative_l2=float(np.mean(values)),
                max_control_relative_l2=float(max(values)),
                true_gradient_relative_mean=float(
                    np.mean([r["trajectory"][-1]["true_gradient_relative"] for r in rows])
                ),
                derivative_relative_mean=float(np.mean(derivative)),
                lbfgs_at_limit=sum(r["lbfgs_iterations"] == 300 for r in rows),
                count=8,
            )
            print("ONLINE", key, report["online"][key], flush=True)
            del model, before
            torch.cuda.empty_cache()
    with (out / "tests.log").open("w") as log:
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "unittest",
                "chapters.ch04_operator_adaptation.test_adaptation",
                "chapters.ch04_operator_learning.test_operators",
                "-v",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    assert p.returncode == 0 and "Ran 19 tests" in (out / "tests.log").read_text()
    protected = {}
    project = Path.cwd()
    for name in ("ch04", "ch04_adaptation", "ch04_operator_learning", "ch05"):
        p = project / "chapters" / name
        m = read(p / "RUNTIME_MANIFEST.json")
        for file, h in m.items():
            assert sha(p / file) == h
        protected[name] = len(m)
    for name in (
        "full_training_v1",
        "adaptation_v1",
        "operator_learning_v1",
        "operator_full_training_v1",
    ):
        m = read(project / "results/ch04" / name / "ARTIFACT_MANIFEST.json")
        for file, h in m.items():
            assert sha(project / file) == h
        protected[name] = len(m)
    report.update(
        status="passed",
        protected=protected,
        tests=19,
        online_controls=48,
        derivative_inputs=48,
        seconds=time.perf_counter() - start,
    )
    write(out / "audit.json", report)
    status("complete")
    (out / "VALIDATION_COMPLETE").write_text(
        "Six full neural runs and six train-only fitted heads; 48 online controls; 19 tests; no test evaluation.\n"
    )
    queue = read(training / "queue_status.json")
    queue["stage"] = "complete"
    write(training / "queue_status.json", queue)
    print("ADAPTATION_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--data", required=True)
    a = p.parse_args()
    run(a.result, a.data)
