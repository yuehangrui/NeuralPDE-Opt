"""Label-free inference for every retained checkpoint.

Book/code notation:
Predictors return physical PDE state y_theta without reading labels. beta_coefficients are reduced output coordinates; load_coordinates are z_f. Historical serialized keys remain unchanged.
"""

import numpy as np
import torch
from bootstrap import ROOT
from integrity import read, require
from neuralpde_opt.chapter3 import reaction_training as reaction
from neuralpde_opt.chapter3.reaction_operators import make_reaction_model
from pod import CoefficientNet, encode_inputs, reconstruct
from model import model_for
from model_v2 import MatrixMap


def array_file(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def build(row, device="cpu"):
    path = ROOT / row["path"]
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    family = row["family"]
    if family == "baseline":
        return reaction.load_model(path, device), None
    if family == "physics":
        model = make_reaction_model(
            saved["identity"]["config"]["model"], saved["normalization"], 0.01
        )
        auxiliary = None
    elif family == "pod":
        config = saved["identity"]["config"]
        rank = int(row["variant"].split("_")[1])
        model = CoefficientNet(289, rank, config["width"], config["depth"])
        auxiliary = (
            array_file(ROOT / "results/ch03/pod_v1/basis.npz"),
            read(ROOT / "results/ch03/pod_v1/normalization.json"),
            rank,
        )
    else:
        basis = array_file(ROOT / "results/ch03/varmion_v1/basis.npz")
        norm = read(ROOT / "results/ch03/varmion_v1/normalization.json")
        require(
            np.isfinite([norm["c_mean"], norm["c_std"]]).all() and norm["c_std"] > 0,
            "Invalid normalization",
        )
        if family == "varmion_v2":
            kind = "spd" if row["variant"].startswith("spd") else "dense"
            model = MatrixMap(kind, norm["c_mean"], norm["c_std"], basis["k0"]).float()
        else:
            model = model_for("varmion" if family == "hints" else row["variant"], norm, basis["k0"])
        auxiliary = basis["basis"]
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), auxiliary


@torch.no_grad()
def predict(row, data, mass, device="cpu", batch_size=4):
    c = np.asarray(data["c"])
    require(
        c.ndim == 3 and c.shape[1:] == (65, 65) and len(c) and np.isfinite(c).all() and c.min() > 0,
        "Invalid inference coefficients",
    )
    model, aux = build(row, device)
    family = row["family"]
    if family in {"varmion", "varmion_v2", "hints"}:
        require(
            "f" in data and data["f"].shape == c.shape and np.isfinite(data["f"]).all(),
            "Missing/invalid source",
        )
        load_coordinates = data["f"].reshape(len(c), -1) @ (mass @ aux)
    result = []
    for start in range(0, len(c), batch_size):
        values = c[start : start + batch_size]
        if family in {"baseline", "physics"}:
            y_prediction = (
                model(torch.as_tensor(values, dtype=torch.float32, device=device))
                .double()
                .cpu()
                .numpy()
            )
        elif family == "pod":
            basis, norm, rank = aux
            x = torch.as_tensor(encode_inputs(values, norm, 4), dtype=torch.float32, device=device)
            beta_coefficients = model(x).double().cpu().numpy() * np.sqrt(
                basis["eigenvalues"][:rank]
            )
            y_prediction = reconstruct(beta_coefficients, basis, rank)
        else:
            sensors = (
                values[:, ::4, ::4].reshape(len(values), -1) if family == "varmion_v2" else values
            )
            beta_coefficients = (
                model(
                    torch.as_tensor(sensors, dtype=torch.float32, device=device),
                    torch.as_tensor(
                        load_coordinates[start : start + batch_size],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                .double()
                .cpu()
                .numpy()
            )
            y_prediction = (beta_coefficients @ aux.T).reshape(-1, 65, 65)
        require(np.isfinite(y_prediction).all(), "Nonfinite inference")
        result.append(y_prediction)
    return np.concatenate(result)
