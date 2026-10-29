# Chapter 3 upstream source notices

These are isolated experimental ports, not replacements for the frozen
Chapter 2 code or the original Chapter 3 teaching models.

DeepXDE: Lu Lu and the DeepXDE contributors, repository lululxvi/deepxde,
commit 99b6620386d18cefb1549dddb7b7fe468cfad607.
https://github.com/lululxvi/deepxde/tree/99b6620386d18cefb1549dddb7b7fe468cfad607
The extracted source in upstream_vendor/deepxde_core.py retains LGPL-2.1.
Original source and full license are included under deepxde/ here. Modified
2026-10-21: extracted original class bodies and replaced the multi-backend
registries with a small, explicit PyTorch-only compatibility shim. All source
needed to modify/rebuild this component is supplied; it is not relicensed as MIT.
DeepONet reference: https://doi.org/10.1038/s42256-021-00302-5

Galerkin Transformer: Shuhao Cao, scaomath/galerkin-transformer,
commit f9e7d6abc24e958b1165f48b22625c5bab688043.
https://github.com/scaomath/galerkin-transformer/tree/f9e7d6abc24e958b1165f48b22625c5bab688043
MIT license retained under galerkin-transformer/LICENSE. Original source and
configuration included. Modified 2026-10-21: remove unrelated training/plotting
imports; call supplied Dropout module instead of unconditional F.dropout;
allocate the SpectralConv2d complex-pair buffer with the input dtype.
Paper: https://arxiv.org/abs/2105.14995

SOURCE_MANIFEST.json records exact original and ported source checksums.
build_ports.py reproduces the extracted core files from these originals into
a fresh patch/ directory. It does not run upstream training scripts.

Adapters and train/val protocol are project-specific. DeepXDE branch uses ReLU
and trunk uses tanh; 17x17 sensors, 300-wide three-hidden-layer nets, rank 56.
An optional 4-mode Fourier coordinate adapter is a separately named ablation.
Galerkin retains author Darcy width 128, six layers, four heads, two spectral
layers (width 32, modes 12), CNN scalers and 0.05 dropout. Resolution scalers
are adapted to 65->33->17 and 17->33->65, and node input channels to a,f.

Both use the project's train-only field normalization and relative consistent
P1 L2 objective. The bounded validation uses AdamW 5e-4, weight decay 1e-5,
clip 1, logical batch 16/microbatch 4, cosine decay over 100 epochs, seed 0.
This matches the project's short adaptation budget, not the authors' full
published training recipes/datasets. Galerkin's boundary clamp differs from
the teaching model's polynomial boundary multiplier. Do not infer a pure
attention effect or architecture-family ranking from this combined change.
