# Response 4 — A6/A7 implementation

Added Stage-A continuation code after Milestone 1:

- Phase A6 baseline comparison:
  - raw resized image baseline
  - random projection baseline
  - extracted VAE conditioning latent baseline
  - selected Instruct layer baselines
  - final UNet prediction baseline
  - ArcFace correlation placeholder if ArcFace embeddings are unavailable

- Phase A7 gradient sanity scan:
  - short 5-step gradient probe only
  - uses top identity-ranked layers by default
  - measures whether gradients reach TPS, Delaunay, rolling, DCT, and FFT phase parameter blocks
  - records `Z`, `loss`, gradient norms, PSNR, SSIM, displacement, foldover, and clamp diagnostics

This is not Stage B and not a full attack. It is only the gate that determines which layers are worth targeting later.

Run commands are in:

```text
instructions/a6000_phase_a6_a7_commands.md
```
