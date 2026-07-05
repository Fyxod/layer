# Response 7 — Stage B smoke implementation

Implemented the first Stage-B targeted geometry smoke.

## What was added

```text
layer/identity_layers/stage_b_attack.py
layer/scripts/run_stage_b_smoke.py
layer/scripts/summarize_stage_b.py
identity_layers/configs/stage_b_smoke.json
instructions/a6000_stage_b_next_commands.md
```

## What this run does

The smoke targets only:

```text
vae_image_latent
unet.conv_in
```

using prompts:

```text
add black sunglasses
add headphones
```

Default smoke size:

```text
max_cases = 3
iterations = 50
starts = 1
execution = sequential
```

The scalar objective is:

```text
Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))
loss = -Z
```

Only differentiable geometry parameters are optimized.

## Geometry components

The smoke reuses the A7/WOOD-style geometry block:

```text
TPS
Delaunay / piecewise
rolling shutter
DCT low-frequency warp
FFT phase
```

All active components are optimized together in one backward pass.

## What is intentionally not included

```text
pixel noise
adversarial patches
finetuning
LoRA
model-weight training
SPSA/CEM/black-box search
landmarks
face alignment / face detection
visual counter-loss
```

## Main command file

Run commands are in:

```text
instructions/a6000_stage_b_next_commands.md
```

## Important gate

This is not the 150-iteration main attack. It is the Gate-4 short optimization smoke.

Proceed to 150 iterations only if:

```text
Z increases
gradients stay finite
input PSNR/SSIM remain usable
geometry does not show severe foldover/clamp saturation
```

If both `vae_image_latent` and `unet.conv_in` fail, revisit the ArcFace-correlated deeper layers with a modified gradient probe.
