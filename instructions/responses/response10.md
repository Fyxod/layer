# Response 10 — Broad 400-run diagnosis

The broad 400-iteration Stage-B run is useful diagnostically, but the top results are not valid final attacks.

## What happened

The top `unet.conv_in` runs maximized `Z` by destroying the input image.

Observed failure mode:

```text
input SSIM collapsed as low as ~0.01
PSNR fell to ~8–11 dB
max displacement hit the 8 px cap
FFT phase saturated at ±pi
perturbed images became green / phase-corrupted
```

This explains the green/destroyed perturbed images.

## Cause

The current Stage-B objective is:

```text
loss = -Z
```

with no visual counter-loss.

At 400 iterations, `unet.conv_in` found an easy destructive loophole:

```text
saturate FFT phase
push DCT/spatial displacement toward the hard cap
destroy low-level input appearance
increase conv_in representation distance
```

This is expected behavior for an unconstrained internal objective. It does not mean the attack succeeded.

## Conclusions

1. `unet.conv_in` is definitely attackable as an internal representation.
2. But unconstrained `unet.conv_in` is too easy to exploit destructively.
3. The best 400-iteration `unet.conv_in` runs are invalid because the perturbed inputs are not acceptable.
4. `unet.mid_block` and `unet.up_blocks.0` still did not move: they remain gradient-dead under this setup.
5. `vae_image_latent` remains more stable but too weak; its strongest run still had visible color artifacts and did not clearly break the edit.

## Best current read

The earlier 50-iteration `unet.conv_in / add headphones / barack_obama_01` result remains the most interesting point, because it weakened headphones before the image fully collapsed.

The 400-iteration run shows that simply running longer is not the answer.

## Next step

Run a constrained follow-up:

```text
unet.conv_in only
200 iterations
max_combined_disp_px = 4
FFT disabled first
lower spatial ranges
```

Commands:

```text
instructions/a6000_stage_b_constrained_commands.md
```

Optional second run:

```text
same setup, but tiny FFT phase limit = 0.35 rad
```

## Decision

If the constrained run preserves the headphone failure while keeping the input recognizable, continue with Stage-C identity/final-edit evaluation.

If the constrained run loses the effect, the current single-layer `unet.conv_in` objective is not enough by itself. Then the next move should be one of:

```text
best-valid checkpoint selection / early stopping by input SSIM
small preservation constraint
modified gradient probe for ArcFace-correlated deeper layers
```
