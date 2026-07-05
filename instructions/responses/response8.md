# Response 8 — Stage B smoke interpretation

The Stage-B smoke completed with 12 runs.

## Main numeric result

`unet.conv_in` was clearly stronger than `vae_image_latent`.

Approximate averages:

```text
unet.conv_in:
  mean Z increase ≈ 0.00249
  mean final Z ≈ 0.00370

vae_image_latent:
  mean Z increase ≈ 0.00062
  mean final Z ≈ 0.00126
```

Best numeric run:

```text
layer: unet.conv_in
prompt: add black sunglasses
image_id: barack_obama_01
Z increase ≈ 0.00623
input SSIM ≈ 0.938
max displacement ≈ 4.28 px
```

But visually, the sunglasses edit still succeeds. This is not a visible edit-failure success.

## Best visual signal

The notable visual candidate is:

```text
layer: unet.conv_in
prompt: add headphones
image_id: barack_obama_01
Z increase ≈ 0.00355
input SSIM ≈ 0.932
max displacement ≈ 4.75 px
output SSIM ≈ 0.878
```

Visual read:

- clean edit clearly adds headphones
- perturbed edit is much weaker and does not clearly preserve the same headphones
- perturbed input is still recognizable but visibly color/geometry shifted

This is promising enough for one focused 150-iteration run, but it is not a clean final result yet.

## Weak or negative signals

- `vae_image_latent` mostly produced weak movement.
- Michelle Obama cases were effectively flat in this smoke.
- Joe Biden headphones changed/weakened the edit but still looked headset-like.
- Sunglasses runs mostly preserved sunglasses.

## Next step

Run one focused Stage-B main test:

```text
layer: unet.conv_in
prompt: add headphones
image_ids:
  - barack_obama_01
  - joe_biden_01
iterations: 150
```

Commands are in:

```text
instructions/a6000_stage_b_focused_main_commands.md
```

After the focused run, inspect whether the headphone failure strengthens or whether the run only adds more input distortion.

If the focused run is promising, the next coding task is Stage-C evaluation with ArcFace / DeepFace identity metrics on original-vs-perturbed and clean-edit-vs-perturbed-edit.
