# Response 13 - Stage B preservation checkpoint run prepared

Prepared the next Stage B run requested after the constrained reports.

## What changed

Added optional Stage B support for:

- `input_preservation_weight`
- best-valid checkpoint selection
- final image generation from the best valid checkpoint instead of always using the last iteration

The old configs still behave as before because the new fields default to disabled.

## New config

```text
identity_layers/configs/stage_b_preservation_checkpoint_250.json
```

This config runs:

```text
layer:
  unet.conv_in

prompts:
  add headphones
  add black sunglasses

images:
  barack_obama_01
  joe_biden_01
  michelle_obama_01
  kamala_harris_01

iterations:
  250
```

Loss:

```text
Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))
loss = -Z + 1.0 * MSE(perturbed, original)
```

Best valid checkpoint rule:

```text
iteration >= 10
input SSIM >= 0.97
input PSNR >= 26.0
combined max displacement <= 5.0 px
score = Z increase from initial
```

FFT phase is disabled.

## Instructions

Added:

```text
instructions/a6000_stage_b_preservation_checkpoint_commands.md
```

Run that command set on the A6000.

## Expected runtime

Estimated:

```text
6-12 minutes
```

## What to look for after running

First inspect:

```text
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_decision_report.md
identity_layers/outputs/stage_b_preservation_checkpoint_250/stage_b_all_runs.csv
identity_layers/outputs/reports/stage_b_preservation_checkpoint_250/layer_stage_b_report.pdf
```

Key fields:

```text
final_source
selected_checkpoint_iter
selected_checkpoint_score
selected_checkpoint_input_ssim
selected_checkpoint_max_disp_px
Z_increase
output_ssim
output_l2
```

Decision rule:

Only treat a result as interesting if the input remains visually acceptable and the perturbed edit visibly weakens/fails. If the final edit still clearly works, call it weak/metric-only even if Z increased.
