# Response 15 - Stage B spatial best-valid 400 run prepared

Prepared the next requested command/config set.

## New config

```text
identity_layers/configs/stage_b_spatial_bestvalid_400.json
```

This run uses:

```text
layer:
  unet.conv_in

loss:
  loss = -Z

FFT:
  disabled

iterations:
  400

checkpoint selection:
  enabled
```

It keeps final artifacts from the best valid checkpoint:

```text
iteration >= 1
input SSIM >= 0.97
input PSNR >= 26.0
combined max displacement <= 5.0 px
score = Z increase from initial
```

## Commands

Added:

```text
instructions/a6000_stage_b_spatial_bestvalid_400_commands.md
```

Run that command file on the A6000.

## Expected runtime

```text
8 runs x 400 iterations
roughly 10-20 minutes
```

## Main reason for this run

The previous preservation run proved:

```text
input_preservation_weight = 1.0 is too strong
```

It made all Z increases negative.

This run returns to the spatial-only `loss = -Z` setup that actually moved the Kamala cases, but uses best-valid checkpoint selection so the saved final images are not just the last iteration.

## What to inspect after push

```text
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_top_sheet.jpg
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_all_runs.csv
identity_layers/outputs/stage_b_spatial_bestvalid_400/stage_b_decision_report.md
identity_layers/outputs/reports/stage_b_spatial_bestvalid_400/layer_stage_b_report.pdf
```

Call it interesting only if the perturbation remains visually acceptable and the perturbed edit visibly weakens/fails.
