# Response 11 — Stage-B report builder

Added a FACE/WOOD-style report builder for LAYER Stage-B outputs.

## Script

```text
layer/scripts/build_report.py
```

## Default report

Input:

```text
identity_layers/outputs/stage_b_broad_400
```

Output:

```text
identity_layers/outputs/reports/stage_b_current/
  layer_stage_b_report.html
  layer_stage_b_report.md
  layer_stage_b_report.pdf
  report_data_summary.json
  missing_artifacts.md
  assets/
```

## Contents

The report includes:

- Stage-A context
- Stage-B run matrix
- aggregate layer/prompt summary
- per-run final values
- validity labels for destroyed/invalid inputs
- image strips
- input and edit difference images
- Z/loss/SSIM/PSNR/displacement/FFT graphs
- PDF and HTML outputs

## Important conclusion encoded in the report

The broad 400-iteration top `unet.conv_in` runs are marked invalid when the input is destroyed. High `Z` alone is not treated as success.

## Commands

Reusable commands are in:

```text
instructions/stage_b_report_commands.md
```
