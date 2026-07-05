# LAYER Stage-B report commands

This builds a FACE/WOOD-style report from existing Stage-B outputs.

Default report target:

```text
identity_layers/outputs/stage_b_broad_400
```

Default output:

```text
identity_layers/outputs/reports/stage_b_current/
  layer_stage_b_report.html
  layer_stage_b_report.md
  layer_stage_b_report.pdf
  report_data_summary.json
  missing_artifacts.md
  assets/
```

## Build the current broad-run report

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.build_report \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_broad_400 \
  --output-root /home/interns/Desktop/layer/identity_layers/outputs/reports/stage_b_current
```

## Build a compressed PDF/report

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.build_report \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_broad_400 \
  --output-root /home/interns/Desktop/layer/identity_layers/outputs/reports/stage_b_current \
  --compress-report
```

## Build a report for another Stage-B folder

Change only `--results-dir` and optionally `--output-root`.

Example:

```bash
$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.build_report \
  --root /home/interns/Desktop/layer \
  --results-dir /home/interns/Desktop/layer/identity_layers/outputs/stage_b_constrained_spatial_200 \
  --output-root /home/interns/Desktop/layer/identity_layers/outputs/reports/stage_b_constrained_spatial_200
```

## Push report outputs

```bash
git status -sb
git add \
  layer/scripts/build_report.py \
  instructions/stage_b_report_commands.md \
  identity_layers/outputs/reports/stage_b_current
git commit -m "Add LAYER Stage B report"
git push origin main
```

## Notes

The report marks runs with severe input collapse as invalid. The green/destroyed broad-run images are diagnostic artifacts, not attack successes.
