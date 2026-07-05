# LAYER Milestone 1 A6000 commands

Run these on the Linux A6000 machine from the repo root.

The configs use the committed Wikimedia identity-probe dataset by default:

```text
identity_layers/datasets/wikimedia_identity_probe/identity_manifest.csv
```

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.inventory_instruct_layers \
  --root /home/interns/Desktop/layer \
  --mat-root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.extract_identity_activations \
  --root /home/interns/Desktop/layer \
  --mat-root /home/interns/Desktop/mat

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.score_identity_layers \
  --root /home/interns/Desktop/layer

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.summarize_identity_layer_scan \
  --root /home/interns/Desktop/layer
```

Expected main outputs:

```text
identity_layers/outputs/layer_inventory/
identity_layers/outputs/activation_scan/
identity_layers/outputs/identity_scores/
identity_layers/outputs/reports/
```

Push results after the run:

```bash
git status -sb
git add identity_layers/outputs
git commit -m "Add InstructPix2Pix identity layer scan outputs"
git push origin main
```

Milestone 1 only inventories/scans activations. It does not run a geometry attack.
