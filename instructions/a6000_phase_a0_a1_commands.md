# LAYER Phase A0/A1 commands only

Use this when you only want to test repository setup and layer inventory before running activation extraction.

These commands use the committed Wikimedia identity-probe dataset by default:

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
```

Inspect after it finishes:

```text
identity_layers/outputs/layer_inventory/environment.json
identity_layers/outputs/layer_inventory/model_config.json
identity_layers/outputs/layer_inventory/experiment_plan.json
identity_layers/outputs/layer_inventory/layer_inventory.csv
identity_layers/outputs/layer_inventory/activation_shapes.csv
identity_layers/outputs/layer_inventory/recommended_initial_layers.json
identity_layers/outputs/layer_inventory/model_tree.txt
```

Also inspect the dataset sheet once:

```text
identity_layers/datasets/wikimedia_identity_probe/dataset_contact_sheet.jpg
```

If the inventory command succeeds, then continue with the full Milestone 1 command file:

```text
instructions/a6000_milestone1_commands.md
```
