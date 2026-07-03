# Response 2 — A0/A1 readiness note

Do not run the previous command set until the latest commit is pulled.

The implementation was tightened after checking the full objective file:

- added named A0 metadata outputs: `environment.json`, `model_config.json`, `experiment_plan.json`
- added fixed-condition A3 metadata: `resolved_timesteps.json`, `fixed_noise_metadata.json`, `prompt_embeddings_metadata.json`, `scan_conditions.csv`
- changed identity scoring to the requested distance definition:
  `identity_separation = mean(different_identity_distance) - mean(same_identity_distance)`
- added the expected A5 output filenames:
  `layer_identity_scores.csv`, `layer_timestep_scores.csv`, `layer_prompt_scores.csv`, `verification_metrics.csv`, `ranked_layers.json`

For A0/A1 only, run:

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.inventory_instruct_layers \
  --root /home/interns/Desktop/layer \
  --mat-root /home/interns/Desktop/mat
```

After A0/A1 succeeds, run the remaining Milestone 1 commands from `instructions/a6000_milestone1_commands.md`.
