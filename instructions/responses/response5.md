# Response 5 — A6/A7 output interpretation and ArcFace rerun

Current pushed outputs complete the first A6/A7 pass.

## What A6 found

- Raw resized image and random projection baselines ranked highest by the current same/different identity split.
- The strongest InstructPix2Pix internal representation remains the image-conditioning side:
  - `vae_image_latent`
  - `unet.conv_in`
- Deeper UNet, attention, and final UNet-prediction baselines showed much smaller identity separation.
- The existing A6 output did not compute ArcFace. It wrote the placeholder:
  - `arcface_status = not_computed`

## What A7 found

- The short gradient sanity scan completed with no failures.
- Only two layers passed the practical gradient gate:
  - `vae_image_latent`
  - `unet.conv_in`
- The other tested layers had finite rows but zero useful gradient increase in the short probe.

## Next step

Because ArcFace is available on the system, rerun only A6 with ArcFace enabled:

```bash
cd /home/interns/Desktop/layer
git pull origin main

$HOME/.local/bin/micromamba run -p /home/interns/Desktop/mat/.micromamba/envs/mat-a6000 \
  python -m layer.scripts.run_baseline_comparison \
  --root /home/interns/Desktop/layer \
  --compute-arcface \
  --face-repo /home/interns/Desktop/face \
  --arcface-checkpoint /home/interns/Desktop/face/models/arcface/iresnet100.pth
```

Then push:

```bash
git status -sb
git add identity_layers/outputs/baseline_comparison
git commit -m "Add ArcFace baseline comparison for identity layers"
git push origin main
```

Do not run a full Stage-B attack until the ArcFace baseline/correlation is checked.
