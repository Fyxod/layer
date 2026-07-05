# Response 9 — Stage B broad 400-run setup

Added a controlled broader Stage-B run config.

## Config

```text
identity_layers/configs/stage_b_broad_400.json
```

Run matrix:

```text
4 layers × 2 prompts × 4 images × 400 iterations
```

Layers:

```text
unet.conv_in
vae_image_latent
unet.mid_block
unet.up_blocks.0
```

Prompts:

```text
add headphones
add black sunglasses
```

Images:

```text
barack_obama_01
joe_biden_01
michelle_obama_01
kamala_harris_01
```

The fourth image is `kamala_harris_01`.

## ETA

Based on actual Stage-B smoke timings:

```text
approximately 25–40 minutes
```

## Commands

Commands are in:

```text
instructions/a6000_stage_b_broad_400_commands.md
```

## Important interpretation note

This broader run is still controlled.

It does not mean all these layers are equally promising:

- `unet.conv_in` is the primary target.
- `vae_image_latent` is the original identity-separation target but looked weaker in Stage-B smoke.
- `unet.mid_block` and `unet.up_blocks.0` are diagnostic additions because they correlated more with ArcFace but failed the earlier gradient gate.

Do not call a run successful unless the final edited output visibly weakens/fails while the perturbed input remains acceptable.
