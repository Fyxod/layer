# Response 17 - deeper modified gradient probe prepared

Prepared the next diagnostic step after the Stage B `unet.conv_in` results stayed weak.

## What changed

Updated the gradient sanity code to support multiple objective variants:

```text
pooled_cosine
mean_std_cosine
normalized_activation_mse
```

The old behavior is still the default:

```text
pooled_cosine
```

Also added support for:

```text
multiple prompts
explicit image_ids
objective_variants from config/CLI
```

## New config

```text
identity_layers/configs/gradient_scan_deeper_modified.json
```

This tests:

```text
vae_image_latent
unet.conv_in
unet.mid_block
unet.up_blocks.0
```

with:

```text
add black sunglasses
add headphones
```

and the 4 current Stage B images.

## New command file

```text
instructions/a6000_deeper_modified_gradient_probe_commands.md
```

Run this on the A6000 next.

## Why this is the right next step

The `unet.conv_in` single-layer attack now looks mostly exhausted:

```text
positive Z only on Kamala
final edit effects are weak / metric-only
```

Earlier ArcFace correlation suggested:

```text
unet.mid_block
unet.up_blocks.0
```

might contain identity-related information. The first A7 gradient probe may have rejected them because pooled cosine was too blunt for deeper activations.

This probe checks whether those deeper layers become usable under a less lossy objective before we give up on them.

## What to look for

After the run, inspect:

```text
identity_layers/outputs/gradient_scan_deeper_modified/layer_gradient_rankings.csv
identity_layers/outputs/gradient_scan_deeper_modified/selected_layers.json
identity_layers/outputs/gradient_scan_deeper_modified/gradient_failures.csv
```

Proceed only if a deeper layer/objective has:

```text
finite gradients
positive mean Z increase
nonzero gradient norm
```

If not, the deeper ArcFace-correlated layers are not practically usable with the current geometry setup.
