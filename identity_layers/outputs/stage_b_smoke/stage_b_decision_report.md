# Stage B targeted layer smoke report

This is a short layer-targeted geometry smoke, not a full attack sweep.

## Layers tested

- `unet.conv_in`
- `vae_image_latent`

## Summary

- Runs completed: 12
- Objective: `Z = 1 - cosine_similarity(pool(layer(original)), pool(layer(perturbed)))`
- Loss: `loss = -Z`
- Trainable values: geometry parameters only

## Top runs by Z increase

| layer | image | prompt | initial Z | final Z | Z increase | input SSIM | max disp px | output SSIM |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| unet.conv_in | barack_obama_01 | add black sunglasses | 0.00166106 | 0.00788665 | 0.00622559 | 0.9379 | 4.28 | 0.9349 |
| unet.conv_in | joe_biden_01 | add black sunglasses | 0.00106478 | 0.00470781 | 0.00364304 | 0.9625 | 3.082 | 0.9625 |
| unet.conv_in | barack_obama_01 | add headphones | 0.00166106 | 0.0052104 | 0.00354934 | 0.9315 | 4.754 | 0.8784 |
| vae_image_latent | joe_biden_01 | add black sunglasses | 0.000706255 | 0.00416523 | 0.00345898 | 0.947 | 3.661 | 0.9389 |
| unet.conv_in | joe_biden_01 | add headphones | 0.00106478 | 0.00258625 | 0.00152147 | 0.9606 | 3.628 | 0.9375 |
| vae_image_latent | joe_biden_01 | add headphones | 0.000706255 | 0.000830591 | 0.000124335 | 0.9539 | 4.328 | 0.929 |
| vae_image_latent | barack_obama_01 | add headphones | 0.000736952 | 0.000790954 | 5.40018e-05 | 0.955 | 2.299 | 0.9066 |
| vae_image_latent | barack_obama_01 | add black sunglasses | 0.000736952 | 0.000790715 | 5.37634e-05 | 0.955 | 2.299 | 0.9488 |
| unet.conv_in | michelle_obama_01 | add black sunglasses | 0.000913262 | 0.000913262 | 0 | 0.9776 | 1.921 | 0.9809 |
| unet.conv_in | michelle_obama_01 | add headphones | 0.000913262 | 0.000913262 | 0 | 0.9776 | 1.921 | 0.9681 |
| vae_image_latent | michelle_obama_01 | add black sunglasses | 0.000502467 | 0.000502467 | 0 | 0.9776 | 1.921 | 0.9809 |
| vae_image_latent | michelle_obama_01 | add headphones | 0.000502467 | 0.000502467 | 0 | 0.9776 | 1.921 | 0.9681 |

## Decision note

Use this smoke only as Gate 4 evidence. Proceed to a 150-iteration run only if Z increases without numerical failure and the geometry diagnostics remain acceptable.
